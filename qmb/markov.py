"""
This file implements a variational Monte Carlo method for solving quantum many-body problems.
"""

import logging
import typing
import dataclasses
import torch
import torch.utils.tensorboard
import tyro
from .common import CommonConfig
from .subcommand_dict import subcommand_dict
from .optimizer import initialize_optimizer


def markov_chain(model, network, initial_config: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Perform Markov chain sampling to generate new configurations, and return a list of chain configurations.
    """
    configs = [initial_config]
    current_config = initial_config
    current_psi = network(current_config.unsqueeze(0)).squeeze(0)
    for _ in range(steps):
        proposed_config, lold, lnew = model.hopping(current_config)
        proposed_psi = network(proposed_config.unsqueeze(0)).squeeze(0)
        acceptance_ratio = (proposed_psi.abs() ** 2) / (current_psi.abs() ** 2)
        acceptance_ratio = acceptance_ratio.clamp(max=1.0)
        if torch.rand(1).item() < acceptance_ratio.item():
            current_config = proposed_config
            current_psi = proposed_psi
        configs.append(current_config)
    return torch.stack(configs)

@dataclasses.dataclass
class MarkovConfig:
    """
    The Markov VMC optimization for solving quantum many-body problems.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # The number of relative configurations to be used in energy calculation
    relative_count: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 40000
    # Whether to use the global optimizer
    global_opt: typing.Annotated[bool, tyro.conf.arg(aliases=["-g"])] = False
    # Whether to use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # The number of steps for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"])] = 1000

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3

    def main(self, *, model_param: typing.Any = None, network_param: typing.Any = None) -> None:
        """
        The main function for the Markov VMC optimization.
        """
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-locals

        model, network, data = self.common.main(model_param=model_param, network_param=network_param)

        logging.info(
            "Arguments Summary: "
            "Sampling Count: %d, "
            "Relative Count: %d, "
            "Global Optimizer: %s, "
            "Use LBFGS: %s, "
            "Learning Rate: %.10f, "
            "Local Steps: %d, ",
            self.sampling_count,
            self.relative_count,
            "Yes" if self.global_opt else "No",
            "Yes" if self.use_lbfgs else "No",
            self.learning_rate,
            self.local_step,
        )

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=self.use_lbfgs,
            learning_rate=self.learning_rate,
            state_dict=data.get("optimizer"),
        )

        if "markov" not in data:
            data["markov"] = {"global": 0, "local": 0}

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.common.folder())  # type: ignore[no-untyped-call]

        def init(L, N):
            a =N//2
            b=N-a
            from .bitspack import unpack_int, pack_int
            x = torch.zeros(L, dtype=torch.uint8)
            x[torch.randperm(L)[:a]] = 1
            y = torch.zeros(L, dtype=torch.uint8)
            y[torch.randperm(L)[:b]] = 1
            return pack_int(torch.stack([x, y], dim=1).reshape([1, -1]), size=1)[0]


        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            configs_i = markov_chain(
                model,
                network,
                initial_config=init(model.m * model.n, model.electron_number).to(self.common.device),
                steps=self.sampling_count,
            )
            psi_i = network(configs_i)
            logging.info("Sampling completed, unique configurations count: %d", len(configs_i))

            logging.info("Calculating relative configurations")
            if self.relative_count <= len(configs_i):
                configs_src = configs_i
                configs_dst = configs_i
            else:
                configs_src = configs_i
                configs_dst = torch.cat([configs_i, model.find_relative(configs_i, psi_i, self.relative_count - len(configs_i))])
            logging.info("Relative configurations calculated, count: %d", len(configs_dst))

            optimizer = initialize_optimizer(
                network.parameters(),
                use_lbfgs=self.use_lbfgs,
                learning_rate=self.learning_rate,
                new_opt=not self.global_opt,
                optimizer=optimizer,
            )

            def closure() -> torch.Tensor:
                # Optimizing energy
                optimizer.zero_grad()
                psi_src = network(configs_src)
                with torch.no_grad():
                    psi_dst = network(configs_dst)
                    hamiltonian_psi_dst = model.apply_within(configs_dst, psi_dst, configs_src)
                num = psi_src.conj() @ hamiltonian_psi_dst
                den = psi_src.conj() @ psi_src.detach()
                energy = num / den
                energy = energy.real
                energy.backward()  # type: ignore[no-untyped-call]
                return energy

            logging.info("Starting local optimization process")

            for i in range(self.local_step):
                energy: torch.Tensor = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                logging.info("Local optimization in progress, step: %d, energy: %.10f, ref energy: %.10f", i, energy.item(), model.ref_energy)
                writer.add_scalar("markov/energy", energy, data["markov"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("markov/error", energy - model.ref_energy, data["markov"]["local"])  # type: ignore[no-untyped-call]
                data["markov"]["local"] += 1
                for config in  configs_src:
                    psi = network(config.unsqueeze(0)).squeeze(0)
                    print(model.show_config(config), psi.item())

            logging.info("Local optimization process completed")

            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["markov"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            self.common.save(data, data["markov"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["markov"] = MarkovConfig
