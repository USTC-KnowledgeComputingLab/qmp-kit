"""
This file implements a VMC method based on the Markov chain for solving quantum many-body problems.
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
from .bitspack import pack_int, unpack_int


@dataclasses.dataclass
class MarkovConfig:
    """
    The VMC optimization based on the Markov chain for solving quantum many-body problems.
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
    # The initial configurations for the first step
    initial_config: typing.Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3

    def main(self) -> None:
        """
        The main function for the VMC optimization based on the Markov chain.
        """
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-locals

        model, network, data = self.common.main()

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
            data["markov"] = {"global": 0, "local": 0, "pool": None}

        # TODO: 如何确认大小?
        configs = pack_int(
            torch.tensor([[int(i) for i in self.initial_config]], dtype=torch.uint8, device=self.common.device),
            size=1,
        )
        if data["markov"]["pool"] is None:
            data["markov"]["pool"] = configs
            logging.info("The initial configuration is imported successfully.")
        else:
            logging.info("The initial configuration is provided, but the pool from the last iteration is not empty, so the initial configuration will be ignored.")

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.common.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Checking the configuration pool")
            config = data["markov"]["pool"]
            old_config = config.repeat(self.sampling_count // config.shape[0] + 1, 1)[:self.sampling_count]

            logging.info("Hopping configurations")
            def hop(config):
                # TODO: use hamiltonian
                x = unpack_int(config, size=1, last_dim=model.m * model.n)
                x = x.view(-1, model.m, model.n)
                batch, L1, L2 = x.shape
                swap_dim = torch.randint(0, 2, (batch,))

                out = x.clone()
                for b in range(batch):
                    if swap_dim[b] == 0:  # 在 L1 方向交换
                        i = torch.randint(0, L1 - 1, (1,)).item()
                        j = torch.randint(0, L2, (1,)).item()
                        out[b, i, j], out[b, i+1, j] = x[b, i+1, j], x[b, i, j]
                    else:  # 在 L2 方向交换
                        i = torch.randint(0, L1, (1,)).item()
                        j = torch.randint(0, L2 - 1, (1,)).item()
                        out[b, i, j], out[b, i, j+1] = x[b, i, j+1], x[b, i, j]
                x = out.view(-1, model.m * model.n)
                return pack_int(x, size=1)
            new_config = hop(old_config)
            old_weight = network(old_config)
            new_weight = network(new_config)
            accept_prob = (new_weight / old_weight).abs().clamp(max=1)**2
            accept = torch.rand_like(accept_prob) < accept_prob
            configs = torch.where(accept.unsqueeze(-1), new_config, old_config)
            configs_i = torch.unique(configs, dim=0)
            psi_i = network(configs_i)
            data["markov"]["pool"] = configs_i
            logging.info("Sampling completed, configurations count: %d", len(configs_i))

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
