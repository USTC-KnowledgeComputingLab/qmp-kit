"""
This file implements a reinforcement learning based imaginary time evolution algorithm.
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


@dataclasses.dataclass
class RlimConfig:
    """
    The reinforcement learning based imaginary time evolution algorithm.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # The number of relative configurations to be used in energy calculation
    relative_count: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 40000
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"])] = 1e-3
    # The learning rate for the imaginary time evolution
    evolution_time: typing.Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1e-3
    # The number of steps for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"])] = 32
    # The dropout of the loss function
    dropout: typing.Annotated[float, tyro.conf.arg(aliases=["-d"])] = 0.5

    def main(self) -> None:
        """
        The main function for the RLIM optimization.
        """
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-locals

        model, network, data = self.common.main()
        ref_network = network

        logging.info(
            "Arguments Summary: "
            "Sampling Count: %d, "
            "Relative Count: %d, "
            "Learning Rate: %.10f, "
            "Evolution Time: %.10f, "
            "Local Steps: %d, "
            "Dropout: %.2f",
            self.sampling_count,
            self.relative_count,
            self.learning_rate,
            self.evolution_time,
            self.local_step,
            self.dropout,
        )

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=False,
            learning_rate=self.learning_rate,
            state_dict=data.get("optimizer"),
        )

        if "rlim" not in data:
            data["rlim"] = {"global": 0, "local": 0}

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.common.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            configs_i, psi_i, _, _ = network.generate_unique(self.sampling_count)
            ref_configs_i, ref_psi_i, _, _ = ref_network.generate_unique(self.sampling_count)
            logging.info("Sampling completed, unique configurations count: %d, reference unique configurations count: %d", len(configs_i), len(ref_configs_i))

            logging.info("Calculating relative configurations")
            if self.relative_count <= len(configs_i):
                configs_src = configs_i
                configs_dst = configs_i
            else:
                configs_src = configs_i
                configs_dst = torch.cat([configs_i, model.find_relative(configs_i, psi_i, self.relative_count - len(configs_i))])
            logging.info("Relative configurations calculated, count: %d", len(configs_dst))
            if self.relative_count <= len(ref_configs_i):
                ref_configs_src = ref_configs_i
                ref_configs_dst = ref_configs_i
            else:
                ref_configs_src = ref_configs_i
                ref_configs_dst = torch.cat([ref_configs_i, model.find_relative(ref_configs_i, ref_psi_i, self.relative_count - len(ref_configs_i))])
            logging.info("Reference relative configurations calculated, count: %d", len(ref_configs_dst))

            def closure() -> torch.Tensor:
                # Optimizing loss
                optimizer.zero_grad()
                psi_src = network(configs_src)  # psi s
                ref_psi_src = network(ref_configs_src)  # psi r
                with torch.no_grad():
                    psi_dst = network(configs_dst)  # psi s'
                    ref_psi_dst = network(ref_configs_dst)  # psi r'
                    hamiltonian_psi_dst = model.apply_within(configs_dst, psi_dst, configs_src)  # H ss' psi s'
                    ref_hamiltonian_psi_dst = model.apply_within(ref_configs_dst, ref_psi_dst, ref_configs_src)  # H rr' psi r'
                a = torch.outer(psi_src.detach(), ref_psi_src) - torch.outer(psi_src, ref_psi_src.detach())
                b = torch.outer(hamiltonian_psi_dst, ref_psi_src) - torch.outer(psi_src, ref_hamiltonian_psi_dst)
                diff = torch.nn.functional.dropout(torch.view_as_real(a - self.evolution_time * b).abs(), p=self.dropout).flatten()
                loss = diff @ diff
                loss.backward()  # type: ignore[no-untyped-call]
                # Calculate energy
                with torch.no_grad():
                    num = psi_src.conj() @ hamiltonian_psi_dst
                    den = psi_src.conj() @ psi_src
                    energy = (num / den).real
                loss.energy = energy  # type: ignore[attr-defined]
                return loss

            logging.info("Starting local optimization process")

            for i in range(self.local_step):
                loss: torch.Tensor = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                energy: float = loss.energy  # type: ignore[attr-defined]
                logging.info("Local optimization in progress, step: %d, loss: %.10f, energy: %.10f, ref energy: %.10f, energy error: %.10f", i, loss.item(), energy, model.ref_energy,
                             energy - model.ref_energy)
                writer.add_scalar("rlim/energy", energy, data["rlim"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("rlim/error", energy - model.ref_energy, data["rlim"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("rlim/loss", loss, data["rlim"]["local"])  # type: ignore[no-untyped-call]
                data["rlim"]["local"] += 1

            logging.info("Local optimization process completed")

            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["rlim"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            self.common.save(data, data["rlim"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["rlim"] = RlimConfig
