"""
This file implements the reinforcement learning based subspace diagonalization algorithm.
"""

import logging
import typing
import dataclasses
import functools
import tyro
import scipy
import torch
from ..common import CommonConfig
from ..subcommand_dict import subcommand_dict
from ..model_dict import ModelProto
from ..optimizer import initialize_optimizer
from ..bitspack import pack_int


def lanczos_energy(model: ModelProto, configs: torch.Tensor, step: int, threshold: float) -> tuple[float, torch.Tensor]:
    """
    Calculate the energy using the Lanczos method.
    """
    vector = torch.randn([configs.size(0)], dtype=torch.complex128, device=configs.device)

    v: list[torch.Tensor] = [vector / torch.linalg.norm(vector)]  # pylint: disable=not-callable
    alpha: list[torch.Tensor] = []
    beta: list[torch.Tensor] = []
    w: torch.Tensor
    w = model.apply_within(configs, v[-1], configs)
    alpha.append((w.conj() @ v[-1]).real)
    w = w - alpha[-1] * v[-1]
    i = 0
    while True:
        norm_w = torch.linalg.norm(w)  # pylint: disable=not-callable
        if norm_w < threshold:
            break
        beta.append(norm_w)
        v.append(w / beta[-1])
        w = model.apply_within(configs, v[-1], configs)
        alpha.append((w.conj() @ v[-1]).real)
        if i == step:
            break
        w = w - alpha[-1] * v[-1] - beta[-1] * v[-2]
        v[-2] = v[-2].cpu()  # v maybe very large, so we need to move it to CPU
        i += 1

    if len(beta) == 0:
        return alpha[0].item(), v[0]
    vals, vecs = scipy.linalg.eigh_tridiagonal(torch.stack(alpha, dim=0).cpu(), torch.stack(beta, dim=0).cpu(), lapack_driver="stebz", select="i", select_range=(0, 0))
    energy = torch.as_tensor(vals[0])
    result = functools.reduce(torch.add, (weight[0] * vector.to(device=configs.device) for weight, vector in zip(vecs, v)))
    return energy.item(), result


def config_contributions(model: ModelProto, base_configs: torch.Tensor, active_configs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """
    Calculate the config contribution for the given configurations pool of the ground state approximation.
    """
    base_state = state[:base_configs.size(0)]
    active_state = state[base_configs.size(0):]
    num = (active_state.conj() * model.apply_within(base_configs, base_state, active_configs)).real * 2
    den = (base_state.conj() @ base_state).real
    result = num / den
    return result


@dataclasses.dataclass
class RldiagConfig:
    """
    The reinforcement learning based subspace diagonalization algorithm.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The initial configuration for the first step, which is usually the Hatree-Fock state for quantum chemistry system
    initial_config: typing.Annotated[typing.Optional[str], tyro.conf.arg(aliases=["-i"])] = None
    # The maximum size of the configuration pool
    max_pool_size: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 32768
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"])] = 1e-3
    # The step of lanczos iteration for calculating the energy
    lanczos_step: typing.Annotated[int, tyro.conf.arg(aliases=["-k"])] = 64
    # The thereshold for the lanczos iteration
    lanczos_threshold: typing.Annotated[float, tyro.conf.arg(aliases=["-d"])] = 1e-8
    # The coefficient of configuration number for the sigma calculation
    alpha: typing.Annotated[float, tyro.conf.arg(aliases=["-a"])] = 0.0

    def main(self, *, model_param: typing.Any = None, network_param: typing.Any = None) -> None:
        """
        The main function for the reinforcement learning based subspace diagonalization algorithm.
        """

        # pylint: disable=too-many-statements
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches

        model, network, data = self.common.main(model_param=model_param, network_param=network_param)

        logging.info(
            "Arguments Summary: "
            "Initial Configuration: %s, "
            "Learning Rate: %.10f, "
            "Lanczos step: %d, "
            "Lanczos threshold: %.10f, "
            "Alpha: %.10f",
            self.initial_config if self.initial_config is not None else "None",
            self.learning_rate,
            self.lanczos_step,
            self.lanczos_threshold,
            self.alpha,
        )

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=False,
            learning_rate=self.learning_rate,
            state_dict=data.get("optimizer"),
        )

        if self.initial_config is None:
            if "rldiag" not in data:  # pylint: disable=no-else-raise
                raise ValueError("The initial configuration is not set, please set it.")
            else:
                configs = data["rldiag"]["configs"].to(device=self.common.device)
        else:
            # The parameter initial_config accepts two formats:
            # 1. The 0/1 string, such as "11111100000000000000000000000011110000000000110000110000"
            # 2. The packed string, such as "63,0,0,192,3,48,12"
            if all(i in "01" for i in self.initial_config):
                # The 0/1 string
                configs = pack_int(
                    torch.tensor([[int(i) for i in self.initial_config]], dtype=torch.bool, device=self.common.device),
                    size=1,
                )
            else:
                # The packed string
                configs = torch.tensor([[int(i) for i in self.initial_config.split(",")]], dtype=torch.uint8, device=self.common.device)
            if "rldiag" not in data:
                data["rldiag"] = {"global": 0, "local": 0, "configs": configs}
            else:
                data["rldiag"]["configs"] = configs
                data["rldiag"]["local"] = 0

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.common.folder())  # type: ignore[no-untyped-call]

        last_state = None
        while True:
            logging.info("Starting a new cycle")

            logging.info("Evaluating each configuration")
            score = network(configs)
            logging.info("All configurations are evaluated")

            logging.info("Applying the action")
            # |  old config pool  |
            # | pruned | remained | expanded |
            #          |   new config pool   |
            action = score.real >= -self.alpha
            _, topk = torch.topk(score.real, k=score.size(0) // 2, dim=0)
            action[topk] = True
            if score.size(0) > self.max_pool_size:
                _, topk = torch.topk(-score.real, k=score.size(0) - self.max_pool_size)
                action[topk] = False
            action[0] = True
            remained_configs = configs[action]
            pruned_configs = configs[torch.logical_not(action)]
            expanded_configs = model.single_relative(remained_configs)  # There are duplicated config here
            effective_expanded_configs, previous_to_effective = torch.unique(expanded_configs, dim=0, return_inverse=True)
            old_configs = torch.cat([remained_configs, pruned_configs])
            new_configs = torch.cat([remained_configs, effective_expanded_configs])
            configs = new_configs
            logging.info("Action has been applied")

            configs_size = configs.size(0)
            logging.info("Configuration pool size: %d", configs_size)
            writer.add_scalar("rldiag/configs/global", configs_size, data["rldiag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/configs/local", configs_size, data["rldiag"]["local"])  # type: ignore[no-untyped-call]

            if last_state is not None:
                old_state = last_state[torch.cat([action.nonzero()[:, 0], torch.logical_not(action).nonzero()[:, 0]])]
            else:
                _, old_state = lanczos_energy(model, old_configs, self.lanczos_step, self.lanczos_threshold)
            new_energy, new_state = lanczos_energy(model, configs, self.lanczos_step, self.lanczos_threshold)
            last_state = new_state
            energy = new_energy
            logging.info("Current energy is %.10f, Reference energy is %.10f, Energy error is %.10f", energy, model.ref_energy, energy - model.ref_energy)
            writer.add_scalar("rldiag/energy/state/global", energy, data["rldiag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/energy/state/local", energy, data["rldiag"]["local"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/energy/error/global", energy - model.ref_energy, data["rldiag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/energy/error/local", energy - model.ref_energy, data["rldiag"]["local"])  # type: ignore[no-untyped-call]
            if "base" not in data["rldiag"]:
                # This is the first time to calculate the energy, which is usually the energy of the Hatree-Fock state for quantum chemistry system
                # This will not be flushed acrossing different cycle chains.
                data["rldiag"]["base"] = energy

            old_contribution = config_contributions(model, remained_configs, pruned_configs, old_state)
            effective_new_contribution = config_contributions(model, remained_configs, effective_expanded_configs, new_state)
            new_contribution = effective_new_contribution[previous_to_effective]
            contribution = torch.cat([old_contribution, new_contribution])
            configs_for_training = torch.cat([pruned_configs, remained_configs])
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                score_for_training = network(configs_for_training)
                loss = torch.linalg.norm(score_for_training + contribution)  # pylint: disable=not-callable
                optimizer.zero_grad()
                loss.backward()
            optimizer.step()  # pylint: disable=no-value-for-parameter

            logging.info("Saving model checkpoint")
            data["rldiag"]["configs"] = configs
            data["rldiag"]["energy"] = energy
            data["rldiag"]["global"] += 1
            data["rldiag"]["local"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            self.common.save(data, data["rldiag"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current cycle completed")


subcommand_dict["rldiag"] = RldiagConfig
