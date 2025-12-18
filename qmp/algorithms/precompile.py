"""
This file precompiles essential extensions to run specific model.
"""

import typing
import dataclasses
import torch
import tyro
from ..model_dict import model_dict, ModelProto, NetworkProto
from ..subcommand_dict import subcommand_dict


@dataclasses.dataclass
class PrecompileConfig:
    """
    The precompilation for solving quantum many-body problems.
    """

    # The model name
    model_name: typing.Annotated[str, tyro.conf.Positional, tyro.conf.arg(metavar="MODEL")]
    # Arguments for physical model
    physics_args: typing.Annotated[tuple[str, ...], tyro.conf.arg(aliases=["-P"]), tyro.conf.UseAppendAction] = ()
    # The device to run on
    device: typing.Annotated[torch.device, tyro.conf.arg(aliases=["-D"])] = torch.device(type="cuda", index=0)

    def main(self) -> None:
        """
        The main function for precompilation.
        """

        model_t = model_dict[self.model_name]
        model_config_t = model_t.config_t
        network_config_t = model_t.network_dict["mlp"]
        model: ModelProto = model_t(tyro.cli(model_config_t, args=self.physics_args))
        network: NetworkProto = network_config_t().create(model).to(device=self.device)
        configs_i, psi_i, _, _ = network.generate_unique(1)
        model.apply_within(configs_i, psi_i, configs_i)
        model.find_relative(configs_i, psi_i, 1)


subcommand_dict["precompile"] = PrecompileConfig
