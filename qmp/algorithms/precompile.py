"""
This file precompiles essential extensions to run specific model.
"""

import typing
import dataclasses
import torch
from ..utility.model_dict import model_dict, ModelProto, NetworkProto
from ..utility.subcommand_dict import subcommand_dict


@dataclasses.dataclass
class PrecompileConfig:
    """
    The precompilation for solving quantum many-body problems.
    """

    # The model name
    model_name: str
    # Arguments for physical model
    physics_args: tuple[str, ...] = ()
    # The device to run on
    device: torch.device = torch.device(type="cuda", index=0)

    def main(self, *, model_param: typing.Any = None) -> None:
        """
        The main function for precompilation.
        """

        model_t = model_dict[self.model_name]
        network_config_t = model_t.network_dict["mlp"]
        if model_param is None:
            raise ValueError(
                "model_param must be provided when calling main(). "
                "This should be an instance of the model's config class, "
                "typically created from Hydra configuration."
            )
        model: ModelProto = model_t(model_param)
        network: NetworkProto = network_config_t().create(model).to(device=self.device)
        configs_i, psi_i, _, _ = network.generate_unique(1)
        model.apply_within(configs_i, psi_i, configs_i)
        model.find_relative(configs_i, psi_i, 1)


subcommand_dict["precompile"] = PrecompileConfig
