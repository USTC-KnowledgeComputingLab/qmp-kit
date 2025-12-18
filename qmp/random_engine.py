"""
Random engine utilities.
"""

import torch


def dump_random_engine_state(device: torch.device) -> torch.Tensor:
    """
    Dump the random engine state for the specified device.
    """
    match device.type:
        case "cpu":
            return torch.get_rng_state()
        case "cuda":
            return torch.cuda.get_rng_state(device)
        case _:
            raise ValueError(f"Unknown device: {device}")


def load_random_engine_state(state: torch.Tensor, device: torch.device) -> None:
    """
    Load the random engine state for the specified device.
    """
    match device.type:
        case "cpu":
            torch.set_rng_state(state)
        case "cuda":
            torch.cuda.set_rng_state(state, device)
        case _:
            raise ValueError(f"Unknown device: {device}")
