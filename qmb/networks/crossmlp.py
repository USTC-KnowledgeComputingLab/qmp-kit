"""
This file implements a cross MLP network.
"""

import itertools
import typing
import torch
from ..bitspack import unpack_int


class FakeLinear(torch.nn.Module):
    """
    A fake linear layer with zero input dimension to avoid PyTorch initialization warnings.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        assert dim_in == 0
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.zeros([dim_out]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the fake linear layer.
        """
        batch, _ = x.shape
        return self.bias.view([1, -1]).expand([batch, -1])


def select_linear_layer(dim_in: int, dim_out: int) -> torch.nn.Module:
    """
    Selects between a fake linear layer and a standard one to avoid initialization warnings when dim_in is zero.
    """
    if dim_in == 0:  # pylint: disable=no-else-return
        return FakeLinear(dim_in, dim_out)
    else:
        return torch.nn.Linear(dim_in, dim_out)


class MLP(torch.nn.Module):
    """
    This module implements multiple layers MLP with given dim_input, dim_output and hidden_size.
    """

    def __init__(self, dim_input: int, dim_output: int, hidden_size: tuple[int, ...]) -> None:
        super().__init__()
        self.dim_input: int = dim_input
        self.dim_output: int = dim_output
        self.hidden_size: tuple[int, ...] = hidden_size

        dimensions: list[int] = [dim_input] + list(hidden_size) + [dim_output]
        linears: list[torch.nn.Module] = [select_linear_layer(i, j) for i, j in itertools.pairwise(dimensions)]
        modules: list[torch.nn.Module] = [module for linear in linears for module in (linear, torch.nn.SiLU())][:-1]
        self.model: torch.nn.Module = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.
        """
        return self.model(x)


class WaveFunction(torch.nn.Module):
    """
    The wave function for the cross MLP network.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            sites: int,  # Number of qubits
            physical_dim: int,  # Dimension of the physical space, which is always 2 for MLP
            is_complex: bool,  # Indicates whether the wave function is complex-valued, which is always true for MLP
            embedding_hidden_size: tuple[int, ...],  # Hidden layer sizes for embedding part
            embedding_size: int,  # The dimension of the embedding
            momentum_hidden_size: tuple[int, ...],  # Hidden layer sizes for momentum part
            momentum_count: int,  # The number of max momentum order
            tail_hidden_size: tuple[int, ...],  # Hidden layer size for tail part
            kind: typing.Literal[0, 1, 2],  # Kind of the crossmlp forward function
            ordering: int | list[int],  # Ordering of sites: +1 for normal order, -1 for reversed order, or a custom order list
    ) -> None:
        super().__init__()
        self.sites: int = sites
        assert physical_dim == 2
        # This module is only used in reinforcement learning, which expects real values for the weights.
        assert is_complex == False  # pylint: disable=singleton-comparison
        self.embedding_hidden_size: tuple[int, ...] = embedding_hidden_size
        self.embedding_size: int = embedding_size
        self.momentum_hidden_size: tuple[int, ...] = momentum_hidden_size
        self.momentum_count: int = momentum_count
        self.tail_hidden_size: tuple[int, ...] = tail_hidden_size
        self.kind: typing.Literal[0, 1, 2] = kind

        self.emb = MLP(self.sites, self.embedding_size, self.embedding_hidden_size)
        self.momentum = torch.nn.ModuleList([MLP(self.embedding_size, self.embedding_size, momentum_hidden_size) for _ in range(self.momentum_count)])
        self.tail = MLP(self.embedding_size, 1, tail_hidden_size)

        # Site Ordering Configuration
        # +1 for normal order, -1 for reversed order
        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.ordering: torch.Tensor
        self.register_buffer("ordering", torch.tensor(ordering, dtype=torch.int64))
        self.ordering_reversed: torch.Tensor
        self.register_buffer("ordering_reversed", torch.scatter(torch.zeros(self.sites, dtype=torch.int64), 0, self.ordering, torch.arange(self.sites, dtype=torch.int64)))

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the wave function psi for the given configurations.
        """
        dtype = self.dummy_param.dtype
        # x: batch_size * sites
        x = unpack_int(x, size=1, last_dim=self.sites)
        # Apply ordering
        x = torch.index_select(x, 1, self.ordering_reversed)
        # Dtype conversion
        x = x.to(dtype=dtype)

        # emb: batch_size * embedding_size
        emb = self.emb(x)

        if self.kind == 0:
            # x' = F(x - E[x]) + x
            for layer in self.momentum:
                new_emb = emb - emb.mean(dim=0, keepdim=True)
                new_emb = layer(new_emb)
                emb = emb + new_emb
                emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        elif self.kind == 1:
            # x' = F(x) - E[F(x)] + x
            for layer in self.momentum:
                new_emb = layer(emb)
                new_emb = new_emb - new_emb.mean(dim=0, keepdim=True)
                emb = emb + new_emb
                emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        elif self.kind == 2:
            # x' = (F(x) + x) - E [F(x) + x]
            for layer in self.momentum:
                new_emb = layer(emb)
                new_emb = new_emb + emb
                emb = new_emb - new_emb.mean(dim=0, keepdim=True)
                emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        else:
            raise ValueError(f"Invalid kind: {self.kind}")

        tail = self.tail(emb).squeeze(-1)
        return tail

    @torch.jit.export
    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        This module does not support generating unique configurations.
        """
        # This module is only used in reinforcement learning, which does not require configurations sampling.
        raise NotImplementedError("The generate_unique method is not implemented for this class.")
