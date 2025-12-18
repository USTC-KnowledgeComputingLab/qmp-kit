"""
This file implements the PEPS tensor network.
"""

import torch
from ..bitspack import unpack_int


class PEPS(torch.nn.Module):
    """
    The PEPS tensor network.
    """

    # pylint: disable=invalid-name

    def __init__(self, L1: int, L2: int, d: int, D: int, Dc: int, use_complex: bool = False) -> None:  # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self.L1: int = L1
        self.L2: int = L2
        self.d: int = d
        self.D: int = D
        self.Dc: int = Dc
        self.use_complex: bool = use_complex

        self.tensors = torch.nn.Parameter(
            torch.randn([L1, L2, d, D, D, D, D], dtype=torch.complex128 if use_complex else torch.float64)
        )

    def _tensor(self, l1: int, l2: int, config: torch.Tensor) -> torch.Tensor:
        """
        Get the tensor for a specific lattice site (l1, l2) and configuration.
        """
        # pylint: disable=unsubscriptable-object
        # Order: L, U, D, R
        tensor: torch.Tensor = self.tensors[l1, l2, config.to(torch.int64)]
        if l2 == 0:
            tensor = tensor[:, :1, :, :, :]
        if l1 == 0:
            tensor = tensor[:, :, :1, :, :]
        if l1 == self.L1 - 1:
            tensor = tensor[:, :, :, :1, :]
        if l2 == self.L2 - 1:
            tensor = tensor[:, :, :, :, :1]
        return tensor

    def _bmps(self, line1: list[torch.Tensor], line2: list[torch.Tensor]) -> list[torch.Tensor]:
        # pylint: disable=too-many-locals
        # tensor in double: blLudrR
        double = [torch.einsum("blumr,bLmdR->blLudrR", tensor1, tensor2) for tensor1, tensor2 in zip(line1, line2)]
        # Merge two left index for the first tensor
        # tensor shape should be: bludrR
        double[0] = double[0].flatten(1, 2)
        for l2 in range(self.L2 - 1):
            # tensor shape: bludrR
            # b for batch
            # lud for q tensor
            # rR for r tensor
            tensor = double[l2]
            b, l, u, d, r, R = tensor.shape
            tensor = tensor.reshape([b, l * u * d, r * R])
            q_tensor, r_tensor = torch.linalg.qr(tensor, mode="reduced")  # pylint: disable=not-callable
            double[l2] = q_tensor.reshape([b, l, u, d, -1])
            remain = r_tensor.reshape([b, -1, r, R])
            double[l2 + 1] = torch.einsum("blmM,bmMudrR->bludrR", remain, double[l2 + 1])
        # Merge two right index for the last tensor
        double[-1] = double[-1].flatten(4, 5)
        # tensor shape is: bludr
        for l2 in range(self.L2 - 1, 0, -1):
            # tensor shape: bludr
            # b for batch
            # l for u tensor
            # udr for v tensor
            tensor = double[l2]
            b, l, u, d, r = tensor.shape
            tensor = tensor.reshape([b, l, u * d * r])
            u_tensor, s_tensor, v_tensor = torch.linalg.svd(tensor, full_matrices=False)  # pylint: disable=not-callable
            middle_size = s_tensor.shape[-1]
            if middle_size > self.Dc:
                u_tensor = u_tensor[:, :, : self.Dc]
                s_tensor = s_tensor[:, : self.Dc]
                v_tensor = v_tensor[:, : self.Dc, :]
            double[l2] = v_tensor.reshape([b, -1, u, d, r])
            double[l2 - 1] = torch.einsum("bludm,bmr,br->bludr", double[l2 - 1], u_tensor, s_tensor)
        # tensor shape is still: b l u d r
        return double

    def _contract(self, tensors: list[list[torch.Tensor]]) -> torch.Tensor:
        candidates = tensors[0]
        for l1 in range(1, self.L1):
            candidates = self._bmps(candidates, tensors[l1])
        result = candidates[0]
        for l2 in range(1, self.L2):
            result = result * candidates[l2]
        return result[:, 0, 0, 0, 0]

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PEPS tensor network.
        """
        tensors: list[list[torch.Tensor]] = [
            [self._tensor(l1, l2, configs[:, (l1 * self.L2) + l2]) for l2 in range(self.L2)] for l1 in range(self.L1)
        ]
        return self._contract(tensors)


class PepsFunction(torch.nn.Module):
    """
    The PEPS tensor network used by qmp interface.
    """

    def __init__(self, L1: int, L2: int, d: int, D: int, Dc: int, use_complex: bool = False) -> None:  # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        assert d == 2
        self.sites = L1 * L2
        self.model = PEPS(L1, L2, d, D, Dc, use_complex)

    @torch.jit.export
    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate a batch of unique configurations.
        """
        raise NotImplementedError("The generate_unique method is not implemented for this class.")

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PEPS tensor network.
        """
        x = unpack_int(x, size=1, last_dim=self.sites)
        return self.model(x)
