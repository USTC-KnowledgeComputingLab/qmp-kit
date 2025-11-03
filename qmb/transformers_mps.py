"""
This file implements the PEPS tensor network.
"""

import torch
from .bitspack import unpack_int

class BinaryTransformerEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        # 0 和 1 各有一个 embedding 向量
        self.embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        
        # Transformer Encoder 层
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True  # 让输入输出都是 [batch, seq, emb]
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [batch, seq], 元素为 0 或 1
        返回: [batch, seq, embedding_dim]
        """
        emb = self.embedding(x.to(dtype=torch.int64))                # [batch, seq, embedding_dim]
        out = self.encoder(emb)                # [batch, seq, embedding_dim]
        return out

class MPS(torch.nn.Module):
    """
    The MPS tensor network.
    """

    # pylint: disable=invalid-name

    def __init__(self, L: int, d: int, D: int, use_complex: bool = False) -> None:  # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self.L: int = L
        self.d: int = d
        self.D: int = D
        self.use_complex: bool = use_complex

        self.mps = BinaryTransformerEncoder(embedding_dim=d * D * D * 2, num_heads=4, num_layers=2, dropout=0.1)

    def _tensor(self, l: int, config: torch.Tensor) -> torch.Tensor:
        """
        Get the tensor for a specific lattice site (l1, l2) and configuration.
        """
        # pylint: disable=unsubscriptable-object
        # Order: L, R
        tensor: torch.Tensor = self.tensors[l, config.to(torch.int64)]
        if l == 0:
            tensor = tensor[:, :1, :]
        if l == self.L - 1:
            tensor = tensor[:, :, :1]
        return tensor

    def _contract(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        # tensors index: batch, l, r
        result = tensors[0]
        for l in range(1, self.L):
            result = torch.einsum("blm,bmr->blr", result, tensors[l])
        return result[:, 0, 0]

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PEPS tensor network.
        """
        mps =torch.view_as_complex( self.mps(configs).reshape([-1, self.L, self.d, self.D, self.D,2]))  # [batch, L, d, D, D]
        tensors: list[list[torch.Tensor]] = [[mps[i, l, int(configs[i, l])] for l in range(self.L)] for i in range(mps.shape[0])]
        stacked: list[torch.Tensor] = [torch.stack([tensors[i][l] for i in range(len(configs))]) for l in range(self.L)]
        return self._contract(stacked)

class MpsFunction(torch.nn.Module):
    """
    The Mps tensor network used by qmb interface.
    """

    def __init__(self, L: int, d: int, D: int, use_complex: bool = False) -> None:  # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        assert d == 2
        self.L = L
        self.model = MPS(L, d, D, use_complex)

    @torch.jit.export
    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate a batch of unique configurations.
        """
        raise NotImplementedError("The generate_unique method is not implemented for this class.")

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MPS tensor network.
        """
        x = unpack_int(x, size=1, last_dim=self.L)
        return self.model(x).contiguous()
