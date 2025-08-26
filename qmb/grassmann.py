"""
This file implements the Grassmann neural network.
"""

import torch
from grassmann_tensor import GrassmannTensor

# Arrow of physics edge is False


class Grassmann2Layer(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.tensor = torch.nn.Parameter(torch.randn(self.dim * 2, self.dim * 2))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # x: b+0 * 1+1 * n+n
        # g: n+n * n+n
        batch_size = data.size(0)
        x = GrassmannTensor((False, False, False), ((batch_size, 0), (1, 1), (self.dim, self.dim)), data).update_mask()
        g = GrassmannTensor((True, False), ((self.dim, self.dim), (self.dim, self.dim)), self.tensor).update_mask()

        return x.matmul(g).tensor


class Grassmann3Layer(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.tensor = torch.nn.Parameter(torch.randn(self.dim * 2, self.dim * 2, self.dim * 2))

    def forward(self, data_1: torch.Tensor, data_2: torch.Tensor) -> torch.Tensor:
        # x_1: b+0 * 1+1 * n+n
        # x_2: b+0 * 1+1 * n+n
        # g: n+n * n+n * n+n
        batch_size = data_1.size(0)
        x_1 = GrassmannTensor((False, False, False), ((batch_size, 0), (1, 1), (self.dim, self.dim)), data_1).update_mask()
        x_2 = GrassmannTensor((False, False, False), ((batch_size, 0), (1, 1), (self.dim, self.dim)), data_2).update_mask()
        g = GrassmannTensor((True, True, False), ((self.dim, self.dim), (self.dim, self.dim), (self.dim, self.dim)), self.tensor).update_mask()

        # gm: n+n * n+n n+n
        g_m = g.reverse((1,)).reshape((-1, (self.dim**2 * 2, self.dim**2 * 2)))
        # gx_1_m: b+0 * 1+1 * n+n n+n
        gx_1_r = x_1.matmul(g_m)
        # gx_1: b+0 * 1+1 * n+n * n+n
        gx_1 = gx_1_r.reshape((-1, -1, (self.dim, self.dim), (self.dim, self.dim))).reverse((2,))
        # gx_1_m: b+0 * n+n * 1+1 n+n
        gx_1_m = gx_1.permute((0, 2, 1, 3)).reshape((-1, -1, (self.dim * 2, self.dim * 2)))
        # gx_12_r: b+0 * 1+1 * 1+1 n+n
        gx_12_r = x_2.matmul(gx_1_m)
        # gx_12: b+0 * 1+1 * 1+1 * n+n
        gx_12 = gx_12_r.reshape((-1, -1, (1, 1), (self.dim, self.dim)))
        # gx_12_m: b+0 * 1+1 1+1 * n+n
        gx_12_m = gx_12.reshape((-1, (2, 2), -1))

        result = gx_12_m.tensor.view((batch_size, 2, 2, self.dim * 2))
        return result.sum(dim=2)


class GrassmannNN(torch.nn.Module):

    def __init__(self, site: int, dim: int) -> None:
        super().__init__()
        self.site = site
        self.dim = dim
        self.embedding = torch.nn.Parameter(torch.randn(site, 2, dim))
        self.head = Grassmann2Layer(dim)
        self.body = torch.nn.ModuleList([Grassmann3Layer(dim) for _ in range(site - 1)])
        self.act = torch.nn.Tanh()

    def embed(self, site: int, data: torch.Tensor) -> torch.Tensor:
        # data: b
        emb = self.embedding[site][data.to(dtype=torch.long)]
        expanded = emb.reshape((-1, 1, self.dim)).expand((-1, 4, self.dim))
        result = torch.where(
            torch.logical_or(
                torch.logical_and(
                    torch.arange(4, device=data.device).reshape((1, -1, 1)) == 0,
                    torch.logical_not(data.reshape((-1, 1, 1))),
                ),
                torch.logical_and(
                    torch.arange(4, device=data.device).reshape((1, -1, 1)) == 3,
                    data.reshape((-1, 1, 1)),
                ),
            ), expanded, 0).reshape((-1, 2, self.dim * 2))
        return result

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: b * s
        for i in range(self.site):
            if i == 0:
                x = self.head(self.embed(i, data[:, i]))
                x = self.act(x)
            else:
                x = self.body[i - 1](x, self.embed(i, data[:, i]))
                x = self.act(x)
        return x
