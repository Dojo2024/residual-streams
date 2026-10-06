"""A minimal pre-norm transformer encoder whose residual connection can be plain, hyper-connections
or frac-connections, to show how the connections plug into a network.

    Encoder(dim=384, depth=12, heads=6, connections="hyper", rate=4)

With hyper-connections the encoder copies each token into rate streams before the first block
and sums the streams after the last one (Zhu et al., arXiv 2409.19606); frac-connections keep the
usual [batch, tokens, dim] shape throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .connections import FracConnection, HyperConnection


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.fc2 = nn.Linear(int(dim * ratio), dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ConnectionBlock(Block):
    """Block with its two residual adds replaced by hyper- or frac-connections (each paper's
    Algorithm 3). Each branch counts as one layer for the hyper-connection init, so the attention
    of block i reads stream 2i mod rate and its MLP stream (2i + 1) mod rate."""

    def __init__(self, dim: int, heads: int, conn: str, rate: int, layer_id: int, **kwargs) -> None:
        super().__init__(dim, heads)
        if conn == "hyper":
            self.conn_attn = HyperConnection(dim, rate, 2 * layer_id, **kwargs)
            self.conn_mlp = HyperConnection(dim, rate, 2 * layer_id + 1, **kwargs)
        else:
            assert conn == "frac", conn
            self.conn_attn = FracConnection(dim, rate, **kwargs)
            self.conn_mlp = FracConnection(dim, rate, **kwargs)

    def forward(self, h: Tensor) -> Tensor:
        x, alpha, beta = self.conn_attn.width_connection(h)
        h = self.conn_attn.depth_connection(h, self.attn(self.norm1(x)), alpha, beta)
        x, alpha, beta = self.conn_mlp.width_connection(h)
        return self.conn_mlp.depth_connection(h, self.mlp(self.norm2(x)), alpha, beta)


class Encoder(nn.Module):
    """connections: "plain", "hyper" or "frac". rate: streams (hyper) or fractions (frac).
    dynamic and stream_dtype are passed to the connections."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        connections: str = "plain",
        rate: int = 4,
        dynamic: bool = True,
        stream_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rate = rate if connections == "hyper" else 0
        if connections == "plain":
            self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        else:
            self.blocks = nn.ModuleList(
                ConnectionBlock(dim, heads, connections, rate, ii, dynamic=dynamic, stream_dtype=stream_dtype)
                for ii in range(depth)
            )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def _output(self, x: Tensor) -> Tensor:
        # with hyper-connections a layer's output is the sum of its streams
        dtype = self.norm.weight.dtype
        x = x.sum(dim=-2, dtype=dtype) if self.rate else x.to(dtype)
        return self.norm(x)

    def forward(self, x: Tensor) -> Tensor:
        """x [batch, tokens, dim] -> [batch, tokens, dim]."""
        if self.rate:
            x = x[..., None, :].expand(*x.shape[:-1], self.rate, x.shape[-1])
        for block in self.blocks:
            x = block(x)
        return self._output(x)

    def forward_layers(self, x: Tensor, layers: tuple[int, ...]) -> list[Tensor]:
        """The normed output of each of the given blocks (1-indexed), for layer readouts."""
        if self.rate:
            x = x[..., None, :].expand(*x.shape[:-1], self.rate, x.shape[-1])
        outputs = []
        for layer, block in enumerate(self.blocks, start=1):
            x = block(x)
            if layer in layers:
                outputs.append(self._output(x))
        return outputs
