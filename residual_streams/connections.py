"""Hyper-connections and frac-connections: learned replacements for the residual add.

Hyper-Connections, Zhu et al., ICLR 2025, arXiv 2409.19606.
Frac-Connections, Zhu et al., arXiv 2503.14125.

Parameter and method names follow each paper's Algorithm 2 (PyTorch pseudo-code), so the code
can be read next to the paper. Both modules wrap one residual branch (attention or MLP):

    x, alpha, beta = conn.width_connection(h)   # the branch input
    h = conn.depth_connection(h, branch(x), alpha, beta)

The math is the papers'; the arrangement differs in one way, to move less memory. The papers'
width_connection returns mix_h, the branch input and the mixed streams stacked, which
depth_connection then reads back. Here width_connection returns only the branch input, and
depth_connection forms the mixed streams itself, so mix_h is never stored.
"""

import torch
import torch.nn as nn
from torch import Tensor


class HyperConnection(nn.Module):
    """One hyper-connection. The hidden state is rate copies ("streams") of each token, H of
    shape [..., rate, dim], instead of one residual vector.

    The matrix [A_m | A_r] (static_alpha, [rate, rate + 1]) mixes the streams: column 0 (A_m)
    forms the branch input, columns 1: (A_r) mix the streams with each other. B (static_beta,
    [rate]) weights the branch output added into each stream:

        H' = A_r^T H + B^T T(A_m^T H)

    Dynamic hyper-connections (DHC, eqs. 10-13) add input-dependent terms tanh(norm(H) W) * s,
    with W = 0 and s = 0.01 at init. The init (eq. 14), B = 1, A_m = e_(layer_id mod rate),
    A_r = I, makes the block an ordinary pre-norm residual, so a network built from these starts
    out identical to a plain one.

    stream_dtype: the dtype the streams are stored in after depth_connection, e.g. torch.bfloat16
    to halve their memory traffic. None keeps the computed dtype (fp32 for fp32 inputs). The
    mixing itself is always computed in the dtype of the (fp32) parameters.
    """

    def __init__(
        self,
        dim: int,
        rate: int,
        layer_id: int,
        dynamic: bool = True,
        stream_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.dynamic = dynamic
        self.stream_dtype = stream_dtype
        self.static_beta = nn.Parameter(torch.ones(rate))
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[layer_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(rate)], dim=1))
        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def extra_repr(self) -> str:
        return f"rate={self.rate}, dynamic={self.dynamic}"

    def width_connection(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """h [..., rate, dim] -> branch input A_m^T H [..., dim], alpha, beta."""
        alpha, beta = self.static_alpha, self.static_beta
        if self.dynamic:
            # one projection for W_alpha and W_beta, in the parameters' dtype even under autocast
            with torch.autocast(h.device.type, enabled=False):
                fn = torch.cat([self.dynamic_alpha_fn, self.dynamic_beta_fn[:, None]], dim=1)
                norm_h = self.layer_norm(h.to(self.layer_norm.weight.dtype))
                dynamic = torch.tanh(norm_h @ fn)
            alpha = alpha + dynamic[..., :-1] * self.dynamic_alpha_scale
            beta = beta + dynamic[..., -1] * self.dynamic_beta_scale
        x = sum(alpha[..., r, 0, None] * h[..., r, :] for r in range(self.rate))
        return x, alpha, beta

    def depth_connection(self, h: Tensor, h_o: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """A_r^T H + B^T h_o, with h_o [..., dim] the branch output -> [..., rate, dim]."""
        mixed = sum(alpha[..., r, 1:, None] * h[..., r, None, :] for r in range(self.rate))
        h = mixed + beta[..., :, None] * h_o[..., None, :]
        return h if self.stream_dtype is None else h.to(self.stream_dtype)


class FracConnection(nn.Module):
    """One frac-connection. Instead of copying the hidden state into rate streams, it is split
    into rate fractions of dim / rate (eq. 6), so it keeps its usual shape [..., dim] and costs no
    extra memory.

    alpha = [Y | A] (static_alpha, [rate, 2 rate]): Y mixes the fractions into the branch input,
    A mixes them with each other. B (static_beta, [rate]) scales each fraction of the branch
    output. With H the hidden state viewed as [rate, dim / rate] (eq. 8):

        H' = A^T H + B * T(Y^T H)

    Dynamic frac-connections (DFC, eqs. 11-14) add tanh(norm(H) W) * s per fraction, with W = 0
    at init. The init (eq. 15), Y = A = I and B = 1, makes the block an ordinary pre-norm
    residual. The paper's code casts mix_h to bf16; set stream_dtype=torch.bfloat16 for that.
    """

    def __init__(
        self,
        dim: int,
        rate: int,
        dynamic: bool = True,
        stream_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert dim % rate == 0, (dim, rate)
        self.rate = rate
        self.dynamic = dynamic
        self.stream_dtype = stream_dtype
        self.static_beta = nn.Parameter(torch.ones(rate))
        self.static_alpha = nn.Parameter(torch.cat([torch.eye(rate), torch.eye(rate)], dim=1))
        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim // rate, 2 * rate))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim // rate))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = nn.LayerNorm(dim // rate, eps=1e-6)

    def extra_repr(self) -> str:
        return f"rate={self.rate}, dynamic={self.dynamic}"

    def width_connection(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """h [..., dim] -> branch input Y^T H [..., dim], alpha, beta."""
        h = h.unflatten(-1, (self.rate, -1))
        alpha, beta = self.static_alpha, self.static_beta
        if self.dynamic:
            with torch.autocast(h.device.type, enabled=False):
                fn = torch.cat([self.dynamic_alpha_fn, self.dynamic_beta_fn[:, None]], dim=1)
                norm_h = self.layer_norm(h.to(self.layer_norm.weight.dtype))
                dynamic = torch.tanh(norm_h @ fn)
            alpha = alpha + dynamic[..., :-1] * self.dynamic_alpha_scale
            beta = beta + dynamic[..., -1] * self.dynamic_beta_scale
        x = sum(alpha[..., i, : self.rate, None] * h[..., i, None, :] for i in range(self.rate))
        return x.flatten(-2), alpha, beta

    def depth_connection(self, h: Tensor, h_o: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """A^T H + B * h_o per fraction, with h_o [..., dim] the branch output -> [..., dim]."""
        h = h.unflatten(-1, (self.rate, -1))
        mixed = sum(alpha[..., i, self.rate :, None] * h[..., i, None, :] for i in range(self.rate))
        h = (mixed + beta[..., None] * h_o.unflatten(-1, (self.rate, -1))).flatten(-2)
        return h if self.stream_dtype is None else h.to(self.stream_dtype)
