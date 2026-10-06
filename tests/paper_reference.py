"""The papers' own PyTorch pseudo-code (Algorithm 2 of each), transcribed as reference
implementations for the tests. Hidden states are [batch, tokens, ...].

Changes from the papers' text, all needed to run the code or to compare in float64:
- device arguments and the `config` object are dropped (FC's use_tanh / use_hc_norm / use_scale
  are all on for the dynamic version, as in the paper's equations);
- FC's mix_h line casts to .float() and then .bfloat16(); both casts are dropped, so float64 inputs
  stay float64 (with them the match is only ~5e-7);
- a static (non-dynamic) path is added for each, using only the static parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperHyperConnection(nn.Module):
    """Hyper-Connections (arXiv 2409.19606), Algorithm 2. h: B x L x N x D."""

    def __init__(self, dim, rate, layer_id, dynamic):
        super().__init__()
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic
        self.static_beta = nn.Parameter(torch.ones((rate,)))
        init_alpha0 = torch.zeros((rate, 1))
        init_alpha0[layer_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye((rate))], dim=1))
        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate + 1)))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim,)))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def width_connection(self, h):
        # get alpha and beta
        if self.dynamic:
            norm_h = self.layer_norm(h)
        if self.dynamic:
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = F.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]
        if self.dynamic:
            dc_weight = norm_h @ self.dynamic_beta_fn
            dc_weight = F.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, ...]
        else:
            beta = self.static_beta[None, None, ...]
        # width connection
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        if beta.ndim == 3:
            return torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]
        # static beta has no batch and token dims
        return torch.einsum("blh,n->blnh", h_o, beta[0, 0]) + mix_h[..., 1:, :]

    def branch_input(self, mix_h):
        # Algorithm 3: the branch reads mix_h[..., 0, :]
        return mix_h[..., 0, :]


class PaperFracConnection(nn.Module):
    """Frac-Connections (arXiv 2503.14125), Algorithm 2. h: B x L x D."""

    def __init__(self, dim, rate, dynamic):
        super().__init__()
        self.rate = rate
        self.dynamic = dynamic
        self.static_beta = nn.Parameter(torch.ones((rate,)))
        self.static_alpha = nn.Parameter(torch.cat([torch.eye((rate)), torch.eye((rate))], dim=1))
        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim // self.rate, rate * 2)))
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim // self.rate,)))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = nn.LayerNorm(dim // self.rate, eps=1e-6)

    def width_connection(self, h):
        h_shape = h.shape
        h_reshape = h.reshape(h_shape[:-1] + (self.rate, h_shape[-1] // self.rate))
        if not self.dynamic:
            return self.static_alpha.transpose(-1, -2) @ h_reshape, self.static_beta
        norm_h = self.layer_norm(h_reshape)
        dynamic_alpha = F.tanh(norm_h @ self.dynamic_alpha_fn)
        dynamic_alpha = dynamic_alpha * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        dynamic_beta = F.tanh(norm_h @ self.dynamic_beta_fn)
        dynamic_beta = dynamic_beta * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta[None, None, ...]
        mix_h = alpha.transpose(-1, -2).contiguous() @ h_reshape
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        h_o_shape = h_o.shape
        h = beta[..., None] * h_o.reshape(h_o_shape[:-1] + (self.rate, h_o_shape[-1] // self.rate)) + mix_h[..., self.rate :, :]
        h_shape = h.shape
        return h.reshape(h_shape[:-2] + (h_shape[-2] * h_shape[-1],))

    def branch_input(self, mix_h):
        # Algorithm 3: the branch reads the first rate rows of mix_h, flattened back to D
        mix_h_shape = mix_h.shape
        return mix_h[..., : self.rate, :].reshape(mix_h_shape[:-2] + (mix_h_shape[-2] // 2 * mix_h_shape[-1],))
