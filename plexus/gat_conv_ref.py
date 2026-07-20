# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
"""
Single-card reference implementation of GATv1 (single- or multi-head, concat).

Convention used here (and in plexus.gat_conv): `out_channels` is the TOTAL
output dim per layer, i.e. heads * head_dim, with concat=True so that layer
chaining keeps a constant feature width.  PyG's `out_channels` is per-head;
to compare against PyG you should pass `out_channels=head_dim, heads=H,
concat=True` to PyG and `out_channels=H*head_dim, heads=H` here.

Reformulated for layout compatibility with the 3D-parallel layer:

    w_src[h] = W[:, h, :] @ att_src[h, :]              # [F_in]
    w_dst[h] = W[:, h, :] @ att_dst[h, :]              # [F_in]
    alpha_src[i, h] = h_i  @ w_src[h]                  # scalar per (i, h)
    alpha_dst[i, h] = h_i  @ w_dst[h]
    e_ij_h         = LeakyReLU(alpha_src[j,h] + alpha_dst[i,h])
    alpha_ij_h     = softmax over j in N(i) of e_ij_h
    out_i_h        = (sum_j alpha_ij_h * h_j) @ W[:, h, :] + b_h
    out_i          = concat_h(out_i_h)                 # [H * F_h]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-head: src [E, H], index [E] -> [dim_size, H] amax-reduced."""
    if src.dim() == 1:
        out = torch.full((dim_size,), float("-inf"), dtype=src.dtype, device=src.device)
        out.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    else:
        H = src.size(1)
        out = torch.full((dim_size, H), float("-inf"), dtype=src.dtype, device=src.device)
        out.scatter_reduce_(
            0, index.unsqueeze(-1).expand(-1, H), src, reduce="amax", include_self=True
        )
    out = torch.where(torch.isinf(out) & (out < 0), torch.zeros_like(out), out)
    return out


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-head sum.  src [E] or [E, H] / [E, H, F]; index [E]."""
    if src.dim() == 1:
        out = torch.zeros((dim_size,), dtype=src.dtype, device=src.device)
        out.scatter_add_(0, index, src)
        return out
    if src.dim() == 2:
        H = src.size(1)
        out = torch.zeros((dim_size, H), dtype=src.dtype, device=src.device)
        out.scatter_add_(0, index.unsqueeze(-1).expand(-1, H), src)
        return out
    if src.dim() == 3:
        H, Fd = src.size(1), src.size(2)
        out = torch.zeros((dim_size, H, Fd), dtype=src.dtype, device=src.device)
        idx = index.view(-1, 1, 1).expand(-1, H, Fd)
        out.scatter_add_(0, idx, src)
        return out
    raise ValueError(f"unsupported src.dim()={src.dim()}")


class GATConvRef(nn.Module):
    """
    Reference multi-head GATv1 layer.  Numerically equivalent to PyG's
    GATConv(in_channels, head_dim, heads=H, concat=True, dropout=0,
            add_self_loops=False) as long as you initialize with the same
    weights.  `out_channels` here is the TOTAL output dim = heads * head_dim.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert out_channels % heads == 0, (
            f"out_channels ({out_channels}) must be divisible by heads ({heads})"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.head_dim = out_channels // heads
        self.negative_slope = negative_slope

        # Weight stored as [in, H * F_h] (heads-outer flat layout, matches PyG).
        self.weight = nn.Parameter(torch.empty(in_channels, heads * self.head_dim))
        # Attention vectors stored as [H, F_h].
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        # Bias is per-output-feature, [H * F_h].
        self.bias = nn.Parameter(torch.zeros(heads * self.head_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        bound = 1.0 / math.sqrt(self.head_dim)
        nn.init.uniform_(self.att_src, -bound, bound)
        nn.init.uniform_(self.att_dst, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int = None,
    ) -> torch.Tensor:
        if num_nodes is None:
            num_nodes = x.size(0)

        H = self.heads
        F_h = self.head_dim
        F_in = self.in_channels

        # View weight as [F_in, H, F_h] for per-head ops.
        W3 = self.weight.view(F_in, H, F_h)

        # Combined attention vectors in F_in space, per head.
        # w_src[i, h] = sum_f W3[i, h, f] * att_src[h, f]
        w_src = torch.einsum("ihf,hf->ih", W3, self.att_src)  # [F_in, H]
        w_dst = torch.einsum("ihf,hf->ih", W3, self.att_dst)  # [F_in, H]

        # Per-node attention scalars (computed from raw H), one per (i, h).
        alpha_src = x @ w_src                                  # [N, H]
        alpha_dst = x @ w_dst                                  # [N, H]

        src, dst = edge_index[0], edge_index[1]
        e = alpha_src[src] + alpha_dst[dst]                    # [E, H]
        e = F.leaky_relu(e, self.negative_slope)

        # Per-head numerically-stable row softmax (group by dst).
        e_max = _scatter_max(e, dst, dim_size=num_nodes).detach()  # [N, H]
        e_shifted = e - e_max[dst]                             # [E, H]
        exp_e = torch.exp(e_shifted)
        row_sum = _scatter_sum(exp_e, dst, dim_size=num_nodes)  # [N, H]
        alpha = exp_e / (row_sum[dst] + 1e-16)                  # [E, H]

        # Weighted aggregation on raw H: agg[i, h, :] = sum_j alpha[ij,h] * x[j].
        # Implement as scatter_add over (j -> i) edges with weights alpha[:, h].
        msg = x[src].unsqueeze(1) * alpha.unsqueeze(-1)         # [E, H, F_in]
        agg = _scatter_sum(msg, dst, dim_size=num_nodes)        # [N, H, F_in]

        # Per-head output projection: out[i, h, f] = sum_in agg[i, h, in] * W3[in, h, f].
        out = torch.einsum("nhi,ihf->nhf", agg, W3)             # [N, H, F_h]
        out = out.reshape(num_nodes, H * F_h)

        if self.bias is not None:
            out = out + self.bias
        return out
