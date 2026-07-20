# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
"""
3D-parallel multi-head GATv1 layer for Plexus -- project-first decomposition.

Mathematically equivalent to torch_geometric.nn.GATConv with
heads=H, concat=True, dropout=0, add_self_loops=False:

    H_prime[n, h, :]   = H_in[n, :] @ W[:, h, :]                         # per-head projection
    alpha_src[n, h]    = H_prime[n, h, :] @ att_src[h, :]
    alpha_dst[n, h]    = H_prime[n, h, :] @ att_dst[h, :]
    e_ij_h            = LeakyReLU(alpha_src[j, h] + alpha_dst[i, h])
    alpha_ij_h        = softmax_i over j of e_ij_h
    out[i, h, :]      = sum_j alpha_ij_h * H_prime[j, h, :]
    out[i]            = concat_h(out[i, h, :]) + bias

vs. the previous "raw-H aggregate, project at the end" form, this version
keeps per-layer activation memory INDEPENDENT of `heads` (saves H_prime of
shape [N, F_h, H] = [N, F_total] instead of AGG of shape [N, H, F_in] which
scales as H x F_in = H x F_total).

Sharding (layer 0, use_3d_linear, 3D groups (x,y,z)):
    groups = (outer=x, inner=z, depth=y)
    H_local      : [N/outer, F_in/inner]
    weight_local : [F_in/inner, F_h_local * heads]   (depth-sharded if gather_weights)
                   F_h is sharded along outer, heads are full per outer rank
    att_src/dst  : [F_h_local * heads]               replicated across (inner, depth)
    bias         : [F_h_local * heads]               replicated across (inner, depth)

Forward gathers W and att across outer into the F_h-full layout used inside
the layer, then drops the layer output (full F_h) back to the F_h_local
shard for the next layer's input.  Because the gathered W / att are tiny
(parameters), the extra comm is negligible; the activation/comm savings on
H_prime, AGG, and the missing w_src/w_dst all-reduces dominate.

Backward: torch.autograd handles everything except the alpha-weighted SpMM,
which uses a custom Function (`_AlphaWeightedSpMM3D`) because PyTorch's
torch.sparse.mm backward through CSR values densifies the gradient to a
[N_dst x N_src] tensor (~11 TB on ogbn-products).  The custom Function
computes grad_alpha per-edge in chunks and grad_H_prime via standard
sparse-dense matmul on A^T (one per head, accumulated).
"""

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from axonn import axonn as ax
from axonn.intra_layer.communication import (
    BackwardAllReduce,
    Drop,
    ForwardGather_BackwardReduceScatter,
    Gather,
    _all_reduce,
    _gather,
)
from axonn.intra_layer.fully_connected import (
    extract_local_params_from_full_params,
)
from torch.nn import Parameter

from plexus import plexus as plx
from plexus.utils.general import get_process_groups_info, pad_dimension


# ---------------------------------------------------------------------------
# Autograd-aware all-reduce: SUM in both forward and backward.
# ---------------------------------------------------------------------------


class _AllReduceSumFwdBwd(torch.autograd.Function):
    """Forward and backward both sum-all-reduce over `process_group`."""

    @staticmethod
    def forward(ctx, input_, process_group):
        ctx.process_group = process_group
        x = input_.contiguous().clone()
        if dist.is_initialized() and dist.get_world_size(process_group) > 1:
            dist.all_reduce(x, group=process_group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized() and dist.get_world_size(ctx.process_group) > 1:
            ax.get_timers().start("gat bwd: ar_sum")
            grad = grad_output.contiguous().clone()
            dist.all_reduce(grad, group=ctx.process_group)
            ax.get_timers().stop("gat bwd: ar_sum")
            return grad, None
        return grad_output, None


def _ar_sum(t: torch.Tensor, process_group) -> torch.Tensor:
    return _AllReduceSumFwdBwd.apply(t, process_group)


# ---------------------------------------------------------------------------
# Sentinel Functions to bracket the layer's overall backward pass.
# Forward is identity; their backward runs at the entry / exit of the
# layer's bwd subgraph so we can wrap a single "gat conv bwd" timer
# around the entire backward without manually instrumenting autograd.
# ---------------------------------------------------------------------------


class _BwdTimerStart(torch.autograd.Function):
    """Applied to the layer's OUT in fwd; its backward fires FIRST when
    grad_OUT arrives -> start the bwd timer."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        ax.get_timers().start("gat conv bwd")
        return grad


class _BwdTimerStop(torch.autograd.Function):
    """Applied to the layer's input x in fwd; its backward fires LAST when
    the grad has fully propagated to x -> stop the bwd timer."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        ax.get_timers().stop("gat conv bwd")
        return grad


def _gather_grad(t: torch.Tensor, process_group, dim: int) -> torch.Tensor:
    """
    Autograd-aware AllGather along an arbitrary dim (forward AllGather,
    backward ReduceScatter).  axonn's ForwardGather_BackwardReduceScatter
    only supports dim=0, so we transpose around it for other dims.

    Treats world_size==1 / non-init as no-op.
    """
    if not dist.is_initialized() or dist.get_world_size(process_group) <= 1:
        return t
    if dim == 0:
        return ForwardGather_BackwardReduceScatter.apply(
            t, process_group, 0, False, False
        )
    t_swap = t.transpose(0, dim).contiguous()
    g = ForwardGather_BackwardReduceScatter.apply(
        t_swap, process_group, 0, False, False
    )
    return g.transpose(0, dim).contiguous()


# ---------------------------------------------------------------------------
# Custom autograd Function: per-head alpha-weighted SpMM on a 3D dense input.
# ---------------------------------------------------------------------------


class _AlphaWeightedSpMM3D(torch.autograd.Function):
    """
    AGG[i, f, h] = sum_{k in row i} alpha[k, h] * H_prime[col[k], f, h]

    Inputs:
        alpha     : [E, H_heads]       per-edge per-head attention weight
        H_prime   : [N_src, F_h, H_heads]  per-head projected features
        crow, col : CSR structure of A_local   (shape [N_dst, N_src])
        num_rows  : N_dst (rows of A_local)

    Output:
        AGG : [N_dst, F_h, H_heads]

    Differentiates only through `alpha` and `H_prime`.  PyTorch's built-in
    torch.sparse.mm backward densifies the grad of the CSR values to an
    [N_dst x N_src] tensor (~11 TB on ogbn-products).  We bypass that path
    by computing grad_alpha per-edge in chunks and grad_H_prime via a
    value-fixed sparse-dense matmul on A^T (one per head, accumulated).
    """

    @staticmethod
    def forward(ctx, alpha, H_prime, crow, col, num_rows):
        device = H_prime.device
        N_src, F_h, H_heads = H_prime.shape

        with torch.no_grad():
            AGG = H_prime.new_zeros(num_rows, F_h, H_heads)
            for h in range(H_heads):
                # H_prime[:, :, h] is a non-contiguous 2D view; sparse.mm
                # requires contiguous.  One transient copy per head keeps
                # peak memory low (vs. permuting the whole [H, N_src, F_h]
                # once, which holds an extra full-size tensor across all heads).
                H_h = H_prime[:, :, h].contiguous()
                csr_h = torch.sparse_csr_tensor(
                    crow, col, alpha[:, h].contiguous(),
                    size=(num_rows, N_src),
                    dtype=alpha.dtype, device=device,
                )
                AGG[:, :, h] = torch.sparse.mm(csr_h, H_h)

        ctx.save_for_backward(alpha, H_prime, crow, col)
        ctx.num_rows = num_rows
        return AGG

    @staticmethod
    def backward(ctx, grad_AGG):
        alpha, H_prime, crow, col = ctx.saved_tensors
        num_rows = ctx.num_rows
        device = H_prime.device
        N_src, F_h, H_heads = H_prime.shape
        col_long = col.long()
        E = col_long.shape[0]

        # ---- grad_alpha: per-edge dot product over F_h, chunked ----
        grad_alpha = None
        if ctx.needs_input_grad[0]:
            ax.get_timers().start("gat bwd: grad_alpha (per-edge)")
            row_idx = torch.repeat_interleave(
                torch.arange(num_rows, device=device, dtype=col_long.dtype),
                (crow[1:] - crow[:-1]).to(col_long.dtype),
            ).long()
            # Each chunk holds grad_msg [chunk, F_h, H], H_chunk [chunk, F_h, H]
            # plus their elementwise product.  Aim for ~512 MiB per chunk.
            target = 512 * (1 << 20)
            per_edge_bytes = max(1, F_h * H_heads * 4 * 4)
            chunk = max(1, target // per_edge_bytes)
            grad_alpha = torch.empty(E, H_heads, dtype=alpha.dtype, device=device)
            for s in range(0, E, chunk):
                e = min(s + chunk, E)
                grad_msg = grad_AGG[row_idx[s:e]]   # [chunk, F_h, H]
                H_chunk = H_prime[col_long[s:e]]    # [chunk, F_h, H]
                # grad_alpha[k, h] = sum_f grad_msg[k, f, h] * H_chunk[k, f, h]
                grad_alpha[s:e] = (grad_msg * H_chunk).sum(dim=1)
            ax.get_timers().stop("gat bwd: grad_alpha (per-edge)")

        # ---- grad_H_prime: per-head, csr_h^T @ grad_AGG[:, :, h] ----
        grad_H_prime = None
        if ctx.needs_input_grad[1]:
            ax.get_timers().start("gat bwd: grad_H_prime (per-head spmm)")
            with torch.no_grad():
                grad_H_prime = H_prime.new_zeros(N_src, F_h, H_heads)
                for h in range(H_heads):
                    # One transient copy per head keeps peak memory low.
                    g_h = grad_AGG[:, :, h].contiguous()
                    att_csr = torch.sparse_csr_tensor(
                        crow, col, alpha[:, h].contiguous(),
                        size=(num_rows, N_src),
                        dtype=alpha.dtype, device=device,
                    )
                    att_t_csr = att_csr.transpose(0, 1).to_sparse_csr()
                    grad_H_prime[:, :, h] = torch.sparse.mm(att_t_csr, g_h)
            ax.get_timers().stop("gat bwd: grad_H_prime (per-head spmm)")

        return grad_alpha, grad_H_prime, None, None, None


def _csr_row_index_per_nnz(crow_indices: torch.Tensor) -> torch.Tensor:
    """Map each CSR nonzero to its row id."""
    num_rows = crow_indices.numel() - 1
    counts = (crow_indices[1:] - crow_indices[:-1]).to(torch.long)
    rows = torch.arange(num_rows, device=crow_indices.device, dtype=torch.long)
    return torch.repeat_interleave(rows, counts)


def _scatter_max_per_head(src, index, dim_size):
    """src [E, H] / [E], index [E] -> amax-reduced over rows."""
    if src.dim() == 1:
        out = torch.full((dim_size,), float("-inf"), dtype=src.dtype, device=src.device)
        out.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
        return out
    H = src.size(1)
    out = torch.full((dim_size, H), float("-inf"), dtype=src.dtype, device=src.device)
    out.scatter_reduce_(
        0, index.unsqueeze(-1).expand(-1, H), src, reduce="amax", include_self=True
    )
    return out


def _scatter_sum_per_head(src, index, dim_size):
    """src [E, H] / [E], index [E] -> sum-reduced over rows."""
    if src.dim() == 1:
        out = torch.zeros((dim_size,), dtype=src.dtype, device=src.device)
        out.scatter_add_(0, index, src)
        return out
    H = src.size(1)
    out = torch.zeros((dim_size, H), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, H), src)
    return out


# ---------------------------------------------------------------------------
# GAT layer (project-first).
# ---------------------------------------------------------------------------


class GATConv(nn.Module):
    """Multi-head GATv1 (concat) layer with Plexus' 3D tensor-parallel
    sharding, project-first decomposition."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_num: int,
        heads: int = 1,
        shard_features_in_depth: bool = False,
        negative_slope: float = 0.2,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert heads >= 1, f"heads must be >= 1, got {heads}"
        assert out_channels % heads == 0, (
            f"out_channels ({out_channels}) must be divisible by heads ({heads}); "
            f"out_channels = heads * head_dim with concat=True."
        )

        self.layer_num = layer_num
        self.negative_slope = float(negative_slope)
        self.heads = heads
        self.head_dim = out_channels // heads  # F_h

        # Group selection identical to GCNConv so that GAT layers chain
        # cleanly with each other (and with GCN layers).
        if plx.use_3d_linear:
            if layer_num % 3 == 0:
                groups = ("x", "z", "y")
            elif layer_num % 3 == 1:
                groups = ("y", "x", "z")
            else:
                groups = ("z", "y", "x")
        else:
            if layer_num % 3 == 0:
                groups = ("x", "y", "z")
            elif layer_num % 3 == 1:
                groups = ("z", "x", "y")
            else:
                groups = ("y", "z", "x")

        self.gather_features = bool(layer_num == 0 and shard_features_in_depth)

        num_gpus, ranks, process_groups = get_process_groups_info(groups)
        self.outer_group, self.inner_group, self.depth_group = process_groups
        self.outer_group_size, self.inner_group_size, self.depth_group_size = num_gpus
        self.outer_rank = ranks[0]

        # Pad F_h so it divides outer_size; heads stay full per outer rank.
        head_dim_padded = pad_dimension(self.head_dim, self.outer_group_size)
        out_channels_padded = head_dim_padded * heads
        self.head_dim_padded = head_dim_padded
        self.local_head_dim = head_dim_padded // self.outer_group_size  # F_h_local

        if layer_num == 0:
            self.gather_weights = True
        else:
            self.gather_weights = (
                pad_dimension(in_channels, self.inner_group_size)
                // self.inner_group_size
            ) % self.depth_group_size == 0

        if self.gather_weights:
            self.in_channels_padded = pad_dimension(
                in_channels, self.inner_group_size, self.depth_group_size
            )
        else:
            self.in_channels_padded = pad_dimension(in_channels, self.inner_group_size)
        self.out_channels_padded = out_channels_padded
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.local_in_channels = self.in_channels_padded // self.inner_group_size
        self.local_out_channels = self.local_head_dim * heads

        # ---- W storage: same as the previous multi-head version. ----
        # Layout: [F_in_padded, F_h_padded * H] flat = [F_in_padded, F_h_padded, H]
        # with F_h-outer flat order, so contiguous outer-slicing along the flat
        # last dim cuts F_h while keeping all heads on every rank.
        full_w = torch.empty(in_channels, heads, self.head_dim, device="cuda")
        with torch.no_grad():
            nn.init.xavier_uniform_(full_w.view(in_channels, heads * self.head_dim))
        full_w = full_w.permute(0, 2, 1).contiguous()  # [in, F_h, H]
        if head_dim_padded > self.head_dim:
            full_w = F.pad(full_w, (0, 0, 0, head_dim_padded - self.head_dim))
        full_w = full_w.view(in_channels, head_dim_padded * heads)
        if self.in_channels_padded > in_channels:
            full_w = F.pad(full_w, (0, 0, 0, self.in_channels_padded - in_channels))

        if self.gather_weights:
            self.weight = Parameter(
                extract_local_params_from_full_params(
                    full_w, self.inner_group, self.outer_group, self.depth_group
                ),
                requires_grad=True,
            )
        else:
            self.weight = Parameter(
                extract_local_params_from_full_params(
                    full_w,
                    self.inner_group,
                    self.outer_group,
                    dist.new_group(ranks=[dist.get_rank()]),
                ),
                requires_grad=True,
            )

        # ---- Attention vectors stored as [F_h_padded * H] flat = [F_h, H]. ----
        bound = 1.0 / math.sqrt(self.head_dim)
        full_att_src = torch.empty(heads, self.head_dim, device="cuda")
        full_att_dst = torch.empty(heads, self.head_dim, device="cuda")
        nn.init.uniform_(full_att_src, -bound, bound)
        nn.init.uniform_(full_att_dst, -bound, bound)
        full_att_src = full_att_src.permute(1, 0).contiguous()  # [F_h, H]
        full_att_dst = full_att_dst.permute(1, 0).contiguous()
        if head_dim_padded > self.head_dim:
            full_att_src = F.pad(full_att_src, (0, 0, 0, head_dim_padded - self.head_dim))
            full_att_dst = F.pad(full_att_dst, (0, 0, 0, head_dim_padded - self.head_dim))
            full_att_src[self.head_dim:, :].zero_()
            full_att_dst[self.head_dim:, :].zero_()

        att_slice = slice(
            self.outer_rank * self.local_head_dim,
            (self.outer_rank + 1) * self.local_head_dim,
        )
        self.att_src = Parameter(
            full_att_src[att_slice].contiguous().view(-1).clone(),
            requires_grad=True,
        )
        self.att_dst = Parameter(
            full_att_dst[att_slice].contiguous().view(-1).clone(),
            requires_grad=True,
        )

        if bias:
            full_bias = torch.zeros(head_dim_padded, heads, device="cuda")
            self.bias = Parameter(
                full_bias[att_slice].contiguous().view(-1).clone(),
                requires_grad=True,
            )
        else:
            self.register_parameter("bias", None)

        # Replicated-grad sync: att_src / att_dst / bias are sharded along
        # outer (F_h axis) but replicated across (inner, depth).
        for p in (self.att_src, self.att_dst):
            p.register_hook(self._make_replica_grad_hook())
        if self.bias is not None:
            self.bias.register_hook(self._make_replica_grad_hook())

    def _make_replica_grad_hook(self):
        inner_pg = self.inner_group
        depth_pg = self.depth_group

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not dist.is_initialized():
                return grad
            g = grad.contiguous().clone()
            if dist.get_world_size(inner_pg) > 1:
                dist.all_reduce(g, group=inner_pg)
            if dist.get_world_size(depth_pg) > 1:
                dist.all_reduce(g, group=depth_pg)
            return g

        return hook

    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index_shards) -> torch.Tensor:
        """
        x                : H_local with shape [N/outer, F_in/inner]  (or depth-sharded)
        edge_index_shards: list of (A_local_csr, A_local_t_csr) tuples produced by
                           DataLoader.  We use A_local for layer self.layer_num.
                           A_local has shape [N/depth, N/outer].
        """
        ax.get_timers().start("gat conv fwd")

        edge_index, _ = edge_index_shards[self.layer_num % len(edge_index_shards)]
        crow = edge_index.crow_indices()
        col = edge_index.col_indices().to(torch.long)
        N_dst_local = edge_index.size(0)  # = N / depth
        device = x.device
        H_heads = self.heads
        F_h_loc = self.local_head_dim
        F_h_padded = self.head_dim_padded

        # Sentinel: this Function's backward is the LAST one to fire in this
        # layer's backward subgraph (it sits closest to x).  Used to stop
        # the "gat conv bwd" timer that _BwdTimerStart starts on the OUT side.
        if x.requires_grad:
            x = _BwdTimerStop.apply(x)

        # ----- (1) optionally all-gather H over depth (only layer 0 + depth-shard) -----
        if self.gather_features:
            ax.get_timers().start("Allgather F")
            x_g = Gather.apply(x, self.depth_group, 0)
            ax.get_timers().stop("Allgather F")
            H_in = x_g.reshape(
                N_dst_local * self.depth_group_size, self.local_in_channels
            )
        else:
            H_in = x

        # ----- (2) materialize the local W tile -----
        if self.gather_weights:
            ax.get_timers().start("Allgather W")
            W = ForwardGather_BackwardReduceScatter.apply(
                self.weight, self.depth_group, 0, False, False
            )
            ax.get_timers().stop("Allgather W")
        else:
            W = BackwardAllReduce.apply(self.weight, self.depth_group, False)
        # [F_in_local, F_h_local * H] -> [F_in_local, F_h_local, H]
        W = W.reshape(self.local_in_channels, F_h_loc, H_heads)

        # ----- (3) gather W and att across outer to get F_h FULL -----
        # Tiny tensors (parameters); negligible comm.  Forward AllGather,
        # backward ReduceScatter so each rank's local F_h slice gets its
        # correct grad after summing across outer ranks.
        ax.get_timers().start("gat: gather W/att over outer")
        W_full = _gather_grad(W, self.outer_group, 1)            # [F_in/inner, F_h, H]
        att_src_full = _gather_grad(
            self.att_src.view(F_h_loc, H_heads), self.outer_group, 0
        )                                                         # [F_h, H]
        att_dst_full = _gather_grad(
            self.att_dst.view(F_h_loc, H_heads), self.outer_group, 0
        )                                                         # [F_h, H]
        ax.get_timers().stop("gat: gather W/att over outer")

        # ----- (4) project: H_prime = H_in @ W_full per head -----
        # H_in [N/outer, F_in/inner], W_full [F_in/inner, F_h, H]
        # -> H_prime_partial [N/outer, F_h, H], partial along inner.
        # AR over inner -> H_prime fully replicated across inner (and full F_h
        # across outer).
        ax.get_timers().start("gat: project H_prime")
        H_prime_partial = torch.einsum("ni,ifh->nfh", H_in, W_full)
        H_prime = _ar_sum(H_prime_partial, self.inner_group)
        ax.get_timers().stop("gat: project H_prime")

        # ----- (5) per-node attention scalars (purely local now) -----
        # alpha_src[n, h] = sum_f H_prime[n, f, h] * att_src[f, h]
        # F_h is full on every rank, so no AR needed.
        ax.get_timers().start("gat: per-node alpha")
        alpha_src = torch.einsum("nfh,fh->nh", H_prime, att_src_full)  # [N/outer, H]
        alpha_dst = torch.einsum("nfh,fh->nh", H_prime, att_dst_full)
        ax.get_timers().stop("gat: per-node alpha")

        # ----- (6) reshard alpha_dst from "rows by outer" to "rows by depth" -----
        # alpha_dst is [N/outer, H] with H replicated.  Forward AllGather over
        # outer (autograd-aware: backward ReduceScatter) and narrow by depth.
        ax.get_timers().start("gat: reshard alpha_dst")
        alpha_dst_full = ForwardGather_BackwardReduceScatter.apply(
            alpha_dst.contiguous(), self.outer_group, 0, False, False
        )

        depth_rank = dist.get_rank(group=self.depth_group)
        depth_required = N_dst_local * self.depth_group_size
        if alpha_dst_full.shape[0] < depth_required:
            pad_rows = depth_required - alpha_dst_full.shape[0]
            alpha_dst_full = F.pad(alpha_dst_full, (0, 0, 0, pad_rows))
        alpha_dst_depth = alpha_dst_full.narrow(
            0, depth_rank * N_dst_local, N_dst_local
        ).contiguous()
        ax.get_timers().stop("gat: reshard alpha_dst")

        # ----- (7) edge LeakyReLU score per head -----
        ax.get_timers().start("gat: edge score")
        row_idx = _csr_row_index_per_nnz(crow)            # [E_local]
        e = alpha_src[col] + alpha_dst_depth[row_idx]     # [E_local, H]
        e = F.leaky_relu(e, self.negative_slope)
        ax.get_timers().stop("gat: edge score")

        # ----- (8) distributed per-row, per-head softmax over outer -----
        ax.get_timers().start("gat: row softmax")
        with torch.no_grad():
            row_max = _scatter_max_per_head(e.detach(), row_idx, dim_size=N_dst_local)
            if dist.is_initialized() and dist.get_world_size(self.outer_group) > 1:
                row_max = torch.where(
                    torch.isinf(row_max) & (row_max < 0),
                    torch.full_like(row_max, torch.finfo(row_max.dtype).min / 2),
                    row_max,
                )
                dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=self.outer_group)
            row_max = torch.where(
                torch.isinf(row_max) & (row_max < 0),
                torch.zeros_like(row_max),
                row_max,
            )

        e_shifted = e - row_max[row_idx]                  # [E, H]
        exp_e = torch.exp(e_shifted)
        row_sum_local = _scatter_sum_per_head(exp_e, row_idx, dim_size=N_dst_local)
        row_sum_global = _ar_sum(row_sum_local, self.outer_group)
        alpha = exp_e / (row_sum_global[row_idx] + 1e-16)  # [E, H]
        ax.get_timers().stop("gat: row softmax")

        # ----- (9) custom-autograd weighted SpMM on H_prime (per-head F_h) -----
        # AGG[i, f, h] = sum_{k in row i} alpha[k, h] * H_prime[col[k], f, h]
        # Output: [N/depth, F_h, H], partial along outer (sum over j across outer).
        ax.get_timers().start("AGG = A_attn * H_prime")
        AGG = _AlphaWeightedSpMM3D.apply(
            alpha,
            H_prime,
            crow,
            edge_index.col_indices(),
            N_dst_local,
        )
        ax.get_timers().stop("AGG = A_attn * H_prime")
        ax.get_timers().start("allreduce H")
        AGG = _ar_sum(AGG, self.outer_group)              # [N/depth, F_h, H]
        ax.get_timers().stop("allreduce H")

        # ----- (10) drop AGG to local F_h slice (autograd: forward slice, backward AllGather) -----
        # Forward: each outer rank keeps its own F_h_local slice (no comm).
        # Backward: AllGather grads along F_h so the local W / H_prime grads
        # see the full-F_h gradient signal.  This re-shards back to the
        # storage layout so the next layer's input matches expectations.
        if dist.is_initialized() and dist.get_world_size(self.outer_group) > 1:
            AGG_local = Drop.apply(AGG, self.outer_group, 1)   # [N_dst, F_h_local, H]
        else:
            AGG_local = AGG

        # ----- (11) reshape and add bias -----
        OUT = AGG_local.reshape(N_dst_local, F_h_loc * H_heads)
        if self.bias is not None:
            OUT = OUT + self.bias

        # Sentinel: this Function's backward is the FIRST one to fire in this
        # layer's backward subgraph (it sits at OUT).  Starts the
        # "gat conv bwd" timer that _BwdTimerStop closes on the input side.
        if OUT.requires_grad:
            OUT = _BwdTimerStart.apply(OUT)

        ax.get_timers().stop("gat conv fwd")
        return OUT
