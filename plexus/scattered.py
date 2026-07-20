# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

"""
Helpers for the scattered activation layout used by GCNConvRS (hybrid /
half-collective schemes, see analysis/gcn_parallel_cost.tex).

Layout contract: between GCN layers the activation is fully row-scattered.
Each GPU holds one contiguous block of `padded_N / P` global rows with FULL
feature columns.  The block index follows the nesting (c > f > r) of the
consuming layer's axis rotation:

    layer l%3==0: (c,f,r) = ("x","z","y")
    layer l%3==1:            ("y","x","z")
    layer l%3==2:            ("z","y","x")

The input of layer 0 (and, because the rotation has period 3, the output of
layer L for L%3==0) therefore lives on contiguous row blocks ordered by
(rank_x, rank_z, rank_y).

Because feature columns are never sharded in this layout:
  - linears are plain replicated-weight row-parallel ops (zero forward
    communication; weight grads are partial sums over the scattered rows and
    must be all-reduced over the whole intra-layer group),
  - RMSNorm reduces over the full local row (no communication),
  - cross-entropy / argmax are local per row (one scalar/statistics
    all-reduce over the intra-layer group).
"""

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from axonn import axonn as ax
from plexus.utils.general import get_process_groups_info


def scattered_nesting_letters(layer_num: int):
    """(c, f, r) axis letters of the input nesting of GCN layer `layer_num`
    (use layer_num = L to get the nesting of the final layer's output)."""
    if layer_num % 3 == 0:
        return ("x", "z", "y")
    if layer_num % 3 == 1:
        return ("y", "x", "z")
    return ("z", "y", "x")


def scattered_bounds(padded_N: int, layer_num: int = 0):
    """Global row range [start, stop) of this rank's scattered block for the
    input of `layer_num`.  padded_N must be divisible by P."""
    letters = scattered_nesting_letters(layer_num)
    num_gpus, ranks, _ = get_process_groups_info(letters)
    P = num_gpus[0] * num_gpus[1] * num_gpus[2]
    assert padded_N % P == 0, f"padded_N={padded_N} not divisible by P={P}"
    n0 = padded_N // P
    block = (ranks[0] * num_gpus[1] + ranks[1]) * num_gpus[2] + ranks[2]
    return block * n0, (block + 1) * n0


def intra_group():
    return ax.comm_handle.intra_layer_group


def steady_cell(n_coarse: int, layer_num: int):
    """[start, stop) of this rank's hierarchical cell WITHIN its coarse
    input block of size n_coarse, for the input nesting of `layer_num`
    (matches GCNConvRS's _splits subdivision).  Used by the minibatch path
    where n_coarse = number of sampled nodes in the coarse block."""
    from plexus.gcn_conv_rs import _splits

    letters = scattered_nesting_letters(layer_num)
    num_gpus, ranks, _ = get_process_groups_info(letters)
    _, Gf, Gr = num_gpus
    _, rf, rr = ranks
    s_f = _splits(n_coarse, Gf)
    a = sum(s_f[:rf])
    s_fr = _splits(s_f[rf], Gr)
    a += sum(s_fr[:rr])
    return a, a + s_fr[rr]


class ScatteredLinear(torch.nn.Module):
    """Replicated-weight linear on row-scattered input: out = x @ W + b.

    Forward/backward for activations are fully local.  Weight/bias grads
    are partial sums over the local rows; call sync_replicated_gradients()
    (or rely on the train loop's sync) to all-reduce them over the
    intra-layer group after backward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # same seed on every rank -> identical replicas.  Weight is stored
        # (in, out); use the nn.Linear bound 1/sqrt(fan_in) explicitly
        # (kaiming_uniform_ would treat dim 1 = out as fan_in here).
        w = torch.empty(in_features, out_features, device="cuda")
        bound = 1.0 / math.sqrt(in_features)
        w.uniform_(-bound, bound)
        self.weight = torch.nn.Parameter(w, requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_features, device="cuda"), requires_grad=True
            )
        else:
            self.bias = None
        self._replicated_grad = True     # marker for sync_replicated_gradients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


def sync_replicated_gradients(model: torch.nn.Module, mean: bool = False):
    """All-reduce grads of replicated parameters (ScatteredLinear weights,
    norm weights with full features) over the intra-layer group.

    GCNConvRS weights are excluded: their backward already all-reduces.
    """
    group = intra_group()
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        return
    for module in model.modules():
        if not getattr(module, "_replicated_grad", False):
            continue
        for p in module.parameters(recurse=False):
            if p.grad is not None:
                dist.all_reduce(p.grad, group=group)
                if mean:
                    p.grad.div_(world)


class ScatteredRMSNorm(torch.nn.Module):
    """Plain RMSNorm over the full (unsharded) feature dimension.  No
    forward communication; weight grad synced via sync_replicated_gradients."""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(size, device="cuda"), requires_grad=True
        )
        self._replicated_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        norm = xf.pow(2).mean(dim=-1, keepdim=True)
        out = xf * torch.rsqrt(norm + self.eps) * self.weight
        return out.to(x.dtype)


def scattered_cross_entropy(logits, target, padded_N, num_nodes, num_layers,
                            node_mask=None):
    """Cross-entropy on row-scattered logits (n_loc x FULL num_classes).

    Rows are this rank's final-layer scattered block (nesting of "layer
    num_layers"); rows whose global id >= num_nodes (padding) and rows
    excluded by node_mask are ignored.  The returned scalar is the global
    mean over all valid rows: each rank's backward contribution is
    local_sum / global_count, so calling .backward() on the returned loss
    yields exact global gradients.
    """
    device = logits.device
    n_loc = logits.shape[0]

    if padded_N is None:
        # minibatch: every local row is a real sampled node
        valid = target >= 0
    else:
        start, stop = scattered_bounds(padded_N, layer_num=num_layers)
        global_idx = torch.arange(n_loc, device=device) + start
        valid = (global_idx < num_nodes) & (target >= 0)
    if node_mask is not None:
        valid = valid & node_mask.bool()

    tgt = target.clone()
    tgt[~valid] = -1
    local_sum = F.cross_entropy(
        logits, tgt, reduction="sum", ignore_index=-1
    )
    count = valid.sum()
    if dist.is_initialized():
        dist.all_reduce(count, group=intra_group())
    count = count.clamp_min(1)

    loss = local_sum / count            # backward-correct local share
    loss_report = loss.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(loss_report, group=intra_group())
    # return a tensor whose value is the global loss but whose grad_fn is
    # the local share (standard data-parallel loss trick)
    return loss - loss.detach() + loss_report


@torch.no_grad()
def scattered_argmax(logits, padded_N, num_nodes, num_layers):
    """Row-local argmax over the full class dim; -1 for padded rows."""
    start, stop = scattered_bounds(padded_N, layer_num=num_layers)
    pred = logits.argmax(dim=1)
    global_idx = torch.arange(pred.shape[0], device=pred.device) + start
    pred[global_idx >= num_nodes] = -1
    return pred
