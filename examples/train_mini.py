# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import torch
import os
import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from axonn import axonn as ax
import torch.nn.functional as F
from torch.profiler import _KinetoProfile, record_function
from axonn.intra_layer.communication import Drop, Gather
from plexus import plexus as plx
import torch.distributed as dist
from plexus.gcn_conv import GCNConv
from plexus.linear import PlexusLinear
from plexus.linear_3d import Plexus3DLinear
from plexus.norm import PlexusRMSNorm, sync_norm_gradients, check_norm_weight_consistency
from plexus.utils.dataloader import DataLoader
from plexus.cross_entropy import parallel_cross_entropy, parallel_bce_with_logits
from plexus.utils.general import set_seed, print_axonn_timer_data, get_process_groups_info
from plexus.utils.subgraph_sampler import (
    compute_steps_per_epoch,
    sample_nodes,
    sample_nodes_epoch_perm,
    sample_nodes_hub_anchor,
    sample_nodes_train_hub,
    sample_nodes_degree_gumbel,
    poisson_inclusion_probs,
    build_layout_indices,
    build_compact_adj_shards,
)

PROFILE_START_EPOCH = 6
PROFILE_END_EPOCH = 9
_KinetoProfile._get_distributed_info = lambda self: None

# arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--G_intra_r", type=int, default=1)
    parser.add_argument("--G_intra_c", type=int, default=1)
    parser.add_argument("--G_intra_d", type=int, default=1)
    parser.add_argument(
        "--G_data",
        type=int,
        default=1,
        help="Number of data-parallel groups (DP dimension).",
    )
    parser.add_argument("--gpus_per_node", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument(
        "--block_aggregation",
        action="store_true",
        default=False,
        help="Enable 1D blocking in aggregation",
    )
    parser.add_argument(
        "--overlap_aggregation",
        action="store_true",
        default=False,
        help="Enable overlap in aggregation",
    )
    parser.add_argument(
        "--overlap_bwd",
        action="store_true",
        default=False,
        help="Overlap GCN backward all-reduce with gradient compute.",
    )
    parser.add_argument(
        "--allreduce_lowp",
        action="store_true",
        default=False,
        help=(
            "Use low-precision communication for selected hotspot all-reduces "
            "(cast before all-reduce, cast back after)."
        ),
    )
    parser.add_argument(
        "--allreduce_lowp_dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp16"),
        help="Communication dtype for --allreduce_lowp.",
    )
    parser.add_argument(
        "--tune_gemms",
        action="store_true",
        default=False,
        help="Enables tuning of dense matrix multiplications",
    )
    parser.add_argument("--timing_start_epoch", type=int, default=None)
    parser.add_argument("--timing_end_epoch", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_gcn_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument(
        "--use_profiler",
        action="store_true",
        default=False,
        help="Enable torch profiler collection (default: disabled).",
    )
    parser.add_argument(
        "--use_3d_linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 3D linear layout rotation (default: enabled).",
    )
    parser.add_argument(
        "--train_features",
        action="store_true",
        default=False,
        help="Enable training of input features (include x in optimizer).",
    )
    parser.add_argument("--train_adj", action="store_true", default=False,
        help="During training, restrict message passing to the train-induced subgraph (edges between train nodes only), if available. Evaluation still uses the full graph.")
    parser.add_argument(
        "--minibatch_nodes",
        type=int,
        default=None,
        help="Number of nodes to sample per iteration (enables mini-batch training).",
    )
    parser.add_argument(
        "--minibatch_ratio",
        type=float,
        default=None,
        help="Fraction of nodes to sample per iteration (enables mini-batch training).",
    )
    parser.add_argument(
        "--minibatch_seed",
        type=int,
        default=None,
        help="Seed for deterministic sampling (defaults to --seed).",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Number of iterations per epoch (defaults to ceil(num_nodes / minibatch_nodes) when sampling).",
    )
    parser.add_argument(
        "--minibatch_unbiased",
        action="store_true",
        default=False,
        help="Apply unbiased edge scaling for node sampling (scale non-self edges by 1/p).",
    )
    parser.add_argument(
        "--minibatch_sampler",
        type=str,
        default="uniform",
        choices=["uniform", "hub", "degree", "train_hub"],
        help=(
            "Node sampling strategy: 'uniform' (default, current behavior), "
            "'hub' (always include top-degree hub nodes, fill the rest "
            "uniformly), or 'degree' (weighted sampling with probability "
            "proportional to degree^alpha via Gumbel top-B). Non-uniform "
            "samplers reweight the loss by 1/pi and, with "
            "--minibatch_unbiased, rescale each non-self edge by 1/pi of its "
            "source node."
        ),
    )
    parser.add_argument(
        "--minibatch_hub_frac",
        type=float,
        default=0.2,
        help="Fraction of the batch budget reserved for hub nodes (sampler=hub).",
    )
    parser.add_argument(
        "--minibatch_degree_alpha",
        type=float,
        default=1.0,
        help="Exponent alpha in sampling weight (degree+1)^alpha (sampler=degree).",
    )
    parser.add_argument(
        "--minibatch_train_steps",
        type=int,
        default=10,
        help=(
            "sampler=train_hub: steps per epoch; each epoch covers every TRAIN "
            "node exactly once as a loss target. --minibatch_ratio then sizes "
            "the extra uniform source draw and --minibatch_hub_frac the hub "
            "anchors (as a fraction of that draw)."
        ),
    )
    parser.add_argument(
        "--minibatch_dp_diverse_src",
        action="store_true",
        default=False,
        help=(
            "sampler=train_hub with G_data>1: each DP group draws its own "
            "uniform sources (targets stay disjoint via the shared seed)."
        ),
    )
    parser.add_argument(
        "--minibatch_epoch_perm",
        action="store_true",
        default=False,
        help=(
            "Draw batches as chunks of one per-epoch random permutation shared "
            "across DP groups (full coverage per epoch, exact same 1/p "
            "rescaling). Supported for sampler=uniform and sampler=hub."
        ),
    )
    parser.add_argument(
        "--eval_degree_buckets",
        action="store_true",
        default=False,
        help="Report eval accuracy stratified by node degree buckets.",
    )
    parser.add_argument(
        "--grad_clip_value",
        type=float,
        default=None,
        help=(
            "Clip each gradient element to [-v, v] before the optimizer step. "
            "Element-wise clipping is exact under sharded parameters (no "
            "cross-group communication needed)."
        ),
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
        help="LR schedule: constant (default) or cosine annealing over num_epochs.",
    )
    parser.add_argument(
        "--lr_warmup_epochs",
        type=int,
        default=0,
        help="Linear LR warmup over this many epochs before the cosine phase.",
    )
    parser.add_argument(
        "--grad_variance_ratios",
        type=str,
        default=None,
        help=(
            "Comma-separated ratios, e.g. '0.02,0.05,0.1'. With "
            "--grad_variance_samples, measure uniform-sampling gradient "
            "variance at ALL these ratios on the same frozen parameters."
        ),
    )
    parser.add_argument(
        "--grad_variance_samples",
        type=int,
        default=0,
        help=(
            "After training, hold parameters fixed and draw K fresh mini-batches "
            "to estimate gradient variance (relative variance = sum Var[g] / "
            "sum E[g]^2). 0 disables."
        ),
    )
    parser.add_argument(
        "--minibatch_compact",
        action="store_true",
        default=False,
        help="Build compact subgraphs (reduces tensor sizes per shard).",
    )
    parser.add_argument(
        "--overlap_samp",
        action="store_true",
        default=False,
        help=(
            "Overlap next-step compact minibatch preparation with current-step "
            "training."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help=(
            "After every optimizer step, verify that norm weights are identical "
            "across replication dimensions. Useful for detecting gradient sync bugs."
        ),
    )
    parser.add_argument(
        "--bf16_spmm",
        action="store_true",
        default=False,
        help="Perform sparse matrix multiplications (SPMM) in BF16.",
    )
    parser.add_argument(
        "--bf16_gemm",
        action="store_true",
        default=False,
        help="Perform dense matrix multiplications (GEMM) in BF16.",
    )
    parser.add_argument(
        "--avg_grad",
        action="store_true",
        default=False,
        help=(
            "Average (instead of sum) replicated-parameter gradients across "
            "their replication dimension."
        ),
    )
    parser.add_argument("--multilabel_metric", type=str, default="rocauc",
        choices=["rocauc", "f1_micro"],
        help="Metric for multi-label evaluation: rocauc (default, for ogbn-proteins) or f1_micro (for yelp).")
    parser.add_argument(
        "--compile_norm",
        action="store_true",
        default=False,
        help="Use torch.compile on RMSNorm forward to fuse elementwise kernels.",
    )
    parser.add_argument(
        "--overlap_fwd_comm",
        action="store_true",
        default=False,
        help="Overlap AR(AGG) with Allgather(W) in GCN forward (parallel async NCCL on different groups).",
    )
    parser.add_argument(
        "--overlap_bwd_comm",
        action="store_true",
        default=False,
        help="Overlap AR(grad_agg) with [GRAD_W + RS(grad_W)] in GCN backward (parallel async NCCL).",
    )
    parser.add_argument(
        "--overlap_linear_bwd",
        action="store_true",
        default=False,
        help="Overlap AR(grad_x) with AR(grad_W+bias) in Linear3D backward (parallel async NCCL).",
    )
    parser.add_argument(
        "--vectorize_dp_grad",
        action="store_true",
        default=False,
        help=(
            "Flatten all parameter gradients into a single buffer before the "
            "data-parallel all-reduce (one large AR instead of many small ones)."
        ),
    )
    parser.add_argument(
        "--fuse_norm_activation",
        action="store_true",
        default=False,
        help=(
            "Fuse RMSNorm post-processing, ReLU, and Dropout into a single "
            "torch.compile kernel (implies --compile_norm)."
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability (default: 0.3).",
    )
    return parser


class Net(torch.nn.Module):
    """
    Define the GCN here
    """

    def __init__(
        self,
        num_gcn_layers,
        input_size,
        hidden_size,
        output_size,
        train_features: bool = False,
        fuse_norm_activation: bool = False,
        dropout: float = 0.3,
    ):
        super(Net, self).__init__()

        self.num_gcn_layers = num_gcn_layers
        self.fuse_norm_activation = fuse_norm_activation
        self.dropout = dropout

        node_group, class_group = _loss_groups(self.num_gcn_layers)
        last_outer, _, _ = _layer_groups(self.num_gcn_layers - 1)

        if plx.use_3d_linear:
            # Pre-linear: input -> hidden (row=x, k=y, col=z)
            self.input_linear = Plexus3DLinear(
                input_size,
                hidden_size,
                row_group="x",
                k_group="y",
                col_group="z",
                pad_in_features_with_depth=True,
                pad_out_features_with_depth=True,
                gather_features_in_depth=train_features,
                matmul_name="input_linear",
            )
        else:
            # Pre-linear: input -> hidden (row-parallel, feature group=y)
            self.input_linear = PlexusLinear(
                input_size,
                hidden_size,
                feature_group="y",
                pad_in_features_with_depth=True,
                pad_out_features_with_depth=True,
                gather_features_in_depth=train_features,
            )

        # Initialize layers and norms as ModuleList to register them with the module
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        # for i in range(self.num_gcn_layers):
        #     if i == 0:
        #         self.layers.append(GCNConv(input_size, hidden_size, i, shard_features_in_depth=False))
        #         self.norms.append(PlexusRMSNorm(hidden_size, feature_group=_outer_group_letter(i)))
        #     else:
        #         self.layers.append(GCNConv(hidden_size, hidden_size, i, shard_features_in_depth=False))
        #         self.norms.append(PlexusRMSNorm(hidden_size, feature_group=_outer_group_letter(i)))
                
        for i in range(self.num_gcn_layers):
            self.layers.append(GCNConv(hidden_size, hidden_size, i, shard_features_in_depth=False))
            _, inner_g, depth_g = _layer_groups(i)
            self.norms.append(PlexusRMSNorm(
                hidden_size,
                feature_group=_outer_group_letter(i),
                data_group=depth_g,
                inner_group=inner_g,
            ))

        if plx.use_3d_linear:
            # Post-linear: hidden -> classes (row=node_group, k=last_outer, col=class_group)
            self.output_linear = Plexus3DLinear(
                hidden_size,
                output_size,
                row_group=node_group,
                k_group=last_outer,
                col_group=class_group,
                pad_in_features_with_depth=False,
                pad_out_features_with_depth=False,
                gather_features_in_depth=False,
                matmul_name="output_linear",
            )
        else:
            # Post-linear: hidden -> classes (row-parallel, feature group=class_group)
            self.output_linear = PlexusLinear(
                hidden_size,
                output_size,
                feature_group=class_group,
                pad_in_features_with_depth=False,
                pad_out_features_with_depth=False,
                gather_features_in_depth=False,
            )

    def forward(self, x, edge_index_shards, layout_metadata=None):
        ax.get_timers().start("input_linear")
        x = self.input_linear(x)
        ax.get_timers().stop("input_linear")
        
        for i in range(self.num_gcn_layers):
            # residual = x
            x = self.layers[i](x, edge_index_shards)
            
            ax.get_timers().start("activation")
            x = self.norms[i](x)
            if not self.fuse_norm_activation:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # if layout_metadata is None:
            #     residual = _reshard_residual(residual, i)
            # else:
            #     residual = _reshard_residual_compact(
            #         residual,
            #         i,
            #         layout_metadata,
            #     )
            # x = x + residual
            ax.get_timers().stop("activation")

        ax.get_timers().start("output_linear")
        x = self.output_linear(x)
        ax.get_timers().stop("output_linear")
        return x


def _sync_data_parallel_gradients(
    optimizer, mean: bool = True, vectorize: bool = False
) -> None:
    if not dist.is_initialized():
        return
    dp_group = ax.comm_handle.data_parallel_group
    if dp_group is None:
        return
    dp_world = dist.get_world_size(dp_group)
    if dp_world <= 1:
        return
    ax.get_timers().start("dp grad sync")
    grads = []
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param.grad is not None:
                grads.append(param.grad)
    if grads:
        if vectorize:
            from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, group=dp_group)
            if mean:
                flat.div_(dp_world)
            for old_tensor, new_tensor in zip(
                grads, _unflatten_dense_tensors(flat, grads)
            ):
                old_tensor.data = new_tensor
        else:
            for grad in grads:
                dist.all_reduce(grad, group=dp_group)
                if mean:
                    grad.div_(dp_world)
    ax.get_timers().stop("dp grad sync")


# called each epoch
def train(
    model,
    optimizer,
    features_local,
    adj_shards,
    labels,
    train_mask,
    num_nodes,
    num_classes,
    layout_metadata=None,
    test: bool = False,
    vectorize_dp_grad: bool = False,
    node_weight=None,
    grad_clip_value: float | None = None,
):
    # set to training mode
    model.train()

    # set gradients of optimized parameters to 0
    optimizer.zero_grad()

    # forward pass
    output = model(features_local, adj_shards, layout_metadata=layout_metadata)

    if labels.ndim == 2 and labels.size(-1) > 1:
        # Multi-label (e.g., ogbn-proteins): BCE-with-logits on raw scores.
        loss = parallel_bce_with_logits(
            output,
            labels,
            model.num_gcn_layers,
            num_nodes,
            num_classes,
            node_mask=train_mask,
            node_weight=node_weight,
        )
    else:
        # Single-label: cross-entropy over classes.
        loss = parallel_cross_entropy(
            output,
            labels,
            model.num_gcn_layers,
            num_nodes,
            num_classes,
            node_mask=train_mask,
            node_weight=node_weight,
        )

    # backward pass
    loss.backward()
    ax.get_timers().start("sync_norm_gradients")
    sync_norm_gradients(model.norms, mean=plx.avg_grad)
    ax.get_timers().stop("sync_norm_gradients")
    ax.get_timers().start("sync_data_parallel_gradients")
    _sync_data_parallel_gradients(optimizer, vectorize=vectorize_dp_grad)
    ax.get_timers().stop("sync_data_parallel_gradients")
    if grad_clip_value is not None:
        params = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_value_(params, grad_clip_value)
    # update weights
    optimizer.step()

    if test:
        check_norm_weight_consistency(model.norms)

    return loss


def _minibatch_size(num_nodes: int, batch_size: int | None, ratio: float | None) -> int:
    if batch_size is not None and ratio is not None:
        raise ValueError("Provide only one of batch_size or ratio.")
    if ratio is not None:
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be in (0, 1].")
        batch_size = max(1, int(math.floor(num_nodes * ratio)))
    if batch_size is None:
        raise ValueError("Provide batch_size or ratio for sampling.")
    return max(1, min(int(batch_size), num_nodes))


def _edge_scale_value(num_nodes: int, batch_size: int) -> float | None:
    if num_nodes > 1 and batch_size > 1:
        p_neighbor = (batch_size - 1) / (num_nodes - 1)
        return 1.0 / p_neighbor
    return None


def _compute_global_degrees(adj_shards, data_loader, num_nodes: int) -> torch.Tensor:
    """Non-self degree of every node, identical on all ranks.

    Each rank counts the nonzeros of its layout-0 adjacency shard per global
    row; a one-time all-reduce over the TP group sums the column blocks. The
    shard grid is replicated along the remaining 3D axis, so the sum
    overcounts by that replication factor.
    """
    adj = adj_shards[0][0]
    crow = adj.crow_indices()
    counts = (crow[1:] - crow[:-1]).to(torch.int64)

    if plx.use_3d_linear:
        dim1_letter, dim2_letter = "y", "x"
    else:
        dim1_letter, dim2_letter = "z", "x"
    rep_letter = ({"x", "y", "z"} - {dim1_letter, dim2_letter}).pop()
    num_gpus, _, _ = get_process_groups_info((dim1_letter, dim2_letter, rep_letter))
    rep = int(num_gpus[2])

    deg = torch.zeros(num_nodes, dtype=torch.int64, device=adj.device)
    row_start = int(data_loader.adj_dim1_start[0])
    row_stop = int(data_loader.adj_dim1_stop[0])
    hi = min(row_stop, num_nodes)
    if hi > row_start:
        deg[row_start:hi] = counts[: hi - row_start]
    dist.all_reduce(deg, op=dist.ReduceOp.SUM, group=ax.comm_handle.intra_layer_group)
    deg = torch.div(deg, rep, rounding_mode="floor")
    # remove the single self-loop each node carries
    deg = (deg - 1).clamp_min(0)
    return deg.to(torch.float32)


def _slice_node_vector(
    vec_global: torch.Tensor, start: int, stop: int, num_nodes: int, fill: float
) -> torch.Tensor:
    out = vec_global.new_full((stop - start,), fill)
    hi = min(stop, num_nodes)
    if hi > start:
        out[: hi - start] = vec_global[start:hi]
    return out


def _compute_global_train_mask(
    train_mask_local: torch.Tensor, labels_start: int, labels_stop: int, num_nodes: int
) -> torch.Tensor:
    """Assemble the global train mask from per-rank label shards (one-time)."""
    mask = torch.zeros(num_nodes, dtype=torch.int32, device=train_mask_local.device)
    hi = min(labels_stop, num_nodes)
    if hi > labels_start:
        mask[labels_start:hi] = train_mask_local[: hi - labels_start].to(torch.int32)
    dist.all_reduce(mask, op=dist.ReduceOp.MAX, group=ax.comm_handle.intra_layer_group)
    return mask.to(torch.bool)


def _build_sampler_ctx(
    args,
    degrees: torch.Tensor | None,
    num_nodes: int,
    batch_size: int,
    seed_base: int,
    minibatch_seed: int,
    dp_rank: int,
    labels_start: int,
    labels_stop: int,
    train_mask_local: torch.Tensor | None = None,
) -> dict:
    """Precompute static state for non-uniform / epoch-permutation sampling.

    All quantities derive from static graph statistics and shared seeds, so
    every rank builds identical tensors without communication (degrees were
    all-reduced once at initialization).
    """
    ctx = {
        "type": args.minibatch_sampler,
        "epoch_perm": bool(args.minibatch_epoch_perm),
        "batch_size": int(batch_size),
        "seed_base": int(seed_base),
        "minibatch_seed": int(minibatch_seed),
        "G_data": int(args.G_data),
        "dp_rank": int(dp_rank),
        "hub_ids": None,
        "hub_mask": None,
        "log_weights": None,
        "col_scale": None,
        "node_weight_local": None,
    }
    if args.minibatch_sampler == "uniform":
        return ctx

    assert degrees is not None
    pi = None
    if args.minibatch_sampler == "hub":
        k = int(args.minibatch_hub_frac * batch_size)
        k = max(1, min(k, batch_size - 1, num_nodes - 1))
        # stable sort so hub selection breaks degree ties identically everywhere
        _, order = torch.sort(degrees, descending=True, stable=True)
        hub_ids = order[:k].contiguous()
        hub_mask = torch.zeros(num_nodes, dtype=torch.bool, device=degrees.device)
        hub_mask[hub_ids] = True
        p_rest = (batch_size - k) / max(1, (num_nodes - k))
        pi = torch.full(
            (num_nodes,), p_rest, dtype=torch.float32, device=degrees.device
        )
        pi[hub_ids] = 1.0
        ctx["hub_ids"] = hub_ids
        ctx["hub_mask"] = hub_mask
        ctx["hub_k"] = k
    elif args.minibatch_sampler == "degree":
        w = (degrees + 1.0).pow(args.minibatch_degree_alpha)
        pi = poisson_inclusion_probs(w, batch_size)
        ctx["log_weights"] = torch.log(w)
    elif args.minibatch_sampler == "train_hub":
        assert train_mask_local is not None
        gmask = _compute_global_train_mask(
            train_mask_local, labels_start, labels_stop, num_nodes
        )
        train_ids = gmask.nonzero(as_tuple=False).squeeze(1).contiguous()
        n_train = int(train_ids.numel())
        T = max(1, int(args.minibatch_train_steps))
        chunks_total = T * max(1, int(args.G_data))
        b_target = max(1, math.ceil(n_train / chunks_total))
        b_uniform = int(batch_size)  # --minibatch_ratio sizes the source draw
        k = int(args.minibatch_hub_frac * b_uniform)
        k = max(0, min(k, num_nodes - 1))
        _, order = torch.sort(degrees, descending=True, stable=True)
        hub_ids = order[:k].contiguous()
        ctx["train_ids"] = train_ids
        ctx["hub_ids"] = hub_ids
        ctx["b_target"] = b_target
        ctx["b_uniform"] = b_uniform
        ctx["train_steps"] = T
        ctx["dp_diverse_src"] = bool(args.minibatch_dp_diverse_src)
        # static inclusion probs: targets p_t (train only), uniform p_u, hubs 1
        p_t = min(1.0, b_target * chunks_total / max(1, n_train))
        p_u = min(1.0, b_uniform / num_nodes)
        pi = torch.full(
            (num_nodes,), p_u, dtype=torch.float32, device=degrees.device
        )
        # per-step target inclusion for train nodes: b_target/n_train, plus p_u
        p_t_step = b_target / max(1, n_train)
        pi[gmask] = 1.0 - (1.0 - p_t_step) * (1.0 - p_u)
        if k > 0:
            pi[hub_ids] = 1.0
        pi = pi.clamp(1e-12, 1.0)
        # unweighted loss: each train node is a target exactly once per epoch
        inv_pi = 1.0 / pi
        if args.minibatch_unbiased:
            ctx["col_scale"] = inv_pi
        ctx["node_weight_local"] = None
        return ctx
    else:
        raise ValueError(f"Unknown sampler {args.minibatch_sampler!r}")

    inv_pi = 1.0 / pi
    if args.minibatch_unbiased:
        ctx["col_scale"] = inv_pi
    ctx["node_weight_local"] = _slice_node_vector(
        inv_pi, labels_start, labels_stop, num_nodes, fill=0.0
    )
    return ctx


def _sample_minibatch_nodes(
    sampler_ctx: dict | None,
    num_nodes: int,
    minibatch_nodes: int | None,
    minibatch_ratio: float | None,
    minibatch_seed: int,
    global_step: int,
    attempt: int,
    steps_per_epoch: int,
    device: torch.device,
) -> torch.Tensor:
    if sampler_ctx is None or (
        sampler_ctx["type"] == "uniform" and not sampler_ctx["epoch_perm"]
    ):
        return sample_nodes(
            num_nodes,
            batch_size=minibatch_nodes,
            ratio=minibatch_ratio,
            seed=minibatch_seed,
            step=global_step + attempt,
            device=device,
        )

    stype = sampler_ctx["type"]
    batch_size = sampler_ctx["batch_size"]
    epoch = global_step // max(1, steps_per_epoch)
    local_chunk = global_step % max(1, steps_per_epoch)
    chunk = local_chunk * sampler_ctx["G_data"] + sampler_ctx["dp_rank"]

    if stype == "uniform":
        # epoch permutation shared across DP groups; chunks are disjoint
        return sample_nodes_epoch_perm(
            num_nodes,
            batch_size,
            seed=sampler_ctx["seed_base"],
            epoch=epoch,
            chunk=chunk,
            device=device,
        )
    if stype == "hub":
        if sampler_ctx["epoch_perm"]:
            return sample_nodes_hub_anchor(
                num_nodes,
                batch_size,
                sampler_ctx["hub_ids"],
                sampler_ctx["hub_mask"],
                seed=sampler_ctx["seed_base"],
                step=0,
                device=device,
                epoch=epoch,
                chunk=chunk,
            )
        return sample_nodes_hub_anchor(
            num_nodes,
            batch_size,
            sampler_ctx["hub_ids"],
            sampler_ctx["hub_mask"],
            seed=sampler_ctx["minibatch_seed"],
            step=global_step + attempt,
            device=device,
        )
    if stype == "train_hub":
        return sample_nodes_train_hub(
            num_nodes,
            sampler_ctx["train_ids"],
            sampler_ctx["b_target"],
            sampler_ctx["hub_ids"],
            sampler_ctx["b_uniform"],
            seed=sampler_ctx["seed_base"],
            epoch=epoch,
            chunk=chunk,
            step=global_step + attempt,
            device=device,
            src_seed=(
                sampler_ctx["minibatch_seed"]
                if sampler_ctx.get("dp_diverse_src")
                else None
            ),
        )
    if stype == "degree":
        return sample_nodes_degree_gumbel(
            sampler_ctx["log_weights"],
            batch_size,
            seed=sampler_ctx["minibatch_seed"],
            step=global_step + attempt,
            device=device,
        )
    raise ValueError(f"Unknown sampler {stype!r}")


def _scale_csr_non_self(
    csr: torch.Tensor, row_offset: int, col_offset: int, scale: float
) -> None:
    if csr._nnz() == 0:
        return
    crow = csr.crow_indices()
    col = csr.col_indices()
    vals = csr.values()
    row_counts = crow[1:] - crow[:-1]
    if row_counts.numel() == 0:
        return
    row_ids = torch.repeat_interleave(
        torch.arange(csr.size(0), device=csr.device, dtype=torch.long),
        row_counts,
    )
    if row_ids.numel() == 0:
        return
    row_global = row_ids + int(row_offset)
    col_global = col + int(col_offset)
    non_self = row_global != col_global
    if non_self.any():
        vals[non_self] = vals[non_self] * float(scale)


def _scale_adj_shards_non_self(
    adj_shards, row_starts, col_starts, scale: float
) -> None:
    if scale is None or scale == 1.0:
        return
    num_layouts = len(row_starts)
    for shard_idx, (adj, adj_t) in enumerate(adj_shards):
        layout_idx = shard_idx % num_layouts
        row_offset = row_starts[layout_idx]
        col_offset = col_starts[layout_idx]
        _scale_csr_non_self(adj, row_offset, col_offset, scale)
        _scale_csr_non_self(adj_t, col_offset, row_offset, scale)


def _prepare_compact_minibatch(
    *,
    model,
    data_loader,
    train_adj_shards,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor | None,
    num_nodes: int,
    minibatch_nodes: int | None,
    minibatch_ratio: float | None,
    minibatch_seed: int,
    global_step: int,
    edge_scale: float | None,
    resample_attempts: int = 3,
    enable_timers: bool = True,
    sampler_ctx: dict | None = None,
    steps_per_epoch: int = 1,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor | None,
        int,
        dict,
        torch.Tensor | None,
    ]
    | None
):
    timers = ax.get_timers() if enable_timers else None
    # epoch-permutation batches are deterministic; retrying cannot change them
    deterministic = bool(sampler_ctx is not None and sampler_ctx["epoch_perm"])
    attempts = 0 if deterministic else resample_attempts
    for attempt in range(attempts + 1):
        if timers is not None:
            timers.start("sample nodes")
        sample_idx = _sample_minibatch_nodes(
            sampler_ctx,
            num_nodes,
            minibatch_nodes,
            minibatch_ratio,
            minibatch_seed,
            global_step,
            attempt,
            steps_per_epoch,
            features.device,
        )
        sample_idx, _ = torch.sort(sample_idx)
        if timers is not None:
            timers.stop("sample nodes")

        if timers is not None:
            timers.start("compact indices")
        row_idx_list = build_layout_indices(
            sample_idx,
            data_loader.adj_dim1_start,
            data_loader.adj_dim1_stop,
            device=features.device,
            assume_sorted=True,
        )
        col_idx_list = build_layout_indices(
            sample_idx,
            data_loader.adj_dim2_start,
            data_loader.adj_dim2_stop,
            device=features.device,
            assume_sorted=True,
        )
        if timers is not None:
            timers.stop("compact indices")
        last_layout = (model.num_gcn_layers - 1) % len(row_idx_list)
        local_rows = int(row_idx_list[last_layout].numel())
        local_cols0 = int(col_idx_list[0].numel())
        empty_flag = torch.tensor(
            1 if local_rows == 0 or local_cols0 == 0 else 0,
            device=features.device,
        )
        # ax.get_timers().start("compact empty check")
        # dist.all_reduce(
        #     empty_flag, op=dist.ReduceOp.SUM, group=ax.comm_handle.intra_layer_group
        # )
        # ax.get_timers().stop("compact empty check")
        if empty_flag.item() == 0:
            break

    if empty_flag.item() > 0:
        if dist.get_rank() == 0:
            print(
                "[warn] compact minibatch produced empty rows after resampling; "
                "skipping this step."
            )
        return None

    if timers is not None:
        timers.start("compact adj shards")
    adj_shards_mb = build_compact_adj_shards(
        train_adj_shards,
        row_idx_list,
        col_idx_list,
        data_loader.adj_dim1_start,
        data_loader.adj_dim2_start,
        edge_scale=edge_scale,
        enable_timers=enable_timers,
        col_scale=sampler_ctx["col_scale"] if sampler_ctx is not None else None,
    )
    if timers is not None:
        timers.stop("compact adj shards")
    col_idx0 = col_idx_list[0]
    if timers is not None:
        timers.start("compact feature slice")
    if col_idx0.numel() > 0:
        features_mb = features.index_select(0, col_idx0)
    else:
        features_mb = features[:0]
    if timers is not None:
        timers.stop("compact feature slice")
    row_idx_last = row_idx_list[last_layout]
    if timers is not None:
        timers.start("compact label slice")
    node_weight_local = (
        sampler_ctx.get("node_weight_local") if sampler_ctx is not None else None
    )
    if row_idx_last.numel() > 0:
        labels_mb = labels.index_select(0, row_idx_last)
        if train_mask is None:
            train_mask_mb = None
        else:
            train_mask_mb = train_mask.index_select(0, row_idx_last)
        node_weight_mb = (
            node_weight_local.index_select(0, row_idx_last)
            if node_weight_local is not None
            else None
        )
    else:
        labels_mb = labels[:0]
        train_mask_mb = None if train_mask is None else train_mask[:0]
        node_weight_mb = None if node_weight_local is None else node_weight_local[:0]
    if timers is not None:
        timers.stop("compact label slice")
    num_nodes_loss = 2**62
    layout_metadata = {
        "row_idx_list": row_idx_list,
        "col_idx_list": col_idx_list,
        "row_starts": data_loader.adj_dim1_start,
        "col_starts": data_loader.adj_dim2_start,
        "allow_missing": False,
    }
    return (
        features_mb,
        labels_mb,
        adj_shards_mb,
        train_mask_mb,
        num_nodes_loss,
        layout_metadata,
        node_weight_mb,
    )


def _build_full_layout_metadata(data_loader, device: torch.device) -> dict:
    return {
        "row_idx_list": [
            torch.arange(stop - start, device=device, dtype=torch.long)
            for start, stop in zip(
                data_loader.adj_dim1_start,
                data_loader.adj_dim1_stop,
            )
        ],
        "col_idx_list": [
            torch.arange(stop - start, device=device, dtype=torch.long)
            for start, stop in zip(
                data_loader.adj_dim2_start,
                data_loader.adj_dim2_stop,
            )
        ],
        "row_starts": data_loader.adj_dim1_start,
        "col_starts": data_loader.adj_dim2_start,
        "allow_missing": True,
    }


def _launch_compact_prefetch(
    *,
    prefetch_executor: ThreadPoolExecutor,
    prefetch_stream: torch.cuda.Stream,
    device_index: int,
    model,
    data_loader,
    train_adj_shards,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor | None,
    num_nodes: int,
    minibatch_nodes: int | None,
    minibatch_ratio: float | None,
    minibatch_seed: int,
    global_step: int,
    edge_scale: float | None,
    sampler_ctx: dict | None = None,
    steps_per_epoch: int = 1,
):
    def _worker():
        torch.cuda.set_device(device_index)
        with torch.cuda.stream(prefetch_stream):
            minibatch = _prepare_compact_minibatch(
                model=model,
                data_loader=data_loader,
                train_adj_shards=train_adj_shards,
                features=features,
                labels=labels,
                train_mask=train_mask,
                num_nodes=num_nodes,
                minibatch_nodes=minibatch_nodes,
                minibatch_ratio=minibatch_ratio,
                minibatch_seed=minibatch_seed,
                global_step=global_step,
                edge_scale=edge_scale,
                enable_timers=False,
                sampler_ctx=sampler_ctx,
                steps_per_epoch=steps_per_epoch,
            )
            ready_event = torch.cuda.Event(blocking=False)
            ready_event.record(prefetch_stream)
        return minibatch, ready_event

    return prefetch_executor.submit(_worker)


def _loss_groups(num_gcn_layers):
    outer_group, inner_group, depth_group = _layer_groups(num_gcn_layers - 1)
    if plx.use_3d_linear:
        return (depth_group, inner_group)
    return (depth_group, outer_group)


def _all_gather_variable_rows(
    tensor: torch.Tensor,
    process_group,
) -> tuple[torch.Tensor, list[int]]:
    if not dist.is_initialized() or dist.get_world_size(process_group) == 1:
        return tensor, [int(tensor.shape[0])]

    local_rows = torch.tensor(
        [int(tensor.shape[0])],
        device=tensor.device,
        dtype=torch.long,
    )
    gathered_sizes = [
        torch.empty_like(local_rows)
        for _ in range(dist.get_world_size(process_group))
    ]
    dist.all_gather(gathered_sizes, local_rows, group=process_group)
    row_sizes = [int(sz.item()) for sz in gathered_sizes]
    max_rows = max(row_sizes, default=0)

    if tensor.ndim == 1:
        padded = tensor.new_empty(max_rows)
        if tensor.shape[0] > 0:
            padded[: tensor.shape[0]] = tensor
        if tensor.shape[0] < max_rows:
            padded[tensor.shape[0] :] = 0
    else:
        padded = tensor.new_empty((max_rows, *tensor.shape[1:]))
        if tensor.shape[0] > 0:
            padded[: tensor.shape[0]] = tensor
        if tensor.shape[0] < max_rows:
            padded[tensor.shape[0] :] = 0

    gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size(process_group))]
    dist.all_gather(gathered, padded.contiguous(), group=process_group)
    pieces = [chunk[:rows] for chunk, rows in zip(gathered, row_sizes)]

    if not pieces:
        return tensor[:0], row_sizes
    return torch.cat(pieces, dim=0), row_sizes


def _outer_group_letter(layer_num: int) -> str:
    return _layer_groups(layer_num)[0]


def _reshard_residual(x: torch.Tensor, layer_num: int) -> torch.Tensor:
    x = x.clone()
    outer_group, inner_group, depth_group = _layer_groups(layer_num)
    _, _, process_groups = get_process_groups_info(
        (outer_group, inner_group, depth_group)
    )
    outer_pg, inner_pg, depth_pg = process_groups

    x_full = Gather.apply(x, outer_pg, 0)
    x_full = Gather.apply(x_full, inner_pg, 1)
    x_full = Drop.apply(x_full, depth_pg, 0)
    x_full = Drop.apply(x_full, outer_pg, 1)
    return x_full


def _reshard_residual_compact(
    x: torch.Tensor,
    layer_num: int,
    layout_metadata: dict,
) -> torch.Tensor:
    layout_idx = layer_num % len(layout_metadata["row_idx_list"])
    row_idx = layout_metadata["row_idx_list"][layout_idx]
    col_idx = layout_metadata["col_idx_list"][layout_idx]
    row_start = int(layout_metadata["row_starts"][layout_idx])
    col_start = int(layout_metadata["col_starts"][layout_idx])

    outer_group, inner_group, _ = _layer_groups(layer_num)
    _, _, process_groups = get_process_groups_info((outer_group, inner_group))
    outer_pg, inner_pg = process_groups

    # Compact minibatch yields uneven local row counts, so the standard
    # Gather/Drop rotation on dim 0 is not valid. Rebuild the sampled rows by
    # global node id, then reshard only the feature dimension.
    x_full_hidden = Gather.apply(x, inner_pg, 1)

    src_global = col_idx + col_start
    gathered_x, _ = _all_gather_variable_rows(x_full_hidden, outer_pg)
    gathered_src_global, _ = _all_gather_variable_rows(src_global, outer_pg)
    if gathered_src_global.numel() > 1:
        order = torch.argsort(gathered_src_global)
        gathered_src_global = gathered_src_global.index_select(0, order)
        gathered_x = gathered_x.index_select(0, order)

    dst_global = row_idx + row_start
    if dst_global.numel() == 0:
        return Drop.apply(gathered_x[:0], outer_pg, 1)

    allow_missing = bool(layout_metadata.get("allow_missing", False))
    positions = torch.searchsorted(gathered_src_global, dst_global)
    valid = positions < gathered_src_global.numel()
    if gathered_src_global.numel() > 0:
        safe_positions = positions.clamp(max=gathered_src_global.numel() - 1)
        valid = valid & (
            gathered_src_global.index_select(0, safe_positions) == dst_global
        )
    if not torch.all(valid) and not allow_missing:
        missing = dst_global[~valid][:8].tolist()
        raise RuntimeError(
            "Residual reshard could not match sampled nodes between layouts. "
            f"layer={layer_num}, missing_global_nodes={missing}"
        )

    if torch.all(valid):
        x_reordered = gathered_x.index_select(0, positions)
    else:
        x_reordered = gathered_x.new_zeros((dst_global.numel(), *gathered_x.shape[1:]))
        if valid.any():
            matched_dst = valid.nonzero(as_tuple=False).squeeze(1)
            matched_src = positions.index_select(0, matched_dst)
            x_reordered.index_copy_(
                0,
                matched_dst,
                gathered_x.index_select(0, matched_src),
            )
    return Drop.apply(x_reordered, outer_pg, 1)


def _layer_groups(layer_num: int):
    if plx.use_3d_linear:
        if layer_num % 3 == 0:
            return ("x", "z", "y")
        if layer_num % 3 == 1:
            return ("y", "x", "z")
        return ("z", "y", "x")
    if layer_num % 3 == 0:
        return ("x", "y", "z")
    if layer_num % 3 == 1:
        return ("z", "x", "y")
    return ("y", "z", "x")


class BestValidTracker:
    def __init__(self, metric_name: str = "acc", percent_scale: bool = True):
        self._results = []
        self._metric_name = metric_name
        self._percent_scale = percent_scale

    def add(self, train_metric: float, val_metric: float, test_metric: float):
        self._results.append((train_metric, val_metric, test_metric))

    def print_ogb_style(self, header: str = "OGB (best val)"):
        if not self._results:
            print(f"{header}: no evaluation results recorded.")
            return

        result = torch.tensor(self._results)
        if self._percent_scale:
            result = 100 * result
        best_epoch = result[:, 1].argmax().item()
        print(header + ":")
        print(f"Highest Train ({self._metric_name}): {result[:, 0].max():.2f}")
        print(f"Highest Valid ({self._metric_name}): {result[:, 1].max():.2f}")
        print(f"  Final Train ({self._metric_name}): {result[best_epoch, 0]:.2f}")
        print(f"   Final Test ({self._metric_name}): {result[best_epoch, 2]:.2f}")


@torch.no_grad()
def _distributed_argmax(logits, num_gcn_layers, num_nodes, num_classes):
    groups = _loss_groups(num_gcn_layers)
    num_gpus, ranks, process_groups = get_process_groups_info(groups)
    class_group = process_groups[1]
    class_rank = ranks[1]

    local_num_classes = logits.shape[1]
    class_offset = class_rank * local_num_classes
    invalid_classes = (
        torch.arange(local_num_classes, device=logits.device) + class_offset
    ) >= num_classes

    logits = logits.clone()
    if invalid_classes.any():
        logits[:, invalid_classes] = float("-inf")

    local_max, local_idx = logits.max(dim=1)
    global_idx = local_idx + class_offset

    gathered_vals = [torch.empty_like(local_max) for _ in range(num_gpus[1])]
    gathered_idx = [torch.empty_like(global_idx) for _ in range(num_gpus[1])]
    dist.all_gather(gathered_vals, local_max, group=class_group)
    dist.all_gather(gathered_idx, global_idx, group=class_group)

    vals = torch.stack(gathered_vals, dim=0)
    idxs = torch.stack(gathered_idx, dim=0)
    best = vals.argmax(dim=0)
    pred = idxs.gather(0, best.unsqueeze(0)).squeeze(0)

    node_rank = ranks[0]
    invalid_nodes = (
        torch.arange(pred.shape[0], device=pred.device) + node_rank * pred.shape[0]
    ) >= num_nodes
    if invalid_nodes.any():
        pred[invalid_nodes] = -1

    return pred


def _compute_split_metrics(pred, labels, mask, num_classes, node_group):
    if mask is None:
        return None

    valid = mask & (labels >= 0) & (pred >= 0)
    pred_eval = pred[valid]
    labels_eval = labels[valid]

    correct = (pred_eval == labels_eval).sum().to(torch.long)
    total = valid.sum().to(torch.long)

    tp = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
    fp = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
    fn = torch.zeros(num_classes, dtype=torch.long, device=pred.device)

    if total.item() > 0:
        match = pred_eval == labels_eval
        tp = torch.bincount(labels_eval[match], minlength=num_classes).to(torch.long)
        pred_cnt = torch.bincount(pred_eval, minlength=num_classes).to(torch.long)
        true_cnt = torch.bincount(labels_eval, minlength=num_classes).to(torch.long)
        fp = pred_cnt - tp
        fn = true_cnt - tp

    dist.all_reduce(correct, op=dist.ReduceOp.SUM, group=node_group)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=node_group)
    dist.all_reduce(tp, op=dist.ReduceOp.SUM, group=node_group)
    dist.all_reduce(fp, op=dist.ReduceOp.SUM, group=node_group)
    dist.all_reduce(fn, op=dist.ReduceOp.SUM, group=node_group)

    correct_f = correct.float()
    total_f = total.float().clamp_min(1.0)
    acc = (correct_f / total_f).item()

    tp_f = tp.float()
    fp_f = fp.float()
    fn_f = fn.float()

    micro_tp = tp_f.sum()
    micro_fp = fp_f.sum()
    micro_fn = fn_f.sum()
    micro_denom = (2 * micro_tp + micro_fp + micro_fn).clamp_min(1.0)
    micro_f1 = ((2 * micro_tp) / micro_denom).item()

    denom = 2 * tp_f + fp_f + fn_f
    f1_per_class = torch.where(
        denom > 0, (2 * tp_f) / denom.clamp_min(1.0), torch.zeros_like(denom)
    )
    support = (tp_f + fn_f) + (tp_f + fp_f)
    valid_classes = support > 0
    macro_f1 = (
        f1_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    )

    return {
        "acc": acc,
        "f1_micro": micro_f1,
        "f1_macro": macro_f1,
        "total": int(total.item()),
    }


_DEG_BUCKET_EDGES = (1, 2, 4, 8, 16, 32, 64)
_DEG_BUCKET_LABELS = ("0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+")


def _degree_bucket_metrics(pred, labels, mask, degree_local, node_group):
    """Accuracy stratified by (non-self) node degree; diagnoses edge starvation."""
    if mask is None:
        return None
    nb = len(_DEG_BUCKET_LABELS)
    valid = mask.to(torch.bool) & (labels >= 0) & (pred >= 0) & (degree_local >= 0)
    boundaries = torch.tensor(
        _DEG_BUCKET_EDGES, device=degree_local.device, dtype=degree_local.dtype
    )
    bucket = torch.bucketize(degree_local.clamp(min=0), boundaries)
    total = torch.zeros(nb, dtype=torch.long, device=pred.device)
    correct = torch.zeros(nb, dtype=torch.long, device=pred.device)
    if valid.any():
        total = torch.bincount(bucket[valid], minlength=nb)
        corr_mask = valid & (pred == labels)
        correct = torch.bincount(bucket[corr_mask], minlength=nb)
    stats = torch.stack([correct, total])
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=node_group)
    correct, total = stats[0], stats[1]
    return [
        (lbl, correct[i].item() / max(1, total[i].item()), int(total[i].item()))
        for i, lbl in enumerate(_DEG_BUCKET_LABELS)
    ]


@torch.no_grad()
def evaluate(
    model,
    features_local,
    adj_shards,
    labels,
    masks,
    num_nodes,
    num_classes,
    layout_metadata=None,
    multilabel_metric="rocauc",
    degree_local=None,
):
    groups = _loss_groups(model.num_gcn_layers)
    _, _, process_groups = get_process_groups_info(groups)
    node_group = process_groups[0]
    class_group = process_groups[1]

    model.eval()
    logits = model(features_local, adj_shards, layout_metadata=layout_metadata)

    # Multi-label (e.g., ogbn-proteins, yelp): gather logits across class+node groups.
    if labels.ndim == 2 and labels.size(-1) > 1:
        if masks is None:
            return {}

        # 1) Gather class shards -> full logits for this node shard.
        class_world = dist.get_world_size(group=class_group)
        gathered_logits = [torch.empty_like(logits) for _ in range(class_world)]
        dist.all_gather(gathered_logits, logits, group=class_group)
        full_logits_local = torch.cat(gathered_logits, dim=1)[:, :num_classes]

        # 2) Gather node shards -> full logits/labels/masks.
        node_world = dist.get_world_size(group=node_group)
        gathered_full_logits = [torch.empty_like(full_logits_local) for _ in range(node_world)]
        dist.all_gather(gathered_full_logits, full_logits_local, group=node_group)
        y_pred = torch.cat(gathered_full_logits, dim=0)[:num_nodes]

        gathered_labels = [torch.empty_like(labels) for _ in range(node_world)]
        dist.all_gather(gathered_labels, labels, group=node_group)
        y_true = torch.cat(gathered_labels, dim=0)[:num_nodes]

        gathered_masks = {}
        for split in ("train", "val", "test"):
            mask = masks.get(split)
            if mask is None:
                gathered_masks[split] = None
                continue
            buf = [torch.empty_like(mask) for _ in range(node_world)]
            dist.all_gather(buf, mask, group=node_group)
            gathered_masks[split] = torch.cat(buf, dim=0)[:num_nodes].to(torch.bool)

        if dist.get_rank() != 0:
            return {}

        if multilabel_metric == "f1_micro":
            from sklearn.metrics import f1_score

            y_pred_binary = (y_pred > 0).cpu().numpy()
            y_true_np = y_true.cpu().numpy()
            results = {}
            for split in ("train", "val", "test"):
                mask = gathered_masks.get(split)
                if mask is None:
                    results[split] = None
                    continue
                mask_np = mask.cpu().numpy()
                results[split] = {
                    "f1_micro": f1_score(
                        y_true_np[mask_np], y_pred_binary[mask_np], average="micro"
                    )
                }
            return results
        else:
            try:
                from ogb.nodeproppred import Evaluator
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import OGB Evaluator (ogb.nodeproppred). Ensure the `ogb` package and its dependencies are installed."
                ) from exc

            evaluator = Evaluator(name="ogbn-proteins")
            results = {}
            for split in ("train", "val", "test"):
                mask = gathered_masks.get(split)
                if mask is None:
                    results[split] = None
                    continue
                results[split] = evaluator.eval({"y_true": y_true[mask], "y_pred": y_pred[mask]})
            return results

    pred = _distributed_argmax(logits, model.num_gcn_layers, num_nodes, num_classes)

    results = {}
    if masks is None:
        return results

    for split in ("train", "val", "test"):
        results[split] = _compute_split_metrics(
            pred,
            labels,
            masks.get(split),
            num_classes,
            node_group,
        )
        if degree_local is not None and results[split] is not None:
            results[split]["deg_buckets"] = _degree_bucket_metrics(
                pred,
                labels,
                masks.get(split),
                degree_local,
                node_group,
            )
    return results


if __name__ == "__main__":
    if os.environ.get("PLEXUS_DEBUG_HANG"):
        import faulthandler

        faulthandler.dump_traceback_later(
            int(os.environ["PLEXUS_DEBUG_HANG"]), exit=True
        )
    parser = create_parser()
    args = parser.parse_args()
    if os.environ.get("RANK", "0") == "0":
        print(f"[args] {vars(args)}", flush=True)
    set_seed(args.seed)

    # initialize distributed environment
    dist.init_process_group(backend="nccl")
    plx.init(
        G_intra_r=args.G_intra_r,
        G_intra_c=args.G_intra_c,
        G_intra_d=args.G_intra_d,
        gpus_per_node=args.gpus_per_node,
        enable_internal_timers=True,
        block_aggregation=args.block_aggregation,
        overlap_aggregation=args.overlap_aggregation,
        overlap_backward=args.overlap_bwd,
        tune_gemms=args.tune_gemms,
        use_3d_linear_flag=args.use_3d_linear,
        allreduce_low_precision=args.allreduce_lowp,
        allreduce_low_precision_dtype=args.allreduce_lowp_dtype,
        G_data=args.G_data,
        bf16_spmm_flag=args.bf16_spmm,
        bf16_gemm_flag=args.bf16_gemm,
        avg_grad_flag=args.avg_grad,
        overlap_fwd_comm_flag=args.overlap_fwd_comm,
        overlap_bwd_comm_flag=args.overlap_bwd_comm,
        overlap_linear_bwd_flag=args.overlap_linear_bwd,
    )
    dp_rank = ax.comm_handle.data_parallel_rank

    # initialize parallel data loader
    data_loader = DataLoader(args.data_dir, args.num_gcn_layers)

    # get the dataset which includes graph, features, and output labels
    (
        adj_shards,
        adj_shards_train,
        features,
        labels,
        masks,
        num_nodes,
        num_features,
        num_classes,
    ) = data_loader.load(
        load_train_adj=args.train_adj,
        train_features=args.train_features,
    )

    # create the model and move to gpu
    model = Net(
        args.num_gcn_layers,
        num_features,
        args.hidden_size,
        num_classes,
        train_features=args.train_features,
        fuse_norm_activation=args.fuse_norm_activation,
        dropout=args.dropout,
    ).to(torch.device("cuda"))

    if args.compile_norm or args.fuse_norm_activation:
        for norm in model.norms:
            norm.compile(
                fuse_activation=args.fuse_norm_activation, dropout_p=args.dropout
            )
        if dist.get_rank() == 0:
            if args.fuse_norm_activation:
                print("[info] RMSNorm + ReLU + Dropout fused and compiled with torch.compile")
            else:
                print("[info] RMSNorm elementwise ops compiled with torch.compile")

    # create optimizer for parameters
    optim_params = list(model.parameters())
    if args.train_features:
        optim_params.append(features)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = None
    if args.lr_schedule == "cosine":
        warm_ep = max(0, int(args.lr_warmup_epochs))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.num_epochs - warm_ep)
        )
        if warm_ep > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_ep
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, cosine], milestones=[warm_ep]
            )
        else:
            lr_scheduler = cosine

    dist.barrier(device_ids=[torch.cuda.current_device()])

    do_eval = bool(args.eval) and (masks is not None)
    if args.eval and not do_eval and dist.get_rank() == 0:
        print(
            "[warn] --eval was set but no train/val/test masks were found in the dataset; skipping evaluation."
        )

    best_valid = BestValidTracker() if do_eval and dist.get_rank() == 0 else None
    if best_valid is not None and labels.ndim == 2 and labels.size(-1) > 1:
        best_valid = BestValidTracker(metric_name=args.multilabel_metric, percent_scale=True)

    train_mask = masks.get("train") if masks is not None else None
    if args.train_adj and adj_shards_train is None and dist.get_rank() == 0:
        print(
            "[warn] --train_adj was set but train-induced adjacency was not found; falling back to full-graph adjacency."
        )
    train_adj_shards = (
        adj_shards_train if args.train_adj and adj_shards_train is not None else adj_shards
    )

    use_minibatch = args.minibatch_nodes is not None or args.minibatch_ratio is not None
    if args.minibatch_nodes is not None and args.minibatch_ratio is not None:
        raise ValueError("Provide only one of --minibatch_nodes or --minibatch_ratio.")
    if args.minibatch_compact and args.train_features:
        raise ValueError("--minibatch_compact is not supported with --train_features yet.")
    if use_minibatch and not args.minibatch_compact:
        raise ValueError(
            "Non-compact minibatch is no longer supported; please enable --minibatch_compact."
        )
    overlap_samp = bool(args.overlap_samp and use_minibatch)
    if args.overlap_samp and not use_minibatch and dist.get_rank() == 0:
        print("[warn] --overlap_samp is ignored because minibatch is disabled.")

    full_layout_metadata = _build_full_layout_metadata(data_loader, features.device)

    minibatch_seed = args.seed if args.minibatch_seed is None else args.minibatch_seed
    minibatch_seed = int(minibatch_seed) + (int(dp_rank) * 1000003)
    if use_minibatch:
        if args.steps_per_epoch is None:
            steps_per_epoch = compute_steps_per_epoch(
                num_nodes,
                batch_size=args.minibatch_nodes,
                ratio=args.minibatch_ratio,
            )
            if args.G_data > 1:
                steps_per_epoch = max(
                    1, int(math.ceil(steps_per_epoch / args.G_data))
                )
        else:
            steps_per_epoch = max(1, int(args.steps_per_epoch))
    else:
        steps_per_epoch = 1

    edge_scale_global = None
    if use_minibatch and args.minibatch_unbiased:
        batch_size = _minibatch_size(
            num_nodes,
            batch_size=args.minibatch_nodes,
            ratio=args.minibatch_ratio,
        )
        edge_scale_global = _edge_scale_value(num_nodes, batch_size)
        if edge_scale_global == 1.0:
            edge_scale_global = None
        if edge_scale_global is None and dist.get_rank() == 0 and batch_size <= 1:
            print(
                "[warn] minibatch_unbiased enabled with batch_size<=1; "
                "skipping edge scaling."
            )
        if (
            edge_scale_global is not None
            and args.train_adj
            and adj_shards_train is not None
        ):
            _scale_adj_shards_non_self(
                train_adj_shards,
                data_loader.adj_dim1_start,
                data_loader.adj_dim2_start,
                edge_scale_global,
            )
            edge_scale_global = None

    # ------- non-uniform / epoch-permutation sampling setup -------
    sampler_ctx = None
    degrees = None
    multilabel = labels.ndim == 2 and labels.size(-1) > 1
    nonuniform = args.minibatch_sampler != "uniform"
    if (nonuniform or args.minibatch_epoch_perm) and not use_minibatch:
        raise ValueError(
            "--minibatch_sampler/--minibatch_epoch_perm require minibatch "
            "training (--minibatch_nodes or --minibatch_ratio)."
        )
    if nonuniform and args.train_adj:
        raise NotImplementedError(
            "--minibatch_sampler hub/degree is not supported with --train_adj."
        )
    if args.minibatch_epoch_perm and args.minibatch_sampler == "degree":
        raise NotImplementedError(
            "--minibatch_epoch_perm is only supported for uniform and hub samplers."
        )
    if nonuniform or args.eval_degree_buckets:
        degrees = _compute_global_degrees(adj_shards, data_loader, num_nodes)

    if use_minibatch and (nonuniform or args.minibatch_epoch_perm):
        batch_size_resolved = _minibatch_size(
            num_nodes,
            batch_size=args.minibatch_nodes,
            ratio=args.minibatch_ratio,
        )
        seed_base = args.seed if args.minibatch_seed is None else args.minibatch_seed
        sampler_ctx = _build_sampler_ctx(
            args,
            degrees,
            num_nodes,
            batch_size_resolved,
            seed_base=int(seed_base),
            minibatch_seed=minibatch_seed,
            dp_rank=dp_rank,
            labels_start=int(data_loader.labels_start),
            labels_stop=int(data_loader.labels_stop),
            train_mask_local=train_mask,
        )
        if nonuniform:
            # per-node 1/pi rescaling replaces the scalar 1/p factor
            edge_scale_global = None
        if args.minibatch_sampler == "train_hub":
            steps_per_epoch = sampler_ctx["train_steps"]
        if args.minibatch_epoch_perm:
            if args.minibatch_sampler == "hub":
                k = sampler_ctx["hub_k"]
                chunks_total = max(0, (num_nodes - k) // (batch_size_resolved - k))
            else:
                chunks_total = num_nodes // batch_size_resolved
            perm_steps = chunks_total // max(1, args.G_data)
            if perm_steps < 1:
                raise ValueError(
                    "epoch permutation needs at least one chunk per DP group "
                    f"(chunks_total={chunks_total}, G_data={args.G_data})."
                )
            if args.steps_per_epoch is not None:
                perm_steps = min(perm_steps, max(1, int(args.steps_per_epoch)))
            steps_per_epoch = perm_steps
        if dist.get_rank() == 0:
            print(
                f"[info] sampler={args.minibatch_sampler} "
                f"epoch_perm={args.minibatch_epoch_perm} "
                f"B={batch_size_resolved} steps_per_epoch={steps_per_epoch} "
                + (
                    f"hub_k={sampler_ctx['hub_k']} "
                    if args.minibatch_sampler == "hub"
                    else ""
                )
                + (
                    f"alpha={args.minibatch_degree_alpha} "
                    if args.minibatch_sampler == "degree"
                    else ""
                )
                + f"edge_rescale={'per-node 1/pi' if (nonuniform and args.minibatch_unbiased) else ('scalar 1/p' if edge_scale_global is not None else 'none')}"
            )

    degree_labels_local = None
    if args.eval_degree_buckets and degrees is not None:
        degree_labels_local = _slice_node_vector(
            degrees,
            int(data_loader.labels_start),
            int(data_loader.labels_stop),
            num_nodes,
            fill=-1.0,
        )

    prof = None
    trace_dir = None
    rank = dist.get_rank()
    if args.use_profiler:
        result_dir = os.environ.get("RESULT_DIR", "./result")
        trace_dir = os.path.join(result_dir, "profiler")
        os.makedirs(trace_dir, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.start()

    print( "steps_per_epoch: ", steps_per_epoch)

    prefetch_executor = None
    prefetch_stream = None
    prefetch_device_index = torch.cuda.current_device()
    if overlap_samp:
        prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="minibatch_prefetch"
        )
        prefetch_stream = torch.cuda.Stream(device=prefetch_device_index)
        
    tag=0
    # Cold-start prefetch for epoch 0 (unavoidable; no training to overlap with)
    prefetch_future = None
    if overlap_samp and use_minibatch:
        prefetch_future = _launch_compact_prefetch(
            prefetch_executor=prefetch_executor,
            prefetch_stream=prefetch_stream,
            device_index=prefetch_device_index,
            model=model,
            data_loader=data_loader,
            train_adj_shards=train_adj_shards,
            features=features,
            labels=labels,
            train_mask=train_mask,
            num_nodes=num_nodes,
            minibatch_nodes=args.minibatch_nodes,
            minibatch_ratio=args.minibatch_ratio,
            minibatch_seed=minibatch_seed,
            global_step=0,
            edge_scale=edge_scale_global,
            sampler_ctx=sampler_ctx,
            steps_per_epoch=steps_per_epoch,
        )

    # training loop
    for i in range(args.num_epochs):
        # if i == PROFILE_START_EPOCH:
        #     torch.cuda.profiler.start()
        # torch.cuda.nvtx.range_push("epoch " + str(i))
        if args.timing_start_epoch is None:
            args.timing_start_epoch = 0

        if args.timing_end_epoch is None:
            args.timing_end_epoch = args.num_epochs - 1

        if i >= args.timing_start_epoch and i <= args.timing_end_epoch:
            ax.get_timers().start("epoch " + str(i))

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            global_step = i * steps_per_epoch + step
            if use_minibatch:
                if overlap_samp:
                    ax.get_timers().start("prefetch wait")
                    minibatch, minibatch_ready = prefetch_future.result()
                    ax.get_timers().stop("prefetch wait")
                    if step + 1 < steps_per_epoch:
                        ax.get_timers().start("prefetch launch")
                        prefetch_future = _launch_compact_prefetch(
                            prefetch_executor=prefetch_executor,
                            prefetch_stream=prefetch_stream,
                            device_index=prefetch_device_index,
                            model=model,
                            data_loader=data_loader,
                            train_adj_shards=train_adj_shards,
                            features=features,
                            labels=labels,
                            train_mask=train_mask,
                            num_nodes=num_nodes,
                            minibatch_nodes=args.minibatch_nodes,
                            minibatch_ratio=args.minibatch_ratio,
                            minibatch_seed=minibatch_seed,
                            global_step=global_step + 1,
                            edge_scale=edge_scale_global,
                            sampler_ctx=sampler_ctx,
                            steps_per_epoch=steps_per_epoch,
                        )
                        ax.get_timers().stop("prefetch launch")
                    elif i + 1 < args.num_epochs:
                        ax.get_timers().start("prefetch launch")
                        prefetch_future = _launch_compact_prefetch(
                            prefetch_executor=prefetch_executor,
                            prefetch_stream=prefetch_stream,
                            device_index=prefetch_device_index,
                            model=model,
                            data_loader=data_loader,
                            train_adj_shards=train_adj_shards,
                            features=features,
                            labels=labels,
                            train_mask=train_mask,
                            num_nodes=num_nodes,
                            minibatch_nodes=args.minibatch_nodes,
                            minibatch_ratio=args.minibatch_ratio,
                            minibatch_seed=minibatch_seed,
                            global_step=(i + 1) * steps_per_epoch,
                            edge_scale=edge_scale_global,
                            sampler_ctx=sampler_ctx,
                            steps_per_epoch=steps_per_epoch,
                        )
                        ax.get_timers().stop("prefetch launch")
                    else:
                        prefetch_future = None
                    torch.cuda.current_stream().wait_event(minibatch_ready)
                else:
                    ax.get_timers().start("minibatch prep")
                    with record_function("sampling"):
                        minibatch = _prepare_compact_minibatch(
                            model=model,
                            data_loader=data_loader,
                            train_adj_shards=train_adj_shards,
                            features=features,
                            labels=labels,
                            train_mask=train_mask,
                            num_nodes=num_nodes,
                            minibatch_nodes=args.minibatch_nodes,
                            minibatch_ratio=args.minibatch_ratio,
                            minibatch_seed=minibatch_seed,
                            global_step=global_step,
                            edge_scale=edge_scale_global,
                            sampler_ctx=sampler_ctx,
                            steps_per_epoch=steps_per_epoch,
                        )
                    ax.get_timers().stop("minibatch prep")
                if minibatch is None:
                    continue
                (
                    features_mb,
                    labels_to_use,
                    adj_to_use,
                    mask_to_use,
                    num_nodes_loss,
                    layout_metadata,
                    node_weight_mb,
                ) = minibatch
            else:
                features_mb = features
                labels_to_use = labels
                adj_to_use = train_adj_shards
                mask_to_use = train_mask
                num_nodes_loss = num_nodes
                layout_metadata = full_layout_metadata
                node_weight_mb = None

            ax.get_timers().start("train step")
            with record_function("train"):
                loss = train(
                    model,
                    optimizer,
                    features_mb,
                    adj_to_use,
                    labels_to_use,
                    mask_to_use,
                    num_nodes_loss,
                    num_classes,
                    layout_metadata=layout_metadata,
                    test=args.test,
                    vectorize_dp_grad=args.vectorize_dp_grad,
                    node_weight=node_weight_mb,
                    grad_clip_value=args.grad_clip_value,
                )
            ax.get_timers().stop("train step")
            epoch_loss += float(loss.detach().item())
        # torch.cuda.nvtx.range_pop()
            
        # if i == PROFILE_END_EPOCH:
        #     torch.cuda.profiler.stop()

        if i >= args.timing_start_epoch and i <= args.timing_end_epoch:
            ax.get_timers().stop("epoch " + str(i))

        if i == args.timing_end_epoch:
            print_axonn_timer_data(ax.get_timers().get_times()[0])

        if lr_scheduler is not None:
            lr_scheduler.step()

        log = "Epoch: {:03d}, Train Loss: {:.4f}"
        if dist.get_rank() == 0:
            avg_loss = epoch_loss / max(1, steps_per_epoch)
            print(log.format(i, avg_loss))
    
        # advance profiler step each epoch
        if prof is not None:
            prof.step()

        if do_eval and args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            if tag < 3:
                ax.get_timers().start("eval")
            metrics = evaluate(
                model,
                features,
                adj_shards,
                labels,
                masks,
                num_nodes,
                num_classes,
                layout_metadata=full_layout_metadata,
                multilabel_metric=args.multilabel_metric,
                degree_local=degree_labels_local,
            )
            if tag < 3:
                ax.get_timers().stop("eval")
                print_axonn_timer_data(ax.get_timers().get_times()[0])
                tag += 1
            if dist.get_rank() == 0:
                for split, m in metrics.items():
                    if m is None:
                        continue
                    if labels.ndim == 2 and labels.size(-1) > 1:
                        mk = args.multilabel_metric
                        print("{}: {} {:.4f}".format(split.upper(), mk, m[mk]))
                    else:
                        print(
                            "{}: acc {:.4f}, f1_micro {:.4f}, f1_macro {:.4f} (n={})".format(
                                split.upper(),
                                m["acc"],
                                m["f1_micro"],
                                m["f1_macro"],
                                m["total"],
                            )
                        )
                        if m.get("deg_buckets"):
                            parts = " ".join(
                                f"[{lbl}]={acc:.4f}(n={tot})"
                                for lbl, acc, tot in m["deg_buckets"]
                            )
                            print(f"  {split} by-degree: {parts}")
                if best_valid is not None:
                    train_m = metrics.get("train")
                    val_m = metrics.get("val")
                    test_m = metrics.get("test")
                    if train_m is not None and val_m is not None and test_m is not None:
                        if labels.ndim == 2 and labels.size(-1) > 1:
                            mk = args.multilabel_metric
                            best_valid.add(
                                train_m[mk], val_m[mk], test_m[mk]
                            )
                        else:
                            best_valid.add(train_m["acc"], val_m["acc"], test_m["acc"])

    if args.grad_variance_samples > 0 and multilabel:
        if dist.get_rank() == 0:
            print("[warn] --grad_variance_samples skipped (multilabel not supported).")
    elif args.grad_variance_samples > 0 and use_minibatch:
        # Fixed-parameter gradient variance over K fresh mini-batches (E3).
        # With --grad_variance_ratios, sweep uniform sampling at several
        # ratios on the SAME frozen parameters (removes the parameter-point
        # confound between ratios).
        K = int(args.grad_variance_samples)
        optim_params_v = [p for g in optimizer.param_groups for p in g["params"]]
        base_step = args.num_epochs * steps_per_epoch
        model.train()
        if args.grad_variance_ratios:
            sweep = [
                (float(r), None, _edge_scale_value(
                    num_nodes, _minibatch_size(num_nodes, None, float(r))
                ) if args.minibatch_unbiased else None)
                for r in args.grad_variance_ratios.split(",")
            ]
        else:
            sweep = [(args.minibatch_ratio, sampler_ctx, edge_scale_global)]
        for ratio_v, ctx_v, escale_v in sweep:
            g_sum = [torch.zeros_like(p) for p in optim_params_v]
            g_sq = [torch.zeros_like(p) for p in optim_params_v]
            n_ok = 0
            for k in range(K):
                mb = _prepare_compact_minibatch(
                    model=model,
                    data_loader=data_loader,
                    train_adj_shards=train_adj_shards,
                    features=features,
                    labels=labels,
                    train_mask=train_mask,
                    num_nodes=num_nodes,
                    minibatch_nodes=None,
                    minibatch_ratio=ratio_v,
                    minibatch_seed=minibatch_seed,
                    global_step=base_step + k,
                    edge_scale=escale_v,
                    sampler_ctx=ctx_v,
                    steps_per_epoch=steps_per_epoch,
                )
                if mb is None:
                    continue
                f_mb, y_mb, adj_mb, m_mb, nn_loss, lm, w_mb = mb
                optimizer.zero_grad()
                out = model(f_mb, adj_mb, layout_metadata=lm)
                loss_v = parallel_cross_entropy(
                    out, y_mb, model.num_gcn_layers, nn_loss, num_classes,
                    node_mask=m_mb, node_weight=w_mb,
                )
                loss_v.backward()
                sync_norm_gradients(model.norms, mean=plx.avg_grad)
                _sync_data_parallel_gradients(optimizer, vectorize=False)
                for p, s, q in zip(optim_params_v, g_sum, g_sq):
                    if p.grad is not None:
                        s.add_(p.grad)
                        q.add_(p.grad * p.grad)
                n_ok += 1
            if n_ok > 1:
                sum_var = torch.zeros(1, device=torch.device("cuda"))
                sum_mean_sq = torch.zeros(1, device=torch.device("cuda"))
                for s, q in zip(g_sum, g_sq):
                    mean = s / n_ok
                    var = q / n_ok - mean * mean
                    sum_var += var.clamp_min(0).sum()
                    sum_mean_sq += (mean * mean).sum()
                if dist.get_rank() == 0:
                    rel = (sum_var / sum_mean_sq.clamp_min(1e-30)).item()
                    print(
                        f"[grad-variance] ratio={ratio_v} K={n_ok} "
                        f"sum_var={sum_var.item():.6e} "
                        f"sum_mean_sq={sum_mean_sq.item():.6e} "
                        f"relative_var={rel:.4f}",
                        flush=True,
                    )

    print(f"rank {rank} Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=True)
    
    if prof is not None:
        prof.stop()
        prof.export_chrome_trace(
            os.path.join(trace_dir, f"profiler_rank_{rank}.json")
        )

    if best_valid is not None and dist.get_rank() == 0:
        best_valid.print_ogb_style()

    dist.destroy_process_group()
