#!/usr/bin/env python
"""
Standalone minibatch trainer for the hybrid half-collective GCN scheme
(GCNConvRS with scattered activations; the `--conv rs` counterpart of
examples/train_mini.py).

This file deliberately does NOT import or modify examples/train_mini.py.
It reuses only library primitives:
  - plexus.utils.dataloader.DataLoader (scattered=True)
  - plexus.utils.subgraph_sampler: sample_nodes / compute_steps_per_epoch /
    build_layout_indices / build_compact_adj_shards  (called as-is)
  - plexus.gcn_conv_rs.GCNConvRS and plexus.scattered helpers

Per step:
  1. every rank draws the SAME sorted global sample (shared seed),
  2. compact adjacency shards are rebuilt with the existing sampler
     machinery (same static per-layer 2D blocks as train_mini),
  3. this rank's minibatch input rows = its hierarchical cell of the
     sampled nodes inside its coarse layer-0 block (steady_cell); labels
     and train mask use the final layer's cell,
  4. GCNConvRS handles the uneven per-block sampled counts natively
     (splits derived from the compact A shard shapes).

v1 restrictions (asserted): G_data == 1, num_gcn_layers % 3 == 0
(so the layer-0 input and final-output coarse blocks are the same axis),
single-label classification.

Run (example, 8 GPUs / 2 nodes):
  srun -n 8 ./get_rank.sh python -u examples/train_mini_rs.py \
      --G_intra_r 2 --G_intra_c 2 --G_intra_d 2 \
      --num_epochs 10 --hidden_size 256 --num_gcn_layers 3 \
      --lr 0.005 --minibatch_ratio 0.05 --allreduce_lowp \
      --data_dir .../products_part8
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import torch.distributed as dist
from axonn import axonn as ax

from plexus import plexus as plx
from plexus.gcn_conv_rs import (GCNConvRS, _all_gather_rows,
                                compact_shard_columns,
                                sparse_gather_plan_local,
                                sparse_gather_exchange)
from plexus.scattered import (
    ScatteredLinear,
    ScatteredRMSNorm,
    scattered_cross_entropy,
    scattered_argmax,
    sync_replicated_gradients,
    steady_cell,
    intra_group,
)
from plexus.utils.dataloader import DataLoader
from plexus.utils.general import (
    set_seed,
    print_axonn_timer_data,
    get_process_groups_info,
)
from plexus.utils.subgraph_sampler import (
    sample_nodes,
    compute_steps_per_epoch,
    build_layout_indices,
    build_compact_adj_shards,
)


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--G_intra_r", type=int, default=1)
    p.add_argument("--G_intra_c", type=int, default=1)
    p.add_argument("--G_intra_d", type=int, default=1)
    p.add_argument("--gpus_per_node", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--num_gcn_layers", type=int, default=3)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minibatch_ratio", type=float, default=0.05)
    p.add_argument("--sampler", type=str, default="epoch_perm",
                   choices=["uniform", "epoch_perm"],
                   help="uniform: independent randperm(N) per step (O(N)); "
                        "epoch_perm: per-epoch permutation sliced per step "
                        "(same pairwise inclusion probability, O(batch)).")
    p.add_argument("--minibatch_unbiased", action="store_true", default=False,
                   help="scale edge weights by 1/p_neighbor (train_mini's "
                        "unbiased estimator)")
    p.add_argument("--eval", action="store_true", default=False)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--allreduce_lowp", action="store_true", default=False)
    p.add_argument("--allreduce_lowp_dtype", type=str, default="bf16",
                   choices=["bf16", "fp16"])
    p.add_argument("--overlap_bwd_comm", action="store_true", default=False,
                   help="async A2A(gAGG) over grad_W GEMM + deferred "
                        "AR(grad_W) hidden behind SpMM^T (cross-comm v2)")
    p.add_argument("--timing_start_epoch", type=int, default=2)
    p.add_argument("--timing_end_epoch", type=int, default=None)
    return p


class NetRS(torch.nn.Module):
    """input linear -> L x (GCNConvRS + local RMSNorm/ReLU/Dropout) ->
    output linear, all on scattered activations."""

    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super().__init__()
        self.num_gcn_layers = num_layers
        num_gpus_xyz, _, _ = get_process_groups_info(("x", "y", "z"))
        self.rs_fixed_axes = (num_gpus_xyz[0] == 1 and num_gpus_xyz[1] == 1)
        self.input_linear = ScatteredLinear(input_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [GCNConvRS(hidden_size, hidden_size, i, gemm_1d=True,
                       fixed_axes=self.rs_fixed_axes)
             for i in range(num_layers)]
        )
        self.norms = torch.nn.ModuleList(
            [ScatteredRMSNorm(hidden_size) for _ in range(num_layers)]
        )
        self.output_linear = ScatteredLinear(hidden_size, output_size)

    def forward(self, x, adj_shards):
        if self.rs_fixed_axes:
            adj_shards = adj_shards[:1]   # featpar: full-A shard every layer
        x = self.input_linear(x)
        for i in range(self.num_gcn_layers):
            x = self.layers[i](x, adj_shards)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        return self.output_linear(x)


def assemble_coarse_block(x_scattered, letters):
    """One-time: gather my COARSE block (e.g. the x-block) from the
    scattered even shards: AG over the innermost (r) group, then the
    middle (f) group -- hierarchical order is preserved."""
    _, _, pgs = get_process_groups_info(letters)   # (c, f, r) groups
    x = _all_gather_rows(x_scattered.contiguous(), pgs[2])   # over r
    x = _all_gather_rows(x, pgs[1])                          # over f
    return x


@torch.no_grad()
def full_graph_eval(model, features, adj_shards, labels, masks,
                    padded_N, num_nodes, num_classes):
    """Full-graph accuracy on the scattered (even) layout."""
    model.eval()
    logits = model(features, adj_shards)
    pred = scattered_argmax(logits, padded_N, num_nodes,
                            model.num_gcn_layers)
    results = {}
    for split in ("train", "val", "test"):
        mask = masks.get(split) if masks else None
        if mask is None:
            results[split] = None
            continue
        valid = mask & (labels >= 0) & (pred >= 0)
        correct = (pred[valid] == labels[valid]).sum()
        total = valid.sum()
        dist.all_reduce(correct, group=intra_group())
        dist.all_reduce(total, group=intra_group())
        results[split] = (correct.item(), max(1, total.item()))
    model.train()
    return results


def main():
    args = create_parser().parse_args()
    set_seed(args.seed)
    assert args.num_gcn_layers % 3 == 0, \
        "v1 requires num_gcn_layers % 3 == 0 (layer-0 and final coarse " \
        "blocks on the same axis)"

    if not dist.is_initialized():
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        elif "SLURM_LOCALID" in os.environ:
            torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
        dist.init_process_group(backend="nccl")

    plx.init(
        G_intra_r=args.G_intra_r,
        G_intra_c=args.G_intra_c,
        G_intra_d=args.G_intra_d,
        gpus_per_node=args.gpus_per_node,
        enable_internal_timers=True,
        allreduce_low_precision=args.allreduce_lowp,
        allreduce_low_precision_dtype=args.allreduce_lowp_dtype,
        overlap_bwd_comm_flag=args.overlap_bwd_comm,
    )

    data_loader = DataLoader(args.data_dir, args.num_gcn_layers,
                             scattered=True)
    (adj_shards, adj_shards_train, features, labels, masks,
     num_nodes, num_features, num_classes) = data_loader.load()
    padded_N = data_loader.padded_num_nodes
    rank0 = dist.get_rank() == 0

    # one-time coarse-block assembly (letters of the layer-0 nesting;
    # identical to the final nesting because L % 3 == 0)
    letters = ("x", "z", "y")
    feats_blk = assemble_coarse_block(features, letters)
    labels_blk = assemble_coarse_block(
        labels.reshape(-1, 1).to(torch.float32), letters
    ).reshape(-1).to(torch.int64)
    train_mask = masks.get("train") if masks else None
    tmask_blk = None
    if train_mask is not None:
        tmask_blk = assemble_coarse_block(
            train_mask.reshape(-1, 1).to(torch.float32), letters
        ).reshape(-1) > 0.5

    model = NetRS(args.num_gcn_layers, num_features, args.hidden_size,
                  num_classes).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    steps = compute_steps_per_epoch(num_nodes, ratio=args.minibatch_ratio)
    batch_size = max(1, int(num_nodes * args.minibatch_ratio))
    edge_scale = None
    if args.minibatch_unbiased and num_nodes > 1 and batch_size > 1:
        edge_scale = 1.0 / ((batch_size - 1) / (num_nodes - 1))

    if rank0:
        print(f"steps_per_epoch: {steps}  batch_size: {batch_size} "
              f"edge_scale: {edge_scale}")

    # featpar (grid (1,1,P)): single layout, full-A shard reused every layer
    if model.rs_fixed_axes:
        adj_shards_mb_src = list(adj_shards[:1])
        d1_starts = list(data_loader.adj_dim1_start[:1])
        d1_stops = list(data_loader.adj_dim1_stop[:1])
        d2_starts = list(data_loader.adj_dim2_start[:1])
        d2_stops = list(data_loader.adj_dim2_stop[:1])
        last_layout = 0
    else:
        adj_shards_mb_src = list(adj_shards)
        d1_starts = list(data_loader.adj_dim1_start)
        d1_stops = list(data_loader.adj_dim1_stop)
        d2_starts = list(data_loader.adj_dim2_start)
        d2_stops = list(data_loader.adj_dim2_stop)
        last_layout = (args.num_gcn_layers - 1) % min(3, args.num_gcn_layers)
    device = torch.device("cuda")

    if args.timing_end_epoch is None:
        args.timing_end_epoch = args.num_epochs - 1

    # ---- prefetched minibatch preparation (side stream + worker thread;
    # no ax timers inside the worker: the timer stack is not thread-safe.
    # Under PLEXUS_SYNC_PREP=1 prepare() runs on the main thread, so the
    # internal prep timers are safe to enable for a breakdown.)
    prefetch_stream = torch.cuda.Stream()
    executor = ThreadPoolExecutor(max_workers=1)
    sync_prep = os.environ.get("PLEXUS_SYNC_PREP") == "1"
    # PLEXUS_SPARSE_GATHER=1: compact each layout's shard to its nonempty
    # columns and gather only the referenced H rows over r (selective
    # a2av instead of AG).  Pays off when the in-batch mean degree is well
    # below F (papers-class); see analysis/dist_spmm_papers_review.md.
    sparse_gather = os.environ.get("PLEXUS_SPARSE_GATHER") == "1"

    # epoch_perm sampler: one O(N) randperm per epoch (cached, int32),
    # each step slices a disjoint chunk -- same pairwise inclusion
    # probability as uniform (see sample_nodes_epoch_perm), so edge_scale
    # stays valid, but the per-step cost drops from O(N) to O(batch).
    max_chunks = max(1, num_nodes // batch_size)
    _perm = {"key": None, "t": None}

    def _epoch_perm_sample(epoch, step):
        rep, chunk = divmod(step, max_chunks)
        key = int(args.seed) + int(epoch) * 7919 + rep * 104729
        if _perm["key"] != key:
            gen = torch.Generator(device=device)
            gen.manual_seed(key)
            _perm["t"] = torch.randperm(num_nodes, generator=gen,
                                        device=device, dtype=torch.int32)
            _perm["key"] = key
        s = chunk * batch_size
        return _perm["t"][s:s + batch_size].to(torch.int64)

    def prepare(epoch, step):
        timers = ax.get_timers() if sync_prep else None
        with torch.cuda.stream(prefetch_stream):
            if timers:
                timers.start("prep sample+sort")
            if args.sampler == "epoch_perm":
                sample = _epoch_perm_sample(epoch, step)
            else:
                sample = sample_nodes(num_nodes, ratio=args.minibatch_ratio,
                                      seed=args.seed + epoch * 100003,
                                      step=step, device=device)
            sample, _ = torch.sort(sample)
            if timers:
                timers.stop("prep sample+sort")
                timers.start("prep layout indices")
            row_idx = build_layout_indices(
                sample, d1_starts, d1_stops, assume_sorted=True)
            col_idx = build_layout_indices(
                sample, d2_starts, d2_stops, assume_sorted=True)
            if timers:
                timers.stop("prep layout indices")
            if (row_idx[last_layout].numel() == 0
                    or col_idx[0].numel() == 0):
                return None
            if timers:
                timers.start("prep compact adj")
            adj_mb = build_compact_adj_shards(
                adj_shards_mb_src, row_idx, col_idx,
                d1_starts, d2_starts,
                edge_scale=edge_scale, enable_timers=sync_prep)
            if timers:
                timers.stop("prep compact adj")
            if sparse_gather:
                if timers:
                    timers.start("prep sparse plan")
                sg_mb = []
                for l, (adj_l, adjt_l) in enumerate(adj_mb):
                    # col compaction (r-axis gather), then row compaction
                    # via the transpose view (c-axis reduce)
                    adj2, adjt2, need = compact_shard_columns(adj_l, adjt_l)
                    adjt3, adj3, rows_nz = compact_shard_columns(adjt2, adj2)
                    lay = model.layers[l]
                    plan = {
                        "r": sparse_gather_plan_local(
                            need, adj_l.shape[1], lay.Gf, lay.Gr),
                        "c": sparse_gather_plan_local(
                            rows_nz, adj_l.shape[0], 1, lay.Gc),
                    }
                    sg_mb.append((adj3, adjt3, plan))
                adj_mb = sg_mb
                if timers:
                    timers.stop("prep sparse plan")
            a, b = steady_cell(int(col_idx[0].numel()), layer_num=0)
            x_mb = feats_blk.index_select(0, col_idx[0][a:b])
            a2, b2 = steady_cell(int(row_idx[last_layout].numel()),
                                 layer_num=args.num_gcn_layers)
            sel = row_idx[last_layout][a2:b2]
            y_mb = labels_blk.index_select(0, sel)
            m_mb = (tmask_blk.index_select(0, sel)
                    if tmask_blk is not None else None)
            ev = torch.cuda.Event()
            ev.record(prefetch_stream)
        return x_mb, y_mb, m_mb, adj_mb, ev

    total_steps = args.num_epochs * steps
    # PLEXUS_SYNC_PREP=1: run prep serially in-loop (no overlap with the
    # train step) -- diagnoses SM/HBM contention between prep and NCCL.
    # "prefetch wait" then measures the full prep cost directly.
    future = None if sync_prep else executor.submit(prepare, 0, 0)

    for epoch in range(args.num_epochs):
        if args.timing_start_epoch <= epoch <= args.timing_end_epoch:
            ax.get_timers().start(f"epoch {epoch}")
        epoch_loss, epoch_steps = 0.0, 0

        for step in range(steps):
            ax.get_timers().start("prefetch wait")
            if sync_prep:
                batch = prepare(epoch, step)
                torch.cuda.synchronize()
            else:
                batch = future.result()
            ax.get_timers().stop("prefetch wait")
            gstep = epoch * steps + step
            if not sync_prep and gstep + 1 < total_steps:
                ne, ns = divmod(gstep + 1, steps)
                future = executor.submit(prepare, ne, ns)
            if batch is None:
                if rank0:
                    print("[warn] empty local block; skipping step")
                continue
            x_mb, y_mb, m_mb, adj_mb, ev = batch
            torch.cuda.current_stream().wait_event(ev)
            if sparse_gather:
                # complete the selective plans (2 tiny collectives per
                # plan; must run on the main thread pre-forward)
                for l, shard in enumerate(adj_mb):
                    if len(shard) == 3:
                        sparse_gather_exchange(shard[2]["r"],
                                               model.layers[l].r_group)
                        sparse_gather_exchange(shard[2]["c"],
                                               model.layers[l].c_group)

            ax.get_timers().start("train step")
            optimizer.zero_grad()
            out = model(x_mb, adj_mb)
            loss = scattered_cross_entropy(
                out, y_mb, None, None, args.num_gcn_layers, node_mask=m_mb)
            loss.backward()
            sync_replicated_gradients(model)
            optimizer.step()
            ax.get_timers().stop("train step")

            epoch_loss += loss.item()
            epoch_steps += 1

        if args.timing_start_epoch <= epoch <= args.timing_end_epoch:
            ax.get_timers().stop(f"epoch {epoch}")
        if epoch == args.timing_end_epoch:
            print_axonn_timer_data(ax.get_timers().get_times()[0])
        if rank0:
            print(f"Epoch: {epoch:03d}, Train Loss: "
                  f"{epoch_loss / max(1, epoch_steps):.4f}")

        if args.eval and (epoch + 1) % args.eval_every == 0:
            res = full_graph_eval(model, features, adj_shards, labels,
                                  masks, padded_N, num_nodes, num_classes)
            if rank0:
                for split, r in res.items():
                    if r is not None:
                        print(f"{split.upper()}: acc {r[0] / r[1]:.4f} "
                              f"(n={r[1]})")

    if rank0:
        print(f"Peak GPU memory: "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
