#!/usr/bin/env python3
"""
Memory estimation for Plexus 3D-parallel GCN training.

Computes per-GPU memory breakdown for full-graph training and
compact minibatch training, following the exact parallelism layout
used in plexus (use_3d_linear=True).

Usage:
    # Run all presets (papers100M, igb_medium, igb_large):
    python scripts/estimate_memory.py

    # Run a single preset:
    python scripts/estimate_memory.py --preset papers100m
    python scripts/estimate_memory.py --preset igb_medium
    python scripts/estimate_memory.py --preset igb_large

    # Custom parameters (overrides preset):
    python scripts/estimate_memory.py --preset igb_medium --G_r 4 --G_c 2 --G_d 2 --p 0.1
    python scripts/estimate_memory.py --N 2449029 --hidden 256 --num_classes 47 \
        --nnz 123718280 --G_r 2 --G_c 2 --G_d 2 --p 0.1
"""

import argparse
import math


# ============================================================
# Presets
# ============================================================

PRESETS = {
    "papers100m": {
        "label": "ogbn-papers100M",
        "N": 111_059_956,
        "num_features": 128,
        "hidden": 128,
        "num_classes": 172,
        "num_layers": 3,
        "nnz": 1_615_685_872,
        "G_r": 8, "G_c": 4, "G_d": 4,   # 128 GPUs
        "p": 0.01,
    },
    "igb_medium": {
        "label": "IGB-Medium (19 classes)",
        "N": 10_000_000,
        "num_features": 1024,
        "hidden": 128,
        "num_classes": 19,
        "num_layers": 3,
        "nnz": 130_077_694,
        "G_r": 2, "G_c": 2, "G_d": 2,   # 8 GPUs
        "p": 0.05,
    },
    "igb_large": {
        "label": "IGB-Large (19 classes)",
        "N": 100_000_000,
        "num_features": 1024,
        "hidden": 128,
        "num_classes": 19,
        "num_layers": 3,
        "nnz": 1_323_571_364,
        "G_r": 8, "G_c": 4, "G_d": 4,   # 128 GPUs
        "p": 0.01,
    },
}


# ============================================================
# 3D layout helpers (mirrors plexus/gcn_conv.py & train.py)
# ============================================================

def layer_groups(layer_num):
    """GCN (outer, inner, depth) groups per layer (use_3d_linear=True)."""
    r = layer_num % 3
    if r == 0:
        return ("x", "z", "y")
    elif r == 1:
        return ("y", "x", "z")
    else:
        return ("z", "y", "x")


def adj_groups(layer_num):
    """Adjacency (dim1_rows, dim2_cols) partitioning per layer."""
    r = layer_num % 3
    if r == 0:
        return ("y", "x")
    elif r == 1:
        return ("z", "y")
    else:
        return ("x", "z")


def loss_groups(num_layers):
    """(node_group, class_group) for cross-entropy."""
    _, inner, depth = layer_groups(num_layers - 1)
    return (depth, inner)


# ============================================================
# Helpers
# ============================================================

def pad_up(value, *divisors):
    """Pad value to be divisible by all divisors."""
    lcm = divisors[0]
    for d in divisors[1:]:
        lcm = lcm * d // math.gcd(lcm, d)
    return math.ceil(value / lcm) * lcm


def fmt_size(nbytes):
    """Format bytes as human-readable string."""
    gb = nbytes / (1024 ** 3)
    if gb >= 0.01:
        return f"{gb:.3f} GB"
    mb = nbytes / (1024 ** 2)
    if mb >= 0.1:
        return f"{mb:.1f} MB"
    kb = nbytes / 1024
    return f"{kb:.1f} KB"


def fmt_shape(rows, cols):
    """Format shape string with K/M suffix."""
    def _s(v):
        if v >= 1e6:
            return f"{v/1e6:.2f}M"
        elif v >= 1e3:
            return f"{v/1e3:.1f}K"
        return str(int(v))
    return f"({_s(rows)}, {_s(cols)})"


# ============================================================
# Core estimation
# ============================================================

def estimate(
    N,
    num_features,
    hidden,
    num_classes,
    num_layers,
    nnz,
    G_r,
    G_c,
    G_d,
    dtype_bytes=4,
    idx_bytes=8,
    p=1.0,
):
    """
    Estimate per-GPU memory.

    Returns:
        fixed_items:  list of (category, description, bytes)
        step_items:   list of (category, description, bytes)
    """
    gs = {"x": G_r, "y": G_c, "z": G_d}
    N_eff = int(N * p) if p < 1.0 else N

    fixed_items = []
    step_items = []

    # ========== FIXED MEMORY ==========

    # 1. Adjacency matrices (CSR: crow + col + values) × (adj + adj_t)
    for lyr in range(num_layers):
        d1, d2 = adj_groups(lyr)
        num_parts = gs[d1] * gs[d2]
        nnz_local = nnz / num_parts
        nrows_local = N / gs[d1]

        crow = (nrows_local + 1) * idx_bytes
        col = nnz_local * idx_bytes
        val = nnz_local * dtype_bytes
        adj_bytes = crow + col + val

        fixed_items.append((
            "Adjacency",
            f"adj  layout {lyr} ({d1}×{d2}={num_parts}p, "
            f"nnz≈{nnz_local/1e6:.1f}M)",
            adj_bytes,
        ))
        fixed_items.append((
            "Adjacency",
            f"adj_t layout {lyr} (transpose)",
            adj_bytes,
        ))

    # 2. Raw features (kept for index_select in minibatch; used directly in full-graph)
    _, feat_col = adj_groups(0)
    feat_rows = N / gs[feat_col]
    feat_bytes = feat_rows * num_features * dtype_bytes
    fixed_items.append((
        "Features",
        f"raw features {fmt_shape(feat_rows, num_features)}",
        feat_bytes,
    ))

    # 3. Labels + masks
    node_g, class_g = loss_groups(num_layers)
    label_rows = N / gs[node_g]
    label_bytes = label_rows * 8          # int64
    mask_bytes = label_rows * 1 * 3       # 3 bool masks (train/val/test)
    fixed_items.append((
        "Labels",
        f"labels ({label_rows/1e6:.2f}M × int64) + masks",
        label_bytes + mask_bytes,
    ))

    # 4. Model parameters
    param_elems = 0
    for lyr in range(num_layers):
        outer, inner, depth = layer_groups(lyr)
        local_in = pad_up(hidden, gs[inner]) // gs[inner]
        local_out = pad_up(hidden, gs[outer]) // gs[outer]
        # weight sharded by depth for layer 0, conditionally for others
        w_shard = local_in * local_out // gs[depth]
        param_elems += w_shard

    # input_linear (3D): W shape (local_out, local_in)
    il_in = pad_up(num_features, gs["y"], gs["y"]) // gs["y"]
    il_out = pad_up(hidden, gs["z"], gs["z"]) // gs["z"]
    param_elems += il_in * il_out + il_out   # weight + bias

    # output_linear (3D)
    last_outer, _, _ = layer_groups(num_layers - 1)
    ol_k = gs[last_outer]
    ol_in = pad_up(hidden, ol_k) // ol_k
    ol_out = pad_up(num_classes, gs[class_g]) // gs[class_g]
    param_elems += ol_in * ol_out + ol_out   # weight + bias

    param_bytes = param_elems * dtype_bytes
    fixed_items.append(("Parameters", f"model weights ({param_elems:,} elems)", param_bytes))

    # 5. Optimizer states (AdamW: m + v = 2× params)
    optim_bytes = param_bytes * 2
    fixed_items.append(("Optimizer", "AdamW m + v (2× params)", optim_bytes))

    # ========== PER-STEP MEMORY ==========

    # --- Compact sub-graph adjacency (minibatch only) ---
    if p < 1.0:
        # expected nnz in compact subgraph ≈ nnz × p²
        compact_nnz_total = nnz * p * p
        for lyr in range(num_layers):
            d1, d2 = adj_groups(lyr)
            num_parts = gs[d1] * gs[d2]
            nnz_c = compact_nnz_total / num_parts
            nrows_c = N_eff / gs[d1]
            c_bytes = ((nrows_c + 1) * idx_bytes +
                       nnz_c * idx_bytes +
                       nnz_c * dtype_bytes)
            step_items.append((
                "Compact Adj",
                f"compact adj+adj_t L{lyr} (nnz≈{nnz_c:.0f})",
                c_bytes * 2,  # adj + adj_t
            ))

    # --- Compact features slice (minibatch only) ---
    if p < 1.0:
        _, fc = adj_groups(0)
        fr = N_eff / gs[fc]
        step_items.append((
            "Features slice",
            f"features_mb {fmt_shape(fr, num_features)}",
            fr * num_features * dtype_bytes,
        ))

    # --- input_linear: saves x + weight for backward ---
    _, fc = adj_groups(0)
    il_rows = N_eff / gs[fc]
    il_local_in = pad_up(num_features, gs["y"], gs["y"]) // gs["y"]
    step_items.append((
        "input_linear",
        f"saved x {fmt_shape(il_rows, il_local_in)}",
        il_rows * il_local_in * dtype_bytes,
    ))

    # --- Per GCN layer: AGG + Norm (x_float + x_normed + rrms) + ReLU ---
    for lyr in range(num_layers):
        outer, inner, depth = layer_groups(lyr)
        d1, d2 = adj_groups(lyr)
        local_in = pad_up(hidden, gs[inner]) // gs[inner]
        local_out = pad_up(hidden, gs[outer]) // gs[outer]
        adj_rows = N_eff / gs[d1]

        # GCN: AGG saved for backward
        agg_bytes = adj_rows * local_in * dtype_bytes
        step_items.append((
            f"GCN L{lyr} AGG",
            f"AGG {fmt_shape(adj_rows, local_in)}",
            agg_bytes,
        ))

        # Norm: x_float (= GCN OUT, norm input) saved by autograd
        out_bytes = adj_rows * local_out * dtype_bytes
        step_items.append((
            f"Norm L{lyr}",
            f"x_float (=OUT) {fmt_shape(adj_rows, local_out)}",
            out_bytes,
        ))

        # Norm: x_normed saved for weight gradient
        step_items.append((
            f"Norm L{lyr}",
            f"x_normed {fmt_shape(adj_rows, local_out)}",
            out_bytes,
        ))

        # Norm: rrms (reciprocal RMS)
        rrms_bytes = adj_rows * 1 * dtype_bytes
        step_items.append((
            f"Norm L{lyr}",
            f"rrms {fmt_shape(adj_rows, 1)}",
            rrms_bytes,
        ))

        # ReLU: saves output
        step_items.append((
            f"ReLU L{lyr}",
            f"relu output {fmt_shape(adj_rows, local_out)}",
            out_bytes,
        ))

    # --- output_linear: saves x + weight ---
    ol_rows = N_eff / gs[node_g]
    last_outer_g, _, _ = layer_groups(num_layers - 1)
    ol_local_in = pad_up(hidden, gs[last_outer_g]) // gs[last_outer_g]
    step_items.append((
        "output_linear",
        f"saved x {fmt_shape(ol_rows, ol_local_in)}",
        ol_rows * ol_local_in * dtype_bytes,
    ))

    # --- Cross-entropy: softmax (fp32) + one-hot target (int64) ---
    local_classes = pad_up(num_classes, gs[class_g]) // gs[class_g]
    ce_rows = N_eff / gs[node_g]
    sm_bytes = ce_rows * local_classes * dtype_bytes   # softmax fp32
    tgt_bytes = ce_rows * local_classes * 8            # one-hot int64
    step_items.append((
        "CrossEntropy",
        f"softmax {fmt_shape(ce_rows, local_classes)}",
        sm_bytes,
    ))
    step_items.append((
        "CrossEntropy",
        f"target one-hot {fmt_shape(ce_rows, local_classes)} int64",
        tgt_bytes,
    ))

    # --- Peak backward gradients ---
    # During backward, temporary gradient tensors are allocated per layer.
    # Peak ≈ grad_agg + grad_x for the largest layer.
    max_grad = 0
    for lyr in range(num_layers):
        outer, inner, _ = layer_groups(lyr)
        d1, d2 = adj_groups(lyr)
        local_in = pad_up(hidden, gs[inner]) // gs[inner]
        local_out = pad_up(hidden, gs[outer]) // gs[outer]
        adj_rows = N_eff / gs[d1]
        adj_cols = N_eff / gs[d2]
        # grad_agg (adj_rows, local_in) + grad_x (adj_cols, local_in) + grad_output (adj_rows, local_out)
        grad_layer = (adj_rows * local_in + adj_cols * local_in + adj_rows * local_out) * dtype_bytes
        max_grad = max(max_grad, grad_layer)

    step_items.append((
        "Gradients",
        "peak backward (grad_agg + grad_x + grad_out)",
        max_grad,
    ))

    return fixed_items, step_items


# ============================================================
# Report printer
# ============================================================

def print_report(args, label=None):
    gs = {"x": args.G_r, "y": args.G_c, "z": args.G_d}
    total_gpus = args.G_r * args.G_c * args.G_d
    N_s = int(args.N * args.p)

    sep = "=" * 90
    dash = "-" * 90

    print(sep)
    title = f"  Plexus 3D-Parallel GCN: Per-GPU Memory Estimation"
    if label:
        title += f"  [{label}]"
    print(title)
    print(sep)
    print(f"  N = {args.N:,}   num_features = {args.num_features}   hidden = {args.hidden}")
    print(f"  num_classes = {args.num_classes}   num_layers = {args.num_layers}")
    print(f"  nnz (edges) = {args.nnz:,}   dtype = FP{args.dtype_bytes*8}   idx = INT{args.idx_bytes*8}")
    print(f"  G_r(x) = {args.G_r}   G_c(y) = {args.G_c}   G_d(z) = {args.G_d}   total = {total_gpus} GPUs")
    print(f"  Minibatch ratio p = {args.p}   N_s = {N_s:,}")
    print()

    # Layout info
    print("  Layer layout (use_3d_linear=True):")
    for lyr in range(args.num_layers):
        outer, inner, depth = layer_groups(lyr)
        d1, d2 = adj_groups(lyr)
        parts = gs[d1] * gs[d2]
        local_in = pad_up(args.hidden, gs[inner]) // gs[inner]
        local_out = pad_up(args.hidden, gs[outer]) // gs[outer]
        print(f"    L{lyr}: GCN(outer={outer}:{gs[outer]}, inner={inner}:{gs[inner]}, "
              f"depth={depth}:{gs[depth]})  adj({d1}×{d2}={parts}p)  "
              f"in={local_in} out={local_out}")
    node_g, class_g = loss_groups(args.num_layers)
    print(f"    Loss: node_group={node_g}:{gs[node_g]}  class_group={class_g}:{gs[class_g]}")
    print()

    # ---------- Full graph ----------
    fixed_full, step_full = estimate(
        args.N, args.num_features, args.hidden, args.num_classes,
        args.num_layers, args.nnz,
        args.G_r, args.G_c, args.G_d,
        args.dtype_bytes, args.idx_bytes,
        p=1.0,
    )

    # ---------- Minibatch ----------
    fixed_mini, step_mini = estimate(
        args.N, args.num_features, args.hidden, args.num_classes,
        args.num_layers, args.nnz,
        args.G_r, args.G_c, args.G_d,
        args.dtype_bytes, args.idx_bytes,
        p=args.p,
    )

    def _print_section(title, items):
        print(dash)
        print(f"  {title}")
        print(dash)
        print(f"  {'Category':<20} {'Description':<45} {'Size':>12}")
        print(f"  {'--------':<20} {'-----------':<45} {'----':>12}")
        total = 0
        for cat, desc, nbytes in items:
            total += nbytes
            print(f"  {cat:<20} {desc:<45} {fmt_size(nbytes):>12}")
        print(f"  {'':20} {'':45} {'----------':>12}")
        print(f"  {'SUBTOTAL':<20} {'':45} {fmt_size(total):>12}")
        return total

    # Print full-graph report
    print(sep)
    print("  FULL-GRAPH TRAINING")
    print(sep)
    t_fixed = _print_section("Fixed Memory (always on GPU)", fixed_full)
    print()
    t_step = _print_section("Per-Step Activations & Gradients", step_full)
    print()
    print(f"  >>> TOTAL (full graph): {fmt_size(t_fixed + t_step)}")
    print()

    # Print minibatch report
    if args.p < 1.0:
        print(sep)
        print(f"  COMPACT MINIBATCH TRAINING  (p = {args.p}, N_s = {N_s:,})")
        print(sep)
        t_fixed_m = _print_section("Fixed Memory (always on GPU)", fixed_mini)
        print()
        t_step_m = _print_section("Per-Step Activations & Gradients", step_mini)
        print()
        print(f"  >>> TOTAL (minibatch):  {fmt_size(t_fixed_m + t_step_m)}")
        print()

    # ---------- Summary ----------
    print(sep)
    print("  SUMMARY")
    print(sep)
    total_full = sum(b for _, _, b in fixed_full) + sum(b for _, _, b in step_full)
    if args.p < 1.0:
        total_mini = sum(b for _, _, b in fixed_mini) + sum(b for _, _, b in step_mini)
        step_full_total = sum(b for _, _, b in step_full)
        step_mini_total = sum(b for _, _, b in step_mini)
        fixed_total = sum(b for _, _, b in fixed_full)
        print(f"  {'':35} {'Full Graph':>14} {'Minibatch':>14} {'Saved':>14}")
        print(f"  {'Fixed memory':35} {fmt_size(fixed_total):>14} {fmt_size(fixed_total):>14} {'---':>14}")
        print(f"  {'Per-step (activations+grads)':35} {fmt_size(step_full_total):>14} {fmt_size(step_mini_total):>14} {fmt_size(step_full_total - step_mini_total):>14}")
        print(f"  {'TOTAL':35} {fmt_size(total_full):>14} {fmt_size(total_mini):>14} {fmt_size(total_full - total_mini):>14}")
        print()
        if step_full_total > 0:
            print(f"  Activation reduction: {(1 - step_mini_total / step_full_total) * 100:.1f}%")
        print(f"  With minibatch, memory bottleneck shifts to fixed adj + features.")
    else:
        print(f"  Total per-GPU memory: {fmt_size(total_full)}")
    print(sep)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-GPU memory for Plexus 3D-parallel GCN training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available presets:\n"
            "  papers100m   ogbn-papers100M   (N=111M, feat=128,  cls=172, nnz=1.6B)  128 GPUs\n"
            "  igb_medium   IGB-Medium 19cls   (N=10M,  feat=1024, cls=19,  nnz=130M)  8 GPUs\n"
            "  igb_large    IGB-Large  19cls   (N=100M, feat=1024, cls=19,  nnz=1.3B)  128 GPUs\n"
            "\n"
            "Examples:\n"
            "  python scripts/estimate_memory.py                            # all presets\n"
            "  python scripts/estimate_memory.py --preset igb_medium        # single preset\n"
            "  python scripts/estimate_memory.py --preset igb_medium --p 0.1 --hidden 256\n"
        ),
    )
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Use a predefined dataset configuration. "
                             "Omit to run ALL presets.")
    parser.add_argument("--N", type=int, default=None, help="Total number of nodes")
    parser.add_argument("--num_features", type=int, default=None, help="Input feature dimension")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of output classes")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of GCN layers")
    parser.add_argument("--nnz", type=int, default=None, help="Number of edges (nnz in adj)")
    parser.add_argument("--G_r", type=int, default=None, help="x-group size (G_intra_r)")
    parser.add_argument("--G_c", type=int, default=None, help="y-group size (G_intra_c)")
    parser.add_argument("--G_d", type=int, default=None, help="z-group size (G_intra_d)")
    parser.add_argument("--dtype_bytes", type=int, default=4, choices=[2, 4],
                        help="Bytes per value (FP32=4, FP16/BF16=2)")
    parser.add_argument("--idx_bytes", type=int, default=8, choices=[4, 8],
                        help="Bytes per CSR index (INT64=8, INT32=4)")
    parser.add_argument("--p", type=float, default=None,
                        help="Minibatch sampling ratio (1.0 = full graph only)")
    args = parser.parse_args()

    # Determine which presets to run
    if args.preset is not None:
        preset_names = [args.preset]
    elif args.N is not None:
        # Fully custom — no preset
        preset_names = [None]
    else:
        # No preset and no custom N → run all presets
        preset_names = list(PRESETS.keys())

    for i, name in enumerate(preset_names):
        if i > 0:
            print("\n\n")

        # Start from preset defaults (if any), then override with CLI args
        if name is not None:
            cfg = dict(PRESETS[name])
            label = cfg.pop("label", name)
        else:
            # Fully custom: require at least N and nnz
            cfg = {
                "N": 111_059_956,
                "num_features": 128,
                "hidden": 128,
                "num_classes": 172,
                "num_layers": 3,
                "nnz": 1_615_685_872,
                "G_r": 8, "G_c": 4, "G_d": 4,
                "p": 0.01,
            }
            label = "custom"

        # Override with any explicitly provided CLI args
        for key in ["N", "num_features", "hidden", "num_classes", "num_layers",
                     "nnz", "G_r", "G_c", "G_d", "p"]:
            cli_val = getattr(args, key, None)
            if cli_val is not None:
                cfg[key] = cli_val

        # Build a namespace for print_report
        ns = argparse.Namespace(
            dtype_bytes=args.dtype_bytes,
            idx_bytes=args.idx_bytes,
            **cfg,
        )
        print_report(ns, label=label)


if __name__ == "__main__":
    main()
