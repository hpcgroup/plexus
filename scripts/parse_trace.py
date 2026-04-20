#!/usr/bin/env python3
"""Parse Nsight Systems trace to extract epoch 8 timing breakdown.

Usage:
    python scripts/parse_trace.py [--trace-dir DIR] [--epoch N] [--rank R]

Reads the .nsys-rep (or its .sqlite export) for a given rank, extracts:
  - Total epoch time
  - Sampling time (epoch time minus train-step time)
  - Train step time, broken down into:
      allreduce, reduce_scatter, allgather, spmm, gemm, element_wise, others
"""

import argparse
import glob
import os
import sqlite3
import subprocess
import sys


def classify_kernel(name: str) -> str:
    """Classify a CUDA kernel name into a category."""
    nl = name.lower()
    if "allreduce" in nl:
        return "allreduce"
    if "reducescatter" in nl:
        return "reduce_scatter"
    if "allgather" in nl:
        return "allgather"
    if "csrmm" in nl or "spmm" in nl:
        return "spmm"
    if "gemm" in nl or "sgemm" in nl or "hgemm" in nl:
        return "gemm"
    if "elementwise" in nl:
        return "element_wise"
    return "others"


def ensure_sqlite(nsys_rep_path: str) -> str:
    """Return path to .sqlite file, exporting from .nsys-rep if needed."""
    sqlite_path = nsys_rep_path.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(sqlite_path):
        print(f"Exporting {nsys_rep_path} -> {sqlite_path} ...")
        subprocess.run(
            ["nsys", "export", "--type=sqlite", nsys_rep_path],
            check=True,
        )
    return sqlite_path


def parse_trace(sqlite_path: str, target_epoch: int):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    # --- Epoch time range ---
    cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = ?",
        (f"epoch {target_epoch}",),
    )
    row = cur.fetchone()
    if row is None:
        print(f"ERROR: epoch {target_epoch} not found in trace.")
        conn.close()
        sys.exit(1)

    epoch_start, epoch_end = row
    epoch_ns = epoch_end - epoch_start

    # --- Train step ranges for this epoch ---
    cur.execute(
        """SELECT start, end FROM NVTX_EVENTS
           WHERE text LIKE ? ORDER BY start""",
        (f"train step % epoch {target_epoch}",),
    )
    steps = cur.fetchall()
    train_step_ns = sum(e - s for s, e in steps)
    sampling_ns = epoch_ns - train_step_ns

    # --- GPU kernel breakdown within train steps ---
    category_ns = {
        "allreduce": 0,
        "reduce_scatter": 0,
        "allgather": 0,
        "spmm": 0,
        "gemm": 0,
        "element_wise": 0,
        "others": 0,
    }

    for s_start, s_end in steps:
        cur.execute(
            """SELECT s.value, (k.end - k.start) AS dur
               FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.shortName = s.id
               WHERE k.start >= ? AND k.start < ?""",
            (s_start, s_end),
        )
        for kernel_name, dur in cur.fetchall():
            cat = classify_kernel(kernel_name)
            category_ns[cat] += dur

    conn.close()

    # --- Print results ---
    def ms(ns):
        return ns / 1e6

    total_kernel_ns = sum(category_ns.values())
    overhead_ns = train_step_ns - total_kernel_ns  # CPU overhead / idle / gaps

    # --- Print results ---
    print(f"{'='*60}")
    print(f"  Epoch {target_epoch} Timing Breakdown  (rank from {os.path.basename(sqlite_path)})")
    print(f"{'='*60}")
    print(f"  Total epoch time:       {ms(epoch_ns):10.2f} ms")
    print(f"  Sampling time:          {ms(sampling_ns):10.2f} ms")
    print(f"  Train step time (CPU):  {ms(train_step_ns):10.2f} ms")
    print(f"  Number of train steps:  {len(steps)}")
    print()
    print(f"  --- Train Step Breakdown ---")
    print(f"  {'Category':<20s} {'Time (ms)':>10s} {'%':>8s}")
    print(f"  {'-'*40}")
    for cat in ["allreduce", "reduce_scatter", "allgather", "spmm", "gemm", "element_wise", "others"]:
        t = category_ns[cat]
        pct = 100.0 * t / train_step_ns if train_step_ns > 0 else 0
        print(f"  {cat:<20s} {ms(t):10.2f} ms {pct:7.1f}%")
    print(f"  {'cpu_overhead':<20s} {ms(overhead_ns):10.2f} ms {100.0*overhead_ns/train_step_ns:7.1f}%")
    print(f"  {'-'*40}")
    print(f"  {'Total':<20s} {ms(train_step_ns):10.2f} ms {100.0:7.1f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Parse nsys trace for epoch breakdown")
    parser.add_argument(
        "--trace-dir",
        default="/pscratch/sd/c/cunyang/gnn/plexus/traces/baseline",
        help="Directory containing .nsys-rep files",
    )
    parser.add_argument("--epoch", type=int, default=8, help="Epoch number to analyze")
    parser.add_argument("--rank", type=int, default=None, help="Rank index (default: first .nsys-rep found)")
    args = parser.parse_args()

    # Find .nsys-rep files
    reps = sorted(glob.glob(os.path.join(args.trace_dir, "*.nsys-rep")))
    if not reps:
        print(f"ERROR: No .nsys-rep files found in {args.trace_dir}")
        sys.exit(1)

    if args.rank is not None:
        # Match by rank suffix
        target = [r for r in reps if f"_{args.rank}.nsys-rep" in r]
        if not target:
            print(f"ERROR: No .nsys-rep file for rank {args.rank}")
            sys.exit(1)
        rep = target[0]
    else:
        rep = reps[0]

    print(f"Using trace: {rep}")
    sqlite_path = ensure_sqlite(rep)
    parse_trace(sqlite_path, args.epoch)


if __name__ == "__main__":
    main()
