#!/usr/bin/env python
"""
Convert a PNNL .bin CSR graph file into a PyG Data .pt file
that can be fed into plexus preprocessing.

Usage:
    python scripts/convert_bin.py \
        --infile /path/to/graph.bin \
        --output /path/to/output.pt \
        --num_features 128 \
        --num_classes 32

Then preprocess with:
    python scripts/preprocess.py \
        --name protein \
        --input_dir /path/to/output.pt \
        --output_dir /path/to/processed \
        --unsupervised \
        --no_double_perm
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch_geometric.data import Data


def read_bin_to_csr(infile: str, elem_dtype=np.int64, weight_dtype=np.float64):
    """Read PNNL .bin CSR format, handling the rowptr[-1]=0 bug in some files."""
    fsize = os.path.getsize(infile)
    edge_rec = np.dtype([("tail", elem_dtype), ("weight", weight_dtype)])

    with open(infile, "rb") as f:
        M = int(np.fromfile(f, dtype=elem_dtype, count=1)[0])
        N_header = int(np.fromfile(f, dtype=elem_dtype, count=1)[0])
        rowptr = np.fromfile(f, dtype=elem_dtype, count=M + 1)
        offset = f.tell()

    remaining = fsize - offset
    actual_edges = remaining // edge_rec.itemsize

    # Fix rowptr[-1]=0 bug present in some files (AFDB, twitter7)
    if rowptr[-1] == 0 and M > 0 and actual_edges > 0:
        rowptr[-1] = actual_edges

    with open(infile, "rb") as f:
        f.seek(offset)
        edges = np.fromfile(f, dtype=edge_rec, count=actual_edges)

    tails = edges["tail"]
    weights = edges["weight"]
    return M, actual_edges, rowptr, tails, weights


def main():
    parser = argparse.ArgumentParser(description="Convert PNNL .bin CSR to PyG .pt")
    parser.add_argument("--infile", type=str, required=True, help="Input .bin file")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reading {args.infile} ...")
    M, nnz, rowptr, tails, weights = read_bin_to_csr(args.infile)
    print(f"  Vertices: {M:,}  Edges: {nnz:,}")

    # CSR -> COO edge_index
    print("Converting CSR to COO ...")
    row = np.empty(nnz, dtype=np.int64)
    for i in range(M):
        row[rowptr[i]:rowptr[i + 1]] = i
    col = tails.astype(np.int64)

    edge_index = torch.from_numpy(np.stack([row, col], axis=0))
    del row, col, tails
    print(f"  edge_index shape: {list(edge_index.shape)}")

    # Generate random train/val/test masks
    print("Generating train/val/test masks ...")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(M)
    train_end = int(args.train_ratio * M)
    val_end = int((args.train_ratio + args.val_ratio) * M)

    train_mask = torch.zeros(M, dtype=torch.bool)
    val_mask = torch.zeros(M, dtype=torch.bool)
    test_mask = torch.zeros(M, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    print(f"  train={train_mask.sum().item():,}  val={val_mask.sum().item():,}  test={test_mask.sum().item():,}")

    # Build PyG Data object
    data = Data(
        edge_index=edge_index,
        num_nodes=M,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")
    print(f"\nNext step — preprocess:")
    print(f"  python scripts/preprocess.py \\")
    print(f"      --name protein \\")
    print(f"      --input_dir {args.output} \\")
    print(f"      --output_dir <output_dir> \\")
    print(f"      --unsupervised --no_double_perm --no_build_train_adj")


if __name__ == "__main__":
    main()
