#!/usr/bin/env python
"""
Standalone preprocessor for IGB Homogeneous datasets.

For tiny/small/medium: loads everything into memory, saves .pt, then partitions.
For large/full: streaming mode — features stay on disk as memmap, permutation
applied on-the-fly during partitioning to avoid OOM.

Usage:
    python scripts/preprocess_igb.py \
        --igb_root /path/to/IGB-Datasets \
        --igb_size medium \
        --num_classes 19 \
        --output_dir dataset/igb_medium/directed_sym/processed \
        --num_partitions 16 \
        --partition_output_dir dataset/igb_medium/directed_sym/igb_medium_part16
"""

import os
import sys
import gc
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plexus.utils.dataset_mag import _apply_normalization, partition_graph_2d
from plexus.utils.general import pad_dimension, set_seed

IGB_NUM_NODES = {
    "tiny": 100_000,
    "small": 1_000_000,
    "medium": 10_000_000,
    "large": 100_000_000,
    "full": 269_346_174,
}

FEAT_DIM = 1024

# Datasets too large to fit features in memory twice (for permutation copy)
STREAMING_SIZES = {"large", "full", "medium"}


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess IGB Homogeneous dataset for plexus")
    parser.add_argument("--igb_root", type=str, required=True,
                        help="Root directory of IGB-Datasets")
    parser.add_argument("--igb_size", type=str, default="medium",
                        choices=["tiny", "small", "medium", "large", "full"])
    parser.add_argument("--num_classes", type=int, default=19, choices=[19, 2983])
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for the preprocessed .pt file")
    parser.add_argument("--num_partitions", type=int, default=0,
                        help="Number of 2D partitions (0 = skip partitioning)")
    parser.add_argument("--partition_output_dir", type=str, default=None,
                        help="Directory for partitioned output (required if num_partitions > 0)")
    parser.add_argument("--partition_workers", type=int, default=8)
    parser.add_argument("--double_perm", action="store_true", default=True)
    parser.add_argument("--no_double_perm", dest="double_perm", action="store_false")
    parser.add_argument("--build_train_adj", action="store_true", default=True)
    parser.add_argument("--no_build_train_adj", dest="build_train_adj", action="store_false")
    parser.add_argument("--norm_type", type=str, default="symmetric",
                        choices=["symmetric", "row"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip data loading, use saved feat_perm)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_NPY_MAGIC = b"\x93NUMPY"

def _is_npy(path):
    """Check if a file is a proper .npy file (has numpy magic header)."""
    with open(path, "rb") as f:
        return f.read(6) == _NPY_MAGIC


def _load_mmap(path, dtype, shape):
    """Load a file as memory-mapped array, handling both .npy and raw binary."""
    if _is_npy(path):
        return np.load(path, mmap_mode="r")
    else:
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _igb_paths(igb_root, size, num_classes):
    feat_path = os.path.join(igb_root, size, "processed", "paper", "node_feat.npy")
    label_file = "node_label_19.npy" if num_classes == 19 else "node_label_2K.npy"
    label_path = os.path.join(igb_root, size, "processed", "paper", label_file)
    edge_path = os.path.join(igb_root, size, "processed", "paper__cites__paper", "edge_index.npy")
    return feat_path, label_path, edge_path


def load_igb(igb_root, size, num_classes):
    """Load IGB dataset. For large/full, features are NOT loaded (returns x=None)."""
    num_nodes = IGB_NUM_NODES[size]
    streaming = size in STREAMING_SIZES
    feat_path, label_path, edge_path = _igb_paths(igb_root, size, num_classes)

    print(f"Loading IGB {size} ({num_classes} classes)...")
    print(f"  Features: {feat_path}")
    print(f"  Labels:   {label_path}")
    print(f"  Edges:    {edge_path}")
    if streaming:
        print(f"  *** Streaming mode: features will NOT be loaded into memory ***")

    # Features
    if streaming:
        x = None  # will read from memmap during partitioning
    else:
        x = torch.from_numpy(np.load(feat_path)).float()
        print(f"  Features loaded: {x.shape}")

    # Labels
    if streaming:
        label_mmap = _load_mmap(label_path, dtype="float32", shape=(num_nodes,))
        y = torch.from_numpy(np.array(label_mmap, dtype=np.int64)).clone()
        del label_mmap
    else:
        y = torch.from_numpy(np.load(label_path)).long()
    print(f"  Labels loaded: {y.shape}, unique classes: {y.unique().numel()}")

    # Edges: IGB stores [E, 2], convert to PyG [2, E]
    edges_np = np.load(edge_path)
    edge_index = torch.from_numpy(edges_np.T).long()
    print(f"  Edges loaded: {edge_index.shape[1]:,} edges")
    del edges_np
    gc.collect()

    # IGB citation edges are already in (src -> dst) orientation.
    # No flip needed.

    # Self-loops
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    print(f"  After self-loops: {edge_index.shape[1]:,} edges")

    # Random 60/20/20 split
    perm_split = torch.randperm(num_nodes)
    n_train = int(num_nodes * 0.6)
    n_val = int(num_nodes * 0.2)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm_split[:n_train]] = True
    val_mask[perm_split[n_train:n_train + n_val]] = True
    test_mask[perm_split[n_train + n_val:]] = True

    if x is not None:
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    else:
        # Placeholder x for normalization (only needs edge_index and num_nodes)
        data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    print(f"  Split: train={n_train:,}, val={n_val:,}, test={num_nodes - n_train - n_val:,}\n")
    return data


# ---------------------------------------------------------------------------
# Train adjacency
# ---------------------------------------------------------------------------

def build_train_adj(data, norm_type):
    """Build train-induced adjacency (edges between train nodes only), then normalize."""
    from plexus.utils.dataset_mag import _normalize_gcn_edges, _normalize_row_edges

    train_mask = data.train_mask
    edge_index_raw = data.edge_index
    row, col = edge_index_raw[0], edge_index_raw[1]
    edge_mask = train_mask[row] & train_mask[col]

    train_edge_index = edge_index_raw[:, edge_mask]
    train_edge_weight = torch.ones(train_edge_index.size(1), dtype=torch.float32)

    train_edge_index, train_edge_weight = remove_self_loops(train_edge_index, train_edge_weight)
    train_nodes = torch.where(train_mask)[0]
    if train_nodes.numel() > 0:
        loop_index = torch.stack((train_nodes, train_nodes), dim=0)
        loop_weight = torch.ones(train_nodes.numel(), dtype=torch.float32)
        train_edge_index = torch.cat((train_edge_index, loop_index), dim=1)
        train_edge_weight = torch.cat((train_edge_weight, loop_weight), dim=0)

    if norm_type == "row":
        train_edge_index, train_edge_weight = _normalize_row_edges(
            train_edge_index, data.num_nodes, train_edge_weight
        )
    else:
        train_edge_index, train_edge_weight = _normalize_gcn_edges(
            train_edge_index, data.num_nodes, train_edge_weight
        )
    return train_edge_index, train_edge_weight


# ---------------------------------------------------------------------------
# Permutation (reindex strategy)
# ---------------------------------------------------------------------------

def permute_reindex(data, double_perm, has_features=True,
                    train_edge_index=None, train_edge_weight=None):
    """
    Permute graph via index re-mapping.
    Returns (data, perm, perm2_or_None) so callers can apply feature perm later.
    If has_features=False, feature permutation is skipped (for streaming mode).
    """
    N = data.num_nodes

    perm = torch.randperm(N)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(N)

    perm2 = None
    if double_perm:
        perm2 = torch.randperm(N)
        inv_perm2 = torch.empty_like(perm2)
        inv_perm2[perm2] = torch.arange(N)

    print("Permuting adjacency via index re-mapping...")

    # Permute main edge_index
    edge_index = data.edge_index
    permuted = torch.empty_like(edge_index)
    permuted[0] = inv_perm[edge_index[0]]
    if double_perm:
        permuted[1] = inv_perm2[edge_index[1]]
    else:
        permuted[1] = inv_perm[edge_index[1]]
    data.edge_index = permuted

    if double_perm:
        permuted2 = torch.empty_like(edge_index)
        permuted2[0] = inv_perm2[edge_index[0]]
        permuted2[1] = inv_perm[edge_index[1]]
        data.edge_index_2 = permuted2
        data.edge_weight_2 = data.edge_weight
        gc.collect()

    # Permute train adjacency
    if train_edge_index is not None and train_edge_weight is not None:
        tr = train_edge_index
        ptr = torch.empty_like(tr)
        ptr[0] = inv_perm[tr[0]]
        if double_perm:
            ptr[1] = inv_perm2[tr[1]]
        else:
            ptr[1] = inv_perm[tr[1]]
        data.edge_index_train = ptr
        data.edge_weight_train = train_edge_weight

        if double_perm:
            ptr2 = torch.empty_like(tr)
            ptr2[0] = inv_perm2[tr[0]]
            ptr2[1] = inv_perm[tr[1]]
            data.edge_index_train_2 = ptr2
            data.edge_weight_train_2 = train_edge_weight
        gc.collect()

    del inv_perm
    if double_perm:
        del inv_perm2
    gc.collect()
    print("Done permuting adjacency.\n")

    # Permute features (skip in streaming mode)
    if has_features and hasattr(data, "x") and data.x is not None:
        if double_perm:
            data.x = data.x[perm2, :]
        else:
            data.x = data.x[perm, :]
        gc.collect()
        print("Done permuting features.\n")

    # Permute labels
    labels = data.y
    data.y = labels[perm]
    if double_perm:
        data.y_2 = labels[perm2]
    del labels
    gc.collect()
    print("Done permuting labels.\n")

    # Permute masks
    if hasattr(data, "train_mask"):
        train_mask = data.train_mask
        val_mask = data.val_mask if hasattr(data, "val_mask") else None
        test_mask = data.test_mask if hasattr(data, "test_mask") else None

        data.train_mask = train_mask[perm]
        if val_mask is not None:
            data.val_mask = val_mask[perm]
        if test_mask is not None:
            data.test_mask = test_mask[perm]

        if double_perm:
            data.train_mask_2 = train_mask[perm2]
            if val_mask is not None:
                data.val_mask_2 = val_mask[perm2]
            if test_mask is not None:
                data.test_mask_2 = test_mask[perm2]

    data.plexus_double_perm = bool(double_perm)
    print("Done permuting masks.\n")

    # Return perm so streaming partitioner can apply it to features
    feat_perm = perm2 if double_perm else perm
    return data, feat_perm


# ---------------------------------------------------------------------------
# Streaming partitioner for large datasets
# ---------------------------------------------------------------------------

def _bucket_edges(edge_index, edge_weight, chunk_size_nodes, num_partitions):
    """
    Bucket-sort edges by (src_partition, dst_partition) in a single pass.
    Returns (sorted_edge_index, sorted_edge_weight, boundaries) where
    boundaries[dim1 * P + dim2] gives the start index for that bucket.
    """
    P = num_partitions
    src_part = (edge_index[0] // chunk_size_nodes).clamp_(max=P - 1)
    dst_part = (edge_index[1] // chunk_size_nodes).clamp_(max=P - 1)
    bucket_key = src_part.long() * P + dst_part.long()

    sort_idx = bucket_key.argsort(stable=True)
    sorted_edge_index = edge_index[:, sort_idx]
    sorted_edge_weight = edge_weight[sort_idx]

    # Find start of each bucket via searchsorted
    boundaries = torch.searchsorted(
        bucket_key[sort_idx],
        torch.arange(P * P + 1, dtype=torch.long),
    )
    del sort_idx, bucket_key, src_part, dst_part
    gc.collect()
    return sorted_edge_index, sorted_edge_weight, boundaries


def partition_streaming(
    data,
    feat_perm,
    feat_path,
    num_nodes,
    num_features,
    num_classes,
    num_partitions,
    output_dir,
    num_workers=1,
):
    """
    Partition graph data where features are read from memmap on-the-fly.

    Optimizations vs. naive per-partition scanning:
      1) Edges: bucket-sort once, then O(1) slice per partition.
      2) Features: batch per node-partition with sorted memmap indices
         for near-sequential disk reads (~P passes instead of P^2 random).
    """
    import time

    resume_mode = data is None
    if not resume_mode:
        double_perm = hasattr(data, "edge_index_2")
        has_train_adj = hasattr(data, "edge_index_train") and hasattr(data, "edge_weight_train")
        has_masks = hasattr(data, "train_mask") or hasattr(data, "val_mask") or hasattr(data, "test_mask")
    else:
        double_perm = False
        has_train_adj = False
        has_masks = False

    P = num_partitions

    chunk_size_nodes = pad_dimension(num_nodes, P) // P
    chunk_size_features = pad_dimension(num_features, P) // P

    feat_mmap = _load_mmap(feat_path, dtype="float32", shape=(num_nodes, num_features))

    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    if not resume_mode:
        # Save metadata
        torch.save((num_nodes, num_features, num_classes), os.path.join(output_dir, "metadata.pt"))

        # Create directories
        os.makedirs("edge_index/0", exist_ok=True)
        os.makedirs("input_features", exist_ok=True)
        os.makedirs("output_labels/0", exist_ok=True)
        if double_perm:
            os.makedirs("edge_index/1", exist_ok=True)
            os.makedirs("output_labels/1", exist_ok=True)
        if has_train_adj:
            os.makedirs("edge_index_train/0", exist_ok=True)
            if double_perm:
                os.makedirs("edge_index_train/1", exist_ok=True)
        if has_masks:
            for split in ("train", "val", "test"):
                os.makedirs(f"masks/{split}/0", exist_ok=True)
                if double_perm:
                    os.makedirs(f"masks/{split}/1", exist_ok=True)
    else:
        os.makedirs("input_features", exist_ok=True)

    train_mask_full = train_mask_full_2 = None
    val_mask_full = val_mask_full_2 = None
    test_mask_full = test_mask_full_2 = None
    if not resume_mode:
        train_mask_full = torch.as_tensor(data.train_mask) if hasattr(data, "train_mask") else None
        val_mask_full = torch.as_tensor(data.val_mask) if hasattr(data, "val_mask") else None
        test_mask_full = torch.as_tensor(data.test_mask) if hasattr(data, "test_mask") else None
        train_mask_full_2 = torch.as_tensor(data.train_mask_2) if hasattr(data, "train_mask_2") else None
        val_mask_full_2 = torch.as_tensor(data.val_mask_2) if hasattr(data, "val_mask_2") else None
        test_mask_full_2 = torch.as_tensor(data.test_mask_2) if hasattr(data, "test_mask_2") else None

    # ---- Save feat_perm checkpoint for resume ----
    ckpt_path = os.path.join(output_dir, "_feat_perm.pt")
    if not os.path.exists(ckpt_path):
        torch.save(feat_perm, ckpt_path)
        print(f"  Saved feat_perm checkpoint to {ckpt_path}")
    feat_perm_np = feat_perm.numpy()

    # ---- Step 1-2: Bucket-sort & write edge partitions ----
    edges_marker = os.path.join(output_dir, "_done_edges")
    if os.path.exists(edges_marker):
        print("  Edges already partitioned, skipping...")
    elif data is not None:
        t0 = time.time()
        print("  Bucket-sorting edges (perm 0)...")
        ei0, ew0, bounds0 = _bucket_edges(data.edge_index, data.edge_weight, chunk_size_nodes, P)
        gc.collect()

        ei1, ew1, bounds1 = None, None, None
        if double_perm:
            print("  Bucket-sorting edges (perm 1)...")
            ei1, ew1, bounds1 = _bucket_edges(data.edge_index_2, data.edge_weight_2, chunk_size_nodes, P)
            gc.collect()

        eiT0, ewT0, boundsT0 = None, None, None
        eiT1, ewT1, boundsT1 = None, None, None
        if has_train_adj:
            print("  Bucket-sorting train edges...")
            eiT0, ewT0, boundsT0 = _bucket_edges(data.edge_index_train, data.edge_weight_train, chunk_size_nodes, P)
            gc.collect()
            if double_perm:
                eiT1, ewT1, boundsT1 = _bucket_edges(data.edge_index_train_2, data.edge_weight_train_2, chunk_size_nodes, P)
                gc.collect()
        print(f"  Edge bucket-sort done in {time.time() - t0:.1f}s")

        t0 = time.time()
        print("  Writing edge partitions...")
        for d1 in range(P):
            for d2 in range(P):
                b = d1 * P + d2
                s, e = bounds0[b].item(), bounds0[b + 1].item()
                torch.save(
                    (ei0[:, s:e].clone(), ew0[s:e].clone()),
                    os.path.join(output_dir, "edge_index", "0", f"{d1}_{d2}.pt"),
                )
                if double_perm:
                    s1, e1 = bounds1[b].item(), bounds1[b + 1].item()
                    torch.save(
                        (ei1[:, s1:e1].clone(), ew1[s1:e1].clone()),
                        os.path.join(output_dir, "edge_index", "1", f"{d1}_{d2}.pt"),
                    )
                if has_train_adj:
                    sT, eT = boundsT0[b].item(), boundsT0[b + 1].item()
                    torch.save(
                        (eiT0[:, sT:eT].clone(), ewT0[sT:eT].clone()),
                        os.path.join(output_dir, "edge_index_train", "0", f"{d1}_{d2}.pt"),
                    )
                    if double_perm:
                        sT1, eT1 = boundsT1[b].item(), boundsT1[b + 1].item()
                        torch.save(
                            (eiT1[:, sT1:eT1].clone(), ewT1[sT1:eT1].clone()),
                            os.path.join(output_dir, "edge_index_train", "1", f"{d1}_{d2}.pt"),
                        )
        del ei0, ew0, bounds0, ei1, ew1, bounds1
        del eiT0, ewT0, boundsT0, eiT1, ewT1, boundsT1
        gc.collect()
        print(f"  Edge partitions done in {time.time() - t0:.1f}s")

        open(edges_marker, "w").close()
    else:
        print("  WARNING: edges not done but no data (resume mode), skipping edges")

    # ---- Step 3: Write labels & masks ----
    labels_marker = os.path.join(output_dir, "_done_labels")
    if os.path.exists(labels_marker):
        print("  Labels & masks already partitioned, skipping...")
    elif data is not None:
        t0 = time.time()
        print("  Writing labels & masks...")
        for d1 in range(P):
            ns1 = d1 * chunk_size_nodes
            ne1 = min((d1 + 1) * chunk_size_nodes, num_nodes)

            torch.save(data.y[ns1:ne1].clone(),
                        os.path.join(output_dir, "output_labels", "0", f"{d1}.pt"))
            if double_perm:
                torch.save(data.y_2[ns1:ne1].clone(),
                            os.path.join(output_dir, "output_labels", "1", f"{d1}.pt"))
            if has_masks:
                for mask_name, m0, m1 in [
                    ("train", train_mask_full, train_mask_full_2),
                    ("val", val_mask_full, val_mask_full_2),
                    ("test", test_mask_full, test_mask_full_2),
                ]:
                    if m0 is not None:
                        torch.save(m0[ns1:ne1].to(torch.bool).clone(),
                                    os.path.join(output_dir, "masks", mask_name, "0", f"{d1}.pt"))
                    if double_perm and m1 is not None:
                        torch.save(m1[ns1:ne1].to(torch.bool).clone(),
                                    os.path.join(output_dir, "masks", mask_name, "1", f"{d1}.pt"))
        print(f"  Labels & masks done in {time.time() - t0:.1f}s")

        open(labels_marker, "w").close()
    else:
        print("  WARNING: labels not done but no data (resume mode), skipping labels")

    # ---- Step 4: Write feature partitions (sorted memmap reads, per-d1 checkpoint) ----
    t0 = time.time()
    print("  Writing feature partitions (sorted memmap reads)...")
    skipped = 0
    for d1 in range(P):
        # Check if this node partition is already done
        all_exist = all(
            os.path.exists(os.path.join(output_dir, "input_features", f"{d1}_{d2}.pt"))
            for d2 in range(P)
        )
        if all_exist:
            skipped += 1
            continue

        ns1 = d1 * chunk_size_nodes
        ne1 = min((d1 + 1) * chunk_size_nodes, num_nodes)
        n_rows = ne1 - ns1

        raw_indices = feat_perm_np[ns1:ne1]
        sort_order = np.argsort(raw_indices)
        sorted_raw = raw_indices[sort_order]

        sorted_feats = feat_mmap[sorted_raw]

        feats = np.empty((n_rows, num_features), dtype=np.float32)
        feats[sort_order] = sorted_feats
        del sorted_feats, sorted_raw, sort_order, raw_indices
        gc.collect()

        for d2 in range(P):
            fs = d2 * chunk_size_features
            fe = min((d2 + 1) * chunk_size_features, num_features)
            torch.save(
                torch.from_numpy(feats[:, fs:fe].copy()),
                os.path.join(output_dir, "input_features", f"{d1}_{d2}.pt"),
            )

        del feats
        gc.collect()
        print(f"    node partition {d1}/{P} done ({time.time() - t0:.1f}s)")

    if skipped > 0:
        print(f"  ({skipped}/{P} node partitions already done, skipped)")
    del feat_mmap
    print(f"  Feature partitions done in {time.time() - t0:.1f}s")
    print(f"Streaming partition done. Output: {output_dir}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    streaming = args.igb_size in STREAMING_SIZES

    # --resume: skip data loading, load feat_perm from checkpoint
    if args.resume and streaming:
        if args.num_partitions <= 0:
            print("ERROR: --resume requires --num_partitions > 0.")
            sys.exit(1)
        if args.partition_output_dir is None:
            out_name = f"igb_{args.igb_size}_{args.num_classes}"
            args.partition_output_dir = os.path.join(
                args.output_dir, "..", f"{out_name}_part{args.num_partitions}"
            )

        ckpt_path = os.path.join(args.partition_output_dir, "_feat_perm.pt")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: checkpoint not found at {ckpt_path}")
            print("Run without --resume first to create the checkpoint.")
            sys.exit(1)

        print(f"Resuming from checkpoint: {ckpt_path}")
        feat_perm = torch.load(ckpt_path, weights_only=True)
        feat_path, _, _ = _igb_paths(args.igb_root, args.igb_size, args.num_classes)
        num_nodes = IGB_NUM_NODES[args.igb_size]

        partition_streaming(
            data=None,
            feat_perm=feat_perm,
            feat_path=feat_path,
            num_nodes=num_nodes,
            num_features=FEAT_DIM,
            num_classes=args.num_classes,
            num_partitions=args.num_partitions,
            output_dir=args.partition_output_dir,
            num_workers=args.partition_workers,
        )
        print("All done!")
        return

    # Step 1: Load IGB data (features skipped for large/full)
    data = load_igb(args.igb_root, args.igb_size, args.num_classes)

    # Step 2: Save raw edge_index before normalization (needed for train adj)
    edge_index_raw = data.edge_index.clone()

    # Step 3: Normalize adjacency
    print(f"Applying {args.norm_type} normalization...")
    data = _apply_normalization(data, args.norm_type)
    gc.collect()
    print("Normalization done.\n")

    # Step 4: Build train-induced adjacency
    train_ei, train_ew = None, None
    if args.build_train_adj and hasattr(data, "train_mask"):
        print("Building train-induced adjacency...")
        train_data = Data(edge_index=edge_index_raw, num_nodes=data.num_nodes,
                          train_mask=data.train_mask)
        train_ei, train_ew = build_train_adj(train_data, args.norm_type)
        del train_data
        gc.collect()
        print(f"Train adjacency: {train_ei.shape[1]:,} edges\n")
    del edge_index_raw
    gc.collect()

    # Step 5: Format labels
    data.y = data.y.reshape(-1)
    data.y = torch.nan_to_num(data.y, nan=-1)
    data.y = data.y.to(torch.long)

    # Step 6: Permute (features skipped in streaming mode)
    data, feat_perm = permute_reindex(
        data, args.double_perm, has_features=not streaming,
        train_edge_index=train_ei, train_edge_weight=train_ew,
    )

    if streaming:
        # --- Streaming path: skip .pt save, partition directly from memmap ---
        if args.num_partitions <= 0:
            print("ERROR: large/full datasets require --num_partitions > 0 (streaming mode).")
            sys.exit(1)

        if args.partition_output_dir is None:
            out_name = f"igb_{args.igb_size}_{args.num_classes}"
            args.partition_output_dir = os.path.join(
                args.output_dir, "..", f"{out_name}_part{args.num_partitions}"
            )

        feat_path, _, _ = _igb_paths(args.igb_root, args.igb_size, args.num_classes)
        num_nodes = IGB_NUM_NODES[args.igb_size]

        partition_streaming(
            data=data,
            feat_perm=feat_perm,
            feat_path=feat_path,
            num_nodes=num_nodes,
            num_features=FEAT_DIM,
            num_classes=args.num_classes,
            num_partitions=args.num_partitions,
            output_dir=args.partition_output_dir,
            num_workers=args.partition_workers,
        )
    else:
        # --- In-memory path for smaller datasets ---
        os.makedirs(args.output_dir, exist_ok=True)
        out_name = f"igb_{args.igb_size}_{args.num_classes}"
        out_path = os.path.join(args.output_dir, f"processed_{out_name}.pt")
        torch.save((data, args.num_classes), out_path)
        print(f"Saved preprocessed data to {out_path}\n")

        if args.num_partitions > 0:
            if args.partition_output_dir is None:
                args.partition_output_dir = os.path.join(
                    args.output_dir, "..", f"{out_name}_part{args.num_partitions}"
                )
            os.makedirs(args.partition_output_dir, exist_ok=True)
            print(f"Partitioning into {args.num_partitions} partitions...")
            partition_graph_2d(
                file_path=out_path,
                num_partitions=args.num_partitions,
                output_dir=args.partition_output_dir,
                num_workers=args.partition_workers,
            )
            print(f"Partitioned data saved to {args.partition_output_dir}\n")

    print("All done!")


if __name__ == "__main__":
    main()
