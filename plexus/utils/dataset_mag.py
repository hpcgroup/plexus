import os
import sys
import gc
import re
import torch
import numpy as np
from numpy import dtype
import scipy.sparse as sp
from typing import Optional
import multiprocessing as mp
from scipy.io import mmread, mmwrite
from torch_geometric.data import Data
import torch_geometric.transforms as T
from numpy.core.multiarray import scalar
from plexus.utils.general import pad_dimension, set_seed
from concurrent.futures import ThreadPoolExecutor

# Prefer the bundled OGB copy (it pins `torch.load(..., weights_only=False)` and
# avoids PyTorch 2.6+ default `weights_only=True` issues when loading PyG data).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_local_ogb_root = os.path.join(_repo_root, "ogb")
if os.path.isdir(_local_ogb_root) and _local_ogb_root not in sys.path:
    sys.path.insert(0, _local_ogb_root)

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.datasets import (
    Reddit,
    SuiteSparseMatrixCollection,
)
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    scatter,
    to_undirected,
)
from torch_geometric.utils.sparse import (
    to_edge_index,
    to_torch_csr_tensor,
)

try:
    torch.serialization.add_safe_globals(
        [GlobalStorage, DataEdgeAttr, DataTensorAttr, scalar, dtype]
    )
except Exception:
    pass


def _load_ogb_dataset_compat(name: str, root: str):
    """
    PyTorch 2.6+ changed torch.load default to weights_only=True, but current OGB
    releases may still call torch.load() without this argument when reading processed
    PyG data. For trusted local datasets, force weights_only=False during OGB init.
    """
    original_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        try:
            return original_torch_load(*args, **kwargs)
        except TypeError as exc:
            if "weights_only" in str(exc):
                kwargs.pop("weights_only", None)
                return original_torch_load(*args, **kwargs)
            raise

    torch.load = _torch_load_compat
    try:
        return PygNodePropPredDataset(name=name, root=root)
    finally:
        torch.load = original_torch_load


def _load_mag_scholar(input_dir: str, variant: str, svd_dim: int = 128):
    """
    Load MAG-Scholar dataset from npz file and reduce sparse BoW features
    via randomized SVD.

    Args:
        input_dir: directory containing mag_coarse.npz / mag_fine.npz
        variant: "coarse" or "fine"
        svd_dim: target dimensionality after SVD (default: 128)

    Returns:
        data: PyG Data object with dense x, edge_index, y, and train/val/test masks
        num_classes: int
    """
    from sklearn.decomposition import TruncatedSVD

    npz_path = os.path.join(input_dir, f"mag_{variant}.npz")
    print(f"Loading MAG-Scholar ({variant}) from {npz_path} ...")
    raw = np.load(npz_path, allow_pickle=True)
    print(f"  npz keys: {raw.files}")

    # Reconstruct adjacency CSR matrix
    adj = sp.csr_matrix((
        raw['adj_matrix.data'],
        raw['adj_matrix.indices'],
        raw['adj_matrix.indptr'],
    ), shape=tuple(raw['adj_matrix.shape']))
    print(f"  Adjacency: {adj.shape[0]:,} nodes, {adj.nnz:,} edges")

    # Reconstruct attribute (sparse BoW) CSR matrix
    attr = sp.csr_matrix((
        raw['attr_matrix.data'],
        raw['attr_matrix.indices'],
        raw['attr_matrix.indptr'],
    ), shape=tuple(raw['attr_matrix.shape']))
    print(f"  Attributes: {attr.shape[0]:,} x {attr.shape[1]:,} (sparse BoW)")

    # Labels
    labels = raw['labels']
    class_names = raw['class_names']
    num_classes = len(class_names)
    print(f"  Classes: {num_classes} ({list(class_names[:5])}...)")

    del raw
    gc.collect()

    # SVD dimensionality reduction: sparse 2.78M-dim -> svd_dim dense
    print(f"  Running TruncatedSVD: {attr.shape[1]:,} -> {svd_dim} dims ...")
    svd = TruncatedSVD(n_components=svd_dim, algorithm='randomized', random_state=42)
    x_dense = svd.fit_transform(attr.astype(np.float32))
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD explained variance ratio: {explained:.4f}")
    x_dense = x_dense.astype(np.float32)

    del attr, svd
    gc.collect()

    # Convert adjacency to PyG edge_index via COO
    adj_coo = adj.tocoo()
    edge_index = torch.tensor(
        np.vstack([adj_coo.row, adj_coo.col]),
        dtype=torch.long,
    )
    del adj, adj_coo
    gc.collect()

    # Construct PyG Data
    N = x_dense.shape[0]
    data = Data(
        x=torch.from_numpy(x_dense),
        edge_index=edge_index,
        y=torch.from_numpy(labels.astype(np.int64)),
        num_nodes=N,
    )
    del x_dense, labels
    gc.collect()

    # Generate random 60/20/20 train/val/test split
    perm = np.random.permutation(N)
    train_end = int(0.6 * N)
    val_end = int(0.8 * N)

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"  Split: train={train_mask.sum().item():,}, "
          f"val={val_mask.sum().item():,}, test={test_mask.sum().item():,}")

    return data, num_classes


def _normalize_gcn_edges(edge_index: torch.Tensor, num_nodes: int, edge_weight=None):
    """
    Apply GCN normalization to a given edge list.
    Expects edge_index in message-passing orientation used by training.
    """
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), dtype=torch.float32, device=edge_index.device
        )
    else:
        edge_weight = torch.as_tensor(
            edge_weight, dtype=torch.float32, device=edge_index.device
        )

    row = edge_index[0]
    col = edge_index[1]

    deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


def _normalize_row_edges(edge_index: torch.Tensor, num_nodes: int, edge_weight=None):
    """
    Apply row normalization (D^{-1} A) to a given edge list.
    Each row sums to 1 (outgoing weights normalized by source degree).
    """
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), dtype=torch.float32, device=edge_index.device
        )
    else:
        edge_weight = torch.as_tensor(
            edge_weight, dtype=torch.float32, device=edge_index.device
        )

    row = edge_index[0]
    deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(torch.isinf(deg_inv), 0.0)
    edge_weight = deg_inv[row] * edge_weight

    return edge_index, edge_weight


def _apply_normalization(data, norm_type: str = "symmetric"):
    """
    Unified normalization entry point.
      - "symmetric": D^{-0.5} A D^{-0.5}  (GCNNorm, adds self-loops automatically)
      - "row":       D^{-1} A              (row-stochastic, manual self-loops)
    """
    if norm_type == "symmetric":
        return T.GCNNorm().forward(data)
    elif norm_type == "row":
        edge_index, edge_weight = remove_self_loops(
            data.edge_index, getattr(data, "edge_weight", None)
        )
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=data.num_nodes
        )
        edge_index, edge_weight = _normalize_row_edges(
            edge_index, data.num_nodes, edge_weight
        )
        data.edge_index = edge_index
        data.edge_weight = edge_weight
        return data
    else:
        raise ValueError(f"Unknown norm_type={norm_type!r}; expected 'symmetric' or 'row'")


def _build_train_induced_adj_for_papers(
    edge_index_raw: torch.Tensor,
    edge_weight_raw,
    train_mask: torch.Tensor,
    num_nodes: int,
    norm_type: str = "symmetric",
):
    """
    Build train-induced adjacency for papers100M without introducing self-loops
    on non-train nodes.
    """
    row = edge_index_raw[0]
    col = edge_index_raw[1]
    edge_mask = train_mask[row] & train_mask[col]

    train_edge_index = edge_index_raw[:, edge_mask]
    if edge_weight_raw is not None:
        train_edge_weight = torch.as_tensor(
            edge_weight_raw, dtype=torch.float32, device=train_edge_index.device
        )[edge_mask]
    else:
        train_edge_weight = torch.ones(
            train_edge_index.size(1),
            dtype=torch.float32,
            device=train_edge_index.device,
        )

    # Enforce exactly one self-loop per train node before normalization.
    train_edge_index, train_edge_weight = remove_self_loops(
        train_edge_index, train_edge_weight
    )
    train_nodes = torch.where(train_mask)[0]
    if train_nodes.numel() > 0:
        loop_index = torch.stack((train_nodes, train_nodes), dim=0)
        loop_weight = torch.ones(
            train_nodes.numel(), dtype=train_edge_weight.dtype, device=train_edge_weight.device
        )
        train_edge_index = torch.cat((train_edge_index, loop_index), dim=1)
        train_edge_weight = torch.cat((train_edge_weight, loop_weight), dim=0)

    if norm_type == "row":
        train_edge_index, train_edge_weight = _normalize_row_edges(
            train_edge_index, num_nodes, train_edge_weight
        )
    else:
        train_edge_index, train_edge_weight = _normalize_gcn_edges(
            train_edge_index, num_nodes, train_edge_weight
        )
    return train_edge_index, train_edge_weight


def preprocess_graph(
    name: str,
    input_dir: str,
    output_dir: str,
    double_perm: Optional[bool] = True,
    unsupervised: Optional[bool] = False,
    num_features: Optional[int] = 128,
    num_classes: Optional[int] = 32,
    directed: Optional[bool] = False,
    permute_strategy: str = "auto",
    build_train_adj: Optional[bool] = True,
    force_no_undirected: bool = False,
    force_undirected: bool = False,
    norm_type: str = "symmetric",
):
    """
    Function to take the raw graph data and preprocess it

    Args:
        name: dataset/graph name
        input_dir: directory where original data is stored
        output_dir: directory where processed data should be saved
        double_perm: whether double permutation optimization should be applied or not
        unsupervised: can specify if features/classes should be generated for the dataset
        num_features: can optionally specify number of input features for graphs without features data
        num_classes: can optionally specify number of classes for unlabeled data
        directed: specifies if graph is directed and uses transpose of adjacency matrix if so for incoming message aggregation
        permute_strategy: "auto" (default), "reindex", or "sparse_mm" for permuting the adjacency
        build_train_adj: whether to precompute train-induced adjacency (`edge_index_train*`)
            for `--train_adj` training mode
    Returns:
        saves the preprocessed data to output_dir as a .pt file
    """

    # retrieve the unprocessed dataset and normalize the input features
    if name == "reddit":
        dataset = Reddit(root=input_dir)
    elif name == "products":
        dataset = _load_ogb_dataset_compat(
            name="ogbn-products",
            root=input_dir,
        )
    elif name == "proteins" or name == "ogbn-proteins":
        dataset = _load_ogb_dataset_compat(
            name="ogbn-proteins",
            root=input_dir,
        )
    elif name == "arxiv" or name == "ogbn-arxiv":
        # directed = True
        dataset = _load_ogb_dataset_compat(
            name="ogbn-arxiv",
            root=input_dir,
        )
    elif name == "papers":
        if not force_undirected:
            directed = True
        dataset = _load_ogb_dataset_compat(
            name="ogbn-papers100M",
            root=input_dir,
        )
    elif name == "europe_osm":
        unsupervised = True
        dataset = SuiteSparseMatrixCollection(input_dir, "DIMACS10", name)
    elif name == "protein":
        # input_dir is actually path for .pt file
        unsupervised = True
        dataset = [torch.load(input_dir, weights_only=False)]
    elif name == "amazon":
        # input_dir is actually path for .pt file
        unsupervised = True
        dataset = [torch.load(input_dir, weights_only=False)]
    elif name in ("mag_coarse", "mag_fine"):
        # MAG-Scholar: already undirected, sparse BoW features reduced via SVD
        variant = "coarse" if "coarse" in name else "fine"
        data, num_classes = _load_mag_scholar(input_dir, variant)
        dataset = None
    else:
        raise Exception(name + " dataset not supported")

    print("Read the original dataset.\n")

    if dataset is not None:
        # Standard dataset extraction path
        split_idx = None
        if not unsupervised and hasattr(dataset, "get_idx_split"):
            try:
                split_idx = dataset.get_idx_split()
            except Exception:
                split_idx = None

        # get the relevant parts of the dataset and discard the rest
        data = dataset[0]
        # Decide whether to make the graph undirected:
        #  - arxiv: undirected by default (tunedGNN reference), unless --no_undirected
        #  - any dataset: forced undirected via --force_undirected
        _do_undirected = force_undirected or (
            (name == "arxiv" or name == "ogbn-arxiv") and not force_no_undirected
        )
        if _do_undirected:
            print(f"Converting {name} to undirected graph...")
            edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
            data.edge_index = edge_index
            print(f"Undirected graph: {edge_index.size(1)} edges (incl. self-loops)")

        if directed:
            # Convert once to incoming-message orientation before any normalization.
            data.edge_index = data.edge_index.flip(0)

        if name == "proteins" or name == "ogbn-proteins":
            # OGBN-Proteins provides edge features; following OGB baselines,
            # compute node features by averaging incident edge features.
            data.x = scatter(
                data.edge_attr,
                data.edge_index[0],
                dim=0,
                dim_size=data.num_nodes,
                reduce="mean",
            )
            data = T.NormalizeFeatures().forward(data)
            try:
                delattr(data, "edge_attr")
            except Exception:
                pass

            # ogbn-proteins is a 112-task multi-label problem.
            num_classes = int(data.y.size(-1))
        elif not unsupervised:
            num_classes = dataset.num_classes

        del dataset
        gc.collect()

        if split_idx is not None and not hasattr(data, "train_mask"):
            num_nodes = int(getattr(data, "num_nodes", data.x.shape[0]))
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_idx = torch.as_tensor(split_idx.get("train", []), dtype=torch.long).view(-1)
            valid_key = "valid" if "valid" in split_idx else "val"
            val_idx = torch.as_tensor(split_idx.get(valid_key, []), dtype=torch.long).view(-1)
            test_idx = torch.as_tensor(split_idx.get("test", []), dtype=torch.long).view(-1)

            if train_idx.numel():
                train_mask[train_idx] = True
            if val_idx.numel():
                val_mask[val_idx] = True
            if test_idx.numel():
                test_mask[test_idx] = True

            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

    if unsupervised:
        data.x = torch.rand(data.num_nodes, num_features)
        data = T.NormalizeFeatures().forward(data)
        gc.collect()

    edge_index_raw = data.edge_index
    edge_weight_raw = getattr(data, "edge_weight", None)

    # normalize the adjacency matrix
    data = _apply_normalization(data, norm_type)
    gc.collect()

    train_edge_index = None
    train_edge_weight = None
    if build_train_adj and hasattr(data, "train_mask"):
        train_mask = torch.as_tensor(data.train_mask).reshape(-1).to(torch.bool)
        if name == "papers":
            train_edge_index, train_edge_weight = _build_train_induced_adj_for_papers(
                edge_index_raw,
                edge_weight_raw,
                train_mask,
                data.num_nodes,
                norm_type=norm_type,
            )
        else:
            row = edge_index_raw[0]
            col = edge_index_raw[1]
            edge_mask = train_mask[row] & train_mask[col]

            train_edge_index_raw = edge_index_raw[:, edge_mask]
            if edge_weight_raw is not None:
                train_edge_weight_raw = torch.as_tensor(edge_weight_raw)[edge_mask]
            else:
                train_edge_weight_raw = None

            train_data = Data(edge_index=train_edge_index_raw, num_nodes=data.num_nodes)
            if train_edge_weight_raw is not None:
                train_data.edge_weight = train_edge_weight_raw
            train_data = _apply_normalization(train_data, norm_type)
            train_edge_index = train_data.edge_index
            train_edge_weight = train_data.edge_weight
            del train_data
            gc.collect()

    # assign labels based on number of edges for unsupervised case
    if unsupervised:
        row_counts = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        row_counts_np = row_counts.numpy()

        sorted_indices = np.argsort(row_counts_np)
        buckets = np.array_split(sorted_indices, num_classes)

        data.y = np.zeros_like(row_counts_np, dtype=int)
        for i, bucket in enumerate(buckets):
            data.y[bucket] = i

        data.y = torch.tensor(data.y, dtype=torch.long, device=row_counts.device)

    # get labels in the format we expect
    if data.y.ndim == 2 and data.y.size(-1) > 1:
        # Multi-label: preserve NaNs so OGB evaluator can ignore missing labels.
        data.y = data.y.to(torch.float)
        print("Normalized the adjacency matrix and kept multi-label targets.\n")
    else:
        data.y = data.y.reshape(-1)
        data.y = torch.nan_to_num(data.y, nan=-1)
        data.y = data.y.type(torch.LongTensor)
        print("Normalized the adjacency matrix and converted labels to long.\n")

    # number of nodes in graph
    N = data.x.shape[0]

    with torch.no_grad():
        permute_strategy = (permute_strategy or "auto").lower()
        valid_strategies = {"auto", "reindex", "sparse_mm"}
        if permute_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid permute_strategy={permute_strategy!r}; expected one of {sorted(valid_strategies)}"
            )

        # permutation order of vertices
        perm = torch.randperm(N)

        if double_perm:
            perm2 = torch.randperm(N)

        if permute_strategy == "auto":
            # Sparse CSR + sparse.mm can be prohibitively memory hungry for very
            # large graphs (e.g., ogbn-papers100M). Reindexing is equivalent for
            # permutations and avoids materializing CSR matrices.
            num_edges = int(data.edge_index.size(1))
            if name == "papers" or N >= 10_000_000 or num_edges >= 200_000_000:
                permute_strategy = "reindex"
            else:
                permute_strategy = "sparse_mm"

        if permute_strategy == "reindex":
            print("Permuting adjacency via index re-mapping (no CSR sparse.mm)...\n")

            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(N, device=perm.device)

            if double_perm:
                inv_perm2 = torch.empty_like(perm2)
                inv_perm2[perm2] = torch.arange(N, device=perm2.device)

            edge_index = data.edge_index

            permuted_edge_index = torch.empty_like(edge_index)
            permuted_edge_index[0] = inv_perm[edge_index[0]]
            if double_perm:
                permuted_edge_index[1] = inv_perm2[edge_index[1]]
            else:
                permuted_edge_index[1] = inv_perm[edge_index[1]]
            data.edge_index = permuted_edge_index
            gc.collect()

            if double_perm:
                permuted_edge_index_2 = torch.empty_like(edge_index)
                permuted_edge_index_2[0] = inv_perm2[edge_index[0]]
                permuted_edge_index_2[1] = inv_perm[edge_index[1]]
                data.edge_index_2 = permuted_edge_index_2
                # reuse weights (same edges, different node ids)
                data.edge_weight_2 = data.edge_weight
                gc.collect()

            if train_edge_index is not None and train_edge_weight is not None:
                tr_edge_index = train_edge_index

                permuted_train_edge_index = torch.empty_like(tr_edge_index)
                permuted_train_edge_index[0] = inv_perm[tr_edge_index[0]]
                if double_perm:
                    permuted_train_edge_index[1] = inv_perm2[tr_edge_index[1]]
                else:
                    permuted_train_edge_index[1] = inv_perm[tr_edge_index[1]]
                data.edge_index_train = permuted_train_edge_index
                data.edge_weight_train = train_edge_weight
                gc.collect()

                if double_perm:
                    permuted_train_edge_index_2 = torch.empty_like(tr_edge_index)
                    permuted_train_edge_index_2[0] = inv_perm2[tr_edge_index[0]]
                    permuted_train_edge_index_2[1] = inv_perm[tr_edge_index[1]]
                    data.edge_index_train_2 = permuted_train_edge_index_2
                    data.edge_weight_train_2 = train_edge_weight
                    gc.collect()

            del inv_perm
            if double_perm:
                del inv_perm2
            gc.collect()

            print("Done permuting the adjacency matrix.\n")
        else:
            # create permutation matrix P
            row_indices = torch.arange(N)
            col_indices = perm
            values = torch.ones(N, dtype=torch.float32)
            P = torch.sparse_coo_tensor(
                torch.stack([row_indices, col_indices]), values, (N, N)
            ).to_sparse_csr()
            gc.collect()

            print("Created P matrix.\n")

            if double_perm:
                col_indices_2 = perm2
                P2 = torch.sparse_coo_tensor(
                    torch.stack([row_indices, col_indices_2]),
                    values,
                    (N, N),
                ).to_sparse_csr()
                gc.collect()

                print("Created P2 matrix.\n")

            # convert the adj matrices to csr format
            adj_full = to_torch_csr_tensor(data.edge_index, data.edge_weight, size=(N, N))
            adj_train = (
                to_torch_csr_tensor(train_edge_index, train_edge_weight, size=(N, N))
                if train_edge_index is not None and train_edge_weight is not None
                else None
            )
            has_train_adj = adj_train is not None
            train_adj_perm = None
            train_adj_perm_2 = None
            gc.collect()

            # A = P * A
            data.edge_index = torch.sparse.mm(P, adj_full)
            print("Completed A = P * A\n")

            if double_perm:
                data.edge_index_2 = torch.sparse.mm(P2, adj_full)
                print("Completed A2 = P2 * A2\n")

            if has_train_adj:
                train_adj_perm = torch.sparse.mm(P, adj_train)
                if double_perm:
                    train_adj_perm_2 = torch.sparse.mm(P2, adj_train)
                print("Completed A_train = P * A_train\n")

            del adj_full
            if has_train_adj:
                del adj_train
            gc.collect()

            # A = A * P.AT

            if double_perm:
                data.edge_index = torch.sparse.mm(
                    data.edge_index, P2.transpose(0, 1).to_sparse_csr()
                )

                data.edge_index_2 = torch.sparse.mm(
                    data.edge_index_2, P.transpose(0, 1).to_sparse_csr()
                )

                print("Completed A = A * P2.T\n")
                print("Completed A2 = A2 * P.T\n")

                if has_train_adj:
                    train_adj_perm = torch.sparse.mm(
                        train_adj_perm, P2.transpose(0, 1).to_sparse_csr()
                    )
                    train_adj_perm_2 = torch.sparse.mm(
                        train_adj_perm_2, P.transpose(0, 1).to_sparse_csr()
                    )
                    print("Completed A_train = A_train * P2.T\n")
                    print("Completed A2_train = A2_train * P.T\n")
            else:
                data.edge_index = torch.sparse.mm(
                    data.edge_index, P.transpose(0, 1).to_sparse_csr()
                )

                print("Completed A = A * P.T\n")
                if has_train_adj:
                    train_adj_perm = torch.sparse.mm(
                        train_adj_perm, P.transpose(0, 1).to_sparse_csr()
                    )
                    print("Completed A_train = A_train * P.T\n")

            del P
            if double_perm:
                del P2
            gc.collect()

            # convert back to edge index format
            data.edge_index = to_edge_index(data.edge_index)
            gc.collect()

            if double_perm:
                data.edge_index_2 = to_edge_index(data.edge_index_2)
                gc.collect()

            if has_train_adj:
                data.edge_index_train = to_edge_index(train_adj_perm)
                gc.collect()
                if double_perm:
                    data.edge_index_train_2 = to_edge_index(train_adj_perm_2)
                    gc.collect()

            # set edge_index and edge_weight to the appropriate tensors
            data.edge_weight = data.edge_index[1]
            data.edge_index = data.edge_index[0]
            gc.collect()

            if double_perm:
                data.edge_weight_2 = data.edge_index_2[1]
                data.edge_index_2 = data.edge_index_2[0]
                gc.collect()

            if has_train_adj:
                data.edge_weight_train = data.edge_index_train[1]
                data.edge_index_train = data.edge_index_train[0]
                gc.collect()

                if double_perm:
                    data.edge_weight_train_2 = data.edge_index_train_2[1]
                    data.edge_index_train_2 = data.edge_index_train_2[0]
                    gc.collect()

            print("Done permuting the adjacency matrix.\n")

        # permute the input features
        if double_perm:
            data.x = data.x[perm2, :]
        else:
            data.x = data.x[perm, :]
        gc.collect()

        print("Done permuting the input features.\n")

        # permute the output labels
        labels = data.y
        data.y = labels[perm]

        if double_perm:
            data.y_2 = labels[perm2]

        del labels
        gc.collect()

        print("Done permuting the output labels.\n")

        # permute train/val/test masks (if present) to match the permuted labels
        if hasattr(data, "train_mask"):
            train_mask = torch.as_tensor(data.train_mask).reshape(-1)
            val_mask = (
                torch.as_tensor(data.val_mask).reshape(-1)
                if hasattr(data, "val_mask")
                else None
            )
            test_mask = (
                torch.as_tensor(data.test_mask).reshape(-1)
                if hasattr(data, "test_mask")
                else None
            )

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

    # save the data object and number of classes
    torch.save((data, num_classes), output_dir + "/processed_" + name + ".pt")
    print(
        "Saved the preprocessed dataset to " + output_dir + "/processed_" + name + ".pt"
    )


def write_to_mtx(file_path: str, output_dir: str):
    # Load the .pt file
    data, _ = torch.load(file_path, weights_only=False)

    # Extract edge index and weights
    edge_index = data.edge_index  # Shape: [2, num_edges]
    edge_weight = data.edge_weight  # Shape: [num_edges]
    num_nodes = data.x.shape[0]

    # Convert to numpy
    row, col = (
        edge_index[0].cpu().numpy(),
        edge_index[1].cpu().numpy(),
    )
    weights = edge_weight.cpu().numpy()  # Use provided weights

    # Create a sparse weighted adjacency matrix
    adj_matrix = sp.coo_matrix((weights, (row, col)), shape=(num_nodes, num_nodes))

    # Save to .mtx format
    match = re.search(r"([^/]+)\.pt$", file_path)
    name = match.group(1)
    output_path = output_dir + "/" + name + ".mtx"
    mmwrite(output_path, adj_matrix)
    print("Saved weighted graph adjacency matrix to " + output_path)


def mtx_to_pyg(mtx_file, output_file):
    # Load the MTX file (only extracts coordinate-based edges)
    matrix = mmread(mtx_file)  # Returns a sparse COO matrix

    print("read mtx file\n")

    # Extract row and column indices (ignoring values)
    row, col = matrix.nonzero()

    # Convert to PyTorch tensor with shape [2, num_edges]
    edge_index_np = np.vstack((row, col))  # Shape [2, num_edges]
    edge_index = torch.from_numpy(edge_index_np).long()  # Convert to torch tensor

    print("created edge index\n")

    # Create a PyTorch Geometric Data object
    data = Data(
        edge_index=edge_index,
        edge_weight=torch.ones(edge_index.shape[1]),
        num_nodes=matrix.shape[0],
    )

    # Save the processed graph to a .pt file
    torch.save(data, output_file)
    print(f"Saved processed graph to {output_file}")


def tsv_to_pyg(tsv_file, output_file, N):
    """
    Reads a TSV file representing a graph, converts it to a PyTorch Geometric Data object,
    and saves it to a file.
    """

    edge_list = []
    values = []

    with open(tsv_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            edge_list.append([int(parts[0]), int(parts[1])])
            values.append(float(parts[2]))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(values, dtype=torch.float)
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)

    torch.save(data, output_file)


def print_nnz_stats(file_path: str, num_partitions: int):
    data, _ = torch.load(file_path, weights_only=False)
    N, edge_index = data.num_nodes, data.edge_index

    # Determine chunk size
    chunk_size = N // num_partitions

    # Count nonzeros in each 2D chunk
    nonzero_counts = torch.zeros((num_partitions, num_partitions))

    for i in range(num_partitions):
        row_start = i * chunk_size
        row_end = (i + 1) * chunk_size if i < num_partitions - 1 else N

        for j in range(num_partitions):
            col_start = j * chunk_size
            col_end = (j + 1) * chunk_size if j < num_partitions - 1 else N

            nonzero_counts[i, j] = torch.sum(
                (edge_index[0, :] >= row_start)
                & (edge_index[0, :] < row_end)
                & (edge_index[1, :] >= col_start)
                & (edge_index[1, :] < col_end)
            )

            print(f"row_partition_num: {i}, col_partition_num: {j}")
            print(str(nonzero_counts[i, j]))
            print("")

    # Compute statistics
    min_count = nonzero_counts.min().item()
    mean_count = nonzero_counts.mean().item()
    median_count = nonzero_counts.median().item()
    max_count = nonzero_counts.max().item()

    # Print results
    print(
        f"Min: {min_count}, Mean: {mean_count}, Median: {median_count}, Max: {max_count}"
    )
    print(f"Max/Mean: {max_count / mean_count}, Max/Min: {max_count / min_count}")


def partition_graph_2d(
    file_path: str,
    num_partitions: int,
    output_dir: str,
    num_workers: Optional[int] = 8,
):
    """
    Function to take preprocessed data and statically shard it into 2D shards
    so that parallel data loading can be used

    Args:
        file_path: path to file that contains the preprocessed data
        num_partitions: number of partitions along one dimension (ex: 8 num_partitions means
            data will be partitioned into 8x8 chunks)
        output_dir: directory to save the partitioned files to
        num_workers: number of worker threads used for partitioning (lower if OOM)

    Returns:
        saves the partitioned data to output_dir
    """

    data, num_classes = torch.load(file_path, weights_only=False)

    # checking if the data was processed with the double permutation optimization
    double_perm = False
    if hasattr(data, "edge_index_2"):
        double_perm = True

    print("done loading data")

    num_nodes, num_features = data.x.shape[0], data.x.shape[1]

    # size of each partition of the features matrix
    chunk_size_nodes = pad_dimension(data.x.shape[0], num_partitions) // num_partitions
    chunk_size_features = (
        pad_dimension(data.x.shape[1], num_partitions) // num_partitions
    )

    edge_index_np = data.edge_index
    edge_weight_np = data.edge_weight

    if double_perm:
        edge_index_np_2 = data.edge_index_2
        edge_weight_np_2 = data.edge_weight_2

    has_train_adj = hasattr(data, "edge_index_train") and hasattr(data, "edge_weight_train")
    if has_train_adj:
        edge_index_train_np = data.edge_index_train
        edge_weight_train_np = data.edge_weight_train
        if double_perm:
            edge_index_train_np_2 = data.edge_index_train_2
            edge_weight_train_np_2 = data.edge_weight_train_2

    gc.collect()

    os.chdir(output_dir)

    # save metadata
    torch.save(
        (num_nodes, num_features, num_classes),
        os.path.join(output_dir, "metadata.pt"),
    )

    # make directories where partitions will be stored
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

    has_masks = (
        hasattr(data, "train_mask") or hasattr(data, "val_mask") or hasattr(data, "test_mask")
    )
    if has_masks:
        os.makedirs("masks/train/0", exist_ok=True)
        os.makedirs("masks/val/0", exist_ok=True)
        os.makedirs("masks/test/0", exist_ok=True)
        if double_perm:
            os.makedirs("masks/train/1", exist_ok=True)
            os.makedirs("masks/val/1", exist_ok=True)
            os.makedirs("masks/test/1", exist_ok=True)

    # materialize masks once (avoid repeated conversions in every worker)
    train_mask_full = (
        torch.as_tensor(getattr(data, "train_mask", None)) if hasattr(data, "train_mask") else None
    )
    val_mask_full = (
        torch.as_tensor(getattr(data, "val_mask", None)) if hasattr(data, "val_mask") else None
    )
    test_mask_full = (
        torch.as_tensor(getattr(data, "test_mask", None)) if hasattr(data, "test_mask") else None
    )

    train_mask_full_2 = (
        torch.as_tensor(getattr(data, "train_mask_2", None))
        if hasattr(data, "train_mask_2")
        else None
    )
    val_mask_full_2 = (
        torch.as_tensor(getattr(data, "val_mask_2", None)) if hasattr(data, "val_mask_2") else None
    )
    test_mask_full_2 = (
        torch.as_tensor(getattr(data, "test_mask_2", None))
        if hasattr(data, "test_mask_2")
        else None
    )

    def process_partition(chunk_idx_dim1, chunk_idx_dim2):
        # following calculations are for indices which will be used
        # in partitioning the data for a given partition

        nodes_start_idx_dim1 = chunk_idx_dim1 * chunk_size_nodes
        nodes_stop_idx_dim1 = min(
            (chunk_idx_dim1 + 1) * chunk_size_nodes, data.x.shape[0]
        )

        nodes_start_idx_dim2 = chunk_idx_dim2 * chunk_size_nodes
        nodes_stop_idx_dim2 = min(
            (chunk_idx_dim2 + 1) * chunk_size_nodes, data.x.shape[0]
        )

        features_start_idx = chunk_idx_dim2 * chunk_size_features
        features_stop_idx = min(
            (chunk_idx_dim2 + 1) * chunk_size_features,
            data.x.shape[1],
        )

        valid_src = (edge_index_np[0, :] >= nodes_start_idx_dim1) & (
            edge_index_np[0, :] < nodes_stop_idx_dim1
        )
        valid_dst = (edge_index_np[1, :] >= nodes_start_idx_dim2) & (
            edge_index_np[1, :] < nodes_stop_idx_dim2
        )
        adj_mask = valid_src & valid_dst

        if double_perm:
            valid_src_2 = (edge_index_np_2[0, :] >= nodes_start_idx_dim1) & (
                edge_index_np_2[0, :] < nodes_stop_idx_dim1
            )
            valid_dst_2 = (edge_index_np_2[1, :] >= nodes_start_idx_dim2) & (
                edge_index_np_2[1, :] < nodes_stop_idx_dim2
            )
            adj_mask_2 = valid_src_2 & valid_dst_2

        # get the relevant adj matrix partition
        adj_chunk = edge_index_np[:, adj_mask]
        adj_weights_chunk = edge_weight_np[adj_mask]

        if double_perm:
            adj_chunk_2 = edge_index_np_2[:, adj_mask_2]
            adj_weights_chunk_2 = edge_weight_np_2[adj_mask_2]

        # save adj partition
        torch.save(
            (adj_chunk.clone(), adj_weights_chunk.clone()),
            os.path.join(
                output_dir,
                "edge_index",
                "0",
                f"{chunk_idx_dim1}_{chunk_idx_dim2}.pt",
            ),
        )

        del adj_chunk
        del adj_weights_chunk
        gc.collect()

        if double_perm:
            torch.save(
                (adj_chunk_2.clone(), adj_weights_chunk_2.clone()),
                os.path.join(
                    output_dir,
                    "edge_index",
                    "1",
                    f"{chunk_idx_dim1}_{chunk_idx_dim2}.pt",
                ),
            )

            del adj_chunk_2
            del adj_weights_chunk_2
            gc.collect()

        if has_train_adj:
            valid_src_tr = (edge_index_train_np[0, :] >= nodes_start_idx_dim1) & (
                edge_index_train_np[0, :] < nodes_stop_idx_dim1
            )
            valid_dst_tr = (edge_index_train_np[1, :] >= nodes_start_idx_dim2) & (
                edge_index_train_np[1, :] < nodes_stop_idx_dim2
            )
            adj_mask_tr = valid_src_tr & valid_dst_tr

            adj_chunk_tr = edge_index_train_np[:, adj_mask_tr]
            adj_weights_chunk_tr = edge_weight_train_np[adj_mask_tr]

            torch.save(
                (adj_chunk_tr.clone(), adj_weights_chunk_tr.clone()),
                os.path.join(
                    output_dir,
                    "edge_index_train",
                    "0",
                    f"{chunk_idx_dim1}_{chunk_idx_dim2}.pt",
                ),
            )

            del adj_chunk_tr
            del adj_weights_chunk_tr
            gc.collect()

            if double_perm:
                valid_src_tr_2 = (edge_index_train_np_2[0, :] >= nodes_start_idx_dim1) & (
                    edge_index_train_np_2[0, :] < nodes_stop_idx_dim1
                )
                valid_dst_tr_2 = (edge_index_train_np_2[1, :] >= nodes_start_idx_dim2) & (
                    edge_index_train_np_2[1, :] < nodes_stop_idx_dim2
                )
                adj_mask_tr_2 = valid_src_tr_2 & valid_dst_tr_2

                adj_chunk_tr_2 = edge_index_train_np_2[:, adj_mask_tr_2]
                adj_weights_chunk_tr_2 = edge_weight_train_np_2[adj_mask_tr_2]

                torch.save(
                    (adj_chunk_tr_2.clone(), adj_weights_chunk_tr_2.clone()),
                    os.path.join(
                        output_dir,
                        "edge_index_train",
                        "1",
                        f"{chunk_idx_dim1}_{chunk_idx_dim2}.pt",
                    ),
                )

                del adj_chunk_tr_2
                del adj_weights_chunk_tr_2
                gc.collect()

        # get features matrix partition and save it
        features_chunk = data.x[
            nodes_start_idx_dim1:nodes_stop_idx_dim1,
            features_start_idx:features_stop_idx,
        ]
        torch.save(
            features_chunk.clone(),
            os.path.join(
                output_dir,
                "input_features",
                f"{chunk_idx_dim1}_{chunk_idx_dim2}.pt",
            ),
        )

        # get the labels partition and save it
        if chunk_idx_dim2 == 0:
            labels_chunk = data.y[nodes_start_idx_dim1:nodes_stop_idx_dim1]
            torch.save(
                labels_chunk.clone(),
                os.path.join(
                    output_dir,
                    "output_labels",
                    "0",
                    f"{chunk_idx_dim1}.pt",
                ),
            )

            del labels_chunk
            gc.collect()

            if double_perm:
                labels_chunk_2 = data.y_2[nodes_start_idx_dim1:nodes_stop_idx_dim1]
                torch.save(
                    labels_chunk_2.clone(),
                    os.path.join(
                        output_dir,
                        "output_labels",
                        "1",
                        f"{chunk_idx_dim1}.pt",
                    ),
                )

            if has_masks:
                if train_mask_full is not None:
                    torch.save(
                        train_mask_full[nodes_start_idx_dim1:nodes_stop_idx_dim1]
                        .to(torch.bool)
                        .clone(),
                        os.path.join(
                            output_dir,
                            "masks",
                            "train",
                            "0",
                            f"{chunk_idx_dim1}.pt",
                        ),
                    )
                if val_mask_full is not None:
                    torch.save(
                        val_mask_full[nodes_start_idx_dim1:nodes_stop_idx_dim1]
                        .to(torch.bool)
                        .clone(),
                        os.path.join(
                            output_dir,
                            "masks",
                            "val",
                            "0",
                            f"{chunk_idx_dim1}.pt",
                        ),
                    )
                if test_mask_full is not None:
                    torch.save(
                        test_mask_full[nodes_start_idx_dim1:nodes_stop_idx_dim1]
                        .to(torch.bool)
                        .clone(),
                        os.path.join(
                            output_dir,
                            "masks",
                            "test",
                            "0",
                            f"{chunk_idx_dim1}.pt",
                        ),
                    )

                if double_perm:
                    if train_mask_full_2 is not None:
                        torch.save(
                            train_mask_full_2[
                                nodes_start_idx_dim1:nodes_stop_idx_dim1
                            ]
                            .to(torch.bool)
                            .clone(),
                            os.path.join(
                                output_dir,
                                "masks",
                                "train",
                                "1",
                                f"{chunk_idx_dim1}.pt",
                            ),
                        )
                    if val_mask_full_2 is not None:
                        torch.save(
                            val_mask_full_2[nodes_start_idx_dim1:nodes_stop_idx_dim1]
                            .to(torch.bool)
                            .clone(),
                            os.path.join(
                                output_dir,
                                "masks",
                                "val",
                                "1",
                                f"{chunk_idx_dim1}.pt",
                            ),
                        )
                    if test_mask_full_2 is not None:
                        torch.save(
                            test_mask_full_2[
                                nodes_start_idx_dim1:nodes_stop_idx_dim1
                            ]
                            .to(torch.bool)
                            .clone(),
                            os.path.join(
                                output_dir,
                                "masks",
                                "test",
                                "1",
                                f"{chunk_idx_dim1}.pt",
                            ),
                        )

        print(str(chunk_idx_dim1))
        print(str(chunk_idx_dim2))
        print("")

    # partition the data using multiple threads
    # to speed up the process (lower # of threads if it goes out of memory)

    print(f"Partitioning with num_workers={num_workers}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk_idx_dim1 in range(num_partitions):
            for chunk_idx_dim2 in range(num_partitions):
                futures.append(
                    executor.submit(
                        process_partition,
                        chunk_idx_dim1,
                        chunk_idx_dim2,
                    )
                )

        for future in futures:
            future.result()  # Ensure completion
