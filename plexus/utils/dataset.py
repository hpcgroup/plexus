import os
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
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.datasets import (
    Reddit,
    SuiteSparseMatrixCollection,
)
from torch_geometric.utils.sparse import (
    to_edge_index,
    to_torch_csr_tensor,
)


def preprocess_graph(
    name: str,
    input_dir: str,
    output_dir: str,
    double_perm: Optional[bool] = True,
    unsupervised: Optional[bool] = False,
    num_features: Optional[int] = 128,
    num_classes: Optional[int] = 32,
    directed: Optional[bool] = False,
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
    Returns:
        saves the preprocessed data to output_dir as a .pt file
    """

    # retrieve the unprocessed dataset and normalize the input features
    unsupervised, directed = False, False
    if name == "reddit":
        dataset = Reddit(root=input_dir, transform=T.NormalizeFeatures())
    elif name == "products":
        dataset = PygNodePropPredDataset(
            name="ogbn-products",
            root=input_dir,
            transform=T.NormalizeFeatures(),
        )
    elif name == "papers":
        directed = True
        dataset = PygNodePropPredDataset(
            name="ogbn-papers100M",
            root=input_dir,
            transform=T.NormalizeFeatures(),
        )
    elif name == "europe_osm":
        unsupervised = True
        dataset = SuiteSparseMatrixCollection(input_dir, "DIMACS10", name)
    elif name == "protein":
        # input_dir is actually path for .pt file
        unsupervised = True
        dataset = [torch.load(input_dir)]
    elif name == "amazon":
        # input_dir is actually path for .pt file
        unsupervised = True
        dataset = [torch.load(input_dir)]
    else:
        raise Exception(name + " dataset not supported")

    print("Read the original dataset.\n")

    # get the relevant parts of the dataset and discard the rest
    if not unsupervised:
        num_classes = dataset.num_classes

    data = dataset[0]
    del dataset
    gc.collect()

    if unsupervised:
        data.x = torch.rand(data.num_nodes, num_features)
        data = T.NormalizeFeatures().forward(data)
        gc.collect()

    # normalize the adjacency matrix
    data = T.GCNNorm().forward(data)
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
    data.y = data.y.reshape(-1)
    data.y = torch.nan_to_num(data.y, nan=-1)
    data.y = data.y.type(torch.LongTensor)

    print("Normalized the adjacency matrix and converted labels to long.\n")

    # number of nodes in graph
    N = data.x.shape[0]

    with torch.no_grad():
        # permutation order of vertices
        perm = torch.randperm(N)

        if double_perm:
            perm2 = torch.randperm(N)

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

        # convert the adj matrix to csr format
        data.edge_index = to_torch_csr_tensor(
            data.edge_index, data.edge_weight, size=(N, N)
        )
        gc.collect()

        adj = data.edge_index

        if directed:
            adj = adj.transpose(0, 1).to_sparse_csr()
            gc.collect()

        # A = P * A
        data.edge_index = torch.sparse.mm(P, adj)
        print("Completed A = P * A\n")

        if double_perm:
            data.edge_index_2 = torch.sparse.mm(P2, adj)
            print("Completed A2 = P2 * A2\n")

        del adj
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
        else:
            data.edge_index = torch.sparse.mm(
                data.edge_index, P.transpose(0, 1).to_sparse_csr()
            )

            print("Completed A = A * P.T\n")

        del P
        del P2
        gc.collect()

        # convert back to edge index format
        data.edge_index = to_edge_index(data.edge_index)
        gc.collect()

        if double_perm:
            data.edge_index_2 = to_edge_index(data.edge_index_2)
            gc.collect()

        # set edge_index and edge_weight to the appropriate tensors
        data.edge_weight = data.edge_index[1]
        data.edge_index = data.edge_index[0]
        gc.collect()

        if double_perm:
            data.edge_weight_2 = data.edge_index_2[1]
            data.edge_index_2 = data.edge_index_2[0]
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

    # save the data object and number of classes
    torch.save((data, num_classes), output_dir + "/processed_" + name + ".pt")
    print(
        "Saved the preprocessed dataset to " + output_dir + "/processed_" + name + ".pt"
    )


def write_to_mtx(file_path: str, output_dir: str):
    # Load the .pt file
    data, _ = torch.load(file_path)

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
    data, _ = torch.load(file_path)
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


def partition_graph_2d(file_path: str, num_partitions: int, output_dir: str):
    """
    Function to take preprocessed data and statically shard it into 2D shards
    so that parallel data loading can be used

    Args:
        file_path: path to file that contains the preprocessed data
        num_partitions: number of partitions along one dimension (ex: 8 num_partitions means
            data will be partitioned into 8x8 chunks)
        output_dir: directory to save the partitioned files to

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

        print(str(chunk_idx_dim1))
        print(str(chunk_idx_dim2))
        print("")

    # partition the data using multiple threads
    # to speed up the process (lower # of threads if it goes out of memory)
    with ThreadPoolExecutor(
        max_workers=min(mp.cpu_count(), num_partitions**2)
    ) as executor:
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
