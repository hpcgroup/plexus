import torch
import torch.sparse
import time
import argparse


def multiply_partitioned_matrices_padded(
    partition_row,
    partition_col_x,
    partition_x_col,
    iterations=25,
    warmup_iterations=5,
    pt_file="/pscratch/sd/a/aranjan/gnn-env/gnn-datasets/products/processed_products.pt",
):
    """
    Reads a .pt file, pads and partitions the edge_index and x matrices,
    multiplies the partitions using sparse matrix multiplication, and measures the time.

    Args:
        pt_file (str): Path to the .pt file containing the data dictionary.
        partition_row (int): The number of partitions for the first dimension of edge_index.
        partition_col_x (int): The number of partitions for the second dimension of edge_index
                             and the first dimension of x.
        partition_x_col (int): The number of partitions for the second dimension of x.
        iterations (int): The total number of multiplication iterations to perform.
        warmup_iterations (int): The number of initial iterations to exclude from timing.
    """
    try:
        data, _ = torch.load(pt_file, weights_only=False)

        print(data.edge_weight[0:100])

        edge_index = data.edge_index
        x = data.x

        original_num_nodes = x.shape[0]
        original_x_cols = x.shape[1]

        # Calculate padded dimensions for edge_index (implied adjacency matrix)
        padded_rows = (
            (original_num_nodes + partition_row - 1) // partition_row * partition_row
        )
        padded_cols_x = (
            (original_num_nodes + partition_col_x - 1)
            // partition_col_x
            * partition_col_x
        )

        # Calculate padded dimensions for x
        padded_x_rows = (
            padded_cols_x  # Align with the padded columns of the adjacency matrix
        )

        if partition_x_col == 1:
            padded_x_cols = 128
        else:
            padded_x_cols = (
                (original_x_cols + partition_x_col - 1)
                // partition_x_col
                * partition_x_col
            )

        # Calculate partition sizes for padded dimensions
        row_partition_size = padded_rows // partition_row
        col_partition_x_size = padded_cols_x // partition_col_x
        x_col_partition_size = padded_x_cols // partition_x_col

        # Extract the first partition of padded_adj_indices
        start_row = 0
        end_row = row_partition_size
        start_col = 0
        end_col = col_partition_x_size

        relevant_edges_mask = (
            (edge_index[0] >= start_row)
            & (edge_index[0] < end_row)
            & (edge_index[1] >= start_col)
            & (edge_index[1] < end_col)
        )
        partitioned_edge_index = edge_index[:, relevant_edges_mask]

        # Adjust the indices in the partitioned_edge_index to be relative to the partition
        partitioned_edge_index[0] = partitioned_edge_index[0] - start_row
        partitioned_edge_index[1] = partitioned_edge_index[1] - start_col

        # Create the sparse adjacency matrix from the partitioned edge_index
        partitioned_adj_t = torch.sparse_coo_tensor(
            partitioned_edge_index,
            data.edge_weight[relevant_edges_mask],
            (row_partition_size, col_partition_x_size),
        ).to_sparse_csr()

        padded_x = torch.zeros((padded_x_rows, padded_x_cols), dtype=x.dtype)
        padded_x[:original_num_nodes, :original_x_cols] = x

        # Extract the first partition of padded_x
        x_start_row = 0
        x_end_row = col_partition_x_size
        x_start_col = 0
        x_end_col = x_col_partition_size
        partitioned_x = padded_x[x_start_row:x_end_row, x_start_col:x_end_col]

        print("Workload: " + str(partitioned_adj_t._nnz() * partitioned_x.shape[1]))

        # Move tensors to CUDA if available
        if torch.cuda.is_available():
            partitioned_adj_t = partitioned_adj_t.cuda()
            partitioned_x = partitioned_x.cuda()

        # Perform sparse matrix multiplication and measure time
        times = []
        for i in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()
            result = torch.sparse.mm(partitioned_adj_t, partitioned_x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if i >= warmup_iterations:
                    times.append(time.time() - start_time)
            else:
                if i >= warmup_iterations:
                    start_time = time.time()
                    result = torch.sparse.mm(partitioned_adj_t, partitioned_x)
                    end_time = time.time()
                    times.append(end_time - start_time)

        if times:
            average_time = sum(times) / len(times)
            print(
                f"Average time per sparse matrix multiplication (after {warmup_iterations} warmup iterations) with padding: {average_time:.6f} seconds"
            )
        else:
            print("Not enough iterations to measure time after warmup.")

    except FileNotFoundError:
        print(f"Error: File not found at {pt_file}")
    except KeyError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiply partitioned sparse and dense tensors with padding."
    )
    parser.add_argument(
        "partition_row",
        type=int,
        help="Number of partitions for the first dimension of edge_index.",
    )
    parser.add_argument(
        "partition_col_x",
        type=int,
        help="Number of partitions for the second dimension of edge_index and the first dimension of x.",
    )
    parser.add_argument(
        "partition_x_col",
        type=int,
        help="Number of partitions for the second dimension of x.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Total number of multiplication iterations.",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations."
    )

    args = parser.parse_args()

    multiply_partitioned_matrices_padded(
        args.partition_row,
        args.partition_col_x,
        args.partition_x_col,
        args.iterations,
        args.warmup,
    )
