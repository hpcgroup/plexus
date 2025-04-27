import time
import torch
import argparse
import torch.sparse


def multiply_sharded_matrices_padded(
    pt_file
    shard_row,
    shard_col,
    shard_x_col,
    iterations,
    warmup_iterations,
):
    """
    Reads a .pt file, pads and shards the edge_index and x matrices,
    multiplies the shards using sparse matrix multiplication, and measures the time.

    Args:
        pt_file (str): Path to the .pt file containing the data.
        shard_row (int): The number of shards for the first dimension of edge_index.
        shard_col (int): The number of shards for the second dimension of edge_index
                             and the first dimension of x.
        shard_x_col (int): The number of shards for the second dimension of x.
        iterations (int): The total number of multiplication iterations to perform.
        warmup_iterations (int): The number of initial iterations to exclude from timing.
    """

    try:
        data, _ = torch.load(pt_file, weights_only=False)

        edge_index = data.edge_index
        x = data.x

        original_num_nodes = x.shape[0]
        original_x_cols = x.shape[1]

        # Calculate padded dimensions for edge_index (implied adjacency matrix)
        padded_rows = (
            (original_num_nodes + shard_row - 1) // shard_row * shard_row
        )
        padded_cols_x = (
            (original_num_nodes + shard_col - 1)
            // shard_col
            * shard_col
        )

        # Calculate padded dimensions for x
        padded_x_rows = (
            padded_cols_x  # Align with the padded columns of the adjacency matrix
        )

        if shard_x_col == 1:
            padded_x_cols = original_x_cols
        else:
            padded_x_cols = (
                (original_x_cols + shard_x_col - 1)
                // shard_x_col
                * shard_x_col
            )

        # Calculate shard sizes for padded dimensions
        row_shard_size = padded_rows // shard_row
        col_shard_x_size = padded_cols_x // shard_col
        x_col_shard_size = padded_x_cols // shard_x_col

        # Extract the first shard of padded_adj_indices
        start_row = 0
        end_row = row_shard_size
        start_col = 0
        end_col = col_shard_x_size

        relevant_edges_mask = (
            (edge_index[0] >= start_row)
            & (edge_index[0] < end_row)
            & (edge_index[1] >= start_col)
            & (edge_index[1] < end_col)
        )
        sharded_edge_index = edge_index[:, relevant_edges_mask]

        # Adjust the indices in the sharded_edge_index to be relative to the shard
        sharded_edge_index[0] = sharded_edge_index[0] - start_row
        sharded_edge_index[1] = sharded_edge_index[1] - start_col

        # Create the sparse adjacency matrix from the sharded edge_index
        sharded_adj_t = torch.sparse_coo_tensor(
            sharded_edge_index,
            data.edge_weight[relevant_edges_mask],
            (row_shard_size, col_shard_x_size),
        ).to_sparse_csr()

        padded_x = torch.zeros((padded_x_rows, padded_x_cols), dtype=x.dtype)
        padded_x[:original_num_nodes, :original_x_cols] = x

        # Extract the first shard of padded_x
        x_start_row = 0
        x_end_row = col_shard_x_size
        x_start_col = 0
        x_end_col = x_col_shard_size
        sharded_x = padded_x[x_start_row:x_end_row, x_start_col:x_end_col]

        print("Theoretical # of FLOPs (2 * NNZ * D): " + str(2 * sharded_adj_t._nnz() * sharded_x.shape[1]))

        # Move tensors to CUDA if available
        if torch.cuda.is_available():
            sharded_adj_t = sharded_adj_t.cuda()
            sharded_x = sharded_x.cuda()

        # Perform sparse matrix multiplication and measure time
        times = []
        for i in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()
            result = torch.sparse.mm(sharded_adj_t, sharded_x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if i >= warmup_iterations:
                    times.append(time.time() - start_time)
            else:
                if i >= warmup_iterations:
                    start_time = time.time()
                    result = torch.sparse.mm(sharded_adj_t, sharded_x)
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
        description="Multiply sharded sparse and dense tensors with padding."
    )
    parser.add_argument(
        "pt_file",
        type=int,
        help="Path to plexus processed .pt file containing the data"
    )
    parser.add_argument(
        "shard_row",
        type=int,
        default=1,
        help="Number of shards for the first dimension of edge_index.",
    )
    parser.add_argument(
        "shard_col",
        type=int,
        default=1,
        help="Number of shards for the second dimension of edge_index and the first dimension of x.",
    )
    parser.add_argument(
        "shard_x_col",
        type=int,
        default=1,
        help="Number of shards for the second dimension of x.",
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

    multiply_sharded_matrices_padded(
        args.pt_file,
        args.shard_row,
        args.shard_col,
        args.shard_x_col,
        args.iterations,
        args.warmup
    )
