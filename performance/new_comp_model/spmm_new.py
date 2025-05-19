# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import gc
import time
import math
import torch
import random
import numpy as np
import torch.sparse
from tqdm import tqdm


def generate_sparse_matrix(N, sparsity):
    """
    Generates a random sparse matrix (N x N) with a specified sparsity level.

    Args:
        N (int): Number of rows and columns.
        sparsity (float): Sparsity level (0.0 to 1.0), where 1.0 is fully sparse.
                          Density is 1.0 - sparsity.

    Returns:
        torch.Tensor: A sparse COO tensor.
    """

    # Calculate number of non-zeros
    nnz = int(N * N * (1.0 - sparsity))

    print(
        f"Generating sparse matrix {N}x{N} with sparsity {sparsity} ({nnz} non-zeros)"
    )

    # Generate random indices for non-zero elements
    row_indices = torch.randint(0, N, (nnz,), dtype=torch.long).to(torch.device("cuda"))
    col_indices = torch.randint(0, N, (nnz,), dtype=torch.long).to(torch.device("cuda"))
    indices = torch.stack([row_indices, col_indices], dim=0)

    # Generate random values for non-zero elements
    values = torch.randn(nnz, dtype=torch.float32).to(torch.device("cuda"))

    # Create sparse tensor in COO format
    sparse_coo = torch.sparse_coo_tensor(indices, values, (N, N)).to(torch.device("cuda"))

    return sparse_coo.coalesce()


def generate_dense_matrix(N, D):
    """
    Generates a random dense matrix (N x D).

    Args:
        N (int): Number of rows.
        D (int): Number of columns.

    Returns:
        torch.Tensor: A dense tensor.
    """
    print(f"Generating dense matrix {N}x{D}")
    return torch.randn(N, D, dtype=torch.float32).to(torch.device("cuda"))


def benchmark_sharded_spmm(
    sparse_matrix_full,
    dense_matrix_full,
    num_shards_N0,
    num_shards_N1,
    num_shards_D,
    iterations,
    warmup_iterations,
):
    """
    Benchmarks the sparse matrix multiplication of the first shard
    of the input matrices based on the given sharding configuration.

    Args:
        sparse_matrix_full (torch.Tensor): The full sparse matrix (M x N) in COO format.
        dense_matrix_full (torch.Tensor): The full dense matrix (N x K).
        num_shards_N0 (int): Number of shards along the N0 dimension (sparse rows).
        num_shards_N1 (int): Number of shards along the N1 dimension (sparse columns, dense rows).
        num_shards_D (int): Number of shards along the D dimension (dense columns).
        iterations (int): The number of iterations to perform and time.
        warmup_iterations (int): The number of warmup iterations.

    Returns:
        tuple: (nnz_shard, N0_shard, N1_shard, D_shard, average_time)
    """

    N0, N1 = sparse_matrix_full.shape
    _, D = dense_matrix_full.shape

    # Calculate padded dimensions to ensure even sharding
    padded_N0 = math.ceil(N0 / num_shards_N0) * num_shards_N0
    padded_N1 = math.ceil(N1 / num_shards_N1) * num_shards_N1
    padded_D = math.ceil(D / num_shards_D) * num_shards_D

    # Calculate shard dimensions (size of each shard)
    N0_shard = padded_N0 // num_shards_N0
    N1_shard = padded_N1 // num_shards_N1
    D_shard = padded_D // num_shards_D

    # Define slice ranges for the first shard
    sparse_row_start, sparse_row_end = 0, N0_shard
    sparse_col_start, sparse_col_end = 0, N1_shard
    dense_row_start, dense_row_end = 0, N1_shard
    dense_col_start, dense_col_end = 0, D_shard

    full_indices = sparse_matrix_full.indices()
    full_values = sparse_matrix_full.values()

    # Mask for indices within the first shard's bounds
    relevant_edges_mask = (
        (full_indices[0] >= sparse_row_start)
        & (full_indices[0] < sparse_row_end)
        & (full_indices[1] >= sparse_col_start)
        & (full_indices[1] < sparse_col_end)
    )

    sharded_indices = full_indices[:, relevant_edges_mask].clone()
    sharded_values = full_values[relevant_edges_mask]

    # Adjust indices to be relative to the shard
    sharded_indices[0] -= sparse_row_start
    sharded_indices[1] -= sparse_col_start

    # Create the sparse shard in COO format and convert to CSR
    sharded_sparse = torch.sparse_coo_tensor(
        sharded_indices, sharded_values, (N0_shard, N1_shard)
    ).to_sparse_csr()

    sharded_sparse = sharded_sparse.to('cuda')

    # Experiment (using theoretical NNZ values instead)
    nnz_shard = sparse_matrix_full.values().shape[0] // (num_shards_N0 * num_shards_N1)

    # nnz_shard = sharded_sparse.values().shape[0]

    # --- Extract the first shard of the dense matrix ---
    sharded_dense = dense_matrix_full[
        dense_row_start:dense_row_end, dense_col_start:dense_col_end
    ].contiguous()

    sharded_dense = sharded_dense.to('cuda')

    # --- Perform sparse matrix multiplication and measure time ---

    # Warmup iterations
    for _ in range(warmup_iterations):
        _ = torch.sparse.mm(sharded_sparse, sharded_dense)

    # Timed iterations
    torch.cuda.synchronize()

    start_time = time.time()

    for _ in range(iterations):
        _ = torch.sparse.mm(sharded_sparse, sharded_dense)

    torch.cuda.synchronize()

    end_time = time.time()

    average_time = (end_time - start_time) / iterations

    # Return metrics for this shard multiplication (time is converted to ms)
    return (nnz_shard, N0_shard, N1_shard, D_shard, average_time * 1000)


def split_into_three_powers_of_two(G):
    log_G = int(math.log2(G))
    splits = []

    # Iterate over all possible values for a and b
    for a in range(log_G + 1):
        for b in range(log_G + 1 - a):
            c = log_G - a - b
            if c >= 0:  # Ensure c is non-negative
                splits.append((2**a, 2**b, 2**c))

    return splits


if __name__ == "__main__":
    # Matrix dimensions for the full matrices A (NxN) and B (NxD)
    FULL_N = [196608, 196608, 393216, 5242880, 5767168]
    FULL_D = 512

    SPARSITY_LEVELS = [0.99, 0.995, 0.9978838555, 0.99999879323, 0.99999993865]

    # List of sharding configurations
    ALL_CONFIGS = []
    for G in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        ALL_CONFIGS.extend(split_into_three_powers_of_two(G))

    # Select a subset of configs to test
    CONFIGS_TO_TEST = random.sample(ALL_CONFIGS, len(ALL_CONFIGS))

    NUM_ITERATIONS = 20
    WARMUP_ITERATIONS = 5
    
    # List to store results
    all_results = []

    i = 0

    # --- Benchmarking Loop ---
    for sparsity in SPARSITY_LEVELS:
        print(f"\n--- Benchmarking for Sparsity: {sparsity} ---")

        # Generate the full sparse matrix for this sparsity level
        sparse_matrix_full = generate_sparse_matrix(FULL_N[i], sparsity)

        # Generate the full dense matrix
        dense_matrix_full = generate_dense_matrix(FULL_N[i], FULL_D)

        for config in tqdm(CONFIGS_TO_TEST):
            num_shards_N0, num_shards_N1, num_shards_D = config        
    
            nnz_shard, N0_shard, N1_shard, D_shard, avg_time = benchmark_sharded_spmm(
                sparse_matrix_full,
                dense_matrix_full,
                num_shards_N0,
                num_shards_N1,
                num_shards_D,
                NUM_ITERATIONS,
                WARMUP_ITERATIONS,
            )

            # Store the results
            all_results.append((nnz_shard, N0_shard, N1_shard, D_shard, avg_time))
       
        gc.collect()

        i += 1

    np.save('spmm_benchmarking_exp', all_results)
