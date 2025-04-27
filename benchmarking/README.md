# benchmarking

This directory contains files used for validating the parallel implementation and benchmarking key operations.

## Files

-   **pyg_serial.py**: This Python script provides a serial implementation of a GNN model using PyTorch Geometric (PyG). It is primarily used for validation purposes, allowing for comparison against the parallelized version. The script is configured to train a model with 3 Graph Convolutional Network (GCN) layers and a hidden dimension size of 128 on the ogbn-products dataset by default.

    The script offers several command-line arguments to customize the training process:
    -   `--download_path`: Specifies the path to the directory where the dataset is stored.
    -   `--num_epochs` (optional): Determines the number of training epochs (default is 2).
    -   `--seed` (optional): Allows setting a specific random seed for reproducible experiments.
    -   Other aspects like the number of GCN layers and the hidden dimension size can be modified by adjusting the model definition within the script or by altering the dataset loading within the `get_dataset` function.

    **Example Usage:**
    ```bash
    python pyg_serial.py --download_path <path/to/dataset> --num_epochs 10
    ```

-   **spmm.py**: This script is designed to test the performance of Sparse Matrix-Matrix Multiplication (SpMM), a fundamental operation in GNN computations. It provides flexibility in configuring the SpMM operation to analyze performance under various conditions.

    It accepts the following command-line arguments:
    -   `--pt_file`: Specifies the path to a `.pt` file. This file is expected to be the output of preprocessing a dataset using Plexus, containing a tuple `(data, num_classes)` where `data` is a processed PyG `Data` object. The dimensions of the sparse matrix and the dense feature matrix used in the SpMM benchmark are derived from this data.
    -   `--shard_row` (optional): Optionally specifies how to shard the row dimension (M) of the sparse matrix (A, sized M x K). This allows for investigating the impact of different row sharding strategies on SpMM performance (default is 1).
    -   `--shard_col` (optional): Optionally specifies how to shard the column dimension (K) of the sparse matrix (A, sized M x K), which corresponds to the row dimension of the dense features matrix (F, sized K x N). This allows for investigating the impact of different sharding strategies along the shared dimension on SpMM performance (default is 1).
    -   `--shard_col_x` (optional): Optionally specifies how to shard the column dimension (N) of the dense feature matrix (F, sized K x N). This allows for investigating the impact of different column sharding strategies on SpMM performance (default is 1).
    -   `--iterations` (optional): Sets the total number of SpMM iterations to run for the benchmark (default is 25). 
    -   `--warmup` (optional): Specifies the number of initial iterations to perform as a warmup. The timing results of these warmup iterations will be ignored to get more stable performance measurements (default is 5)).
    - Note that for the arguments related to sharding the matrices, the matrices are padded so their sizes are divisible by these arguments.

    **Example Usage:**
    ```bash
    python spmm.py --pt_file <path/to/data/processed_data.pt>
    ```
