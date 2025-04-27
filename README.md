# Plexus

Plexus is a 3D parallel framework designed for large-scale distributed GNN training.

## Directory Structure

-   **benchmarking**: Contains a serial implementation using PyTorch Geometric (PyG) for validation and testing. Additionally, it includes utilities for benchmarking Sparse Matrix-Matrix Multiplication (SpMM) operations, a key component in GNN computations.
-   **examples**: Offers a practical demonstration of how to leverage Plexus to parallelize a GNN model. This directory includes example scripts for running the parallelized training, as well as utilities for parsing the resulting performance data.
-   **performance**: Houses files dedicated to modeling the performance characteristics of parallel GNN training. This includes models for communication overhead, computation costs (specifically SpMM), and memory utilization.
-   **plexus**: Contains the core logic of the Plexus framework. This includes the parallel implementation of a Graph Convolutional Network (GCN) layer, along with utility functions for dataset preprocessing, efficient data loading, and other essential components for distributed GNN training.
