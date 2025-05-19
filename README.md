# Plexus

Plexus is a 3D parallel framework for large-scale distributed GNN training.

## Dependencies

To use Plexus, you'll need the following dependencies:

* **Python 3.11.7:** It's recommended to use a virtual environment to manage your Python dependencies. You can create one using `venv`:

    ```bash
    python -m venv <your_env_name>
    source <your_env_name>/bin/activate
    ```

* **CUDA 12.4:** For running Plexus on GPUs, you'll need CUDA 12.4. On systems where modules are used to manage software, you can load it with the following command (this is present in the run script under the examples directory for Perlmutter):

    ```bash
    module load cudatoolkit/12.4
    ```

    For AMD Systems like Frontier, the latest available version of ROCm can be loaded.

* **Python Dependencies:** Once your virtual environment is set up, you can install the required Python packages using `pip` and the `requirements.txt` file provided in the repository:

    ```bash
    pip install -r requirements.txt
    ```

* **NCCL/RCCL:** When PyTorch is installed, it will be built with its own internal NCCL/RCCL.

* **AWS OFI NCCL/RCCL:** This is a plugin that allows NCCL/RCCL to use libfabric as a network provider and improve performance.
These are available at the following links - [AWS-OFI-NCCL](https://github.com/aws/aws-ofi-nccl), [AWS-OFI-RCCL](https://github.com/ROCm/aws-ofi-rccl).

    Instructions for Frontier can be found here - [Frontier AWS-OFI-RCCL Plugin](https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#aws-ofi-rccl-plugin). Instead of building RCCL, you can point to the lib folder of your installed PyTorch package.

    On Systems like Perlmutter, the plugin can be loaded with the following module:
    ```bash
    module load nccl
    ```

## Directory Structure

* **benchmarking**: Contains a serial implementation using PyTorch Geometric (PyG) for validation and testing. Additionally, it includes utilities for benchmarking Sparse Matrix-Matrix Multiplication (SpMM) operations, a key component in GNN computations.
* **examples**: Offers a practical demonstration of how to leverage Plexus to parallelize a GNN model. This directory includes example scripts for running the parallelized training, as well as utilities for parsing the resulting performance data.
* **performance**: Houses files dedicated to modeling the performance characteristics of parallel GNN training. This includes models for communication overhead, computation costs (specifically SpMM), and memory utilization.
* **plexus**: Contains the core logic of the Plexus framework. This includes the parallel implementation of a Graph Convolutional Network (GCN) layer, along with utility functions for dataset preprocessing, efficient data loading, and other essential components for distributed GNN training.

## PYTHONPATH

* Once you have cloned the repo and have all dependencies installed, you should also append the path to your clone of the Plexus repo to the PYTHONPATH environment variable.
