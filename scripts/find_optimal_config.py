import argparse
import sys
import os

# Ensure the script can find modules under the performance directory.
# This assumes the script is run from the plexus root directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from performance.comm_model import compute_config_costs
    from performance.comp_model import comp_model
except ImportError:
    print("Error: Failed to import performance models.")
    print("Please make sure this script is located in the 'plexus-AE' project root,")
    print("and that 'performance/comm_model.py' and 'performance/comp_model.py' exist.")
    sys.exit(1)


# Dataset statistics
DATASET_STATS = {
    "Reddit": {
        "nodes": 232965,
        "non_zeros": 114848857,
        "features": 602,
        "classes": 41,
    },
    "ogbn-products": {
        "nodes": 2449029,
        "non_zeros": 126167053,
        "features": 100,
        "classes": 47,
    },
    "Isolate-3-8M": {
        "nodes": 8745542,
        "non_zeros": 1317986044,
        "features": 128,
        "classes": 32,
    },
    "products-14M": {
        "nodes": 14249639,
        "non_zeros": 245036907,
        "features": 128,
        "classes": 32,
    },
    "europe_osm": {
        "nodes": 50912018,
        "non_zeros": 159021338,
        "features": 128,
        "classes": 32,
    },
    "ogbn-papers100M": {
        "nodes": 111059956,
        "non_zeros": 1726745828,
        "features": 100,
        "classes": 172,
    },
    "friendster": {
        "nodes": 65608366,
        "non_zeros": 1806067135,
        "features": 128,
        "classes": 32,
    },
    "mycielskian19": {
        "nodes": 393215,
        "non_zeros": 903194710,
        "features": 128,
        "classes": 32,
    },
    "nlpkkt200": {
        "nodes": 16240000,
        "non_zeros": 440225632,
        "features": 128,
        "classes": 32,
    },
    "nlpkkt160": {
        "nodes": 8345600,
        "non_zeros": 225422112,
        "features": 128,
        "classes": 32,
    },
    "ml_geer": {
        "nodes": 1504002,
        "non_zeros": 110686677,
        "features": 128,
        "classes": 32,
    },
    "hugebubbles-00000": {
        "nodes": 18318143,
        "non_zeros": 54940162,
        "features": 128,
        "classes": 32,
    },
    "ml_laplace": {
        "nodes": 377002,
        "non_zeros": 27582698,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale18-ef16": {
        "nodes": 262144,
        "non_zeros": 4194304,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale19-ef16": {
        "nodes": 524288,
        "non_zeros": 8388608,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale20-ef16": {
        "nodes": 1048576,
        "non_zeros": 16777216,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale21-ef16": {
        "nodes": 2097152,
        "non_zeros": 33554432,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale22-ef16": {
        "nodes": 4194304,
        "non_zeros": 67108864,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale23-ef16": {
        "nodes": 8388608,
        "non_zeros": 134217728,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale24-ef16": {
        "nodes": 16777216,
        "non_zeros": 268435456,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale25-ef16": {
        "nodes": 33554432,
        "non_zeros": 536870912,
        "features": 128,
        "classes": 32,
    },
    "igb-medium": {
        "nodes": 10000000,
        "non_zeros": 130077694,
        "features": 1024,
        "classes": 19,
    },
}

def is_power_of_two(n):
    """Check whether a number is a power of two."""
    return (n > 0) and (n & (n - 1) == 0)

def find_optimal_configuration(dataset_name, total_gpus, hidden_dim=128):
    """
    Compute and return the optimal 3D parallel configuration for a given
    dataset and GPU count.

    Args:
        dataset_name (str): Name of the dataset.
        total_gpus (int): Total number of GPUs for training.
        hidden_dim (int): Hidden dimension of the GNN model.
    """
    if dataset_name not in DATASET_STATS:
        print(f"Error: Unknown dataset '{dataset_name}'.")
        print(f"Available datasets: {', '.join(DATASET_STATS.keys())}")
        return

    if not is_power_of_two(total_gpus):
        print(f"Error: Total GPU count ({total_gpus}) must be a power of two.")
        return

    stats = DATASET_STATS[dataset_name]
    N = stats["nodes"]
    NNZ = stats["non_zeros"]
    num_features = stats["features"]
    num_classes = stats["classes"]

    print(f"Searching for optimal configuration for dataset '{dataset_name}' on {total_gpus} GPUs...")
    print(f"Params: N={N}, NNZ={NNZ}, Features={num_features}, Classes={num_classes}, HiddenDim={hidden_dim}")
    print("-" * 40)

    # Model architecture: input_linear + 3 GCN layers + output_linear
    # input_linear: num_features -> hidden_dim (pure GEMM, no SpMM)
    # GCN 0/1/2:   hidden_dim -> hidden_dim   (SpMM + GEMM)
    # output_linear: hidden_dim -> num_classes (pure GEMM, no SpMM)

    # D_list for computation model: only GCN layers with SpMM, excludes pure linear layers and classes.
    # input_linear already projects features to hidden_dim, so all 3 GCN layers do SpMM at hidden_dim.
    d_list_comp = [hidden_dim, hidden_dim, hidden_dim]
    # D_list for communication model: all layer feature-dimension transitions (5 transitions)
    d_list_comm = [num_features, hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_classes]

    # --- Run models ---
    # Use Perlmutter empirical bandwidth model
    comm_costs = compute_config_costs(total_gpus, N, d_list_comm, version="v3", machine="perlmutter")
    # Use default coefficients for computation model
    comp_costs = comp_model(N, NNZ, total_gpus, d_list_comp)

    # --- Combine results ---
    total_costs = {}
    for config, comm_time in comm_costs.items():
        if config in comp_costs:
            total_costs[config] = comm_time + comp_costs[config]

    if not total_costs:
        print("Failed to compute costs for any configuration. Please check the model implementation.")
        return

    # Sort by total time
    sorted_configs = sorted(total_costs.items(), key=lambda item: item[1])

    # --- Print results ---
    print("Estimated total cost (communication + computation) for each 3D configuration, sorted best to worst:")
    for config, total_time in sorted_configs:
        config_str = f"(X={config[0]}, Y={config[1]}, Z={config[2]})"
        print(f"  - Config {config_str:<20}: {total_time:.4f} ms")

    print("-" * 40)
    best_config, min_time = sorted_configs[0]
    best_config_str = f"(X={best_config[0]}, Y={best_config[1]}, Z={best_config[2]})"
    print(f"Best configuration: {best_config_str}, estimated total time: {min_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the optimal 3D parallel configuration for Plexus GNN training using performance models."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_STATS.keys()),
        help="Name of the dataset to use."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        required=True,
        help="Total number of GPUs for training (must be a power of two)."
    )

    args = parser.parse_args()

    find_optimal_configuration(args.dataset, args.gpus)
