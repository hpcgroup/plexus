import argparse

# Dataset information sourced from user-provided images and tables in the paper
DATASET_STATS = {
    "reddit": {"nodes": 232965, "features": 602, "classes": 41},
    "ogbn-products": {"nodes": 2449029, "features": 100, "classes": 47},
    "isolate-3-8m": {"nodes": 8745542, "features": 128, "classes": 32},
    "products-14m": {"nodes": 14249639, "features": 128, "classes": 32},
    "europe_osm": {"nodes": 50912018, "features": 128, "classes": 32},
    "ogbn-papers100m": {"nodes": 111059956, "features": 100, "classes": 172},
}

def pad_dimension(dim_value, num_gpus, second_num_gpus=1):
    """
    Round up (pad) the dimension based on the number of GPUs so it is divisible.
    This logic is consistent with the implementation in plexus/utils/general.py.
    """
    prod_num_gpus = num_gpus * second_num_gpus
    remainder = dim_value % prod_num_gpus
    if remainder == 0:
        return dim_value
    return dim_value + prod_num_gpus - remainder

def calculate_message_sizes(gx, gy, gz, dataset_name, hidden_dim=128):
    """
    Compute All-reduce and Reduce-scatter message sizes for a 3-layer GCN in Plexus training.
    """
    if dataset_name.lower() not in DATASET_STATS:
        print(f"Error: Dataset '{dataset_name}' not found.")
        print(f"Available datasets: {', '.join(DATASET_STATS.keys())}")
        return

    stats = DATASET_STATS[dataset_name.lower()]
    N = stats["nodes"]
    D_features = stats["features"]
    D_classes = stats["classes"]
    
    sizeof_float = 4  # FP32

    print("-" * 60)
    print(f"Dataset: {dataset_name.capitalize()}")
    print(f"GPU config: Gx={gx}, Gy={gy}, Gz={gz}")
    print(f"Model config: 3-layer GCN, hidden_dim={hidden_dim}, FP32 training")
    print("-" * 60)

    layer_dims = [
        (D_features, hidden_dim),
        (hidden_dim, hidden_dim),
        (hidden_dim, D_classes)
    ]

    gpu_rotations = [
        (gx, gy, gz), (gz, gx, gy), (gy, gz, gx)
    ]

    forward_pass_outputs = []
    backward_pass_outputs = []

    for i, (d_in, d_out) in enumerate(layer_dims):
        gx_i, gy_i, gz_i = gpu_rotations[i]

        # Padding
        if i == 0:
            d_in_padded = pad_dimension(d_in, gy_i, gz_i)
        else:
            d_in_padded = pad_dimension(d_in, gy_i)
        
        n_padded_z = pad_dimension(N, gz_i)
        d_out_padded = pad_dimension(d_out, gx_i)

        # --- Forward Pass Calculations ---
        elements_h = (n_padded_z / gz_i) * (d_in_padded / gy_i)
        size_h_mb = (elements_h * sizeof_float) / (1024 * 1024)
        
        elements_q = (n_padded_z / gz_i) * (d_out_padded / gx_i)
        size_q_mb = (elements_q * sizeof_float) / (1024 * 1024)
        
        forward_pass_outputs.append(
            f"Layer {i} (Gx={gx_i}, Gy={gy_i}, Gz={gz_i}):\n"
            f"  [All-reduce] Aggregation (H): {size_h_mb:.2f} MB\n"
            f"  [All-reduce] Combination (Q): {size_q_mb:.2f} MB"
        )

        # --- Backward Pass Calculations ---
        backward_msg = (
            f"Layer {i} (Gx={gx_i}, Gy={gy_i}, Gz={gz_i}):\n"
            f"  [All-reduce] Aggregation Gradient (grad_agg): {size_h_mb:.2f} MB"
        )

        gather_weights = (i == 0) or ((pad_dimension(d_in, gy_i) // gy_i) % gz_i == 0)
        if gather_weights and gz_i > 1:
            elements_grad_w = (d_in_padded / gy_i) * (d_out_padded / gx_i)
            size_grad_w_mb = (elements_grad_w * sizeof_float) / (1024 * 1024)
            backward_msg += f"\n  [Reduce-scatter] Weight Gradient (grad_W): {size_grad_w_mb:.2f} MB"

        if i == 0 and gz_i > 1:
            elements_grad_x = (n_padded_z / gz_i) * (d_in_padded / gy_i)
            size_grad_x_mb = (elements_grad_x * sizeof_float) / (1024 * 1024)
            backward_msg += f"\n  [Reduce-scatter] Input Gradient (grad_X): {size_grad_x_mb:.2f} MB"
        
        backward_pass_outputs.append(backward_msg)

    # --- Print results in the new order ---
    print("\n--- Forward Pass Communication ---")
    for output in forward_pass_outputs:
        print(output)

    print("\n--- Backward Pass Communication (Actual Order) ---")
    # Print backward communication in reverse order
    for output in reversed(backward_pass_outputs):
        print(output)


def main():
    parser = argparse.ArgumentParser(
        description="Compute All-reduce and Reduce-scatter message sizes in Plexus GNN training.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("gx", type=int, help="Number of GPUs along Gx dimension")
    parser.add_argument("gy", type=int, help="Number of GPUs along Gy dimension")
    parser.add_argument("gz", type=int, help="Number of GPUs along Gz dimension")
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name.\nAvailable options: " + ", ".join(DATASET_STATS.keys())
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of the GCN model (default: 128)"
    )

    args = parser.parse_args()
    calculate_message_sizes(args.gx, args.gy, args.gz, args.dataset, args.hidden_dim)

if __name__ == "__main__":
    main()