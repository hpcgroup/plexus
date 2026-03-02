import torch
import random
import numpy as np
from axonn import axonn as ax
import torch.distributed as dist
import time
import os
import csv
import json


# each gpu has the same random seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# returns new padded dimension that can be divided
# by num_gpus and optionally also by second_num_gpus
# where second_num_gpus is usually used for the depth
# process group
def pad_dimension(dim_value, num_gpus, second_num_gpus=1):
    prod_num_gpus = num_gpus * second_num_gpus
    remainder = dim_value % prod_num_gpus
    return dim_value if remainder == 0 else dim_value + prod_num_gpus - remainder


# helper function that takes in a tuple of letters indicating
# which dimensions of the 3D grid to split across and returns relevant info
def get_process_groups_info(groups):
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert groups[i] != groups[j], "Please input different groups."

    num_gpus, ranks, process_groups = [], [], []
    for group in groups:
        if group.lower() == "x":
            num_gpus.append(ax.comm_handle.G_intra_r)
            ranks.append(ax.comm_handle.intra_layer_row_parallel_rank)
            process_groups.append(ax.comm_handle.outer_intra_layer_parallel_group)
        elif group.lower() == "y":
            num_gpus.append(ax.comm_handle.G_intra_c)
            ranks.append(ax.comm_handle.intra_layer_column_parallel_rank)
            process_groups.append(ax.comm_handle.inner_intra_layer_parallel_group)
        elif group.lower() == "z":
            num_gpus.append(ax.comm_handle.G_intra_d)
            ranks.append(ax.comm_handle.intra_layer_depth_parallel_rank)
            process_groups.append(ax.comm_handle.depth_intra_layer_parallel_group)
        else:
            raise ValueError("A group can only be x, y, or z.")
    return (num_gpus, ranks, process_groups)


def analyze_csr_tensor(sparse_tensor):
    """
    Analyzes a sparse CSR tensor.

    Args:
        sparse_tensor (torch.Tensor): A 2D sparse CSR tensor.

    Returns:
        tuple: Contains the following:
            - int: Number of nonzero values.
            - float: Frobenius norm of the matrix.
    """
    # Accessing CSR components
    values = sparse_tensor.values()

    # Number of nonzero values
    nnz = values.numel()

    # Frobenius norm
    frobenius_norm = torch.sqrt(torch.sum(values**2)).item()

    return nnz, frobenius_norm


class color:
    """
    courtesy - https://gist.github.com/nazwadi/ca00352cd0d20b640efd
    """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.level_color_map = [
            color.PURPLE,
            color.GREEN,
            color.CYAN,
            color.RED,
        ]

    def add_children(self, child):
        self.children.append(child)

    def __str__(self, level=0):
        this_color = self.level_color_map[level % len(self.level_color_map)]
        ret = (
            "\t" * level + "|--" + f"{this_color} {repr(self.value)} {color.END}" + "\n"
        )
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return f"\n{self.value}"


def print_axonn_timer_data(times):
    sorted_call_stacks = list(times.keys())
    sorted_call_stacks.sort(key=lambda x: len(x))
    head_nodes = []
    node_map = {}
    for call_stack in sorted_call_stacks:
        avg_time = torch.tensor(times[call_stack]).to(torch.device("cuda"))
        max_time = avg_time.clone().to(torch.device("cuda"))

        # Perform a reduce operation where root is 0
        dist.reduce(max_time, dst=0, op=dist.ReduceOp.MAX)
        dist.reduce(avg_time, dst=0, op=dist.ReduceOp.SUM)

        # Only on the root process (rank 0), compute the average and load imbalance
        if dist.get_rank() == 0:
            avg_time = avg_time / dist.get_world_size()
            load_imbalance = max_time / avg_time

            # Create the node with the desired formatted string
            node = Node(
                f"{call_stack[-1]} | Max Time: {max_time.item():.3f} ms | "
                f"Avg Time: {avg_time.item():.3f} ms | Load Imbalance: {load_imbalance.item():.3f}"
            )

            node_map[call_stack] = node
            if len(call_stack) > 1:
                parent_node = call_stack[:-1]
                assert parent_node in node_map
                node_map[parent_node].add_children(node)
            else:
                head_nodes.append(node)

    if dist.get_rank() == 0:
        for node in head_nodes:
            print(str(node))

def _log_collective_message_size(op_type, tensor, step_tag, process_group):
    """
    Log collective message size (in MB) and sparsity to a CSV using rank 0.

    Args:
        op_type: "all_reduce" or "reduce_scatter"
        tensor: torch.Tensor participating in the collective
        step_tag: short label for where in the pipeline this occurs (e.g., "AGG")
        process_group: torch.distributed group for the collective
    """

    try:
        is_dist = dist.is_initialized()
        world_rank = dist.get_rank() if is_dist else 0

        if is_dist:
            try:
                group_rank = (
                    dist.get_rank(process_group)
                    if process_group is not None
                    else dist.get_rank()
                )
                group_size = (
                    dist.get_world_size(process_group)
                    if process_group is not None
                    else dist.get_world_size()
                )
            except Exception:
                group_rank = dist.get_rank()
                group_size = dist.get_world_size()
        else:
            group_rank = 0
            group_size = 1

        with torch.no_grad():
            tensor_view = tensor.detach()
            num_bytes = tensor_view.numel() * tensor_view.element_size()
            size_mb = num_bytes / (1024.0 * 1024.0)

            if tensor_view.is_sparse:
                dense_elements = 1
                for dim in tensor_view.shape:
                    dense_elements *= dim
                non_zero_elements = tensor_view._nnz()
                total_elements = dense_elements

                if tensor_view.layout == torch.sparse_csr:
                    crow = tensor_view.crow_indices()
                    zero_rows = torch.sum((crow[1:] - crow[:-1]) == 0).item()
                    total_rows = tensor_view.size(0)
                else:
                    total_rows = tensor_view.size(0)
                    row_indices = tensor_view.indices()[0]
                    zero_rows = int(total_rows - torch.unique(row_indices).numel())
            else:
                total_elements = tensor_view.numel()
                non_zero_elements = torch.count_nonzero(tensor_view).item()

                if tensor_view.dim() == 0:
                    total_rows = 1
                    zero_rows = int(non_zero_elements == 0)
                elif tensor_view.dim() == 1:
                    total_rows = tensor_view.size(0)
                    zero_rows = int(non_zero_elements == 0)
                else:
                    total_rows = tensor_view.size(0)
                    reshaped = tensor_view.reshape(total_rows, -1)
                    non_zero_per_row = torch.sum(reshaped != 0, dim=1)
                    zero_rows = torch.sum(non_zero_per_row == 0).item()

            sparsity = (
                (total_elements - non_zero_elements) / total_elements
                if total_elements > 0
                else 0.0
            )

            zero_row_ratio = (zero_rows / total_rows) if total_rows > 0 else 0.0

            stats_tensor = torch.tensor(
                [sparsity, zero_row_ratio],
                dtype=torch.float64,
                device=tensor_view.device,
            )

            if is_dist and group_size > 1:
                gathered_stats = [
                    torch.zeros_like(stats_tensor) for _ in range(group_size)
                ]
                dist.all_gather(gathered_stats, stats_tensor, group=process_group)

                rank_tensor = torch.tensor(
                    [world_rank],
                    dtype=torch.int64,
                    device=tensor_view.device,
                )
                gathered_rank_tensors = [
                    torch.zeros_like(rank_tensor) for _ in range(group_size)
                ]
                dist.all_gather(
                    gathered_rank_tensors, rank_tensor, group=process_group
                )
            else:
                gathered_stats = [stats_tensor]
                gathered_rank_tensors = [
                    torch.tensor(
                        [world_rank],
                        dtype=torch.int64,
                        device=tensor_view.device,
                    )
                ]

        sparsity_values = [stat.cpu().tolist()[0] for stat in gathered_stats]
        zero_row_ratio_values = [stat.cpu().tolist()[1] for stat in gathered_stats]
        group_members = [int(rank.cpu().item()) for rank in gathered_rank_tensors]

        payload = None
        if group_rank == 0:
            payload = {
                "group_members": sorted(group_members),
                "sparsity": sparsity_values,
                "zero_row_ratio": zero_row_ratio_values,
                "size_mb": size_mb,
            }

        if is_dist:
            world_size = dist.get_world_size()
            gather_list = [None] * world_size
            dist.all_gather_object(gather_list, payload)

            if world_rank != 0:
                return

            payloads = [item for item in gather_list if item is not None]
        else:
            payloads = [payload] if payload is not None else []

        if not payloads:
            return

        payloads.sort(key=lambda p: min(p["group_members"]))

        timestamp = time.time()
        log_path = os.environ.get("PLEXUS_COMM_LOG", "comm_log.csv")
        file_exists = os.path.exists(log_path)
        group_count = len(payloads)
        group_column_name = f"group_{group_count}"
        zero_row_column_name = f"{group_column_name}_zero_row_ratio"
        sparsity_matrix = [group_payload["sparsity"] for group_payload in payloads]
        zero_row_ratio_matrix = [
            group_payload["zero_row_ratio"] for group_payload in payloads
        ]

        with open(log_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                header = [
                    "timestamp",
                    "op",
                    "step",
                    "size_mb",
                    group_column_name,
                    zero_row_column_name,
                ]
                writer.writerow(header)

            row = [
                f"{timestamp:.6f}",
                op_type,
                step_tag,
                f"{payloads[0]['size_mb']:.6f}",
                json.dumps(sparsity_matrix),
                json.dumps(zero_row_ratio_matrix),
            ]

            writer.writerow(row)
    except Exception:
        # Best-effort logging; never break training due to logging failures
        pass
