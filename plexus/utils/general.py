import torch
import random
import numpy as np
from axonn import axonn as ax
import torch.distributed as dist
import time
import os
import csv
import json
from collections import defaultdict


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


def build_pccl_process_groups(axonn_group_letter):
    """
    Build a PCCL ProcessGroups from an AxoNN TP dimension (x, y, or z).

    Decomposes each AxoNN sub-group into a 2D grid:
      - inner groups: ranks on the same physical node (NCCL)
      - outer groups: ranks across nodes, one per node (MPI)

    This is needed because AxoNN's 3D rank layout may produce sub-groups
    whose intra-node members are non-contiguous, which PCCL's default
    ProcessGroups constructor cannot handle.

    IMPORTANT: This function performs collective operations (dist.new_group,
    MPI.COMM_WORLD.Split) so ALL ranks must call it, not just those in a
    particular sub-group.

    Args:
        axonn_group_letter: "x", "y", or "z" — the AxoNN TP dimension.

    Returns:
        A pccl.ProcessGroups instance for the caller's sub-group, or None
        if the caller's sub-group is entirely intra-node (no PCCL benefit).
        Note: even when returning None, the collective operations have still
        been performed for the benefit of other ranks whose sub-groups ARE
        cross-node.
    """
    from pccl import ProcessGroups

    gpus_per_node = ax.comm_handle.gpus_per_node
    world_rank = dist.get_rank()
    G_intra = ax.comm_handle.G_intra
    G_inter = ax.comm_handle.G_inter
    G_data = ax.comm_handle.G_data
    G_intra_r = ax.comm_handle.G_intra_r
    G_intra_c = ax.comm_handle.G_intra_c
    G_intra_d = ax.comm_handle.G_intra_d

    all_tp_ranks_3d = np.arange(G_intra).reshape(G_intra_d, G_intra_r, G_intra_c)

    dim = axonn_group_letter.lower()
    if dim == "x":
        axis = 1
    elif dim == "y":
        axis = 2
    elif dim == "z":
        axis = 0
    else:
        raise ValueError(f"axonn_group_letter must be x, y, or z, got '{axonn_group_letter}'")

    base_subgroups = []
    for idx in np.ndindex(*[s for i, s in enumerate(all_tp_ranks_3d.shape) if i != axis]):
        sl = list(idx)
        sl.insert(axis, slice(None))
        base_subgroups.append(sorted(all_tp_ranks_3d[tuple(sl)].tolist()))

    all_subgroups = list(base_subgroups)
    for data_idx in range(G_data):
        for inter_idx in range(G_inter):
            offset = data_idx * G_inter * G_intra + inter_idx * G_intra
            if data_idx == 0 and inter_idx == 0:
                continue
            for sg in base_subgroups:
                all_subgroups.append(sorted([r + offset for r in sg]))

    all_inner_lists = []
    all_outer_lists = []
    my_group_cross_node = False

    for sg in all_subgroups:
        node_buckets = defaultdict(list)
        for r in sg:
            node_buckets[r // gpus_per_node].append(r)

        sorted_nodes = sorted(node_buckets)
        ranks_per_node = len(node_buckets[sorted_nodes[0]])

        inner = [sorted(node_buckets[n]) for n in sorted_nodes]
        outer = [[sorted(node_buckets[n])[pos] for n in sorted_nodes] for pos in range(ranks_per_node)]

        all_inner_lists.extend(inner)
        all_outer_lists.extend(outer)

        if len(node_buckets) > 1 and world_rank in sg:
            my_group_cross_node = True

    any_cross_node = any(len(set(r // gpus_per_node for r in sg)) > 1 for sg in all_subgroups)

    if not any_cross_node:
        return None

    pg = ProcessGroups.from_rank_lists(
        inner_rank_lists=all_inner_lists,
        outer_rank_lists=all_outer_lists,
        inner_group_backend="nccl",
        outer_group_backend="mpi",
    )

    if not my_group_cross_node:
        return None

    return pg


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

# def _log_collective_message_size(op_type, tensor, step_tag, process_group):
#     """
#     Log collective message size (in MB) and sparsity to a CSV using rank 0.
# 
#     Args:
#         op_type: "all_reduce" or "reduce_scatter"
#         tensor: torch.Tensor participating in the collective
#         step_tag: short label for where in the pipeline this occurs (e.g., "AGG")
#         process_group: torch.distributed group for the collective
#     """
# 
#     try:
#         is_dist = dist.is_initialized()
#         world_rank = dist.get_rank() if is_dist else 0
# 
#         if is_dist:
#             try:
#                 group_rank = (
#                     dist.get_rank(process_group)
#                     if process_group is not None
#                     else dist.get_rank()
#                 )
#                 group_size = (
#                     dist.get_world_size(process_group)
#                     if process_group is not None
#                     else dist.get_world_size()
#                 )
#             except Exception:
#                 group_rank = dist.get_rank()
#                 group_size = dist.get_world_size()
#         else:
#             group_rank = 0
#             group_size = 1
# 
#         with torch.no_grad():
#             tensor_view = tensor.detach()
#             num_bytes = tensor_view.numel() * tensor_view.element_size()
#             size_mb = num_bytes / (1024.0 * 1024.0)
# 
#             if tensor_view.is_sparse:
#                 dense_elements = 1
#                 for dim in tensor_view.shape:
#                     dense_elements *= dim
#                 non_zero_elements = tensor_view._nnz()
#                 total_elements = dense_elements
# 
#                 if tensor_view.layout == torch.sparse_csr:
#                     crow = tensor_view.crow_indices()
#                     zero_rows = torch.sum((crow[1:] - crow[:-1]) == 0).item()
#                     total_rows = tensor_view.size(0)
#                 else:
#                     total_rows = tensor_view.size(0)
#                     row_indices = tensor_view.indices()[0]
#                     zero_rows = int(total_rows - torch.unique(row_indices).numel())
#             else:
#                 total_elements = tensor_view.numel()
#                 non_zero_elements = torch.count_nonzero(tensor_view).item()
# 
#                 if tensor_view.dim() == 0:
#                     total_rows = 1
#                     zero_rows = int(non_zero_elements == 0)
#                 elif tensor_view.dim() == 1:
#                     total_rows = tensor_view.size(0)
#                     zero_rows = int(non_zero_elements == 0)
#                 else:
#                     total_rows = tensor_view.size(0)
#                     reshaped = tensor_view.reshape(total_rows, -1)
#                     non_zero_per_row = torch.sum(reshaped != 0, dim=1)
#                     zero_rows = torch.sum(non_zero_per_row == 0).item()
# 
#             sparsity = (
#                 (total_elements - non_zero_elements) / total_elements
#                 if total_elements > 0
#                 else 0.0
#             )
# 
#             zero_row_ratio = (zero_rows / total_rows) if total_rows > 0 else 0.0
# 
#             stats_tensor = torch.tensor(
#                 [sparsity, zero_row_ratio],
#                 dtype=torch.float64,
#                 device=tensor_view.device,
#             )
# 
#             if is_dist and group_size > 1:
#                 gathered_stats = [
#                     torch.zeros_like(stats_tensor) for _ in range(group_size)
#                 ]
#                 dist.all_gather(gathered_stats, stats_tensor, group=process_group)
# 
#                 rank_tensor = torch.tensor(
#                     [world_rank],
#                     dtype=torch.int64,
#                     device=tensor_view.device,
#                 )
#                 gathered_rank_tensors = [
#                     torch.zeros_like(rank_tensor) for _ in range(group_size)
#                 ]
#                 dist.all_gather(
#                     gathered_rank_tensors, rank_tensor, group=process_group
#                 )
#             else:
#                 gathered_stats = [stats_tensor]
#                 gathered_rank_tensors = [
#                     torch.tensor(
#                         [world_rank],
#                         dtype=torch.int64,
#                         device=tensor_view.device,
#                     )
#                 ]
# 
#         sparsity_values = [stat.cpu().tolist()[0] for stat in gathered_stats]
#         zero_row_ratio_values = [stat.cpu().tolist()[1] for stat in gathered_stats]
#         group_members = [int(rank.cpu().item()) for rank in gathered_rank_tensors]
# 
#         payload = None
#         if group_rank == 0:
#             payload = {
#                 "group_members": sorted(group_members),
#                 "sparsity": sparsity_values,
#                 "zero_row_ratio": zero_row_ratio_values,
#                 "size_mb": size_mb,
#             }
# 
#         if is_dist:
#             world_size = dist.get_world_size()
#             gather_list = [None] * world_size
#             dist.all_gather_object(gather_list, payload)
# 
#             if world_rank != 0:
#                 return
# 
#             payloads = [item for item in gather_list if item is not None]
#         else:
#             payloads = [payload] if payload is not None else []
# 
#         if not payloads:
#             return
# 
#         payloads.sort(key=lambda p: min(p["group_members"]))
# 
#         timestamp = time.time()
#         log_path = os.environ.get("PLEXUS_COMM_LOG", "comm_log.csv")
#         file_exists = os.path.exists(log_path)
#         group_count = len(payloads)
#         group_column_name = f"group_{group_count}"
#         zero_row_column_name = f"{group_column_name}_zero_row_ratio"
#         sparsity_matrix = [group_payload["sparsity"] for group_payload in payloads]
#         zero_row_ratio_matrix = [
#             group_payload["zero_row_ratio"] for group_payload in payloads
#         ]
# 
#         with open(log_path, mode="a", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             if not file_exists:
#                 header = [
#                     "timestamp",
#                     "op",
#                     "step",
#                     "size_mb",
#                     group_column_name,
#                     zero_row_column_name,
#                 ]
#                 writer.writerow(header)
# 
#             row = [
#                 f"{timestamp:.6f}",
#                 op_type,
#                 step_tag,
#                 f"{payloads[0]['size_mb']:.6f}",
#                 json.dumps(sparsity_matrix),
#                 json.dumps(zero_row_ratio_matrix),
#             ]
# 
#             writer.writerow(row)
#     except Exception:
#         # Best-effort logging; never break training due to logging failures
#         pass
