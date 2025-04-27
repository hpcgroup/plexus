import torch
import random
import numpy as np
from axonn import axonn as ax
import torch.distributed as dist


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
        self.level_color_map = [color.PURPLE, color.GREEN, color.CYAN, color.RED]

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
