# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import time
import torch
from functools import partial
from plexus import plexus as plx
from axonn import axonn as ax


# helper function to time a GPU operation
def time_gpu(op, warmup=5, trials=20):
    for _ in range(warmup):
        op()

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(trials):
        op()

    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / trials


# matmul wrapper function that takes in the layout change flags and transpose order
def matmul(A, B, change_layout, transpose=False):
    if transpose:
        temp_A = A
        A = B.t()
        B = temp_A.t()

    if change_layout[0]:
        A = A.t().contiguous().t()
    else:
        A = A.contiguous()

    if change_layout[1]:
        B = B.t().contiguous().t()
    else:
        B = B.contiguous()

    if plx.bf16_gemm:
        C = torch.mm(A.to(torch.bfloat16), B.to(torch.bfloat16)).to(torch.float32)
    else:
        C = torch.mm(A, B)

    if transpose:
        C = C.t().contiguous()

    return C


# configurations to test when tuning the matrix multiplication
transpose_orders = [
    False,
    False,
    False,
    False,
    True,
    True,
    True,
    True,
]
change_layouts = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]

# cached best configurations for tuned matrix multiplications
matmul_to_index = dict()


def tuned_matmul(A, B, matmul_name):
    """
    Tune the matrix multiplication operation for the given matrices A and B.
    Then use the best performing configuration to perform the multiplication.

    Parameters:
    A: The first matrix.
    B: The second matrix.
    transpose (bool): Whether to transpose the matrices and the result.

    Returns:
    the result of the matrix multiplication
    """

    if not plx.tune_gemm:
        if plx.bf16_gemm:
            orig_dtype = A.dtype
            ax.get_timers().start("convert to bf16")
            A = A.to(torch.bfloat16)
            B = B.to(torch.bfloat16)
            ax.get_timers().stop("convert to bf16")
            ax.get_timers().start("gemm bf16")
            result = torch.mm(A, B)
            ax.get_timers().stop("gemm bf16")
            ax.get_timers().start("convert to fp32")
            result = result.to(orig_dtype)
            ax.get_timers().stop("convert to fp32")
            return result
        return torch.mm(A, B)

    if matmul_name in matmul_to_index:
        best_index = matmul_to_index[matmul_name]
        return matmul(
            A,
            B,
            change_layouts[best_index],
            transpose_orders[best_index],
        )

    # minimum time taken and index of the best mode
    min_time, min_idx = None, None
    for i in range(len(transpose_orders)):
        transpose = transpose_orders[i]
        change_layout = change_layouts[i]

        # partial function to pass the layout change flags and transpose order
        op = partial(matmul, A, B, change_layout, transpose)

        # time the operation
        curr_time = time_gpu(op)

        if min_time is None or curr_time < min_time:
            min_time = curr_time
            min_idx = i

    matmul_to_index[matmul_name] = min_idx

    return matmul(A, B, change_layouts[min_idx], transpose_orders[min_idx])
