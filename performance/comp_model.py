# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import math


def split_into_three_powers_of_two(G):
    if G <= 0 or (G & (G - 1)) != 0:
        raise ValueError("G must be a positive power of 2.")

    log_G = int(math.log2(G))
    splits = []

    # Iterate over all possible values for a and b
    for a in range(log_G + 1):
        for b in range(log_G + 1 - a):
            c = log_G - a - b
            if c >= 0:  # Ensure c is non-negative
                splits.append((2**a, 2**b, 2**c))

    return splits


# don't include number of classes in D_list
def comp_model(N, NNZ, G, D_list, coef=[1, 1, 1]):
    """
    Args:
        N - number of nodes in graph
        NNZ - number of nonzeros in graph's adjacency matrix
        G - number of GPUs
        D_list - list of features at each layer excluding number of classes (ex: 3 GCN layers with 128 hidden dim, 100 feature size, [100, 128, 128])
        coef - coefficients to multiply the three terms of the model by to get times in ms (default coefficients don't result in meaningful times, but give an ordering of the configs

    Returns:
        Estimated SpMM time (ms) for each 3D config
    """

    cost_dict = dict()
    for x, y, z in split_into_three_powers_of_two(G):
        flops_cost, fwd_penalty, bwd_penalty = 0, 0, 0
        for i in range(len(D_list)):
            flops_cost += (
                NNZ
                * D_list[i]
                / ([x, z, y][(i + 1) % 3] * [x, z, y][i % 3] * [x, z, y][(i + 2) % 3])
            )

            fwd_penalty += (N * [x, z, y][(i + 2) % 3]) / (D_list[i] * [x, z, y][i % 3])

            bwd_penalty += (N * [x, z, y][(i + 2) % 3]) / (
                D_list[i] * [x, z, y][(i + 1) % 3]
            )

        cost_dict[(x, y, z)] = (
            (coef[0] * (flops_cost**0.5))
            + (coef[1] * (flops_cost**0.5) * fwd_penalty)
            + (coef[2] * (flops_cost**0.5) * bwd_penalty)
        )

    cost_dict = dict(sorted(cost_dict.items(), key=lambda kv: kv[1]))

    return cost_dict
