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


def compute_config_mem(G, N, E, D_list):
    """
    Args:
        G - number of GPUs
        N - number of nodes in graph
        E - number of nonzeros in graph's adjacency matrix
        D_list - list of features at each layer (ex: 3 GCN layers with 128 hidden dim, 100 feature size, 60 classes [100, 128, 128, 60])

    Returns:
        Approximate gpu memory usage (GB) for each 3D config
    """

    config_to_mem = {}
    for X, Y, Z in split_into_three_powers_of_two(G):
        # assuming E is roughly evenly distributed after permutation
        adj_mem, divide_list = 0, [(Z, X), (Y, Z), (X, Y)]
        for i in range(min(3, len(D_list))):
            # +1 for A.T, CSR format
            adj_mem += 2 * (
                (E / (divide_list[i][0] * divide_list[i][1]) * 3)
                + (N / divide_list[i][0] * 2)
            )

        # +1 for grad, +2 for optimizer states
        # weights are sharded across depth dimension
        weight_mem = 0
        for i in range(len(D_list) - 1):
            weight_mem += 4 * D_list[i] * D_list[i + 1] / G

        # +1 for grad
        agg_mem = 0
        divide_list = [(Z * Y), (Y * X), (X * Z)]
        for i in range(len(D_list) - 1):
            agg_mem += 2 * N * D_list[i] / divide_list[i % 3]

        # +1 for grad
        activation_mem = 0
        divide_list = [(X * Y), (Z * X), (Y * Z)]
        for i in range(1, len(D_list)):
            activation_mem += 2 * N * D_list[i] / divide_list[i % 3]

        # +1 for grad, +2 for optimizer states
        # input features are sharded across depth dimension
        activation_mem += 4 * N * D_list[0] / G

        # accounting for max one time mem of gathering
        # input features and weights for first layer
        one_time_mem = ((N * D_list[0]) / (X * Y)) + ((D_list[0] * D_list[1]) / (Y * X))

        # * 2 since labels are int64
        if len(D_list) % 3 == 2:
            divide_amt = Z
        elif len(D_list) % 3 == 0:
            divide_amt = Y
        else:
            divide_amt = X
        labels_mem = N * 2 / divide_amt

        # total_number of 32 bit elements
        tot_elts = (
            adj_mem + weight_mem + agg_mem + activation_mem + one_time_mem + labels_mem
        )

        config_to_mem[f"X{X}Y{Y}Z{Z}"] = tot_elts * 4 / 1024 / 1024 / 1024

    config_to_mem = dict(sorted(config_to_mem.items(), key=lambda item: item[1]))

    return config_to_mem
