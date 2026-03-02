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


def compute_config_mem(G, N, E, D_list, eval_mode=False):
    """
    Args:
        G - number of GPUs
        N - number of nodes in graph
        E - number of nonzeros in graph's adjacency matrix
        D_list - list of features at each layer (ex: 3 GCN layers with 128 hidden dim, 100 feature size, 60 classes [100, 128, 128, 60])
        eval_mode - if True, estimate peak memory during evaluation (full-graph forward, no gradients);
                    training state (weights + optimizer) is assumed to persist in memory.

    Returns:
        Approximate gpu memory usage (GB) for each 3D config
    """

    config_to_mem = {}
    for X, Y, Z in split_into_three_powers_of_two(G):
        # assuming E is roughly evenly distributed after permutation
        adj_mem, divide_list = 0, [(Y, X), (Z, Y), (X, Z)]
        for i in range(min(3, len(D_list))):
            # +1 for A.T, CSR format
            adj_mem += 2 * (
                (E / (divide_list[i][0] * divide_list[i][1]) * 3)
                + (N / divide_list[i][0] * 2)
            )

        # +1 for grad, +2 for optimizer states
        # weights are sharded across depth dimension
        # (optimizer states persist even during eval)
        weight_mem = 0
        for i in range(len(D_list) - 1):
            weight_mem += 4 * D_list[i] * D_list[i + 1] / G

        # aggregation buffers: +1 for grad during training, forward-only during eval
        agg_factor = 1 if eval_mode else 2
        agg_mem = 0
        divide_list = [(Y * Z), (Z * X), (X * Y)]
        for i in range(len(D_list) - 1):
            agg_mem += agg_factor * N * D_list[i] / divide_list[i % 3]

        # activations: +1 for grad during training, forward-only during eval
        act_factor = 1 if eval_mode else 2
        activation_mem = 0
        divide_list = [(X * Z), (Y * X), (Z * Y)]
        for i in range(1, len(D_list)):
            activation_mem += act_factor * N * D_list[i] / divide_list[i % 3]

        # input features are sharded across depth dimension
        # training: +1 for grad, +2 for optimizer states
        # eval: just the features themselves (no grad/optimizer for input data)
        input_factor = 1 if eval_mode else 4
        activation_mem += input_factor * N * D_list[0] / G

        # accounting for max one time mem of gathering
        # input features and weights for first layer
        one_time_mem = ((N * D_list[0]) / (X * Y)) + ((D_list[0] * D_list[1]) / (Y * X))

        # * 2 since labels are int64
        if len(D_list) % 3 == 2:
            divide_amt = Y
        elif len(D_list) % 3 == 0:
            divide_amt = Z
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
