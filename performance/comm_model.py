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


def get_bw(ip, my, GPUS_PER_NODE, version, machine):
    if version == "v1":
        # placement-agnostic
        return 1
    elif version == "v2":
        # placement-aware theoretical
        if machine == "perlmutter":
            if ip * my <= GPUS_PER_NODE:
                return 200
            else:
                return 100 / min(GPUS_PER_NODE, ip)
    elif version == "v3":
        # placement-aware empirical
        if machine == "perlmutter":
            if my == 1:
                return 1

            if ip == 1:
                if my == 2:
                    return 76
                elif my == 4:
                    return 225
                elif my >= 8:
                    return 66.59
            elif ip == 2:
                if my == 2:
                    return 76
                elif my >= 4:
                    return 66.59 / 2
            elif ip >= 4:
                return 66.59 / 4
        elif machine == "frontier":
            if ip == 1:
                if my == 2:
                    return 129
                elif my == 4:
                    return 52
                elif my == 8:
                    return 135
                else:
                    return 80 / 2
            elif ip == 2:
                if my == 2:
                    return 50
                elif my == 4:
                    return 72
                else:
                    return 40 / 2
            elif ip == 4:
                if my == 2:
                    return 36
                else:
                    return 20 / 2
            elif ip >= 8:
                return 10 / 2


def compute_config_costs(G, N, D_list, version, machine):
    """
    Args:
        G - number of gpus
        N - number of nodes
        D_list - list of features at each layer (ex: 3 GCN layers with 128 hidden dim, 100 feature size, 60 classes [100, 128, 128, 60])
        version - "v1 for placement/bandwidth agnostic, v2 for placement aware with theoretical bandwidth, v3 for placement aware with empirical bandwidths"
        machine - currently supports perlmutter and frontier, but bandwidths for other machines can also be added

    Returns:
        Estimated communication time (ms) for each 3D config
    """

    if machine == "perlmutter":
        GPUS_PER_NODE = 4
    elif machine == "frontier":
        GPUS_PER_NODE = 8

    config_to_cost = {}
    for X, Y, Z in split_into_three_powers_of_two(G):
        AR_agg_dict, AR_out_dict, AG_dict = (
            {0: "X", 1: "Z", 2: "Y"},
            {0: "Y", 1: "X", 2: "Z"},
            {0: "Z", 1: "Y", 2: "X"},
        )
        num_gpus_dict = {"X": X, "Y": Y, "Z": Z}

        # hierarchy is column, row, depth (y, x, z)
        bandwidth_dict = dict()
        bandwidth_dict["Y"] = get_bw(1, Y, GPUS_PER_NODE, version, machine)
        bandwidth_dict["X"] = get_bw(Y, X, GPUS_PER_NODE, version, machine)
        bandwidth_dict["Z"] = get_bw(Y * X, Z, GPUS_PER_NODE, version, machine)

        comm_cost = 0
        for i in range(len(D_list) - 1):
            D, D_next = D_list[i], D_list[i + 1]
            AR_agg, AR_out, AG = (
                num_gpus_dict[AR_agg_dict[i % 3]],
                num_gpus_dict[AR_out_dict[i % 3]],
                num_gpus_dict[AG_dict[i % 3]],
            )
            AR_agg_bw, AR_out_bw, AG_bw = (
                bandwidth_dict[AR_agg_dict[i % 3]],
                bandwidth_dict[AR_out_dict[i % 3]],
                bandwidth_dict[AG_dict[i % 3]],
            )

            fwd_comm = (
                (
                    (
                        ((AG - 1) / AG * (N * D / (AR_agg * AR_out)) * (4 / (1024**3)))
                        / AG_bw
                    )
                    if i == 0
                    else 0
                )
                + (
                    (
                        2
                        * (AR_agg - 1)
                        / AR_agg
                        * (N * D / (AG * AR_out))
                        * (4 / (1024**3))
                    )
                    / AR_agg_bw
                )
                + (
                    ((AG - 1) / AG * (D * D_next / (AR_agg * AR_out)) * (4 / (1024**3)))
                    / AG_bw
                )
                + (
                    (
                        2
                        * (AR_out - 1)
                        / AR_out
                        * (N * D_next / (AG * AR_agg))
                        * (4 / (1024**3))
                    )
                    / AR_out_bw
                )
            )

            bwd_comm = (
                (
                    ((AG - 1) / AG * (D * D_next / (AR_agg * AR_out)) * (4 / (1024**3)))
                    / AG_bw
                )
                + (
                    ((AG - 1) / AG * (D * D_next / (AR_agg * AR_out)) * (4 / (1024**3)))
                    / AG_bw
                )
                + (
                    (
                        2
                        * (AR_agg - 1)
                        / AR_agg
                        * (N * D / (AG * AR_out))
                        * (4 / (1024**3))
                    )
                    / AR_agg_bw
                )
                + (
                    (
                        ((AG - 1) / AG * (N * D / (AR_agg * AR_out)) * (4 / (1024**3)))
                        / AG_bw
                    )
                    if i == 0
                    else 0
                )
                + (
                    (
                        (
                            2
                            * (AG - 1)
                            / AG
                            * (N * D / (AR_agg * AR_out))
                            * (4 / (1024**3))
                        )
                        / AG_bw
                    )
                    if i != 0
                    else 0
                )
            )

            comm_cost += fwd_comm + bwd_comm

        config_to_cost[(X, Y, Z)] = comm_cost

    config_to_cost = dict(sorted(config_to_cost.items(), key=lambda item: item[1]))

    return config_to_cost
