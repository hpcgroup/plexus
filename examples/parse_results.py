# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import re


def extract_avg_time(line):
    match = re.search(r"Avg Time: ([0-9]*\.?[0-9]+)", line)
    return float(match.group(1)) if match else 0


def process_log_file(filename, warmup):
    """
    Args:
        filename - path to file to parse
        warmup - number of initial epochs to ignore in the calculation

    Returns:
        tuple containing the epoch time and communication time, averaged across non-warmup epochs
    """

    comm_times, epoch_times = [], []
    comm_time, comp_time, cross_time = None, None, None

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            if (
                "epoch " in line
                and comm_time is not None
                and comp_time is not None
                and cross_time is not None
            ):
                epoch_times.append(comp_time + comm_time + cross_time)
                comm_times.append(comm_time)
                comm_time = 0
                comp_time = 0
                cross_time = 0
            elif "epoch " in line:
                comm_time = 0
                comp_time = 0
                cross_time = 0
            elif comm_time is not None and any(
                keyword in line
                for keyword in [
                    "gather ",
                    "all-reduce ",
                    "reduce-scatter ",
                ]
            ):
                comm_time += extract_avg_time(line)
            elif comp_time is not None and any(
                keyword in line
                for keyword in [
                    "AGG = A * H ",
                    "OUT = AGG * W ",
                    "GRAD_W = AGG.T * GRAD_OUT ",
                    "GRAD_AGG = GRAD_OUT * W.T ",
                    "GRAD_H = A.T * GRAD_AGG ",
                ]
            ):
                comp_time += extract_avg_time(line)
            elif cross_time is not None and any(
                keyword in line for keyword in ["cross entropy"]
            ):
                cross_time += extract_avg_time(line)

    if comm_time is not None and comp_time is not None and cross_time is not None:
        epoch_times.append(comp_time + comm_time + cross_time)
        comm_times.append(comm_time)

    return sum(epoch_times[warmup:]) / (len(epoch_times) - warmup), sum(
        comm_times[warmup:]
    ) / (len(comm_times) - warmup)
