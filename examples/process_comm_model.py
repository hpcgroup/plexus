import re
import os
import numpy as np
import matplotlib.pyplot as plt
from comm_model import compute_config_costs
from comp_model import comp_model


def extract_avg_time(line):
    match = re.search(r"Avg Time: ([0-9]*\.?[0-9]+)", line)
    return float(match.group(1)) if match else 0


def process_log_file(filename):
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
                for keyword in ["gather ", "all-reduce ", "reduce-scatter "]
            ):
                comm_time += extract_avg_time(line)
            elif comp_time is not None and any(
                keyword in line
                for keyword in [
                    "AGG = A * H ",
                    # "OUT = AGG * W ",
                    # "GRAD_W = AGG.T * GRAD_OUT ",
                    # "GRAD_AGG = GRAD_OUT * W.T ",
                    "GRAD_H = A.T * GRAD_AGG ",
                ]
            ):
                comp_time += extract_avg_time(line)
            elif cross_time is not None and any(
                keyword in line for keyword in ["cross entropy"]
            ):
                cross_time += extract_avg_time(line)

    if comm_time is not None and comp_time is not None and cross_time is not None:
        # epoch_times.append(comp_time + comm_time + cross_time)
        epoch_times.append(comp_time + comm_time)
        comm_times.append(comm_time)

    return sum(epoch_times[1:]) / (len(epoch_times) - 1), sum(comm_times[1:]) / (
        len(comm_times) - 1
    )


def parse_config(filename):
    match = re.search(r"reddit_X(\d+)Y(\d+)Z(\d+)\.txt", filename)
    if match:
        x, y, z = map(int, match.groups())
        return (x, y, z)
    return None


def main():
    num_configs = len([f for f in os.listdir() if f.endswith(".txt")])

    epoch_times = [0] * num_configs
    comm_times = [0] * num_configs

    num_gpus = None
    for filename in os.listdir():
        if filename.startswith("reddit_") and filename.endswith(".txt"):
            config = parse_config(filename)
            num_gpus = config[0] * config[1] * config[2]

            """
            CONFIG_RANKS = compute_config_costs(
                num_gpus, 232965, [602, 128, 128, 41], "v3", "perlmutter"
            )
            """

            CONFIG_RANKS = comp_model(232965, 114848857, num_gpus, [602, 128, 128])

            sorted_items = sorted(CONFIG_RANKS.items(), key=lambda x: x[1])

            for i in range(len(sorted_items)):
                CONFIG_RANKS[sorted_items[i][0]] = i

            if config and config in CONFIG_RANKS:
                rank = CONFIG_RANKS[config]
                avg_epoch_time, avg_comm_time = process_log_file(filename)

                if avg_comm_time > 0 and avg_epoch_time > 0:
                    rank = 0
                    comm_times[rank] = avg_comm_time
                    epoch_times[rank] = avg_epoch_time

    x_ticks = list(range(len(CONFIG_RANKS)))
    x_labels = list(range(len(CONFIG_RANKS)))

    np.save("times", (comm_times, epoch_times))


if __name__ == "__main__":
    main()
