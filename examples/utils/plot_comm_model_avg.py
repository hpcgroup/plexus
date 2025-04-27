import matplotlib.pyplot as plt
import numpy as np
import os


def aggregate_npy_data(directory):
    num_gpus = [4, 8, 16, 32, 64, 128]

    mean_comm_times, std_comm_times, mean_epoch_times, std_epoch_times = [], [], [], []

    # Read all .npy files in the directory
    for i in range(len(num_gpus)):
        comm_times_list, epoch_times_list = [], []
        for file in os.listdir(directory):
            if "_" + str(num_gpus[i]) + "_" in file and file.endswith(".npy"):
                data = np.load(os.path.join(directory, file), allow_pickle=True)
                if len(data) >= 2:
                    comm_times, epoch_times = data
                    comm_times_list.append(comm_times)
                    epoch_times_list.append(epoch_times)
        mean_comm_times.append(np.mean(np.array(comm_times_list), axis=0))
        std_comm_times.append(np.std(np.array(comm_times_list), axis=0))
        mean_epoch_times.append(np.mean(np.array(epoch_times_list), axis=0))
        std_epoch_times.append(np.std(np.array(comm_times_list), axis=0))

    print((np.array(mean_epoch_times) - np.array(mean_comm_times)).flatten().tolist())

    np.save(
        "scaling_perlmutter_reddit.npy",
        (mean_comm_times, std_comm_times, mean_epoch_times, std_epoch_times),
    )


aggregate_npy_data(os.getcwd())
