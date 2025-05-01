# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import torch
import argparse
from gcn_conv import GCNConv
import torch.nn.functional as F
from plexus import plexus as plx
import torch.distributed as dist
from utils.dataloader import DataLoader
from cross_entropy import parallel_cross_entropy
from utils.general import set_seed, print_axonn_timer_data


# arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--G_intra_r", type=int, default=1)
    parser.add_argument("--G_intra_c", type=int, default=1)
    parser.add_argument("--G_intra_d", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument(
        "--block_aggregation",
        action="store_true",
        default=False,
        help="Enable 1D blocking in aggregation",
    )
    parser.add_argument(
        "--overlap_aggregation",
        action="store_true",
        default=False,
        help="Enable overlap in aggregation",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


class Net(torch.nn.Module):
    """
    Define the GCN here
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.num_gcn_layers = 3
        self.conv1 = GCNConv(input_size, hidden_size, 0)
        self.conv2 = GCNConv(hidden_size, hidden_size, 1)
        self.conv3 = GCNConv(hidden_size, output_size, 2)

    def forward(self, x, edge_index_shards):
        x = self.conv1(x, edge_index_shards)
        x = F.relu(x)
        x = self.conv2(x, edge_index_shards)
        x = F.relu(x)
        x = self.conv3(x, edge_index_shards)
        return x


# called each epoch
def train(
    model,
    optimizer,
    features_local,
    adj_shards,
    labels,
    num_nodes,
    num_classes,
):
    # set to training mode
    model.train()

    # set gradients of optimized parameters to 0
    optimizer.zero_grad()

    # forward pass
    output = model(features_local, adj_shards)

    # trains on entire graph, doesn't use a data.train_mask
    loss = parallel_cross_entropy(
        output, labels, model.num_gcn_layers, num_nodes, num_classes
    )

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    return loss


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    # initialize distributed environment
    dist.init_process_group(backend="nccl")
    plx.init(
        G_intra_r=args.G_intra_r,
        G_intra_c=args.G_intra_c,
        G_intra_d=args.G_intra_d,
        gpus_per_node=args.gpus_per_node,
        enable_internal_timers=True,
        block_aggregation=args.block_aggregation,
        overlap_aggregation=args.overlap_aggregation,
    )

    # initialize parallel data loader
    data_loader = DataLoader(args.data_dir, 3)

    # get the dataset which includes graph, features, and output labels
    adj_shards, features, labels, num_nodes, num_features, num_classes = (
        data_loader.load()
    )

    # create the model and move to gpu
    model = Net(num_features, 128, num_classes).to(torch.device("cuda"))

    # create optimizer for parameters
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [features], lr=3e-3, weight_decay=0
    )

    dist.barrier(device_ids=[torch.cuda.current_device()])

    # training loop
    for i in range(args.num_epochs):
        # range of epochs to time (inclusive of both endpoints)
        timing_start_epoch, timing_end_epoch = 1, 9

        if i >= timing_start_epoch and i <= timing_end_epoch:
            ax.get_timers().start("epoch " + str(i))

        loss = train(
            model,
            optimizer,
            features,
            adj_shards,
            labels,
            num_nodes,
            num_classes,
        )

        if i >= timing_start_epoch and i <= timing_end_epoch:
            ax.get_timers().stop("epoch " + str(i))

        if i == timing_end_epoch:
            print_axonn_timer_data(ax.get_timers().get_times()[0])

        log = "Epoch: {:03d}, Train Loss: {:.4f}"
        if dist.get_rank() == 0:
            print(log.format(i, loss))

    dist.destroy_process_group()
