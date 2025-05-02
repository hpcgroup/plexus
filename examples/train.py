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
    parser.add_argument("--timing_start_epoch", type=int, default=None)
    parser.add_argument("--timing_end_epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_gcn_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser


class Net(torch.nn.Module):
    """
    Define the GCN here
    """

    def __init__(self, num_gcn_layers, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.num_gcn_layers = num_gcn_layers

        self.layers = []
        for i in range(self.num_gcn_layers):
            if i == 0:
                self.layers.append(GCNConv(input_size, hidden_size, i))
            elif i == self.num_gcn_layers - 1:
                self.layers.append(GCNConv(hidden_size, output_size, i))
            else:
                self.layers.append(GCNConv(hidden_size, hidden_size, i))

    def forward(self, x, edge_index_shards):
        for i in range(self.num_gcn_layers):
            x = self.layers[i](x, edge_index_shards)
            if i != self.num_gcn_layers - 1:
                x = F.relu(x)
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

    # trains on entire graph, data.train_mask not used
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
    data_loader = DataLoader(args.data_dir, args.num_gcn_layers)

    # get the dataset which includes graph, features, and output labels
    adj_shards, features, labels, num_nodes, num_features, num_classes = (
        data_loader.load()
    )

    # create the model and move to gpu
    model = Net(args.num_gcn_layers, num_features, args.hidden_size, num_classes).to(
        torch.device("cuda")
    )

    # create optimizer for parameters
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [features],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dist.barrier(device_ids=[torch.cuda.current_device()])

    # training loop
    for i in range(args.num_epochs):
        # range of epochs to time (inclusive of both endpoints)
        if args.timing_start_epoch is None:
            args.timing_start_epoch = 0

        if args.timing_end_epoch is None:
            args.timing_end_epoch = args.num_epochs - 1

        if i >= args.timing_start_epoch and i <= args.timing_epoch:
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

        if i >= args.timing_start_epoch and i <= args.timing_end_epoch:
            ax.get_timers().stop("epoch " + str(i))

        if i == args.timing_end_epoch:
            print_axonn_timer_data(ax.get_timers().get_times()[0])

        log = "Epoch: {:03d}, Train Loss: {:.4f}"
        if dist.get_rank() == 0:
            print(log.format(i, loss))

    dist.destroy_process_group()
