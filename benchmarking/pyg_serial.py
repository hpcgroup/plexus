# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os
import math
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Reddit
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download_path", type=str)
    parser.add_argument("--num_epochs", type=int, default=10)
    return parser


def get_dataset(download_path=None):
    dataset = PygNodePropPredDataset(
        name="ogbn-products",
        root=input_dir,
        transform=T.NormalizeFeatures(),
    )
    gcn_norm = T.GCNNorm()
    return (gcn_norm.forward(dataset[0]), dataset.num_classes)


class Net(torch.nn.Module):
    def __init__(self, num_input_features, num_classes):
        super(Net, self).__init__()

        self.conv1 = GCNConv(num_input_features, 128, normalize=False, bias=False)
        self.conv2 = GCNConv(128, 128, normalize=False, bias=False)
        self.conv3 = GCNConv(128, num_classes, normalize=False, bias=False)

        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.conv3.lin.weight, a=math.sqrt(5))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


def train(model, optimizer, input_features, adj, labels):
    model.train()

    optimizer.zero_grad()

    output = model(input_features, adj)

    loss = F.cross_entropy(output, labels)

    loss.backward()

    optimizer.step()

    return loss


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    data, num_classes = get_dataset(args.download_path)
    num_input_features = data.x.shape[1]

    data.y = data.y.type(torch.LongTensor)
    data.y = data.y.to(torch.device("cuda"))

    features_local = data.x.to(torch.device("cuda")).requires_grad_()

    model = Net(num_input_features, num_classes).to(torch.device("cuda"))

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [features_local],
        lr=3e-3,
        weight_decay=0,
    )

    adj = torch.sparse_coo_tensor(
        data.edge_index,
        data.edge_weight,
        (data.x.shape[0], data.x.shape[0]),
    )
    adj = adj.to_sparse_csr()
    adj = adj.to(torch.device("cuda"))

    losses = []
    for i in range(args.num_epochs):
        loss = train(model, optimizer, features_local, adj, data.y)
        losses.append(loss.item())
        log = "Epoch: {:03d}, Train Loss: {:.4f}"
        print(log.format(i, loss))
