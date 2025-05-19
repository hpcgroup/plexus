import gc
import os
import glob
import torch
from typing import Optional
import torch.nn.functional as F
from torch_geometric.data import Data
from plexus.utils.general import pad_dimension, get_process_groups_info


class DataLoader:
    """
    Parallel Data Loader to get the relevant shards of the adjacency matrix,
    features matrix, and output labels for each gpu given a 3D configuration
    """

    def __init__(
        self,
        data_dir: str,
        num_gcn_layers: int,
    ):
        """
        Args:
           data_dir: directory containing the preprocessed data (unpartitioned or partitioned)
           num_gcn_layers: number of gcn layers in the model
        """

        self.data_dir = data_dir
        self.num_gcn_layers = num_gcn_layers

        # Determine if data is partitioned based on the existence of metadata.pt
        self.partitioned = os.path.exists(os.path.join(data_dir, "metadata.pt"))

        self.double_perm = os.path.exists(os.path.join(data_dir, "edge_index", "1"))

        # Set the number of partitions dimension based on the number of .pt files in output_labels
        self.labels_dir = os.path.join(data_dir, "output_labels")
        self.num_partitions_dim = len(
            glob.glob(os.path.join(self.labels_dir, "0", "*.pt"))
        )

        # directory where the partitioned adj matrix and features are stored
        if self.partitioned:
            self.adj_dir = os.path.join(data_dir, "edge_index")
            self.features_dir = os.path.join(data_dir, "input_features")

    def __set_graph_attributes(self):
        # get number of nodes and features
        if not self.partitioned:
            self.num_nodes = self.data.x.shape[0]
            self.num_features = self.data.x.shape[1]
        else:
            self.num_nodes, self.num_features, self.num_classes = torch.load(
                f"{self.data_dir}/metadata.pt"
            )

        num_gpus, ranks, _ = get_process_groups_info(("z", "x", "y"))

        # following calculations are to get the indices that will be used
        # to shard the input features matrix

        num_nodes_step = pad_dimension(self.num_nodes, num_gpus[1]) // num_gpus[1]
        num_features_step = (
            pad_dimension(self.num_features, num_gpus[2], num_gpus[0]) // num_gpus[2]
        )

        self.nodes_start, self.nodes_stop = (
            ranks[1] * num_nodes_step,
            (ranks[1] + 1) * num_nodes_step,
        )
        self.features_start, self.features_stop = (
            ranks[2] * num_features_step,
            (ranks[2] + 1) * num_features_step,
        )

        depth_step = (
            pad_dimension(
                (self.nodes_stop - self.nodes_start)
                * (self.features_stop - self.features_start),
                num_gpus[0],
            )
            // num_gpus[0]
        )
        self.depth_start, self.depth_stop = (
            ranks[0] * depth_step,
            (ranks[0] + 1) * depth_step,
        )

        # following calculations are for indices to shard
        # the output labels by

        if self.num_gcn_layers % 3 == 1:
            labels_gpu_idx = 0
        elif self.num_gcn_layers % 3 == 2:
            labels_gpu_idx = 2
        else:
            labels_gpu_idx = 1

        self.labels_start = ranks[labels_gpu_idx] * (
            pad_dimension(self.num_nodes, num_gpus[labels_gpu_idx])
            // num_gpus[labels_gpu_idx]
        )
        self.labels_stop = (ranks[labels_gpu_idx] + 1) * (
            pad_dimension(self.num_nodes, num_gpus[labels_gpu_idx])
            // num_gpus[labels_gpu_idx]
        )

        # following calculations are for indices to get
        # all of the adjacency matrix shards

        (
            self.adj_dim1_start,
            self.adj_dim1_stop,
            self.adj_dim2_start,
            self.adj_dim2_stop,
        ) = ([], [], [], [])
        for i in range(min(3, self.num_gcn_layers)):
            if i == 0:
                dim1_num_gpus, dim2_num_gpus = (
                    num_gpus[0],
                    num_gpus[1],
                )
                rank1, rank2 = ranks[0], ranks[1]
            elif i == 1:
                dim1_num_gpus, dim2_num_gpus = (
                    num_gpus[2],
                    num_gpus[0],
                )
                rank1, rank2 = ranks[2], ranks[0]
            else:
                dim1_num_gpus, dim2_num_gpus = (
                    num_gpus[1],
                    num_gpus[2],
                )
                rank1, rank2 = ranks[1], ranks[2]

            dim1_step = pad_dimension(self.num_nodes, dim1_num_gpus) // dim1_num_gpus
            dim2_step = pad_dimension(self.num_nodes, dim2_num_gpus) // dim2_num_gpus

            dim1_start, dim1_stop = (
                rank1 * dim1_step,
                (rank1 + 1) * dim1_step,
            )
            dim2_start, dim2_stop = (
                rank2 * dim2_step,
                (rank2 + 1) * dim2_step,
            )

            self.adj_dim1_start.append(dim1_start)
            self.adj_dim1_stop.append(dim1_stop)

            self.adj_dim2_start.append(dim2_start)
            self.adj_dim2_stop.append(dim2_stop)

    def __merge_adj_partitions(self, layer_num):
        """
        loads the relevant partitioned files for the layer_num adj matrix shard
        and merges them into one tensor
        """

        partition_size = (
            pad_dimension(self.num_nodes, self.num_partitions_dim)
            // self.num_partitions_dim
        )

        merged_indices = torch.empty((2, 0))
        merged_values = torch.empty(0)

        layer_idx = layer_num % 3

        adj_dim1_start, adj_dim1_stop = (
            self.adj_dim1_start[layer_idx],
            self.adj_dim1_stop[layer_idx],
        )
        adj_dim2_start, adj_dim2_stop = (
            self.adj_dim2_start[layer_idx],
            self.adj_dim2_stop[layer_idx],
        )

        adj_num = 0
        if self.double_perm and layer_num % 2 != 0:
            adj_num = 1

        for partition_idx_dim1 in range(
            min(
                adj_dim1_start // partition_size,
                self.num_partitions_dim,
            ),
            min(
                ((adj_dim1_stop - 1) // partition_size) + 1,
                self.num_partitions_dim,
            ),
        ):
            for partition_idx_dim2 in range(
                min(
                    adj_dim2_start // partition_size,
                    self.num_partitions_dim,
                ),
                min(
                    ((adj_dim2_stop - 1) // partition_size) + 1,
                    self.num_partitions_dim,
                ),
            ):
                curr_edge_index, curr_values = torch.load(
                    f"{self.adj_dir}/{adj_num}/{partition_idx_dim1}_{partition_idx_dim2}.pt"
                )
                merged_indices = torch.cat((merged_indices, curr_edge_index), dim=1)
                merged_values = torch.cat((merged_values, curr_values), dim=0)

                del curr_edge_index
                del curr_values
                gc.collect()

        return merged_indices, merged_values

    def __merge_features_partitions(self):
        """
        loads the relevant partitioned files for the input features matrix
        and merges them into one tensor
        """

        partition_size_dim1 = (
            pad_dimension(self.num_nodes, self.num_partitions_dim)
            // self.num_partitions_dim
        )
        partition_size_dim2 = (
            pad_dimension(self.num_features, self.num_partitions_dim)
            // self.num_partitions_dim
        )

        self.partition_nodes_start = (
            self.nodes_start // partition_size_dim1
        ) * partition_size_dim1
        self.partition_features_start = (
            self.features_start // partition_size_dim2
        ) * partition_size_dim2

        merged_features = None

        for partition_idx_dim1 in range(
            min(
                self.nodes_start // partition_size_dim1,
                self.num_partitions_dim,
            ),
            min(
                ((self.nodes_stop - 1) // partition_size_dim1) + 1,
                self.num_partitions_dim,
            ),
        ):
            curr_merged = None
            for partition_idx_dim2 in range(
                min(
                    self.features_start // partition_size_dim2,
                    self.num_partitions_dim,
                ),
                min(
                    ((self.features_stop - 1) // partition_size_dim2) + 1,
                    self.num_partitions_dim,
                ),
            ):
                curr_partition_tensor = torch.load(
                    f"{self.features_dir}/{partition_idx_dim1}_{partition_idx_dim2}.pt"
                )

                if curr_merged is None:
                    curr_merged = curr_partition_tensor
                else:
                    curr_merged = torch.cat((curr_merged, curr_partition_tensor), dim=1)

                del curr_partition_tensor
                gc.collect()

            if merged_features is None:
                merged_features = curr_merged
            else:
                merged_features = torch.cat((merged_features, curr_merged), dim=0)

            del curr_merged
            gc.collect()

        return merged_features

    def __merge_labels_partitions(self):
        """
        loads the relevant partitioned files for the labels
        and merges them into one tensor
        """

        partition_size = (
            pad_dimension(self.num_nodes, self.num_partitions_dim)
            // self.num_partitions_dim
        )

        self.partition_labels_start = (
            self.labels_start // partition_size
        ) * partition_size

        merged_labels = torch.empty(0)

        labels_num = 0
        if self.double_perm and self.num_gcn_layers % 2 == 0:
            labels_num = 1

        for partition_idx in range(
            min(
                self.labels_start // partition_size,
                self.num_partitions_dim,
            ),
            min(
                ((self.labels_stop - 1) // partition_size) + 1,
                self.num_partitions_dim,
            ),
        ):
            curr_partition = torch.load(
                f"{self.labels_dir}/{labels_num}/{partition_idx}.pt"
            )
            merged_labels = torch.cat(
                (merged_labels, curr_partition),
                dim=0,
            )

            del curr_partition
            gc.collect()

        return merged_labels

    def __split_adj(self, edge_index, edge_weight, layer_num):
        """
        gets the layer_num shard of the adj matrix
        """

        layer_idx = layer_num % 3

        dim1_start_idx, dim1_stop_idx = (
            self.adj_dim1_start[layer_idx],
            self.adj_dim1_stop[layer_idx],
        )
        dim2_start_idx, dim2_stop_idx = (
            self.adj_dim2_start[layer_idx],
            self.adj_dim2_stop[layer_idx],
        )

        adj_mask = (
            (edge_index[0, :] >= dim1_start_idx)
            & (edge_index[0, :] < dim1_stop_idx)
            & (edge_index[1, :] >= dim2_start_idx)
            & (edge_index[1, :] < dim2_stop_idx)
        )

        adj_indices = edge_index[:, adj_mask]
        adj_indices[0, :] -= dim1_start_idx
        adj_indices[1, :] -= dim2_start_idx

        adj_local = torch.sparse_coo_tensor(
            adj_indices,
            edge_weight[adj_mask],
            (
                dim1_stop_idx - dim1_start_idx,
                dim2_stop_idx - dim2_start_idx,
            ),
        )

        del edge_index
        del edge_weight
        gc.collect()

        adj_local = adj_local.to_sparse_csr()
        adj_local_t = adj_local.transpose(0, 1).to_sparse_csr()

        adj_local = adj_local.to(torch.device("cuda"))
        adj_local_t = adj_local_t.to(torch.device("cuda"))

        gc.collect()

        return adj_local, adj_local_t

    def __split_features(self, features_matrix):
        """
        gets the relevant shard of the input features matrix
        """

        if not self.partitioned:
            features_local = features_matrix[
                self.nodes_start : self.nodes_stop,
                self.features_start : self.features_stop,
            ]
        else:
            local_nodes_start, local_nodes_stop = (
                self.nodes_start - self.partition_nodes_start,
                self.nodes_stop - self.partition_nodes_start,
            )
            local_features_start, local_features_stop = (
                self.features_start - self.partition_features_start,
                self.features_stop - self.partition_features_start,
            )

            if features_matrix is None:
                features_matrix = torch.zeros(
                    local_nodes_stop - local_nodes_start,
                    local_features_stop - local_features_start,
                )

            features_local = features_matrix[
                local_nodes_start:local_nodes_stop,
                local_features_start:local_features_stop,
            ]

        del features_matrix
        gc.collect()

        num_nodes_to_pad = (self.nodes_stop - self.nodes_start) - features_local.shape[
            0
        ]
        num_features_to_pad = (
            self.features_stop - self.features_start
        ) - features_local.shape[1]
        features_local = F.pad(
            features_local,
            (0, num_features_to_pad, 0, num_nodes_to_pad),
        )

        features_local = features_local.reshape(-1)[self.depth_start : self.depth_stop]

        features_local = features_local.to(torch.device("cuda")).requires_grad_()

        gc.collect()

        return features_local

    def __split_output(self, output_labels):
        """
        gets the relevant shard of the output labels
        """

        if not self.partitioned:
            output_local = output_labels[self.labels_start : self.labels_stop]
        else:
            local_labels_start, local_labels_stop = (
                self.labels_start - self.partition_labels_start,
                self.labels_stop - self.partition_labels_start,
            )

            output_local = output_labels[local_labels_start:local_labels_stop]

        del output_labels
        gc.collect()

        num_labels_to_pad = (self.labels_stop - self.labels_start) - output_local.shape[
            0
        ]
        output_local = F.pad(output_local, (0, num_labels_to_pad)).to(torch.int64)

        output_local = output_local.to(torch.device("cuda"))

        gc.collect()

        return output_local

    def load(self):
        """
        Call this function on the DataLoader to load the relevant shards of data
        """

        with torch.no_grad():
            adj_shards = []

            # number of adjacency shards needed doubles if double permutation optimization is applied
            if self.double_perm:
                num_adj_shards = min(6, self.num_gcn_layers)
            else:
                num_adj_shards = min(3, self.num_gcn_layers)

            if not self.partitioned:
                # load unpartitioned data
                pt_files = [f for f in os.listdir(self.data_dir) if f.endswith(".pt")]
                if len(pt_files) != 1:
                    raise ValueError("Expected exactly one .pt file in the directory.")
                self.data, self.num_classes = torch.load(
                    os.path.join(self.data_dir, pt_files[0]),
                    weights_only=False,
                )

                self.__set_graph_attributes()

                # get the adj matrix shards
                for i in range(num_adj_shards):
                    if not self.double_perm or i % 2 == 0:
                        adj_shards.append(
                            self.__split_adj(
                                self.data.edge_index,
                                self.data.edge_weight,
                                i,
                            )
                        )
                    else:
                        adj_shards.append(
                            self.__split_adj(
                                self.data.edge_index_2,
                                self.data.edge_weight,
                                i,
                            )
                        )
            else:
                self.__set_graph_attributes()

                # merge the partitioned features and labels
                self.data = Data(
                    x=self.__merge_features_partitions(),
                    y=self.__merge_labels_partitions(),
                )

                # get the adj matrix shards
                for i in range(num_adj_shards):
                    self.edge_index, self.edge_weight = self.__merge_adj_partitions(i)
                    adj_shards.append(
                        self.__split_adj(self.edge_index, self.edge_weight, i)
                    )

            del self.data.edge_index
            del self.data.edge_weight
            gc.collect()

            # get the features shard
            input_features = self.__split_features(self.data.x)

            # get the labels shard
            if not self.double_perm or self.num_gcn_layers % 2 != 0:
                output_labels = self.__split_output(self.data.y)
            else:
                output_labels = self.__split_output(self.data.y_2)

            # return relevant shards of the data and some metadata
            return (
                adj_shards,
                input_features,
                output_labels,
                self.num_nodes,
                self.num_features,
                self.num_classes,
            )
