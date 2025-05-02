# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import math
import torch
from axonn import axonn as ax
from torch.nn import Parameter
import torch.nn.functional as F
from plexus import plexus as plx
import torch.distributed as dist
from utils.general import pad_dimension, get_process_groups_info
from axonn.intra_layer.communication import _gather, _all_reduce, _reduce_scatter
from axonn.intra_layer.fully_connected import extract_local_params_from_full_params


def extract_csr_submatrix(csr_matrix, start_row, end_row):
    """
    Retrieves a row-chunk of a csr matrix [start_row, end_row)
    """

    # Get row offsets, col indices, and values
    crow_indices = csr_matrix.crow_indices()
    col_indices = csr_matrix.col_indices()
    values = csr_matrix.values()

    # Get the range of nonzero elements for the specified rows
    start_ptr = crow_indices[start_row].item()
    end_ptr = crow_indices[end_row].item()

    # Extract the relevant columns and values
    sub_col_indices = col_indices[start_ptr:end_ptr]
    sub_values = values[start_ptr:end_ptr]

    # Adjust row indices to be zero-based for the submatrix
    sub_crow_indices = crow_indices[start_row : end_row + 1] - start_ptr
    sub_crow_indices = torch.cat(
        (torch.tensor([0], device=sub_crow_indices.device), sub_crow_indices[1:])
    )

    # Create new CSR tensor
    num_rows = end_row - start_row
    num_cols = csr_matrix.size(1)
    sub_csr = torch.sparse_csr_tensor(
        sub_crow_indices,
        sub_col_indices,
        sub_values,
        size=(num_rows, num_cols),
        dtype=csr_matrix.dtype,
        device=csr_matrix.device,
    )

    return sub_csr


def chunked_spmm_all_reduce(csr_matrix, H, ar_group):
    """
    Performs SpMM of a CSR matrix with a dense matrix H,
    followed by an all-reduce operation on the result, optionally
    overlapping the all-reduce of the current chunk with the SpMM of the next.
    """

    timer_name = "AGG = A * H and All-Reduce AGG"
    ax.get_timers().start(timer_name)

    # calculate number of rows per chunk
    num_rows = csr_matrix.size(0)
    max_rows_per_chunk = 1000000  # can adjust as needed
    num_chunks = (num_rows + max_rows_per_chunk - 1) // max_rows_per_chunk
    rows_per_chunk = num_rows // num_chunks

    results = [None] * num_chunks
    async_handles = [None] * num_chunks

    # iterate through each chunk
    for i in range(num_chunks):
        # extract the current chunk
        start_row = i * rows_per_chunk
        end_row = num_rows if i == num_chunks - 1 else (i + 1) * rows_per_chunk
        chunk_edge_index = extract_csr_submatrix(csr_matrix, start_row, end_row)

        # spmm for current chunk
        results[i] = torch.sparse.mm(chunk_edge_index, H)

        if plx.overlap_agg:
            # once previous chunk is complete, launch async all-reduce
            # which should allow for overlap with the next chunk's spmm
            async_handles[i] = (
                dist.all_reduce(results[i], group=ar_group, async_op=True)
                if dist.is_initialized()
                else _all_reduce(results[i], ar_group)
            )
        else:
            # Perform all-reduce on the chunk result
            _all_reduce(results[i], ar_group)

    if plx.overlap_agg:
        # Wait for all asynchronous all-reduce operations to complete.
        if dist.is_initialized():
            for handle in async_handles:
                if handle is not None:
                    handle.wait()

    # concatenate all results to form the final output
    AGG = torch.cat(results, dim=0)
    ax.get_timers().stop(timer_name)
    return AGG


class GCNConvFunction(torch.autograd.Function):
    """
    3D Tensor Parallel GCN Conv FWD and BWD
    """

    @staticmethod
    def forward(
        ctx,
        x,
        edge_index,
        edge_index_t,
        weight,
        local_features_shape,
        local_weight_shape,
        all_gather_group,
        aggregation_all_reduce_group,
        combination_all_reduce_group,
        gather_features,
        gather_weights,
    ):
        """
        Forward pass of GCN layer

        Args:
            x: input matrix to the layer
            edge_index: adj matrix
            edge_index_t: transpose of adj matrix
            weight: weights matrix
            local_features_shape: shape of x
            local_weight_shape: shape of weight
            all_gather_group: depth process group
            aggregation_all_reduce_group: process group along which output of aggregation is all-reduced
            combination_all_reduce_group: process group along which output of layer is all-reduced
            gather_features: flag indicating whether x is sharded across depth group or not
            gather_weights: flag indicating whether weight is sharded across depth group or not

        Returns:
            output matrix of current GCN layer
        """

        ax.get_timers().start("gcn conv fwd")

        # gather features if sharded
        if gather_features:
            H = _gather(x, dim=0, process_group=all_gather_group)
            H = H.reshape(local_features_shape)
        else:
            H = x

        # compute aggregation (A * H) and all-reduce the result

        if plx.block_agg:
            AGG = chunked_spmm_all_reduce(edge_index, H, aggregation_all_reduce_group)
        else:
            ax.get_timers.start("AGG = A * H")
            AGG = torch.sparse.mm(edge_index, H)
            ax.get_timers.stop("AGG = A * H")

            _all_reduce(AGG, aggregation_all_reduce_group)

        # save AGG = A*H, weight, and adj matrix for backward pass
        ctx.save_for_backward(AGG, weight, edge_index_t)
        ctx.backward_depth_group = all_gather_group
        ctx.backward_all_reduce_group = aggregation_all_reduce_group
        ctx.local_weight_shape = local_weight_shape
        ctx.bwd_reduce_scatter_grad_x = gather_features
        ctx.bwd_reduce_scatter_grad_weights = gather_weights

        # gather weights - assuming that we always have this matrix sharded
        if gather_weights:
            W = _gather(weight, dim=0, process_group=all_gather_group)
        else:
            W = weight
        W = W.reshape(local_weight_shape)

        # combination - (A * H) * W
        ax.get_timers().start("OUT = AGG * W")
        OUT = torch.mm(AGG, W)
        ax.get_timers().stop("OUT = AGG * W")

        # all reduce output of layer
        _all_reduce(OUT, combination_all_reduce_group)

        ax.get_timers().stop("gcn conv fwd")

        return OUT

    @staticmethod
    def backward(ctx, grad_output):
        ax.get_timers().start("gcn conv bwd")

        # get agg and weight which are needed for
        # TODO: implement activation checkpointing
        agg, weight, adj_t = ctx.saved_tensors

        # gather the weights - assume that this matrix is always sharded
        if ctx.bwd_reduce_scatter_grad_weights:
            weight = _gather(weight, dim=0, process_group=ctx.backward_depth_group)
        weight = weight.reshape(ctx.local_weight_shape)

        # calculate gradient with respect to weight (AGG.T * GRAD_OUTPUT)
        # and reduce scatter it so they're sharded
        ax.get_timers().start("GRAD_W = AGG.T * GRAD_OUT")
        grad_weight = torch.mm(torch.t(agg), grad_output)
        ax.get_timers().stop("GRAD_W = AGG.T * GRAD_OUT")

        if ctx.bwd_reduce_scatter_grad_weights:
            grad_weight = grad_weight.reshape(-1)
            grad_weight = _reduce_scatter(
                grad_weight, dim=0, process_group=ctx.backward_depth_group
            )
        else:
            # all-reduce instead of reduce-scatter if weights aren't sharded
            _all_reduce(grad_weight, process_group=ctx.backward_depth_group)
            grad_weight = grad_weight.reshape(-1)

        # calculate gradient with respect to AGG and all-reduce
        ax.get_timers().start("GRAD_AGG = GRAD_OUT * W.T")
        grad_agg = torch.mm(grad_output, torch.t(weight))
        ax.get_timers().stop("GRAD_AGG = GRAD_OUT * W.T")

        _all_reduce(grad_agg, ctx.backward_all_reduce_group)

        # calculate gradient with respect to features (output of the previous layer)
        ax.get_timers().start("GRAD_H = A.T * GRAD_AGG")
        grad_x = torch.sparse.mm(adj_t, grad_agg)
        ax.get_timers().stop("GRAD_H = A.T * GRAD_AGG")

        if ctx.bwd_reduce_scatter_grad_x:
            # first layer's x is sharded across depth group,
            # so reduce-scatter grad_x
            grad_x = grad_x.reshape(-1)
            grad_x = _reduce_scatter(
                grad_x, dim=0, process_group=ctx.backward_depth_group
            )
        else:
            # x is replicated across depth group after first layer,
            # so all-reduce grad_x
            _all_reduce(grad_x, ctx.backward_depth_group)

        ax.get_timers().stop("gcn conv bwd")

        return (
            grad_x,
            None,
            None,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class GCNConv(torch.nn.Module):
    """
    3D Parallel GCNConv Layer
    """

    def __init__(self, in_channels, out_channels, layer_num, **kwargs):
        super(GCNConv, self).__init__()

        self.layer_num = layer_num

        # groups is the three process groups in a tuple (outer, inner, depth)
        # H matrix divided by outer and inner, depth is for sharding
        if layer_num % 3 == 0:
            groups = ("x", "y", "z")
        elif layer_num % 3 == 1:
            groups = ("z", "x", "y")
        elif layer_num % 3 == 2:
            groups = ("y", "z", "x")

        # only input features (layer 0) sharded
        self.gather_features = True if layer_num == 0 else False

        num_gpus, _, process_groups = get_process_groups_info(groups)

        self.outer_group = process_groups[0]
        self.inner_group = process_groups[1]
        self.depth_group = process_groups[2]

        self.outer_group_size = num_gpus[0]
        self.inner_group_size = num_gpus[1]
        self.depth_group_size = num_gpus[2]

        # initialize full weights matrix
        full_weight = torch.empty(in_channels, out_channels, device="cuda")
        torch.nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))

        # shard weights across depth group if possible
        if layer_num == 0:
            self.gather_weights = True
        else:
            self.gather_weights = (
                pad_dimension(in_channels, self.inner_group_size)
                // self.inner_group_size
            ) % self.depth_group_size == 0

        # pad weight matrix dimensions
        if self.gather_weights:
            self.in_channels = pad_dimension(
                in_channels, self.inner_group_size, self.depth_group_size
            )
        else:
            self.in_channels = pad_dimension(in_channels, self.inner_group_size)

        self.out_channels = pad_dimension(out_channels, self.outer_group_size)

        # pad weights matrix
        full_weight = F.pad(
            full_weight,
            (0, self.out_channels - out_channels, 0, self.in_channels - in_channels),
        )

        self.local_in_channels = self.in_channels // self.inner_group_size
        self.local_out_channels = self.out_channels // self.outer_group_size

        # get local shard of weights
        if self.gather_weights:
            self.weight = Parameter(
                extract_local_params_from_full_params(
                    full_weight, self.inner_group, self.outer_group, self.depth_group
                ),
                requires_grad=True,
            )
        else:
            self.weight = Parameter(
                extract_local_params_from_full_params(
                    full_weight,
                    self.inner_group,
                    self.outer_group,
                    dist.new_group(ranks=[dist.get_rank()]),
                ),
                requires_grad=True,
            )

    # assumes that adjacency matrix and input feature matrix are already sharded on the gpu
    def forward(self, x, edge_index_shards):
        edge_index, edge_index_t = edge_index_shards[
            self.layer_num % len(edge_index_shards)
        ]

        return GCNConvFunction.apply(
            x,
            edge_index,
            edge_index_t,
            self.weight,
            (edge_index.shape[1], self.local_in_channels),
            (self.local_in_channels, self.local_out_channels),
            self.depth_group,
            self.outer_group,
            self.inner_group,
            self.gather_features,
            self.gather_weights,
        )
