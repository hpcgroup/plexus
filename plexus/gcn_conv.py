# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import math
from typing import Any, Optional, Tuple
import torch
from axonn import axonn as ax
from torch.nn import Parameter
import torch.nn.functional as F
from plexus import plexus as plx
import torch.distributed as dist
from plexus.utils.matmul_tuning import tuned_matmul
from plexus.utils.general import pad_dimension, get_process_groups_info, _log_collective_message_size
from axonn.intra_layer.communication import (
    _gather,
    _all_reduce,
    _reduce_scatter,
)
from axonn.intra_layer.fully_connected import (
    extract_local_params_from_full_params,
)

_BWD_AR_STREAMS = {}
_LOWP_AR_CAST_BUFS = {}


def _get_bwd_allreduce_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _BWD_AR_STREAMS.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device_index)
        _BWD_AR_STREAMS[device_index] = stream
    return stream


def _get_lowp_cast_buffer(
    tensor: torch.Tensor,
    comm_dtype: torch.dtype,
) -> torch.Tensor:
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, comm_dtype)
    needed_numel = tensor.numel()
    buf = _LOWP_AR_CAST_BUFS.get(key)
    if buf is None or buf.numel() < needed_numel:
        buf = torch.empty(needed_numel, device=tensor.device, dtype=comm_dtype)
        _LOWP_AR_CAST_BUFS[key] = buf
    return buf[:needed_numel]


def _lowp_comm_dtype() -> torch.dtype:
    return torch.float16 if plx.lowp_allreduce_dtype == "fp16" else torch.bfloat16


def _should_use_lowp_allreduce(tensor: torch.Tensor, enable_lowp: bool) -> bool:
    if not enable_lowp or not plx.lowp_allreduce:
        return False
    if not dist.is_initialized():
        return False
    return tensor.is_cuda and tensor.is_floating_point()


def _all_reduce_with_optional_lowp(
    tensor: torch.Tensor,
    process_group: Any,
    enable_lowp: bool = False,
) -> None:
    if not _should_use_lowp_allreduce(tensor, enable_lowp):
        _all_reduce(tensor, process_group)
        return

    comm_dtype = _lowp_comm_dtype()
    ax.get_timers().start("allreduce lowp comm dtype")
    if tensor.dtype == comm_dtype:
        reduce_buf = tensor
    else:
        cast_buf = _get_lowp_cast_buffer(tensor, comm_dtype)
        cast_buf.copy_(tensor.reshape(-1))
        reduce_buf = cast_buf.view_as(tensor)
    ax.get_timers().stop("allreduce lowp comm dtype")
    ax.get_timers().start("allreduce lowp")
    dist.all_reduce(reduce_buf, group=process_group)
    ax.get_timers().stop("allreduce lowp")
    if reduce_buf is not tensor:
        ax.get_timers().start("allreduce copy back")
        tensor.copy_(reduce_buf)
        ax.get_timers().stop("allreduce copy back")


def _all_reduce_async_with_optional_lowp(
    tensor: torch.Tensor,
    process_group: Any,
    enable_lowp: bool = False,
) -> Tuple[Optional[Any], torch.Tensor]:
    if not dist.is_initialized():
        _all_reduce(tensor, process_group)
        return None, tensor

    if _should_use_lowp_allreduce(tensor, enable_lowp):
        comm_dtype = _lowp_comm_dtype()
        reduce_buf = tensor if tensor.dtype == comm_dtype else tensor.to(dtype=comm_dtype)
        work = dist.all_reduce(reduce_buf, group=process_group, async_op=True)
        return work, reduce_buf

    work = dist.all_reduce(tensor, group=process_group, async_op=True)
    return work, tensor


def _copy_back_if_needed(
    original: torch.Tensor,
    reduced: torch.Tensor,
) -> None:
    if reduced is not original:
        original.copy_(reduced)


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
        (
            torch.tensor([0], device=sub_crow_indices.device),
            sub_crow_indices[1:],
        )
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

    if plx.overlap_agg:
        ax.get_timers().start("AGG = A * H and All-Reduce AGG")

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

        if not plx.overlap_agg:
            ax.get_timers().start("AGG = A * H")

        results[i] = torch.sparse.mm(chunk_edge_index, H)

        if not plx.overlap_agg:
            ax.get_timers().stop("AGG = A * H")

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

    if plx.overlap_agg:
        ax.get_timers().stop("AGG = A * H and All-Reduce AGG")

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
        layer_num,
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
            layer_num: indicates which layer it is

        Returns:
            output matrix of current GCN layer
        """

        ax.get_timers().start("gcn conv fwd")

        # gather features if sharded
        if gather_features:
            ax.get_timers().start("Allgather F")
            _log_collective_message_size("allgather", x, "F", all_gather_group)
            H = _gather(x, dim=0, process_group=all_gather_group)
            ax.get_timers().stop("Allgather F")
            H = H.reshape(local_features_shape)
        else:
            H = x

        # compute aggregation (A * H) and all-reduce the result

        if plx.block_agg:
            AGG = chunked_spmm_all_reduce(edge_index, H, aggregation_all_reduce_group)
        else:
            ax.get_timers().start("AGG = A * H")
            AGG = torch.sparse.mm(edge_index, H)
            ax.get_timers().stop("AGG = A * H")
            # TODO "AGG"
            # _log_collective_message_size("all_reduce", AGG, "AGG", aggregation_all_reduce_group)
            ax.get_timers().start("allreduce H")
            _all_reduce_with_optional_lowp(
                AGG, aggregation_all_reduce_group, enable_lowp=True
            )
            ax.get_timers().stop("allreduce H")

        # save tensors for backward pass
        ctx.use_checkpoint = plx.activation_checkpoint
        ctx.use_no_adj_t = (edge_index_t is None)
        ctx.backward_depth_group = all_gather_group
        ctx.backward_all_reduce_group = aggregation_all_reduce_group
        ctx.local_weight_shape = local_weight_shape
        ctx.bwd_reduce_scatter_grad_x = gather_features
        ctx.bwd_reduce_scatter_grad_weights = gather_weights
        ctx.layer_num = layer_num

        if ctx.use_checkpoint:
            # Save H (pre-spmm features) + A for recomputing AGG in backward
            if ctx.use_no_adj_t:
                ctx.save_for_backward(H, weight, edge_index)
            else:
                ctx.save_for_backward(H, weight, edge_index, edge_index_t)
        else:
            # Original: save AGG directly
            if ctx.use_no_adj_t:
                ctx.save_for_backward(AGG, weight, edge_index)
            else:
                ctx.save_for_backward(AGG, weight, edge_index_t)

        # gather weights - assuming that we always have this matrix sharded
        if gather_weights:
            ax.get_timers().start("Allgather W")
            # _log_collective_message_size("allgather", weight, "W", all_gather_group)
            W = _gather(weight, dim=0, process_group=all_gather_group)
            ax.get_timers().stop("Allgather W")
        else:
            W = weight
        W = W.reshape(local_weight_shape)

        # combination - (A * H) * W
        ax.get_timers().start("OUT = AGG * W")
        OUT = tuned_matmul(AGG, W, "AGG * W " + str(layer_num))
        ax.get_timers().stop("OUT = AGG * W")

        # all reduce output of layer
        # TODO "OUT"
        # _log_collective_message_size("all_reduce", OUT, "OUT", combination_all_reduce_group)
        ax.get_timers().start("allreduce Q")
        _all_reduce_with_optional_lowp(
            OUT, combination_all_reduce_group, enable_lowp=True
        )
        ax.get_timers().stop("allreduce Q")

        ax.get_timers().stop("gcn conv fwd")

        return OUT

    @staticmethod
    def backward(ctx, grad_output):
        ax.get_timers().start("gcn conv bwd")

        # unpack saved tensors — layout depends on optimisation flags
        saved = ctx.saved_tensors

        if ctx.use_checkpoint:
            # Recompute AGG = spmm(A, H) + all_reduce
            H_saved, weight, edge_index = saved[0], saved[1], saved[2]
            agg = torch.sparse.mm(edge_index, H_saved)
            _all_reduce_with_optional_lowp(
                agg, ctx.backward_all_reduce_group, enable_lowp=True
            )
            # adj_t: use saved A^T if available, otherwise transpose on the fly
            if len(saved) > 3:
                adj_t = saved[3]
            else:
                adj_t = edge_index.transpose(0, 1).to_sparse_csr()
        else:
            agg = saved[0]
            weight = saved[1]
            if ctx.use_no_adj_t:
                edge_index = saved[2]
                adj_t = edge_index.transpose(0, 1).to_sparse_csr()
            else:
                adj_t = saved[2]

        # gather the weights - assume that this matrix is always sharded
        if ctx.bwd_reduce_scatter_grad_weights:
            weight = _gather(weight, dim=0, process_group=ctx.backward_depth_group)
        weight = weight.reshape(ctx.local_weight_shape)

        # calculate gradient with respect to AGG and all-reduce
        ax.get_timers().start("GRAD_AGG = GRAD_OUT * W.T")
        grad_agg = tuned_matmul(
            grad_output, torch.t(weight), "GRAD_OUT * W.T " + str(ctx.layer_num)
        )
        ax.get_timers().stop("GRAD_AGG = GRAD_OUT * W.T")

        overlap_bwd_allreduce = (
            plx.overlap_bwd
            and dist.is_initialized()
            and dist.get_world_size(ctx.backward_all_reduce_group) > 1
        )
        grad_agg_done_event = None
        grad_agg_work = None
        grad_agg_reduced = grad_agg
        if overlap_bwd_allreduce:
            compute_stream = torch.cuda.current_stream(device=grad_agg.device)
            comm_stream = _get_bwd_allreduce_stream(grad_agg.device)
            grad_agg_ready_event = torch.cuda.Event(blocking=False)
            grad_agg_done_event = torch.cuda.Event(blocking=False)
            grad_agg_ready_event.record(compute_stream)

            ax.get_timers().start("all-reduce launch")
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(grad_agg_ready_event)
                grad_agg_work, grad_agg_reduced = _all_reduce_async_with_optional_lowp(
                    grad_agg,
                    ctx.backward_all_reduce_group,
                    enable_lowp=True,
                )
                grad_agg_done_event.record(comm_stream)
            ax.get_timers().stop("all-reduce launch")
        else:
            # TODO "GRAD_AGG"
            # _log_collective_message_size("all_reduce", grad_agg, "GRAD_AGG", ctx.backward_all_reduce_group)
            _all_reduce_with_optional_lowp(
                grad_agg, ctx.backward_all_reduce_group, enable_lowp=True
            )

        # calculate gradient with respect to weight (AGG.T * GRAD_OUTPUT)
        # and reduce scatter it so they're sharded
        ax.get_timers().start("GRAD_W = AGG.T * GRAD_OUT")
        grad_weight = tuned_matmul(
            torch.t(agg), grad_output, "AGG.T * GRAD_OUT " + str(ctx.layer_num)
        )
        ax.get_timers().stop("GRAD_W = AGG.T * GRAD_OUT")

        if ctx.bwd_reduce_scatter_grad_weights:
            grad_weight = grad_weight.reshape(-1)
            ax.get_timers().start("ReduceScatter grad_weight")
            grad_weight = _reduce_scatter(
                grad_weight,
                dim=0,
                process_group=ctx.backward_depth_group,
            )
            ax.get_timers().stop("ReduceScatter grad_weight")
        else:
            # all-reduce instead of reduce-scatter if weights aren't sharded
            # _all_reduce(grad_weight, ctx.backward_depth_group)
            _all_reduce_with_optional_lowp(grad_weight, ctx.backward_depth_group, enable_lowp=True)
            grad_weight = grad_weight.reshape(-1)

        if grad_agg_done_event is not None:
            ax.get_timers().start("all-reduce")
            torch.cuda.current_stream(device=grad_agg.device).wait_event(
                grad_agg_done_event
            )
            if grad_agg_work is not None:
                grad_agg_work.wait()
            _copy_back_if_needed(grad_agg, grad_agg_reduced)
            ax.get_timers().stop("all-reduce")

        # calculate gradient with respect to features (output of the previous layer)
        ax.get_timers().start("GRAD_H = A.T * GRAD_AGG")
        grad_x = torch.sparse.mm(adj_t, grad_agg)
        ax.get_timers().stop("GRAD_H = A.T * GRAD_AGG")

        if ctx.bwd_reduce_scatter_grad_x:
            # first layer's x is sharded across depth group,
            # so reduce-scatter grad_x
            grad_x = grad_x.reshape(-1)
            # TODO "GRAD_H"
            # _log_collective_message_size("reduce_scatter", grad_x, "GRAD_H", ctx.backward_depth_group)
            ax.get_timers().start("ReduceScatter grad_x")
            grad_x = _reduce_scatter(
                grad_x, dim=0, process_group=ctx.backward_depth_group
            )
            ax.get_timers().stop("ReduceScatter grad_x")
        else:
            # x is replicated across depth group after first layer,
            # so all-reduce grad_x
            ax.get_timers().start("allreduce grad_x")
            _all_reduce_with_optional_lowp(
                grad_x, ctx.backward_depth_group, enable_lowp=True
            )
            ax.get_timers().stop("allreduce grad_x")
            # _all_reduce(grad_x, ctx.backward_depth_group)

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
            None,
        )


class GCNConv(torch.nn.Module):
    """
    3D Parallel GCNConv Layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        layer_num,
        shard_features_in_depth: bool = True,
        **kwargs,
    ):
        super(GCNConv, self).__init__()

        self.layer_num = layer_num

        # groups is the three process groups in a tuple (outer, inner, depth)
        # H matrix divided by outer and inner, depth is for sharding
        if plx.use_3d_linear:
            if layer_num % 3 == 0:
                groups = ("x", "z", "y")
            elif layer_num % 3 == 1:
                groups = ("y", "x", "z")
            elif layer_num % 3 == 2:
                groups = ("z", "y", "x")
        else:
            if layer_num % 3 == 0:
                groups = ("x", "y", "z")
            elif layer_num % 3 == 1:
                groups = ("z", "x", "y")
            elif layer_num % 3 == 2:
                groups = ("y", "z", "x")

        # only input features (layer 0) sharded when requested
        self.gather_features = bool(layer_num == 0 and shard_features_in_depth)

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
        # torch.nn.init.xavier_uniform_(full_weight)

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
                in_channels,
                self.inner_group_size,
                self.depth_group_size,
            )
        else:
            self.in_channels = pad_dimension(in_channels, self.inner_group_size)

        self.out_channels = pad_dimension(out_channels, self.outer_group_size)

        # pad weights matrix
        full_weight = F.pad(
            full_weight,
            (
                0,
                self.out_channels - out_channels,
                0,
                self.in_channels - in_channels,
            ),
        )

        self.local_in_channels = self.in_channels // self.inner_group_size
        self.local_out_channels = self.out_channels // self.outer_group_size

        # get local shard of weights
        if self.gather_weights:
            self.weight = Parameter(
                extract_local_params_from_full_params(
                    full_weight,
                    self.inner_group,
                    self.outer_group,
                    self.depth_group,
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
            self.layer_num,
        )
