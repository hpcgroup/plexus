import os
import traceback
from typing import Optional
import torch.distributed as dist
from axonn import axonn as ax

_original_all_reduce = None
_allreduce_caller_cache = {}

block_agg = False
overlap_agg = False
overlap_bwd = False
tune_gemm = False
use_3d_linear = True
lowp_allreduce = False
lowp_allreduce_dtype = "bf16"
activation_checkpoint = False
no_adj_transpose = False
int32_indices = False
bf16_activations = False
bf16_spmm = False
bf16_gemm = False
avg_grad = False
overlap_fwd_comm = False
overlap_bwd_comm = False
overlap_linear_bwd = False
use_pccl_allreduce = False

_pccl_process_groups = {}


def init(
    G_intra_r: int = 1,
    G_intra_c: int = 1,
    G_intra_d: int = 1,
    gpus_per_node: Optional[int] = None,
    enable_internal_timers: bool = False,
    block_aggregation: bool = False,
    overlap_aggregation: bool = False,
    overlap_backward: bool = False,
    tune_gemms: bool = False,
    use_3d_linear_flag: bool = True,
    allreduce_low_precision: bool = False,
    allreduce_low_precision_dtype: str = "bf16",
    G_data: int = 1,
    activation_checkpointing: bool = False,
    no_adj_transpose_flag: bool = False,
    int32_csr_indices: bool = False,
    bf16_activations_flag: bool = False,
    bf16_spmm_flag: bool = False,
    bf16_gemm_flag: bool = False,
    avg_grad_flag: bool = False,
    overlap_fwd_comm_flag: bool = False,
    overlap_bwd_comm_flag: bool = False,
    overlap_linear_bwd_flag: bool = False,
    use_pccl_allreduce_flag: bool = False,
) -> None:
    """
    Initialize Plexus' 3D parallelism (optionally with data parallelism).

    Arguments:
        G_intra_r (int): number of GPUs for row group G_x
        G_intra_c (int): number of GPUs for column group G_y
        G_intra_d (int): number of GPUs for depth group G_z
        gpus_per_node (int, optional):  number of GPUs per node (inferred from PyTorch if not set)
        enable_internal_timers (bool): enable AxoNN's internal timers. This will give
        you information about time spent in synchronous communication regions
        and matrix multiplications.
        block_aggregation (bool): 1D block the aggregation
        overlap_aggregation (bool): enable overlap in the aggregation
        overlap_backward (bool): enable overlap in GCN backward communication
        tune_gemms (bool): enable tuning of dense matrix multiplications
        use_3d_linear_flag (bool): enable 3D linear layout rotation
        allreduce_low_precision (bool): enable low-precision communication for
        selected all-reduce hotspots.
        allreduce_low_precision_dtype (str): communication dtype for low-precision
        all-reduce. Supported: "bf16", "fp16".
        G_data (int): number of data-parallel groups (DP dimension)
        activation_checkpointing (bool): recompute AGG in backward instead of
        saving it, trading compute for memory.
        no_adj_transpose_flag (bool): do not pre-store the transposed adjacency
        matrix; compute it on the fly during backward.
        int32_csr_indices (bool): use INT32 instead of INT64 for CSR
        crow_indices and col_indices (valid when N < 2^31).
        bf16_activations_flag (bool): store dense activations saved for backward
        in BF16 instead of FP32.  Halves activation memory at the cost of
        slight numerical precision loss in gradients.
        bf16_spmm_flag (bool): perform sparse matrix multiplications (SPMM)
        in BF16.  Inputs are cast to BF16 before torch.sparse.mm and the
        result is cast back to FP32.
        bf16_gemm_flag (bool): perform dense matrix multiplications (GEMM)
        in BF16.  Inputs are cast to BF16 before torch.mm and the result
        is cast back to FP32.
        avg_grad_flag (bool): average (instead of sum) replicated-parameter
        gradients across their replication dimension during backward.
        overlap_fwd_comm_flag (bool): overlap AR(AGG) with Allgather(W) in GCN
        forward by issuing both as async NCCL ops on different process groups.
        overlap_bwd_comm_flag (bool): overlap AR(grad_agg) with
        [GRAD_W compute + RS(grad_W)] in GCN backward by issuing async NCCL
        ops on different process groups.  Takes priority over overlap_backward.
        overlap_linear_bwd_flag (bool): overlap AR(grad_x) with AR(grad_W+bias)
        in Plexus3DLinear backward by issuing async NCCL ops on col_group and
        row_group simultaneously; also fuses grad_W and grad_bias into a single
        all-reduce call.
        use_pccl_allreduce_flag (bool): replace synchronous all-reduce calls
        in the non-overlapped (sync) paths with PCCL's hierarchical
        reduce_scatter_2D + all_gather_2D.  Only affects gcn_conv and
        linear_3d sync all-reduces on cross-node process groups.
    """

    # overlap_aggregation can only be used with block_aggregation
    assert not overlap_aggregation or block_aggregation

    # use AxoNN for process group creation and timers
    ax.init(
        G_data=G_data,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
        G_intra_d=G_intra_d,
        gpus_per_node=gpus_per_node,
        enable_internal_timers=enable_internal_timers,
    )

    global block_agg, overlap_agg, overlap_bwd, tune_gemm
    global use_3d_linear, lowp_allreduce, lowp_allreduce_dtype
    global activation_checkpoint, no_adj_transpose, int32_indices, bf16_activations
    global bf16_spmm, bf16_gemm, avg_grad
    global overlap_fwd_comm, overlap_bwd_comm, overlap_linear_bwd
    global use_pccl_allreduce, _pccl_process_groups
    block_agg, overlap_agg, overlap_bwd, tune_gemm = (
        block_aggregation,
        overlap_aggregation,
        overlap_backward,
        tune_gemms,
    )
    use_3d_linear = bool(use_3d_linear_flag)
    lowp_allreduce = bool(allreduce_low_precision)
    dtype_key = str(allreduce_low_precision_dtype).lower()
    if dtype_key not in ("bf16", "fp16"):
        raise ValueError(
            "allreduce_low_precision_dtype must be one of: 'bf16', 'fp16'"
        )
    lowp_allreduce_dtype = dtype_key
    activation_checkpoint = bool(activation_checkpointing)
    no_adj_transpose = bool(no_adj_transpose_flag)
    int32_indices = bool(int32_csr_indices)
    bf16_activations = bool(bf16_activations_flag)
    bf16_spmm = bool(bf16_spmm_flag)
    bf16_gemm = bool(bf16_gemm_flag)
    avg_grad = bool(avg_grad_flag)
    overlap_fwd_comm = bool(overlap_fwd_comm_flag)
    overlap_bwd_comm = bool(overlap_bwd_comm_flag)
    overlap_linear_bwd = bool(overlap_linear_bwd_flag)
    use_pccl_allreduce = bool(use_pccl_allreduce_flag) or os.environ.get("PLEXUS_USE_PCCL_ALLREDUCE", "0") == "1"

    if use_pccl_allreduce:
        from mpi4py import MPI
        from pccl.build_kernels import build as build_pccl
        from plexus.utils.general import build_pccl_process_groups

        rank = dist.get_rank()
        print(f"[rank {rank}] PCCL: starting kernel build...", flush=True)
        if rank == 0:
            build_pccl()
            print(f"[rank {rank}] PCCL: build done, waiting at barrier...", flush=True)
            MPI.COMM_WORLD.Barrier()
        else:
            print(f"[rank {rank}] PCCL: waiting at barrier...", flush=True)
            MPI.COMM_WORLD.Barrier()
            build_pccl()
            print(f"[rank {rank}] PCCL: build done", flush=True)

        for dim in ("x", "y", "z"):
            print(f"[rank {rank}] PCCL: building process groups for dim={dim}...", flush=True)
            pg = build_pccl_process_groups(dim)
            axonn_pg = {
                "x": ax.comm_handle.outer_intra_layer_parallel_group,
                "y": ax.comm_handle.inner_intra_layer_parallel_group,
                "z": ax.comm_handle.depth_intra_layer_parallel_group,
            }[dim]
            _pccl_process_groups[axonn_pg] = pg
        print(f"[rank {rank}] PCCL: all process groups ready", flush=True)

    # if os.environ.get("PLEXUS_PROFILE_ALLREDUCE", "0") == "1":
    #     _install_allreduce_profiler()


# def _get_caller_label():
#     stack = traceback.extract_stack()
#     profiler_file = __file__
#     first_axonn = None
#     for frame in reversed(stack):
#         if frame.filename == profiler_file:
#             continue
#         fn = frame.filename
#         if "axonn/" in fn:
#             if first_axonn is None:
#                 first_axonn = frame
#             continue
#         if "plexus/plexus/" in fn:
#             short = fn.rsplit("plexus/", 1)[-1]
#             return f"{short}:{frame.lineno} {frame.name}"
#     if first_axonn is not None:
#         short = first_axonn.filename.rsplit("axonn/", 1)[-1]
#         return f"[axonn] {short}:{first_axonn.lineno} {first_axonn.name}"
#     return "unknown"


# def _install_allreduce_profiler():
#     global _original_all_reduce
#     if _original_all_reduce is not None:
#         return
#     _original_all_reduce = dist.all_reduce
# 
#     rank = dist.get_rank()
#     log_path = os.environ.get(
#         "PLEXUS_ALLREDUCE_LOG",
#         f"allreduce_profile_rank{rank}.csv",
#     )
#     _fh = open(log_path, "w", newline="")
#     _writer = csv.writer(_fh)
#     _writer.writerow([
#         "rank", "caller", "group_size", "numel", "dtype",
#         "msg_bytes", "msg_MB", "async",
#     ])
#     _fh.flush()
#     _call_count = [0]
# 
#     def _patched_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
#         group_size = dist.get_world_size(group)
#         msg_bytes = tensor.numel() * tensor.element_size()
#         caller = _get_caller_label()
#         _call_count[0] += 1
#         _writer.writerow([
#             rank, caller, group_size, tensor.numel(), str(tensor.dtype),
#             msg_bytes, f"{msg_bytes / (1024*1024):.4f}", async_op,
#         ])
#         if _call_count[0] % 50 == 0:
#             _fh.flush()
#         return _original_all_reduce(tensor, op=op, group=group, async_op=async_op)
# 
#     dist.all_reduce = _patched_all_reduce
