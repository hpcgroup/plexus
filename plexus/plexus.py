from typing import Optional
from axonn import axonn as ax

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
    global bf16_spmm, bf16_gemm
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
