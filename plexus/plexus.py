from typing import Optional
from axonn import axonn as ax


def init(
    G_intra_r: int = 1,
    G_intra_c: int = 1,
    G_intra_d: int = 1,
    gpus_per_node: Optional[int] = None,
    enable_internal_timers: bool = False,
    block_aggregation: bool = False,
    overlap_aggregation: bool = False,
    tune_gemms: bool = False,
) -> None:
    """
    Initialize Plexus' 3D parallelism

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
        tune_gemms (bool): enable tuning of dense matrix multiplications
    """

    # overlap_aggregation can only be used with block_aggregation
    assert not overlap_aggregation or block_aggregation

    # use AxoNN for process group creation and timers
    ax.init(
        G_intra_r,
        G_intra_c,
        G_intra_d,
        gpus_per_node,
        enable_internal_timers,
    )

    global block_agg, overlap_agg, tune_gemm
    block_agg, overlap_agg, tune_gemm = (
        block_aggregation,
        overlap_aggregation,
        tune_gemms,
    )
