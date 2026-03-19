import math
import torch
from torch import nn
import torch.distributed as dist
from axonn import axonn as ax
from axonn.intra_layer.communication import Gather, _all_reduce
from plexus import plexus as plx
from plexus.utils.general import pad_dimension, get_process_groups_info
from plexus.utils.matmul_tuning import tuned_matmul


class Plexus3DLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        row_group,
        col_group,
        k_group,
        matmul_name,
    ):
        ctx.save_for_backward(x, weight)
        ctx.row_group = row_group
        ctx.col_group = col_group
        ctx.k_group = k_group
        ctx.matmul_name = matmul_name
        ctx.has_bias = bias is not None
        ax.get_timers().start(matmul_name + " X * W")
        out = tuned_matmul(x, weight.t(), matmul_name + " X * W")
        ax.get_timers().stop(matmul_name + " X * W")
        
        _all_reduce(out, k_group)

        if bias is not None:
            ax.get_timers().start("OUT + BIAS")
            out = out + bias
            ax.get_timers().stop("OUT + BIAS")
        return out

    @staticmethod
    def backward(ctx, grad_output):
        ax.get_timers().start("linear_3d_bwd")
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        # ── choose: parallel AR on different groups vs. original serial ──
        _use_parallel = (
            plx.overlap_linear_bwd
            and ctx.needs_input_grad[0]
            and ctx.needs_input_grad[1]
            and dist.is_initialized()
        )

        if _use_parallel:
            # =============================================================
            # PARALLEL path: AR(grad_x, col_group) ∥ AR(grad_W+bias, row_group)
            #
            # 1. compute grad_x  → async AR on col_group
            # 2. compute grad_W  (overlaps with AR above on NCCL stream)
            #    + grad_bias     → fused async AR on row_group
            # 3. wait both       → both ARs ran concurrently
            # =============================================================

            # 1. compute grad_x and launch async AR on col_group
            ax.get_timers().start(ctx.matmul_name + " GRAD_X")
            grad_x = tuned_matmul(
                grad_output, weight, ctx.matmul_name + " GRAD_X"
            )
            ax.get_timers().stop(ctx.matmul_name + " GRAD_X")

            col_world = dist.get_world_size(ctx.col_group)
            if col_world > 1:
                ax.get_timers().start("async AR(grad_x) launch")
                grad_x = grad_x.contiguous()
                work_x = dist.all_reduce(
                    grad_x, group=ctx.col_group, async_op=True
                )
                ax.get_timers().stop("async AR(grad_x) launch")
            else:
                work_x = None

            # 2. compute grad_W (concurrent with AR(grad_x) on NCCL stream)
            ax.get_timers().start(ctx.matmul_name + " GRAD_W")
            grad_weight = tuned_matmul(
                grad_output.t(), x, ctx.matmul_name + " GRAD_W"
            )
            ax.get_timers().stop(ctx.matmul_name + " GRAD_W")

            # 3. fuse grad_W and grad_bias into a single AR on row_group
            row_world = dist.get_world_size(ctx.row_group)
            has_bias = ctx.has_bias and ctx.needs_input_grad[2]
            if has_bias:
                grad_bias = grad_output.sum(dim=0)

            if row_world > 1:
                if has_bias:
                    weight_numel = grad_weight.numel()
                    combined = torch.cat(
                        [grad_weight.reshape(-1), grad_bias]
                    )
                    ax.get_timers().start("async AR(grad_W+bias) launch")
                    work_w = dist.all_reduce(
                        combined, group=ctx.row_group, async_op=True
                    )
                    ax.get_timers().stop("async AR(grad_W+bias) launch")
                else:
                    ax.get_timers().start("async AR(grad_W) launch")
                    work_w = dist.all_reduce(
                        grad_weight, group=ctx.row_group, async_op=True
                    )
                    ax.get_timers().stop("async AR(grad_W) launch")
            else:
                work_w = None

            # 4. wait for both
            if work_x is not None:
                ax.get_timers().start("wait AR(grad_x)")
                work_x.wait()
                ax.get_timers().stop("wait AR(grad_x)")
            if work_w is not None:
                ax.get_timers().start("wait AR(grad_W)")
                work_w.wait()
                ax.get_timers().stop("wait AR(grad_W)")

            # 5. unpack combined buffer & avg_grad
            if row_world > 1 and has_bias:
                grad_weight = combined[:weight_numel].reshape(
                    grad_weight.shape
                )
                grad_bias = combined[weight_numel:]

            if plx.avg_grad and row_world > 1:
                grad_weight.div_(row_world)
                if has_bias:
                    grad_bias.div_(row_world)

        else:
            # =============================================================
            # ORIGINAL serial path (unchanged)
            # =============================================================
            if ctx.needs_input_grad[0]:
                ax.get_timers().start(ctx.matmul_name + " GRAD_X")
                grad_x = tuned_matmul(
                    grad_output, weight, ctx.matmul_name + " GRAD_X"
                )
                ax.get_timers().stop(ctx.matmul_name + " GRAD_X")
                _all_reduce(grad_x, ctx.col_group)

            if ctx.needs_input_grad[1]:
                ax.get_timers().start(ctx.matmul_name + " GRAD_W")
                grad_weight = tuned_matmul(
                    grad_output.t(), x, ctx.matmul_name + " GRAD_W"
                )
                ax.get_timers().stop(ctx.matmul_name + " GRAD_W")
                _all_reduce(grad_weight, ctx.row_group)
                if plx.avg_grad:
                    row_world = dist.get_world_size(ctx.row_group)
                    if row_world > 1:
                        grad_weight.div_(row_world)

            if ctx.has_bias and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(dim=0)
                _all_reduce(grad_bias, ctx.row_group)
                if plx.avg_grad:
                    row_world = dist.get_world_size(ctx.row_group)
                    if row_world > 1:
                        grad_bias.div_(row_world)

        ax.get_timers().stop("linear_3d_bwd")

        return grad_x, grad_weight, grad_bias, None, None, None, None


class Plexus3DLinear(nn.Module):
    """
    3D GEMM linear layer.

    - X is sharded across (row_group, k_group) and replicated across col_group.
    - W is sharded across (col_group, k_group) and replicated across row_group.
    - Output is sharded across (row_group, col_group) and replicated across k_group.
    - Forward all-reduces across k_group.
    - Backward all-reduces grad_x across col_group and grad_w/grad_bias across row_group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        row_group: str,
        k_group: str,
        col_group: str,
        pad_in_features_with_depth: bool = False,
        pad_out_features_with_depth: bool = False,
        gather_features_in_depth: bool = False,
        bias: bool = True,
        matmul_name: str = "3d_linear",
    ):
        super().__init__()

        self.row_group_letter = row_group
        self.k_group_letter = k_group
        self.col_group_letter = col_group

        num_gpus, ranks, process_groups = get_process_groups_info(
            (row_group, k_group, col_group)
        )
        self.row_group = process_groups[0]
        self.k_group = process_groups[1]
        self.col_group = process_groups[2]

        self.row_group_size = num_gpus[0]
        self.k_group_size = num_gpus[1]
        self.col_group_size = num_gpus[2]

        self.row_rank = ranks[0]
        self.k_rank = ranks[1]
        self.col_rank = ranks[2]

        self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group
        self.depth_group_size = ax.comm_handle.G_intra_d

        depth_mult_in = self.depth_group_size if pad_in_features_with_depth else 1
        depth_mult_out = self.depth_group_size if pad_out_features_with_depth else 1

        self.in_features = in_features
        self.out_features = out_features
        self.in_features_padded = pad_dimension(
            in_features, self.k_group_size, depth_mult_in
        )
        self.out_features_padded = pad_dimension(
            out_features, self.col_group_size, depth_mult_out
        )

        self.local_in_features = self.in_features_padded // self.k_group_size
        self.local_out_features = self.out_features_padded // self.col_group_size

        self.gather_features_in_depth = gather_features_in_depth
        self.matmul_name = matmul_name

        full_weight = torch.empty(
            self.out_features_padded, self.in_features_padded, device="cuda"
        )
        torch.nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))

        row_start = self.col_rank * self.local_out_features
        row_end = row_start + self.local_out_features
        col_start = self.k_rank * self.local_in_features
        col_end = col_start + self.local_in_features
        local_weight = full_weight[row_start:row_end, col_start:col_end].contiguous()
        del full_weight

        self.weight = nn.Parameter(local_weight, requires_grad=True)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.local_out_features, device="cuda")
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            if not self.gather_features_in_depth:
                raise ValueError(
                    "Received flattened input but gather_features_in_depth=False."
                )
            x = Gather.apply(x, self.depth_group, 0)
            if x.numel() % self.local_in_features != 0:
                raise ValueError(
                    "Flattened input size is not divisible by local_in_features."
                )
            x = x.reshape(-1, self.local_in_features)

        if x.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.local_in_features:
            raise ValueError(
                f"Expected local_in_features={self.local_in_features}, got {x.shape[1]}."
            )

        return Plexus3DLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.row_group,
            self.col_group,
            self.k_group,
            self.matmul_name,
        )
