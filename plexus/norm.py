import torch
from torch import nn
from axonn import axonn as ax
import torch.distributed as dist
from axonn.intra_layer.communication import ForwardAllReduce
from plexus.utils.general import pad_dimension, get_process_groups_info
from axonn.intra_layer.communication import _all_reduce


def _rmsnorm_pre(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast to float32 and compute local mean of squares."""
    x_float = x.float()
    norm = (x_float * x_float).mean(dim=-1, keepdim=True)
    return x_float, norm


def _rmsnorm_post(
    x_float: torch.Tensor, norm: torch.Tensor, weight: torch.Tensor, eps: float, dtype: torch.dtype
) -> torch.Tensor:
    """Normalize and scale."""
    x_normed = x_float * torch.rsqrt(norm + eps)
    return (x_normed * weight.float()).to(dtype=dtype)


def _rmsnorm_post_relu_dropout(
    x_float: torch.Tensor,
    norm: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Normalize, scale, ReLU, and dropout — fused for a single compiled kernel."""
    x_normed = x_float * torch.rsqrt(norm + eps)
    out = (x_normed * weight.float()).to(dtype=dtype)
    out = torch.nn.functional.relu(out)
    out = torch.nn.functional.dropout(out, p=dropout_p, training=training)
    return out


class PlexusRMSNorm(nn.Module):
    """
    Tensor-parallel RMSNorm over the feature dimension.

    Assumes the feature dimension is sharded across a single process group
    (the layer's outer group in Plexus). The per-token RMS is computed by
    all-reducing the local mean of squared activations.
    """

    def __init__(
        self,
        size: int,
        feature_group: str,
        data_group: str | None = None,
        inner_group: str | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        num_gpus, _, process_groups = get_process_groups_info((feature_group,))
        self.feature_group = process_groups[0]
        self.feature_group_size = num_gpus[0]
        self.eps = eps

        # Data-dimension group: the group across which the node/data
        # dimension of this norm's input is partitioned (the layer's
        # depth group).  Used by sync_norm_gradients().
        if data_group is not None:
            dg_gpus, _, dg_pgs = get_process_groups_info((data_group,))
            self.data_group = dg_pgs[0]
            self.data_group_size = dg_gpus[0]
            self.data_group_letter = data_group
        else:
            self.data_group = None
            self.data_group_size = 1
            self.data_group_letter = None

        # Inner group: the group across which the norm input is
        # replicated (after the GCN combination all-reduce).
        # Used by check_norm_weight_consistency().
        if inner_group is not None:
            ig_gpus, _, ig_pgs = get_process_groups_info((inner_group,))
            self.inner_group = ig_pgs[0]
            self.inner_group_size = ig_gpus[0]
            self.inner_group_letter = inner_group
        else:
            self.inner_group = None
            self.inner_group_size = 1
            self.inner_group_letter = None

        self.global_size = size
        self.local_size = (
            pad_dimension(size, self.feature_group_size) // self.feature_group_size
        )
        self.weight = nn.Parameter(
            torch.ones(self.local_size, device="cuda"), requires_grad=True
        )

        self._pre = _rmsnorm_pre
        self._post = _rmsnorm_post
        self._fuse_activation = False
        self._dropout_p = 0.0

    def compile(self, fuse_activation=False, dropout_p=0.0):
        """Compile the elementwise parts of RMSNorm with torch.compile.

        Args:
            fuse_activation: If True, fuse ReLU + Dropout into the post-norm
                kernel, eliminating two extra memory round-trips.
            dropout_p: Dropout probability (only used when *fuse_activation*).
        """
        self._pre = torch.compile(_rmsnorm_pre)
        if fuse_activation:
            self._post = torch.compile(_rmsnorm_post_relu_dropout)
            self._fuse_activation = True
            self._dropout_p = dropout_p
        else:
            self._post = torch.compile(_rmsnorm_post)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"PlexusRMSNorm expects 2D input, got {tuple(x.shape)}.")
        if x.shape[1] != self.local_size:
            raise ValueError(
                f"Expected local_size={self.local_size}, got {x.shape[1]}."
            )

        x_float, norm = self._pre(x)
        if self.feature_group_size > 1:
            norm = ForwardAllReduce.apply(norm, self.feature_group)
            norm = norm / self.feature_group_size
        if self._fuse_activation:
            return self._post(
                x_float, norm, self.weight, self.eps, x.dtype,
                self._dropout_p, self.training,
            )
        return self._post(x_float, norm, self.weight, self.eps, x.dtype)


def sync_norm_gradients(norms, mean: bool = False) -> None:
    """All-reduce norm weight gradients across each norm's data group.

    The output of GCN layer *i* has its data (node) dimension partitioned
    across the depth group.  The norm weight gradient is therefore a partial
    sum over local nodes and must be combined across this group before the
    optimizer step.
    """
    if not dist.is_initialized():
        return
    for norm in norms:
        if norm.weight.grad is None:
            continue
        if norm.data_group is None:
            continue
        depth_world = norm.data_group_size
        if depth_world <= 1:
            continue
        dist.all_reduce(norm.weight.grad, group=norm.data_group)
        if mean:
            norm.weight.grad.div_(depth_world)


def check_norm_weight_consistency(norms) -> None:
    """Verify that norm weights are identical across replication dimensions.

    For each norm layer the weight is sharded across the outer (feature)
    group, but should be identical across the inner group and the depth
    group (and data-parallel replicas).  Any divergence indicates a
    gradient synchronisation bug.
    """
    for i, norm in enumerate(norms):
        w = norm.weight.data.clone()

        # Check across the data group (depth / data-dimension group).
        if norm.data_group is not None and norm.data_group_size > 1:
            w_ref = w.clone()
            dist.broadcast(
                w_ref,
                src=dist.get_process_group_ranks(norm.data_group)[0],
                group=norm.data_group,
            )
            if not torch.allclose(w, w_ref, atol=1e-6):
                diff = (w - w_ref).abs().max().item()
                print(
                    f"[TEST FAIL] Rank {dist.get_rank()}: norm[{i}] weight "
                    f"diverged across data group '{norm.data_group_letter}', "
                    f"max diff = {diff}"
                )

        # Check across the inner group (should be replicated).
        if norm.inner_group is not None and norm.inner_group_size > 1:
            w_ref2 = w.clone()
            dist.broadcast(
                w_ref2,
                src=dist.get_process_group_ranks(norm.inner_group)[0],
                group=norm.inner_group,
            )
            if not torch.allclose(w, w_ref2, atol=1e-6):
                diff = (w - w_ref2).abs().max().item()
                print(
                    f"[TEST FAIL] Rank {dist.get_rank()}: norm[{i}] weight "
                    f"diverged across inner group '{norm.inner_group_letter}', "
                    f"max diff = {diff}"
                )

        # Check across the data-parallel group.
        dp_group = ax.comm_handle.data_parallel_group
        if dp_group is not None and dist.get_world_size(dp_group) > 1:
            w_ref3 = w.clone()
            dist.broadcast(
                w_ref3,
                src=dist.get_process_group_ranks(dp_group)[0],
                group=dp_group,
            )
            if not torch.allclose(w, w_ref3, atol=1e-6):
                diff = (w - w_ref3).abs().max().item()
                print(
                    f"[TEST FAIL] Rank {dist.get_rank()}: norm[{i}] weight "
                    f"diverged across data-parallel group, max diff = {diff}"
                )
