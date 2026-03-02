import torch
from torch import nn
from axonn.intra_layer.communication import ForwardAllReduce
from plexus.utils.general import pad_dimension, get_process_groups_info
from axonn.intra_layer.communication import _all_reduce

class PlexusRMSNorm(nn.Module):
    """
    Tensor-parallel RMSNorm over the feature dimension.

    Assumes the feature dimension is sharded across a single process group
    (the layer's outer group in Plexus). The per-token RMS is computed by
    all-reducing the local mean of squared activations.
    """

    def __init__(self, size: int, feature_group: str, eps: float = 1e-6) -> None:
        super().__init__()
        num_gpus, _, process_groups = get_process_groups_info((feature_group,))
        self.feature_group = process_groups[0]
        self.feature_group_size = num_gpus[0]
        self.eps = eps

        self.global_size = size
        self.local_size = (
            pad_dimension(size, self.feature_group_size) // self.feature_group_size
        )
        self.weight = nn.Parameter(
            torch.ones(self.local_size, device="cuda"), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"PlexusRMSNorm expects 2D input, got {tuple(x.shape)}.")
        if x.shape[1] != self.local_size:
            raise ValueError(
                f"Expected local_size={self.local_size}, got {x.shape[1]}."
            )

        dtype = x.dtype
        x_float = x.float()
        norm = (x_float * x_float).mean(dim=-1, keepdim=True)
        if self.feature_group_size > 1:
            # norm = _all_reduce(norm, self.feature_group) / self.feature_group_size
            norm = ForwardAllReduce.apply(norm, self.feature_group)
            norm = norm / self.feature_group_size
        x_normed = x_float * torch.rsqrt(norm + self.eps)
        return (x_normed * self.weight.float()).to(dtype=dtype)
