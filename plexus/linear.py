import math
import torch
from torch import nn
from axonn import axonn as ax
from axonn.intra_layer.communication import Gather
from plexus import plexus as plx
from plexus.utils.general import pad_dimension, get_process_groups_info


class PlexusLinear(nn.Module):
    """
    Row-parallel linear layer.

    - Input features are sharded across `feature_group` (columns of X).
    - Output features are sharded across `feature_group` (rows of W).
    - Input is gathered across `feature_group` to compute local output shard.
    - No all-reduce on grad_input (nodes are sharded elsewhere).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        feature_group: str,
        *,
        pad_in_features_with_depth: bool = False,
        pad_out_features_with_depth: bool = False,
        gather_features_in_depth: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.feature_group_letter = feature_group
        num_gpus, ranks, process_groups = get_process_groups_info((feature_group,))
        self.feature_group = process_groups[0]
        self.feature_group_size = num_gpus[0]
        self.feature_rank = ranks[0]

        self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group
        self.depth_group_size = ax.comm_handle.G_intra_d

        depth_mult_in = self.depth_group_size if pad_in_features_with_depth else 1
        depth_mult_out = self.depth_group_size if pad_out_features_with_depth else 1

        self.in_features = in_features
        self.out_features = out_features
        self.in_features_padded = pad_dimension(
            in_features, self.feature_group_size, depth_mult_in
        )
        self.out_features_padded = pad_dimension(
            out_features, self.feature_group_size, depth_mult_out
        )

        self.local_in_features = self.in_features_padded // self.feature_group_size
        self.local_out_features = self.out_features_padded // self.feature_group_size

        self.gather_features_in_depth = gather_features_in_depth

        # initialize full weight then take local output rows (row-parallel)
        full_weight = torch.empty(
            self.out_features_padded, self.in_features_padded, device="cuda"
        )
        torch.nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        start = self.feature_rank * self.local_out_features
        end = start + self.local_out_features
        local_weight = full_weight[start:end, :].contiguous()
        del full_weight
        self.weight = nn.Parameter(local_weight, requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out_features, device="cuda"))
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

        # gather input features across feature group (row-parallel)
        x_full = Gather.apply(x, self.feature_group, 1)
        if plx.bf16_gemm:
            out = x_full.to(torch.bfloat16).matmul(
                self.weight.t().to(torch.bfloat16)
            ).to(x_full.dtype)
        else:
            out = x_full.matmul(self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out
