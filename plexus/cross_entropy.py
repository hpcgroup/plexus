# Copyright 2025 Parallel Software and Systems Group, University of Maryland._all_reducetimer
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import torch
from axonn import axonn as ax
import torch.nn.functional as F
import torch.distributed as dist
from plexus import plexus as plx
from plexus.utils.general import get_process_groups_info


def _loss_groups(num_layers: int):
    if plx.use_3d_linear:
        if num_layers % 3 == 1:
            return ("y", "z")
        if num_layers % 3 == 2:
            return ("z", "x")
        return ("x", "y")
    if num_layers % 3 == 1:
        return ("z", "x")
    if num_layers % 3 == 2:
        return ("y", "z")
    return ("x", "y")


class TensorParallelCrossEntropy(torch.autograd.Function):
    """
    Parallel Cross-entropy Implementation
    """

    @staticmethod
    def forward(ctx, logits, target, num_layers, num_nodes, num_classes, node_weight=None):
        ax.get_timers().start("cross entropy fwd")

        # select appropriate process groups for last layer
        groups = _loss_groups(num_layers)

        num_gpus, ranks, process_groups = get_process_groups_info(groups)

        # create a mask for the padded classes and
        # change those logits to -inf so their softmax is 0
        invalid_classes = (
            torch.arange(logits.shape[1], device=target.device)
            + (ranks[1] * logits.shape[1])
        ) >= num_classes

        logits[:, invalid_classes] = float("-inf")

        # calculate local max of logits
        logits_max = torch.max(logits, dim=1)[0]

        # all reduce to get max across all logits
        # ax.get_timers().start("logits_max all reduce")
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_groups[1])
        # ax.get_timers().stop("logits_max all reduce")

        # calculate numerator expression
        numerator = torch.exp(logits - logits_max.unsqueeze(1))

        # calculate local sum across numerator to get denominator
        # all reduce to get sum across all classes
        denominator = torch.sum(numerator, dim=1)
        # ax.get_timers().start("denominator all reduce")
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=process_groups[1])
        # ax.get_timers().stop("denominator all reduce")

        # calculate the softmax based on the numerator and denominator
        softmax = numerator / denominator.unsqueeze(1)

        global_node_idx = torch.arange(logits.shape[0], device=target.device) + (
            ranks[0] * logits.shape[0]
        )
        invalid_nodes = global_node_idx >= num_nodes

        target = target.clone()
        invalid_target = target < 0
        if invalid_target.any():
            target[invalid_target] = 0
            invalid_nodes = invalid_nodes | invalid_target

        # # create mask for classes that are outside the local range of classes
        # invalid_logits_mask = (target < (ranks[1] * logits.shape[1])) | (
        #     target >= ((ranks[1] + 1) * logits.shape[1])
        # )

        # # convert from global label to local label
        # target[~invalid_logits_mask] -= ranks[1] * logits.shape[1]
        # target[invalid_logits_mask] = 0

        # # create one hot vector from the labels
        # target = F.one_hot(target, num_classes=logits.shape[1])

        # # for labels out of the local range, make the target vector 0
        # target[invalid_logits_mask] = 0
        
        target = F.one_hot(target, num_classes=(logits.shape[1] * num_gpus[1]))
        target = target[
            :, (ranks[1] * logits.shape[1]) : ((ranks[1] + 1) * logits.shape[1])
        ]

        if invalid_nodes.any():
            softmax[invalid_nodes, :] = 0.0
            target[invalid_nodes, :] = 0

        # optional per-node loss weights (e.g., 1/pi for non-uniform sampling);
        # weights of invalid (padded or masked) nodes are zeroed
        weight_valid = None
        if node_weight is not None:
            if node_weight.ndim != 1:
                node_weight = node_weight.reshape(-1)
            if node_weight.shape[0] != target.shape[0]:
                raise ValueError(
                    f"node_weight must match target length "
                    f"(got {node_weight.shape[0]} vs {target.shape[0]})"
                )
            weight_valid = node_weight.to(softmax.dtype).clone()
            if invalid_nodes.any():
                weight_valid[invalid_nodes] = 0.0

        # save softmax and target for backward pass
        if weight_valid is not None:
            ctx.save_for_backward(softmax, target, weight_valid)
        else:
            ctx.save_for_backward(softmax, target)
        ctx.has_node_weight = weight_valid is not None

        # calculate loss
        epsilon = 1e-9
        loss = torch.sum(-torch.log(softmax.clamp(min=epsilon)) * target, dim=1)

        # all reduce loss across all classes
        # ax.get_timers().start("loss all reduce")
        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=process_groups[1])
        # ax.get_timers().stop("loss all reduce")

        # sum losses for all nodes and then all reduce across all nodes
        ctx.num_nodes = num_nodes

        if weight_valid is not None:
            loss = loss * weight_valid
        loss_sum = torch.sum(loss)
        # ax.get_timers().start("loss_sum all reduce")
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=process_groups[0])
        # ax.get_timers().stop("loss_sum all reduce")

        if weight_valid is not None:
            # self-normalized weighted mean: divide by the total sampled weight
            weight_sum = weight_valid.sum()
            dist.all_reduce(weight_sum, op=dist.ReduceOp.SUM, group=process_groups[0])
            ctx.loss_divisor = weight_sum.clamp_min(1e-12)
        else:
            # average over valid (non-padded, non-masked) nodes
            valid_count = (~invalid_nodes).sum().to(torch.long)
            dist.all_reduce(valid_count, op=dist.ReduceOp.SUM, group=process_groups[0])
            ctx.loss_divisor = valid_count.clamp_min(1)

        avg_loss = loss_sum / ctx.loss_divisor

        ax.get_timers().stop("cross entropy fwd")

        return avg_loss

    @staticmethod
    def backward(ctx, grad_output):
        # calculate gradient of loss with respect to the logits
        ax.get_timers().start("cross entropy bwd")
        if ctx.has_node_weight:
            softmax, target, weight_valid = ctx.saved_tensors
            grad_input = (
                (softmax - target) * weight_valid.unsqueeze(1) / ctx.loss_divisor
            )
        else:
            softmax, target = ctx.saved_tensors
            grad_input = (softmax - target) / ctx.loss_divisor
        ax.get_timers().stop("cross entropy bwd")
        return grad_input * grad_output, None, None, None, None, None


def parallel_cross_entropy(
    logits, target, groups, num_nodes, num_classes, node_mask=None, node_weight=None
):
    if node_mask is not None:
        if node_mask.ndim != 1:
            node_mask = node_mask.reshape(-1)
        if node_mask.shape[0] != target.shape[0]:
            raise ValueError(
                f"node_mask must match target length (got {node_mask.shape[0]} vs {target.shape[0]})"
            )
        target = target.clone()
        target[~node_mask.to(torch.bool)] = -1
    return TensorParallelCrossEntropy.apply(
        logits, target, groups, num_nodes, num_classes, node_weight
    )


class TensorParallelBCEWithLogits(torch.autograd.Function):
    """
    Parallel BCE-with-logits implementation for multi-label classification.
    Assumes logits are sharded across the class group (like TensorParallelCrossEntropy).
    """

    @staticmethod
    def forward(ctx, logits, target, num_layers, num_nodes, num_classes, node_weight=None):
        ax.get_timers().start("bce fwd")

        groups = _loss_groups(num_layers)

        num_gpus, ranks, process_groups = get_process_groups_info(groups)

        local_num_classes = int(logits.shape[1])
        class_offset = ranks[1] * local_num_classes

        invalid_classes = (
            torch.arange(local_num_classes, device=logits.device) + class_offset
        ) >= num_classes

        global_node_idx = torch.arange(logits.shape[0], device=logits.device) + (
            ranks[0] * logits.shape[0]
        )
        invalid_nodes = global_node_idx >= num_nodes

        if target.ndim != 2:
            raise ValueError(
                f"TensorParallelBCEWithLogits expects 2D multi-label targets; got shape {tuple(target.shape)}"
            )

        # Select this rank's class shard.
        target_local = target[:, class_offset : class_offset + local_num_classes]
        target_local = target_local.to(dtype=logits.dtype)

        labeled = torch.isfinite(target_local)
        if invalid_classes.any():
            labeled[:, invalid_classes] = False
        if invalid_nodes.any():
            labeled[invalid_nodes, :] = False

        target_filled = torch.nan_to_num(target_local, nan=0.0)

        per_entry = F.binary_cross_entropy_with_logits(
            logits, target_filled, reduction="none"
        )
        labeled_f = labeled.to(per_entry.dtype)
        if node_weight is not None:
            # fold per-node 1/pi weights into the labeled mask; the rest of the
            # (self-normalized) weighted-mean math then falls out unchanged
            if node_weight.ndim != 1:
                node_weight = node_weight.reshape(-1)
            if node_weight.shape[0] != labeled_f.shape[0]:
                raise ValueError(
                    f"node_weight must match row count "
                    f"(got {node_weight.shape[0]} vs {labeled_f.shape[0]})"
                )
            labeled_f = labeled_f * node_weight.to(per_entry.dtype).unsqueeze(1)
        per_entry = per_entry * labeled_f

        # Reduce total loss and total (weighted) labeled count across both groups.
        loss_sum = per_entry.sum()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=process_groups[1])
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=process_groups[0])

        if node_weight is not None:
            weight_sum = labeled_f.sum()
            dist.all_reduce(weight_sum, op=dist.ReduceOp.SUM, group=process_groups[1])
            dist.all_reduce(weight_sum, op=dist.ReduceOp.SUM, group=process_groups[0])
            ctx.loss_divisor = weight_sum.clamp_min(1e-12)
        else:
            labeled_count = labeled.sum().to(torch.long)
            dist.all_reduce(labeled_count, op=dist.ReduceOp.SUM, group=process_groups[1])
            dist.all_reduce(labeled_count, op=dist.ReduceOp.SUM, group=process_groups[0])
            ctx.loss_divisor = labeled_count.clamp_min(1)

        sigmoid = torch.sigmoid(logits)
        ctx.save_for_backward(sigmoid, target_filled, labeled_f)

        avg_loss = loss_sum / ctx.loss_divisor

        ax.get_timers().stop("bce fwd")
        return avg_loss

    @staticmethod
    def backward(ctx, grad_output):
        ax.get_timers().start("bce bwd")
        sigmoid, target_filled, labeled_f = ctx.saved_tensors
        grad_input = (sigmoid - target_filled) * labeled_f / ctx.loss_divisor
        ax.get_timers().stop("bce bwd")
        return grad_input * grad_output, None, None, None, None, None


def parallel_bce_with_logits(logits, target, groups, num_nodes, num_classes, node_mask=None, node_weight=None):
    if node_mask is not None:
        if node_mask.ndim != 1:
            node_mask = node_mask.reshape(-1)
        if node_mask.shape[0] != target.shape[0]:
            raise ValueError(
                f"node_mask must match target length (got {node_mask.shape[0]} vs {target.shape[0]})"
            )
        target = target.clone()
        target[~node_mask.to(torch.bool)] = float("nan")
    return TensorParallelBCEWithLogits.apply(logits, target, groups, num_nodes, num_classes, node_weight)
