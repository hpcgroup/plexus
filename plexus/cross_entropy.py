import torch
from axonn import axonn as ax
import torch.nn.functional as F
import torch.distributed as dist
from utils.general import get_process_groups_info


class TensorParallelCrossEntropy(torch.autograd.Function):
    """
    Parallel Cross-entropy Implementation
    """

    @staticmethod
    def forward(ctx, logits, target, num_layers, num_nodes, num_classes):
        ax.get_timers().start("cross entropy fwd")

        # select appropriate process groups for last layer depending
        # on the number of layers
        if num_layers % 3 == 1:
            groups = ("z", "x")
        elif num_layers % 3 == 2:
            groups = ("y", "z")
        else:
            groups = ("x", "y")

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
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_groups[1])

        # calculate numerator expression
        numerator = torch.exp(logits - logits_max.unsqueeze(1))

        # calculate local sum across numerator to get denominator
        # all reduce to get sum across all classes
        denominator = torch.sum(numerator, dim=1)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=process_groups[1])

        # calculate the softmax based on the numerator and denominator
        softmax = numerator / denominator.unsqueeze(1)

        # invalid nodes are those that are padded and/or don't have a positive label
        invalid_nodes = (
            (
                torch.arange(logits.shape[0], device=target.device)
                + (ranks[0] * logits.shape[0])
            )
            >= num_nodes
        ) | (target < 0)
        softmax[invalid_nodes, :] = 0.0

        # create mask for classes that are outside the local range of classes
        invalid_logits_mask = (target < (ranks[1] * logits.shape[1])) | (
            target >= ((ranks[1] + 1) * logits.shape[1])
        )

        # convert from global label to local label
        target[~invalid_logits_mask] -= ranks[1] * logits.shape[1]
        target[invalid_logits_mask] = 0

        # create one hot vector from the labels
        target = F.one_hot(target, num_classes=logits.shape[1])

        # for labels out of the local range, make the target vector 0
        target[invalid_logits_mask] = 0

        # save softmax and target for backward pass
        ctx.save_for_backward(softmax, target)

        # calculate loss
        epsilon = 1e-9
        loss = torch.sum(-torch.log(softmax.clamp(min=epsilon)) * target, dim=1)

        # all reduce loss across all classes
        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=process_groups[1])

        # sum losses for all nodes and then all reduce across all nodes
        ctx.num_nodes = num_nodes

        loss_sum = torch.sum(loss)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=process_groups[0])

        # divide loss by number of nodes in graph
        avg_loss = loss_sum / ctx.num_nodes

        ax.get_timers().stop("cross entropy fwd")

        return avg_loss

    @staticmethod
    def backward(ctx, grad_output):
        # calculate gradient of loss with respect to the logits
        ax.get_timers().start("cross entropy bwd")
        softmax, target = ctx.saved_tensors
        grad_input = (softmax - target) / ctx.num_nodes
        ax.get_timers().stop("cross entropy bwd")
        return grad_input * grad_output, None, None, None, None


def parallel_cross_entropy(logits, target, groups, num_nodes, num_classes):
    return TensorParallelCrossEntropy.apply(
        logits, target, groups, num_nodes, num_classes
    )
