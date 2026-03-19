# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import torch
import os
import argparse
from contextlib import nullcontext
from axonn import axonn as ax
import torch.nn.functional as F
from axonn.intra_layer.communication import Drop, Gather
from plexus import plexus as plx
import torch.distributed as dist
from plexus.gcn_conv import GCNConv
from plexus.linear import PlexusLinear
from plexus.linear_3d import Plexus3DLinear
from plexus.norm import PlexusRMSNorm
from plexus.utils.dataloader import DataLoader
from plexus.cross_entropy import parallel_cross_entropy, parallel_bce_with_logits
from plexus.utils.general import set_seed, print_axonn_timer_data, get_process_groups_info


# arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--G_intra_r", type=int, default=1)
    parser.add_argument("--G_intra_c", type=int, default=1)
    parser.add_argument("--G_intra_d", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument(
        "--block_aggregation",
        action="store_true",
        default=False,
        help="Enable 1D blocking in aggregation",
    )
    parser.add_argument(
        "--overlap_aggregation",
        action="store_true",
        default=False,
        help="Enable overlap in aggregation",
    )
    parser.add_argument(
        "--tune_gemms",
        action="store_true",
        default=False,
        help="Enables tuning of dense matrix multiplications",
    )
    parser.add_argument("--timing_start_epoch", type=int, default=None)
    parser.add_argument("--timing_end_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_gcn_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument(
        "--use_3d_linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 3D linear layout rotation (default: enabled).",
    )
    parser.add_argument(
        "--train_features",
        action="store_true",
        default=False,
        help="Enable training of input features (include x in optimizer).",
    )
    parser.add_argument("--train_adj", action="store_true", default=False,
        help="During training, restrict message passing to the train-induced subgraph (edges between train nodes only), if available. Evaluation still uses the full graph.")
    
    
    # beta features
    parser.add_argument("--activation_checkpoint", action="store_true", default=False,
        help="Recompute AGG in backward instead of saving it (trades compute for memory).")
    parser.add_argument("--no_adj_transpose", action="store_true", default=False,
        help="Do not pre-store A^T; compute transpose on the fly during backward.")
    parser.add_argument("--int32_indices", action="store_true", default=False,
        help="Use INT32 CSR indices instead of INT64 (valid when N < 2^31).")
    parser.add_argument("--bf16_activations", action="store_true", default=False,
        help="Store saved-for-backward activations in BF16 instead of FP32 (halves activation memory).")
    parser.add_argument("--multilabel_metric", type=str, default="rocauc",
        choices=["rocauc", "f1_micro"],
        help="Metric for multi-label evaluation: rocauc (default, for ogbn-proteins) or f1_micro (for yelp).")
    return parser


class Net(torch.nn.Module):
    """
    Define the GCN here
    """

    def __init__(
        self,
        num_gcn_layers,
        input_size,
        hidden_size,
        output_size,
        train_features: bool = False,
    ):
        super(Net, self).__init__()

        self.num_gcn_layers = num_gcn_layers

        node_group, class_group = _loss_groups(self.num_gcn_layers)
        last_outer, _, _ = _layer_groups(self.num_gcn_layers - 1)

        if plx.use_3d_linear:
            # Pre-linear: input -> hidden (row=x, k=y, col=z)
            self.input_linear = Plexus3DLinear(
                input_size,
                hidden_size,
                row_group="x",
                k_group="y",
                col_group="z",
                pad_in_features_with_depth=True,
                pad_out_features_with_depth=True,
                gather_features_in_depth=train_features,
                matmul_name="input_linear",
            )
        else:
            # Pre-linear: input -> hidden (row-parallel, feature group=y)
            self.input_linear = PlexusLinear(
                input_size,
                hidden_size,
                feature_group="y",
                pad_in_features_with_depth=True,
                pad_out_features_with_depth=True,
                gather_features_in_depth=train_features,
            )

        # GCN stack: all hidden -> hidden
        self.layers = torch.nn.ModuleList(
            [
                GCNConv(
                    hidden_size,
                    hidden_size,
                    i,
                    shard_features_in_depth=False,
                )
                for i in range(self.num_gcn_layers)
            ]
        )
        self.norms = torch.nn.ModuleList(
            [
                PlexusRMSNorm(
                    hidden_size,
                    feature_group=_outer_group_letter(i),
                )
                for i in range(self.num_gcn_layers)
            ]
        )

        if plx.use_3d_linear:
            # Post-linear: hidden -> classes (row=node_group, k=last_outer, col=class_group)
            self.output_linear = Plexus3DLinear(
                hidden_size,
                output_size,
                row_group=node_group,
                k_group=last_outer,
                col_group=class_group,
                pad_in_features_with_depth=False,
                pad_out_features_with_depth=False,
                gather_features_in_depth=False,
                matmul_name="output_linear",
            )
        else:
            # Post-linear: hidden -> classes (row-parallel, feature group=class_group)
            self.output_linear = PlexusLinear(
                hidden_size,
                output_size,
                feature_group=class_group,
                pad_in_features_with_depth=False,
                pad_out_features_with_depth=False,
                gather_features_in_depth=False,
            )

    # def forward(self, x, edge_index_shards):
    #     x = self.input_linear(x)
    #     for i in range(self.num_gcn_layers):
    #         residual = x
    #         x = self.norms[i](x)
    #         x = F.relu(x)
    #         x = F.dropout(x, p=0.5, training=self.training)

    #         x = self.layers[i](x, edge_index_shards)
    #         x = x + _reshard_residual(residual, i)
            
    #     x = self.norms[i](x)
    #     x = F.relu(x)
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.output_linear(x)
    #     return x
    
    def forward(self, x, edge_index_shards):
        ax.get_timers().start("input_linear")
        x = self.input_linear(x)
        ax.get_timers().stop("input_linear")
        
        for i in range(self.num_gcn_layers):
            residual = x
            x = self.layers[i](x, edge_index_shards)
            
            ax.get_timers().start("activation")
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            # x = x + _reshard_residual(residual, i)
            ax.get_timers().stop("activation")
            
        ax.get_timers().start("output_linear")
        x = self.output_linear(x)
        ax.get_timers().stop("output_linear")
        return x


# called each epoch
def train(
    model,
    optimizer,
    features_local,
    adj_shards,
    labels,
    train_mask,
    num_nodes,
    num_classes,
):
    # set to training mode
    model.train()

    # set gradients of optimized parameters to 0
    optimizer.zero_grad()

    # BF16 activation storage: pack dense FP32 saved tensors as BF16,
    # unpack back to original dtype during backward.
    def _bf16_pack(tensor):
        if tensor.is_floating_point() and tensor.layout == torch.strided:
            return (tensor.to(torch.bfloat16), tensor.dtype)
        return tensor

    def _bf16_unpack(packed):
        if isinstance(packed, tuple):
            tensor, orig_dtype = packed
            return tensor.to(orig_dtype)
        return packed

    hook_ctx = (
        torch.autograd.graph.saved_tensors_hooks(_bf16_pack, _bf16_unpack)
        if plx.bf16_activations
        else nullcontext()
    )

    with hook_ctx:
        # forward pass
        output = model(features_local, adj_shards)

        if labels.ndim == 2 and labels.size(-1) > 1:
            # Multi-label (e.g., ogbn-proteins): BCE-with-logits on raw scores.
            loss = parallel_bce_with_logits(
                output,
                labels,
                model.num_gcn_layers,
                num_nodes,
                num_classes,
                node_mask=train_mask,
            )
        else:
            # Single-label: cross-entropy over classes.
            loss = parallel_cross_entropy(
                output,
                labels,
                model.num_gcn_layers,
                num_nodes,
                num_classes,
                node_mask=train_mask,
            )

        # backward pass
        loss.backward()

    # update weights
    optimizer.step()

    return loss


def _loss_groups(num_gcn_layers):
    outer_group, inner_group, depth_group = _layer_groups(num_gcn_layers - 1)
    if plx.use_3d_linear:
        return (depth_group, inner_group)
    return (depth_group, outer_group)


def _outer_group_letter(layer_num: int) -> str:
    return _layer_groups(layer_num)[0]


def _reshard_residual(x: torch.Tensor, layer_num: int) -> torch.Tensor:
    x = x.clone()
    outer_group, inner_group, depth_group = _layer_groups(layer_num)
    _, _, process_groups = get_process_groups_info(
        (outer_group, inner_group, depth_group)
    )
    outer_pg, inner_pg, depth_pg = process_groups

    x_full = Gather.apply(x, outer_pg, 0)
    x_full = Gather.apply(x_full, inner_pg, 1)
    x_full = Drop.apply(x_full, depth_pg, 0)
    x_full = Drop.apply(x_full, outer_pg, 1)
    return x_full


def _layer_groups(layer_num: int):
    if plx.use_3d_linear:
        if layer_num % 3 == 0:
            return ("x", "z", "y")
        if layer_num % 3 == 1:
            return ("y", "x", "z")
        return ("z", "y", "x")
    if layer_num % 3 == 0:
        return ("x", "y", "z")
    if layer_num % 3 == 1:
        return ("z", "x", "y")
    return ("y", "z", "x")


class BestValidTracker:
    def __init__(self, metric_name: str = "acc", percent_scale: bool = True):
        self._results = []
        self._metric_name = metric_name
        self._percent_scale = percent_scale

    def add(self, train_metric: float, val_metric: float, test_metric: float):
        self._results.append((train_metric, val_metric, test_metric))

    def print_ogb_style(self, header: str = "OGB (best val)"):
        if not self._results:
            print(f"{header}: no evaluation results recorded.")
            return

        result = torch.tensor(self._results)
        if self._percent_scale:
            result = 100 * result
        best_epoch = result[:, 1].argmax().item()
        print(header + ":")
        print(f"Highest Train ({self._metric_name}): {result[:, 0].max():.2f}")
        print(f"Highest Valid ({self._metric_name}): {result[:, 1].max():.2f}")
        print(f"  Final Train ({self._metric_name}): {result[best_epoch, 0]:.2f}")
        print(f"   Final Test ({self._metric_name}): {result[best_epoch, 2]:.2f}")


@torch.no_grad()
def _distributed_argmax(logits, num_gcn_layers, num_nodes, num_classes):
    groups = _loss_groups(num_gcn_layers)
    num_gpus, ranks, process_groups = get_process_groups_info(groups)
    class_group = process_groups[1]
    class_rank = ranks[1]

    local_num_classes = logits.shape[1]
    class_offset = class_rank * local_num_classes
    invalid_classes = (
        torch.arange(local_num_classes, device=logits.device) + class_offset
    ) >= num_classes

    logits = logits.clone()
    if invalid_classes.any():
        logits[:, invalid_classes] = float("-inf")

    local_max, local_idx = logits.max(dim=1)
    global_idx = local_idx + class_offset

    gathered_vals = [torch.empty_like(local_max) for _ in range(num_gpus[1])]
    gathered_idx = [torch.empty_like(global_idx) for _ in range(num_gpus[1])]
    dist.all_gather(gathered_vals, local_max, group=class_group)
    dist.all_gather(gathered_idx, global_idx, group=class_group)

    vals = torch.stack(gathered_vals, dim=0)
    idxs = torch.stack(gathered_idx, dim=0)
    best = vals.argmax(dim=0)
    pred = idxs.gather(0, best.unsqueeze(0)).squeeze(0)

    node_rank = ranks[0]
    invalid_nodes = (
        torch.arange(pred.shape[0], device=pred.device) + node_rank * pred.shape[0]
    ) >= num_nodes
    if invalid_nodes.any():
        pred[invalid_nodes] = -1

    return pred


def _compute_split_metrics(pred, labels, mask, num_classes, node_group):
    if mask is None:
        return None

    valid = mask & (labels >= 0) & (pred >= 0)
    pred_eval = pred[valid]
    labels_eval = labels[valid]

    correct = (pred_eval == labels_eval).sum().to(torch.long)
    total = valid.sum().to(torch.long)

    tp = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
    fp = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
    fn = torch.zeros(num_classes, dtype=torch.long, device=pred.device)

    if total.item() > 0:
        match = pred_eval == labels_eval
        tp = torch.bincount(labels_eval[match], minlength=num_classes).to(torch.long)
        pred_cnt = torch.bincount(pred_eval, minlength=num_classes).to(torch.long)
        true_cnt = torch.bincount(labels_eval, minlength=num_classes).to(torch.long)
        fp = pred_cnt - tp
        fn = true_cnt - tp

    dist.all_reduce(correct, op=dist.ReduceOp.SUM, group=node_group)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=node_group)
    dist.all_reduce(tp, op=dist.ReduceOp.SUM, group=node_group)
    dist.all_reduce(fp, op=dist.ReduceOp.SUM, group=node_group)
    dist.all_reduce(fn, op=dist.ReduceOp.SUM, group=node_group)

    correct_f = correct.float()
    total_f = total.float().clamp_min(1.0)
    acc = (correct_f / total_f).item()

    tp_f = tp.float()
    fp_f = fp.float()
    fn_f = fn.float()

    micro_tp = tp_f.sum()
    micro_fp = fp_f.sum()
    micro_fn = fn_f.sum()
    micro_denom = (2 * micro_tp + micro_fp + micro_fn).clamp_min(1.0)
    micro_f1 = ((2 * micro_tp) / micro_denom).item()

    denom = 2 * tp_f + fp_f + fn_f
    f1_per_class = torch.where(
        denom > 0, (2 * tp_f) / denom.clamp_min(1.0), torch.zeros_like(denom)
    )
    support = (tp_f + fn_f) + (tp_f + fp_f)
    valid_classes = support > 0
    macro_f1 = (
        f1_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    )

    return {
        "acc": acc,
        "f1_micro": micro_f1,
        "f1_macro": macro_f1,
        "total": int(total.item()),
    }


@torch.no_grad()
def evaluate(
    model,
    features_local,
    adj_shards,
    labels,
    masks,
    num_nodes,
    num_classes,
    multilabel_metric="rocauc",
):
    groups = _loss_groups(model.num_gcn_layers)
    _, _, process_groups = get_process_groups_info(groups)
    node_group = process_groups[0]
    class_group = process_groups[1]

    model.eval()
    logits = model(features_local, adj_shards)

    # Multi-label (e.g., ogbn-proteins, yelp): gather logits across class+node groups.
    if labels.ndim == 2 and labels.size(-1) > 1:
        if masks is None:
            return {}

        # 1) Gather class shards -> full logits for this node shard.
        class_world = dist.get_world_size(group=class_group)
        gathered_logits = [torch.empty_like(logits) for _ in range(class_world)]
        dist.all_gather(gathered_logits, logits, group=class_group)
        full_logits_local = torch.cat(gathered_logits, dim=1)[:, :num_classes]

        # 2) Gather node shards -> full logits/labels/masks.
        node_world = dist.get_world_size(group=node_group)
        gathered_full_logits = [torch.empty_like(full_logits_local) for _ in range(node_world)]
        dist.all_gather(gathered_full_logits, full_logits_local, group=node_group)
        y_pred = torch.cat(gathered_full_logits, dim=0)[:num_nodes]

        gathered_labels = [torch.empty_like(labels) for _ in range(node_world)]
        dist.all_gather(gathered_labels, labels, group=node_group)
        y_true = torch.cat(gathered_labels, dim=0)[:num_nodes]

        gathered_masks = {}
        for split in ("train", "val", "test"):
            mask = masks.get(split)
            if mask is None:
                gathered_masks[split] = None
                continue
            buf = [torch.empty_like(mask) for _ in range(node_world)]
            dist.all_gather(buf, mask, group=node_group)
            gathered_masks[split] = torch.cat(buf, dim=0)[:num_nodes].to(torch.bool)

        if dist.get_rank() != 0:
            return {}

        if multilabel_metric == "f1_micro":
            from sklearn.metrics import f1_score

            y_pred_binary = (y_pred > 0).cpu().numpy()
            y_true_np = y_true.cpu().numpy()
            results = {}
            for split in ("train", "val", "test"):
                mask = gathered_masks.get(split)
                if mask is None:
                    results[split] = None
                    continue
                mask_np = mask.cpu().numpy()
                results[split] = {
                    "f1_micro": f1_score(
                        y_true_np[mask_np], y_pred_binary[mask_np], average="micro"
                    )
                }
            return results
        else:
            try:
                from ogb.nodeproppred import Evaluator
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import OGB Evaluator (ogb.nodeproppred). Ensure the `ogb` package and its dependencies are installed."
                ) from exc

            evaluator = Evaluator(name="ogbn-proteins")
            results = {}
            for split in ("train", "val", "test"):
                mask = gathered_masks.get(split)
                if mask is None:
                    results[split] = None
                    continue
                results[split] = evaluator.eval({"y_true": y_true[mask], "y_pred": y_pred[mask]})
            return results

    pred = _distributed_argmax(logits, model.num_gcn_layers, num_nodes, num_classes)

    results = {}
    if masks is None:
        return results

    for split in ("train", "val", "test"):
        results[split] = _compute_split_metrics(
            pred,
            labels,
            masks.get(split),
            num_classes,
            node_group,
        )
    return results


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    # initialize distributed environment
    dist.init_process_group(backend="nccl")
    plx.init(
        G_intra_r=args.G_intra_r,
        G_intra_c=args.G_intra_c,
        G_intra_d=args.G_intra_d,
        gpus_per_node=args.gpus_per_node,
        enable_internal_timers=True,
        block_aggregation=args.block_aggregation,
        overlap_aggregation=args.overlap_aggregation,
        tune_gemms=args.tune_gemms,
        use_3d_linear_flag=args.use_3d_linear,
        activation_checkpointing=args.activation_checkpoint,
        no_adj_transpose_flag=args.no_adj_transpose,
        int32_csr_indices=args.int32_indices,
        bf16_activations_flag=args.bf16_activations,
    )

    # initialize parallel data loader
    data_loader = DataLoader(args.data_dir, args.num_gcn_layers)

    # get the dataset which includes graph, features, and output labels
    (
        adj_shards,
        adj_shards_train,
        features,
        labels,
        masks,
        num_nodes,
        num_features,
        num_classes,
    ) = data_loader.load(
        load_train_adj=args.train_adj,
        train_features=args.train_features,
    )
    
    # create the model and move to gpu
    model = Net(
        args.num_gcn_layers,
        num_features,
        args.hidden_size,
        num_classes,
        train_features=args.train_features,
    ).to(torch.device("cuda"))

    # create optimizer for parameters
    optim_params = list(model.parameters())
    if args.train_features:
        optim_params.append(features)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dist.barrier(device_ids=[torch.cuda.current_device()])

    do_eval = bool(args.eval) and (masks is not None)
    if args.eval and not do_eval and dist.get_rank() == 0:
        print(
            "[warn] --eval was set but no train/val/test masks were found in the dataset; skipping evaluation."
        )

    best_valid = BestValidTracker() if do_eval and dist.get_rank() == 0 else None
    if best_valid is not None and labels.ndim == 2 and labels.size(-1) > 1:
        best_valid = BestValidTracker(metric_name=args.multilabel_metric, percent_scale=True)

    train_mask = masks.get("train") if masks is not None else None
    if args.train_adj and adj_shards_train is None and dist.get_rank() == 0:
        print(
            "[warn] --train_adj was set but train-induced adjacency was not found; falling back to full-graph adjacency."
        )
    train_adj_shards = (
        adj_shards_train if args.train_adj and adj_shards_train is not None else adj_shards
    )
    
    # result_dir = os.environ.get("RESULT_DIR", "./result")
    # rank = dist.get_rank()
    # trace_dir = os.path.join(result_dir, f"profiler")
    # os.makedirs(trace_dir, exist_ok=True)
    # prof = torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
    #     record_shapes=False,
    #     profile_memory=False,
    #     with_stack=False,
    # )
    # prof.start()

    # training loop
    for i in range(args.num_epochs):
        # range of epochs to time (inclusive of both endpoints)
        if args.timing_start_epoch is None:
            args.timing_start_epoch = 0

        if args.timing_end_epoch is None:
            args.timing_end_epoch = args.num_epochs - 1

        if i >= args.timing_start_epoch and i <= args.timing_end_epoch:
            ax.get_timers().start("epoch " + str(i))

        loss = train(
            model,
            optimizer,
            features,
            train_adj_shards,
            labels,
            train_mask,
            num_nodes,
            num_classes,
        )

        if i >= args.timing_start_epoch and i <= args.timing_end_epoch:
            ax.get_timers().stop("epoch " + str(i))

        if i == args.timing_end_epoch:
            print_axonn_timer_data(ax.get_timers().get_times()[0])

        log = "Epoch: {:03d}, Train Loss: {:.4f}"
        if dist.get_rank() == 0:
            print(log.format(i, loss))

    #     # advance profiler step each epoch
    #     prof.step()

    # prof.stop()
    # prof.export_chrome_trace(os.path.join(trace_dir, f"profiler_rank_{rank}.json"))

        if do_eval and args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            metrics = evaluate(
                model,
                features,
                adj_shards,
                labels,
                masks,
                num_nodes,
                num_classes,
                multilabel_metric=args.multilabel_metric,
            )
            if dist.get_rank() == 0:
                for split, m in metrics.items():
                    if m is None:
                        continue
                    if labels.ndim == 2 and labels.size(-1) > 1:
                        mk = args.multilabel_metric
                        print("{}: {} {:.4f}".format(split.upper(), mk, m[mk]))
                    else:
                        print(
                            "{}: acc {:.4f}, f1_micro {:.4f}, f1_macro {:.4f} (n={})".format(
                                split.upper(),
                                m["acc"],
                                m["f1_micro"],
                                m["f1_macro"],
                                m["total"],
                            )
                        )
                if best_valid is not None:
                    train_m = metrics.get("train")
                    val_m = metrics.get("val")
                    test_m = metrics.get("test")
                    if train_m is not None and val_m is not None and test_m is not None:
                        if labels.ndim == 2 and labels.size(-1) > 1:
                            mk = args.multilabel_metric
                            best_valid.add(
                                train_m[mk], val_m[mk], test_m[mk]
                            )
                        else:
                            best_valid.add(train_m["acc"], val_m["acc"], test_m["acc"])
    if dist.get_rank() == 0:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    if best_valid is not None and dist.get_rank() == 0:
        best_valid.print_ogb_style()

    dist.destroy_process_group()
