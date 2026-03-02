# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os
import sys
import math
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Reddit
from torch_geometric.utils import scatter
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch.nn import BatchNorm1d, LayerNorm, Linear

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# Prefer the bundled OGB copy when running from the Plexus repo.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOCAL_OGB_ROOT = os.path.join(_REPO_ROOT, "ogb")
if os.path.isdir(_LOCAL_OGB_ROOT) and _LOCAL_OGB_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_OGB_ROOT)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_parser():
    def _str2bool(value):
        if isinstance(value, bool):
            return value
        value = str(value).strip().lower()
        if value in ("1", "true", "t", "yes", "y", "on"):
            return True
        if value in ("0", "false", "f", "no", "n", "off"):
            return False
        raise argparse.ArgumentTypeError(
            "Boolean value expected (true/false, 1/0, yes/no)."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        choices=("reddit", "products", "ogbn-products", "proteins", "ogbn-proteins"),
        help=(
            "Dataset to benchmark. "
            "Use 'products'/'ogbn-products' for OGBN-Products, "
            "and 'proteins'/'ogbn-proteins' for OGBN-Proteins."
        ),
    )
    parser.add_argument("--download_path", type=str)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=512,
        help="Hidden feature dimension.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of GNN layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--convmodel",
        "--model",
        dest="convmodel",
        type=str,
        default="gcn",
        choices=("gcn", "gat", "sage"),
        help="GNN convolution type.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="layer",
        choices=("layer", "batch", "none"),
        help="Normalization type.",
    )
    parser.add_argument(
        "--jk",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable/disable Jumping Knowledge (JK).",
    )
    parser.add_argument(
        "--res",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable/disable residual connections.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Enable train/val/test evaluation using dataset masks (if available).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Evaluate every N epochs when --eval is set.",
    )
    parser.add_argument(
        "--train_adj",
        action="store_true",
        default=False,
        help="During training, restrict message passing to the train-induced subgraph (edges between train nodes only), if available. Evaluation still uses the full graph.",
    )
    return parser


def _ensure_masks_from_split_idx(data, split_idx):
    if hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(
        data, "test_mask"
    ):
        return data

    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0]))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_idx = torch.as_tensor(split_idx.get("train", []), dtype=torch.long).view(-1)
    valid_key = "valid" if "valid" in split_idx else "val"
    val_idx = torch.as_tensor(split_idx.get(valid_key, []), dtype=torch.long).view(-1)
    test_idx = torch.as_tensor(split_idx.get("test", []), dtype=torch.long).view(-1)

    if train_idx.numel():
        train_mask[train_idx] = True
    if val_idx.numel():
        val_mask[val_idx] = True
    if test_idx.numel():
        test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def get_dataset(dataset_name: str, download_path=None, build_train_adj: bool = False):
    dataset_name = dataset_name.lower().strip()
    num_classes = None
    if dataset_name in ("products", "ogbn-products"):
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except Exception as exc:
            raise RuntimeError(
                "Failed to import OGB (ogb.nodeproppred). Ensure the `ogb` package and its dependencies are installed."
            ) from exc

        dataset = PygNodePropPredDataset(
            name="ogbn-products",
            root=download_path,
            # transform=T.NormalizeFeatures(),
        )
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        data = _ensure_masks_from_split_idx(data, split_idx)
        num_classes = dataset.num_classes
    elif dataset_name in ("proteins", "ogbn-proteins"):
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except Exception as exc:
            raise RuntimeError(
                "Failed to import OGB (ogb.nodeproppred). Ensure the `ogb` package and its dependencies are installed."
            ) from exc

        dataset = PygNodePropPredDataset(
            name="ogbn-proteins",
            root=download_path,
        )
        split_idx = dataset.get_idx_split()
        data = dataset[0]

        # OGBN-Proteins provides edge features; following OGB baselines,
        # we compute node features by averaging incident edge features.
        data.x = scatter(
            data.edge_attr,
            data.edge_index[0],
            dim=0,
            dim_size=data.num_nodes,
            reduce="mean",
        )
        data = _ensure_masks_from_split_idx(data, split_idx)

        # ogbn-proteins is a 112-task multi-label problem.
        num_classes = int(data.y.size(-1))
    else:
        dataset = Reddit(root=download_path)
        data = dataset[0]
        num_classes = dataset.num_classes

    gcn_norm = T.GCNNorm()

    edge_index_raw = data.edge_index
    edge_weight_raw = getattr(data, "edge_weight", None)

    train_edge_index = None
    train_edge_weight = None
    if build_train_adj and hasattr(data, "train_mask"):
        train_mask = torch.as_tensor(data.train_mask).reshape(-1).to(torch.bool)
        row = edge_index_raw[0]
        col = edge_index_raw[1]
        edge_mask = train_mask[row] & train_mask[col]

        train_edge_index_raw = edge_index_raw[:, edge_mask]
        if edge_weight_raw is not None:
            train_edge_weight_raw = torch.as_tensor(edge_weight_raw)[edge_mask]
        else:
            train_edge_weight_raw = None

        train_data = Data(edge_index=train_edge_index_raw, num_nodes=data.num_nodes)
        if train_edge_weight_raw is not None:
            train_data.edge_weight = train_edge_weight_raw
        train_data = gcn_norm.forward(train_data)
        train_edge_index = train_data.edge_index
        train_edge_weight = train_data.edge_weight

    data = gcn_norm.forward(data)
    if train_edge_index is not None and train_edge_weight is not None:
        data.edge_index_train = train_edge_index
        data.edge_weight_train = train_edge_weight

    return (data, num_classes)


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


def _make_norm(norm: str, channels: int):
    if norm == "none":
        return None
    if norm == "layer":
        return LayerNorm(channels, elementwise_affine=True)
    if norm == "batch":
        return BatchNorm1d(channels)
    raise ValueError(f"Unknown norm='{norm}'")

class GNNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, convmodel="gcn", norm="layer"):
        super().__init__()
        self.norm = _make_norm(norm, in_channels)
        if convmodel == "gcn":
            self.conv = GCNConv(in_channels, out_channels, normalize=False, bias=False)
            self._takes_edge_weight = True
        elif convmodel == "gat":
            self.conv = GATConv(in_channels, out_channels // 8, heads=8)
            self._takes_edge_weight = False
        elif convmodel == "sage":
            self.conv = SAGEConv(in_channels, out_channels)
            self._takes_edge_weight = False
        else:
            raise ValueError(f"Unknown model='{convmodel}'")
        self.dropout = dropout
        self.norm_type = norm
    def forward(self, x, edge_index, edge_weight=None):
        if self.norm is not None:
            x = self.norm(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self._takes_edge_weight:
            return self.conv(x, edge_index, edge_weight=edge_weight)
        return self.conv(x, edge_index)
    
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=512, num_layers=3,
                 dropout=0.3, convmodel="gcn", norm="layer", jk=False, res=True):
        super().__init__()

        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.norm = _make_norm(norm, hidden_channels)
        self.norm_type = norm
        self.jk = jk
        self.res = res
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GNNConv(
                hidden_channels,
                hidden_channels,
                dropout,
                convmodel=convmodel,
                norm=norm,
            )
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_weight=None):
        x_final = 0
        x = self.lin1(x)
        x_final += x
        for (conv) in self.convs:
            if self.res:
                x = conv(x, edge_index, edge_weight=edge_weight) + x
            else:
                x = conv(x, edge_index, edge_weight=edge_weight)
            x_final += x
        if self.norm is not None:
            x = self.norm(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.jk:
            x = x_final
        else:
            pass

        return self.lin2(x)

def train(model, optimizer, input_features, adj, labels, train_mask):
    model.train()

    optimizer.zero_grad()

    output = model(input_features, adj)

    if labels.ndim == 2:
        if train_mask is None:
            out = output
            y = labels
        else:
            out = output[train_mask]
            y = labels[train_mask]

        y = y.to(dtype=torch.float)

        # OGB proteins may contain missing labels (NaN). Compute loss only on
        # finite targets to avoid propagating NaNs.
        labeled = torch.isfinite(y)
        if labeled.any():
            per_entry = F.binary_cross_entropy_with_logits(
                out, torch.nan_to_num(y, nan=0.0), reduction="none"
            )
            loss = per_entry[labeled].mean()
        else:
            loss = out.sum() * 0.0
    else:
        if train_mask is not None:
            loss = F.cross_entropy(output[train_mask], labels[train_mask])
        else:
            loss = F.cross_entropy(output, labels)

    loss.backward()

    optimizer.step()

    return loss


def _compute_split_metrics(pred, labels, mask, num_classes):
    if mask is None:
        return None

    valid = mask & (labels >= 0) & (pred >= 0)
    pred_eval = pred[valid]
    labels_eval = labels[valid]

    total = int(valid.sum().item())
    correct = int((pred_eval == labels_eval).sum().item())
    acc = correct / max(total, 1)

    tp = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
    fp = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
    fn = torch.zeros(num_classes, dtype=torch.long, device=pred.device)

    if total > 0:
        match = pred_eval == labels_eval
        tp = torch.bincount(labels_eval[match], minlength=num_classes).to(torch.long)
        pred_cnt = torch.bincount(pred_eval, minlength=num_classes).to(torch.long)
        true_cnt = torch.bincount(labels_eval, minlength=num_classes).to(torch.long)
        fp = pred_cnt - tp
        fn = true_cnt - tp

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
        "total": total,
    }


@torch.no_grad()
def evaluate(model, input_features, adj, labels, masks, num_classes):
    model.eval()
    logits = model(input_features, adj)

    if masks is None:
        return {}

    # Multi-label case (e.g., ogbn-proteins): use OGB ROC-AUC evaluator on
    # raw scores (logits) without thresholding.
    if labels.ndim == 2:
        try:
            from ogb.nodeproppred import Evaluator
        except Exception as exc:
            raise RuntimeError(
                "Failed to import OGB Evaluator (ogb.nodeproppred). Ensure the `ogb` package and its dependencies are installed."
            ) from exc

        evaluator = Evaluator(name="ogbn-proteins")
        results = {}
        for split in ("train", "val", "test"):
            mask = masks.get(split)
            if mask is None:
                results[split] = None
                continue
            results[split] = evaluator.eval(
                {"y_true": labels[mask], "y_pred": logits[mask]}
            )
        return results

    # Single-label multi-class: accuracy/F1 on argmax predictions.
    pred = logits.argmax(dim=1)
    results = {}
    for split in ("train", "val", "test"):
        results[split] = _compute_split_metrics(
            pred,
            labels,
            masks.get(split),
            num_classes,
        )
    return results


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    data, num_classes = get_dataset(
        args.dataset, args.download_path, build_train_adj=args.train_adj
    )
    num_input_features = data.x.shape[1]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if data.y.ndim == 2 and data.y.size(-1) > 1:
        labels = data.y.to(device)
    else:
        labels = data.y.reshape(-1).to(torch.long).to(device)

    features_local = data.x.to(device)

    model = Net(
        num_input_features,
        num_classes,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        convmodel=args.convmodel,
        norm=args.norm,
        jk=args.jk,
        res=args.res,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.lr,
    )

    adj_full = torch.sparse_coo_tensor(
        data.edge_index,
        data.edge_weight,
        (data.x.shape[0], data.x.shape[0]),
    )
    adj_full = adj_full.to_sparse_csr()
    adj_full = adj_full.to(device)

    adj_train = None
    if args.train_adj and hasattr(data, "edge_index_train") and hasattr(data, "edge_weight_train"):
        adj_train = torch.sparse_coo_tensor(
            data.edge_index_train,
            data.edge_weight_train,
            (data.x.shape[0], data.x.shape[0]),
        )
        adj_train = adj_train.to_sparse_csr()
        adj_train = adj_train.to(device)

    masks = None
    train_mask = getattr(data, "train_mask", None)
    val_mask = getattr(data, "val_mask", None)
    test_mask = getattr(data, "test_mask", None)
    if train_mask is not None or val_mask is not None or test_mask is not None:
        masks = {
            "train": train_mask.to(device) if train_mask is not None else None,
            "val": val_mask.to(device) if val_mask is not None else None,
            "test": test_mask.to(device) if test_mask is not None else None,
        }
        if all(v is None for v in masks.values()):
            masks = None

    do_eval = bool(args.eval) and (masks is not None)
    if args.eval and not do_eval:
        print(
            "[warn] --eval was set but no train/val/test masks were found in the dataset; skipping evaluation."
        )

    best_valid = None
    if do_eval:
        if labels.ndim == 2:
            best_valid = BestValidTracker(metric_name="rocauc", percent_scale=True)
        else:
            best_valid = BestValidTracker(metric_name="acc", percent_scale=True)

    if args.train_adj and adj_train is None:
        print(
            "[warn] --train_adj was set but train-induced adjacency was not found; falling back to full-graph adjacency."
        )
    train_adj = adj_train if args.train_adj and adj_train is not None else adj_full

    losses = []
    for i in range(args.num_epochs):
        loss = train(
            model,
            optimizer,
            features_local,
            train_adj,
            labels,
            masks.get("train") if masks is not None else None,
        )
        losses.append(loss.item())
        log = "Epoch: {:03d}, Train Loss: {:.4f}"
        print(log.format(i, loss))

        if do_eval and args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            metrics = evaluate(
                model,
                features_local,
                adj_full,
                labels,
                masks,
                num_classes,
            )
            for split, m in metrics.items():
                if m is None:
                    continue
                if labels.ndim == 2:
                    print("{}: rocauc {:.4f}".format(split.upper(), m["rocauc"]))
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
                    if labels.ndim == 2:
                        best_valid.add(
                            train_m["rocauc"], val_m["rocauc"], test_m["rocauc"]
                        )
                    else:
                        best_valid.add(train_m["acc"], val_m["acc"], test_m["acc"])

    if best_valid is not None:
        best_valid.print_ogb_style()
