# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import os

import torch
from axonn import axonn as ax

_TAG_DTYPE = torch.int32
_TAG_MAX = (2**31) - 1
_COL_MAP_CACHE = {}
_USE_COLMAP_CACHE = os.environ.get("PLEXUS_COMPACT_COLMAP_CACHE", "1").lower() not in (
    "0",
    "false",
    "off",
)
_USE_ADJ_T_TRANSPOSE = os.environ.get("PLEXUS_COMPACT_TRANSPOSE_T", "1").lower() not in (
    "0",
    "false",
    "off",
)
_USE_ADJ_T_DIRECT = os.environ.get("PLEXUS_COMPACT_ADJ_T_DIRECT", "1").lower() not in (
    "0",
    "false",
    "off",
)


def _get_tag_cache(device: torch.device, dim: int, role: str, scope: Optional[int]):
    key = (str(device), int(dim), role, int(scope) if scope is not None else None)
    cache = _COL_MAP_CACHE.get(key)
    if cache is None or cache["map"].numel() != dim or cache["map"].device != device:
        cache = {
            "map": torch.empty(dim, dtype=torch.long, device=device),
            "tag": torch.zeros(dim, dtype=_TAG_DTYPE, device=device),
            "tag_val": 0,
        }
        _COL_MAP_CACHE[key] = cache
    return cache


def _prepare_col_map(cache, idx: torch.Tensor):
    if idx.numel() == 0:
        return cache["map"], cache["tag"], -1

    tag = cache["tag_val"] + 1
    if tag >= _TAG_MAX:
        cache["tag"].zero_()
        tag = 1
    cache["tag_val"] = tag

    cache["map"][idx] = torch.arange(idx.numel(), device=cache["map"].device)
    cache["tag"][idx] = tag
    return cache["map"], cache["tag"], tag

def local_indices_from_sample(
    sample_idx: torch.Tensor,
    start: int,
    stop: int,
    device: Optional[torch.device] = None,
    assume_sorted: bool = False,
) -> torch.Tensor:
    if device is None:
        device = sample_idx.device
    if start >= stop or sample_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if assume_sorted:
        sample_dev = sample_idx.to(device=device) if sample_idx.device != device else sample_idx
        bounds = torch.tensor(
            [start, stop], device=sample_dev.device, dtype=sample_dev.dtype
        )
        idx = torch.searchsorted(sample_dev, bounds, right=False)
        left = int(idx[0].item())
        right = int(idx[1].item())
        if right <= left:
            return torch.empty(0, dtype=torch.long, device=device)
        local = sample_dev[left:right] - int(start)
        return local
    local = sample_idx[(sample_idx >= start) & (sample_idx < stop)] - start
    if local.numel() > 1:
        local, _ = torch.sort(local)
    return local.to(device=device)


def build_layout_indices(
    sample_idx: torch.Tensor,
    starts: Sequence[int],
    stops: Sequence[int],
    device: Optional[torch.device] = None,
    assume_sorted: bool = False,
) -> List[torch.Tensor]:
    return [
        local_indices_from_sample(
            sample_idx, s, t, device=device, assume_sorted=assume_sorted
        )
        for s, t in zip(starts, stops)
    ]


def compute_steps_per_epoch(
    num_nodes: int, batch_size: Optional[int] = None, ratio: Optional[float] = None
) -> int:
    if batch_size is None and ratio is None:
        return 1
    if batch_size is not None and ratio is not None:
        raise ValueError("Provide only one of batch_size or ratio.")
    if ratio is not None:
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be in (0, 1].")
        batch_size = max(1, int(math.floor(num_nodes * ratio)))
    batch_size = max(1, min(int(batch_size), num_nodes))
    return max(1, int(math.ceil(num_nodes / batch_size)))


def sample_nodes(
    num_nodes: int,
    batch_size: Optional[int] = None,
    ratio: Optional[float] = None,
    seed: int = 0,
    step: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if batch_size is None and ratio is None:
        raise ValueError("Provide batch_size or ratio for sampling.")
    if batch_size is not None and ratio is not None:
        raise ValueError("Provide only one of batch_size or ratio.")
    if ratio is not None:
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be in (0, 1].")
        batch_size = max(1, int(math.floor(num_nodes * ratio)))
    batch_size = max(1, min(int(batch_size), num_nodes))

    if device is None:
        gen_device = torch.device("cpu")
        out_device = None
    else:
        out_device = device
        gen_device = device if device.type == "cuda" else torch.device("cpu")

    gen = torch.Generator(device=gen_device)
    gen.manual_seed(int(seed) + int(step))
    perm = torch.randperm(num_nodes, generator=gen, device=out_device)
    return perm[:batch_size]


def _make_generator(seed: int, device: Optional[torch.device]):
    if device is None:
        gen_device = torch.device("cpu")
        out_device = None
    else:
        out_device = device
        gen_device = device if device.type == "cuda" else torch.device("cpu")
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(int(seed))
    return gen, out_device


def sample_nodes_epoch_perm(
    num_nodes: int,
    batch_size: int,
    seed: int,
    epoch: int,
    chunk: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Chunk `chunk` of a per-epoch random permutation (without-replacement
    across steps). P[u,v in the same chunk] = (B-1)/(N-1) exactly, so the
    uniform 1/p edge rescaling remains valid. Caller must keep
    chunk < floor(num_nodes / batch_size)."""
    gen, out_device = _make_generator(int(seed) + int(epoch) * 7919, device)
    perm = torch.randperm(num_nodes, generator=gen, device=out_device)
    start = int(chunk) * int(batch_size)
    return perm[start : start + int(batch_size)]


def sample_nodes_hub_anchor(
    num_nodes: int,
    batch_size: int,
    hub_ids: torch.Tensor,
    hub_mask: torch.Tensor,
    seed: int,
    step: int,
    device: Optional[torch.device] = None,
    epoch: Optional[int] = None,
    chunk: Optional[int] = None,
) -> torch.Tensor:
    """S = fixed top-degree hub set (always included) plus a uniform sample of
    the remaining nodes. With epoch/chunk given, the uniform part is a chunk of
    a per-epoch permutation over non-hub nodes instead of an independent draw."""
    rest_size = int(batch_size) - int(hub_ids.numel())
    if rest_size <= 0:
        return hub_ids
    if epoch is not None and chunk is not None:
        gen, out_device = _make_generator(int(seed) + int(epoch) * 7919, device)
    else:
        gen, out_device = _make_generator(int(seed) + int(step), device)
    perm = torch.randperm(num_nodes, generator=gen, device=out_device)
    rest_all = perm[~hub_mask[perm]]
    if epoch is not None and chunk is not None:
        start = int(chunk) * rest_size
        rest = rest_all[start : start + rest_size]
    else:
        rest = rest_all[:rest_size]
    return torch.cat([hub_ids.to(rest.device), rest])


def sample_nodes_train_hub(
    num_nodes: int,
    train_ids: torch.Tensor,
    target_chunk_size: int,
    hub_ids: torch.Tensor,
    uniform_size: int,
    seed: int,
    epoch: int,
    chunk: int,
    step: int,
    device: Optional[torch.device] = None,
    src_seed: Optional[int] = None,
) -> torch.Tensor:
    """Train-anchored batch: one chunk of a per-epoch permutation over the
    TRAIN nodes (loss targets; every train node is a target exactly once per
    epoch) + fixed hub anchors + a uniform draw of extra aggregation sources.
    Work per epoch scales with |train| + sources instead of |V|.
    src_seed lets DP groups draw DIFFERENT uniform sources (targets must keep
    the shared seed so their chunks stay disjoint across groups)."""
    gen_t, out_device = _make_generator(int(seed) + int(epoch) * 7919, device)
    perm_t = torch.randperm(int(train_ids.numel()), generator=gen_t, device=out_device)
    start = int(chunk) * int(target_chunk_size)
    targets = train_ids[perm_t[start : start + int(target_chunk_size)]]
    parts = [targets, hub_ids.to(targets.device)]
    if uniform_size > 0:
        u_seed = int(seed if src_seed is None else src_seed)
        gen_u, _ = _make_generator(u_seed + 31 + int(step), device)
        perm_u = torch.randperm(num_nodes, generator=gen_u, device=out_device)
        parts.append(perm_u[: int(uniform_size)])
    return torch.unique(torch.cat(parts))


def sample_nodes_degree_gumbel(
    log_weights: torch.Tensor,
    batch_size: int,
    seed: int,
    step: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Weighted sampling without replacement via the Gumbel top-k trick:
    keys = log w + Gumbel noise, take the top-B. All ranks derive the same
    sample from the shared seed."""
    gen, _ = _make_generator(int(seed) + int(step), device)
    u = torch.rand(
        log_weights.shape[0],
        generator=gen,
        device=log_weights.device,
        dtype=torch.float32,
    )
    eps = 1e-20
    gumbel = -torch.log(-torch.log(u.clamp_min(eps)).clamp_min(eps))
    keys = log_weights + gumbel
    # stable sort, not topk: CUDA topk breaks float ties nondeterministically
    # (atomic scheduling), which would let ranks derive different samples and
    # deadlock the TP group's collectives
    _, order = torch.sort(keys, descending=True, stable=True)
    return order[: int(batch_size)]


def poisson_inclusion_probs(
    weights: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Approximate inclusion probability of weighted without-replacement
    sampling with budget B: pi_v = 1 - (1 - w_v / W)^B (exact for the
    Poissonized process). Used for 1/pi edge rescaling and loss reweighting."""
    w = weights.to(torch.float64)
    frac = (w / w.sum()).clamp(max=1.0 - 1e-12)
    pi = 1.0 - torch.exp(float(batch_size) * torch.log1p(-frac))
    return pi.clamp(min=1e-12, max=1.0).to(torch.float32)


def compact_subgraph_csr_native(
    csr: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    row_offset: int = 0,
    col_offset: int = 0,
    edge_scale: Optional[float] = None,
    col_map: Optional[torch.Tensor] = None,
    col_tag: Optional[torch.Tensor] = None,
    col_tag_value: Optional[int] = None,
    self_col_compact: Optional[torch.Tensor] = None,
    return_edges: bool = False,
    enable_timers: bool = True,
    col_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = csr.device
    row_count = int(row_idx.numel())
    col_count = int(col_idx.numel())

    if row_count == 0 or col_count == 0 or csr._nnz() == 0:
        empty = torch.sparse_csr_tensor(
            torch.zeros(row_count + 1, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=csr.dtype, device=device),
            size=(row_count, col_count),
            device=device,
        )
        if return_edges:
            return empty, torch.empty(0, dtype=torch.long, device=device), torch.empty(
                0, dtype=torch.long, device=device
            ), torch.empty(0, dtype=csr.dtype, device=device)
        return empty

    if enable_timers:
        ax.get_timers().start("compact csr: indices")
    crow = csr.crow_indices()
    col = csr.col_indices()
    val = csr.values()
    if enable_timers:
        ax.get_timers().stop("compact csr: indices")

    if enable_timers:
        ax.get_timers().start("compact csr: row ptrs")
    row_ptrs = crow[row_idx]
    row_ptrs_next = crow[row_idx + 1]
    row_counts = row_ptrs_next - row_ptrs
    total = int(row_counts.sum().item())
    if enable_timers:
        ax.get_timers().stop("compact csr: row ptrs")
    if total == 0:
        empty = torch.sparse_csr_tensor(
            torch.zeros(row_count + 1, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=csr.dtype, device=device),
            size=(row_count, col_count),
            device=device,
        )
        if return_edges:
            return empty, torch.empty(0, dtype=torch.long, device=device), torch.empty(
                0, dtype=torch.long, device=device
            ), torch.empty(0, dtype=csr.dtype, device=device)
        return empty

    if enable_timers:
        ax.get_timers().start("compact csr: nnz index")
    prefix = torch.cumsum(row_counts, dim=0)
    nnz_range = torch.arange(total, device=device, dtype=prefix.dtype)
    row_ids = torch.searchsorted(prefix, nnz_range, right=True)
    row_start = prefix - row_counts
    nnz_idx = row_ptrs[row_ids] + (nnz_range - row_start[row_ids])
    if enable_timers:
        ax.get_timers().stop("compact csr: nnz index")

    if enable_timers:
        ax.get_timers().start("compact csr: select nnz")
    col_sel = col[nnz_idx]
    val_sel = val[nnz_idx]
    if enable_timers:
        ax.get_timers().stop("compact csr: select nnz")

    if enable_timers:
        ax.get_timers().start("compact csr: col map")
    if col_map is None:
        col_dim = int(csr.size(1))
        col_map = torch.full(
            (col_dim,),
            -1,
            dtype=torch.long,
            device=device,
        )
        if col_count > 0:
            col_map[col_idx] = torch.arange(col_count, device=device)
    elif col_map.device != device:
        col_map = col_map.to(device=device)
        if col_tag is not None and col_tag.device != device:
            col_tag = col_tag.to(device=device)
    col_compact = col_map[col_sel]
    if col_tag is not None and col_tag_value is not None:
        keep = col_tag[col_sel] == int(col_tag_value)
    else:
        keep = col_compact >= 0
    if enable_timers:
        ax.get_timers().stop("compact csr: col map")
    if keep.any():
        col_compact = col_compact[keep]
        val_compact = val_sel[keep]
        row_ids = row_ids[keep]
    else:
        col_compact = col_sel[:0]
        val_compact = val_sel[:0]
        row_ids = row_ids[:0]

    if enable_timers:
        ax.get_timers().start("compact csr: edge scale")
    if (
        (col_scale is not None or (edge_scale is not None and edge_scale != 1.0))
        and val_compact.numel() > 0
    ):
        if self_col_compact is not None and self_col_compact.numel() > 0:
            if self_col_compact.device != device:
                self_col_compact = self_col_compact.to(device=device)
            self_cols = self_col_compact[row_ids]
            non_self = col_compact != self_cols
        else:
            row_global = row_idx[row_ids] + int(row_offset)
            col_global = col_idx[col_compact] + int(col_offset)
            non_self = row_global != col_global
        if non_self.any():
            if col_scale is not None:
                # Per-node inverse inclusion probability, indexed by global
                # column id. Self-loops keep their original weight.
                if col_scale.device != device:
                    col_scale = col_scale.to(device=device)
                col_global_scale = col_idx[col_compact] + int(col_offset)
                factors = col_scale[col_global_scale].to(val_compact.dtype)
                val_compact[non_self] = val_compact[non_self] * factors[non_self]
            else:
                val_compact[non_self] = val_compact[non_self] * float(edge_scale)
    if enable_timers:
        ax.get_timers().stop("compact csr: edge scale")

    if enable_timers:
        ax.get_timers().start("compact csr: crow")
    if row_ids.numel() > 0:
        row_nnz_keep = torch.bincount(
            row_ids,
            minlength=row_count,
        ).to(torch.int64)
    else:
        row_nnz_keep = torch.zeros(row_count, dtype=torch.int64, device=device)
    crow_new = torch.zeros(row_count + 1, dtype=torch.int64, device=device)
    crow_new[1:] = torch.cumsum(row_nnz_keep, dim=0)
    if enable_timers:
        ax.get_timers().stop("compact csr: crow")

    csr_out = torch.sparse_csr_tensor(
        crow_new,
        col_compact,
        val_compact,
        size=(row_count, col_count),
        device=device,
        dtype=csr.dtype,
    )
    if return_edges:
        return csr_out, row_ids, col_compact, val_compact
    return csr_out


def _build_transpose_csr(
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    vals: torch.Tensor,
    num_rows: int,
    num_cols: int,
) -> torch.Tensor:
    device = vals.device
    if num_rows == 0 or num_cols == 0 or vals.numel() == 0:
        return torch.sparse_csr_tensor(
            torch.zeros(num_rows + 1, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=vals.dtype, device=device),
            size=(num_rows, num_cols),
            device=device,
            dtype=vals.dtype,
        )

    row_ids_t = col_ids
    col_ids_t = row_ids
    order = torch.argsort(row_ids_t)
    row_sorted = row_ids_t[order]
    col_sorted = col_ids_t[order]
    val_sorted = vals[order]
    row_counts = torch.bincount(row_sorted, minlength=num_rows).to(torch.int64)
    crow = torch.zeros(num_rows + 1, dtype=torch.int64, device=device)
    crow[1:] = torch.cumsum(row_counts, dim=0)
    return torch.sparse_csr_tensor(
        crow,
        col_sorted,
        val_sorted,
        size=(num_rows, num_cols),
        device=device,
        dtype=vals.dtype,
    )


def _make_col_map(
    dim: int,
    idx: torch.Tensor,
    device: torch.device,
    role: str,
    scope: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
    if _USE_COLMAP_CACHE:
        col_cache = _get_tag_cache(device, dim, role, scope)
        return _prepare_col_map(col_cache, idx)
    col_map = torch.full(
        (dim,),
        -1,
        dtype=torch.long,
        device=device,
    )
    if idx.numel() > 0:
        col_map[idx] = torch.arange(idx.numel(), device=device)
    return col_map, None, None


def _build_self_col(
    base_idx: torch.Tensor,
    delta: int,
    other_dim: int,
    other_map: torch.Tensor,
    other_tag: Optional[torch.Tensor],
    other_tag_val: Optional[int],
) -> torch.Tensor:
    count = int(base_idx.numel())
    device = base_idx.device
    self_col = torch.full(
        (count,),
        -1,
        dtype=torch.long,
        device=device,
    )
    if count == 0:
        return self_col
    mapped_idx = base_idx + int(delta)
    valid = (mapped_idx >= 0) & (mapped_idx < other_dim)
    if not valid.any():
        return self_col
    mapped_idx_valid = mapped_idx[valid]
    mapped = other_map[mapped_idx_valid]
    if other_tag is not None and other_tag_val is not None:
        valid2 = other_tag[mapped_idx_valid] == int(other_tag_val)
        if valid2.any():
            valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
            self_col[valid_idx[valid2]] = mapped[valid2]
    else:
        self_col[valid] = mapped
    return self_col


def _prepare_compact_layout(
    adj_shards: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    row_indices: Sequence[torch.Tensor],
    col_indices: Sequence[torch.Tensor],
    row_starts: Sequence[int],
    col_starts: Sequence[int],
    edge_scale: Optional[float],
    col_scale: Optional[torch.Tensor] = None,
) -> tuple[
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[Optional[int]],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[Optional[int]],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
]:
    num_layouts = len(row_indices)
    col_maps_cols: List[Optional[torch.Tensor]] = [None] * num_layouts
    col_tags_cols: List[Optional[torch.Tensor]] = [None] * num_layouts
    col_tag_vals_cols: List[Optional[int]] = [None] * num_layouts
    col_maps_rows: List[Optional[torch.Tensor]] = [None] * num_layouts
    col_tags_rows: List[Optional[torch.Tensor]] = [None] * num_layouts
    col_tag_vals_rows: List[Optional[int]] = [None] * num_layouts
    self_cols_rows: List[Optional[torch.Tensor]] = [None] * num_layouts
    self_cols_cols: List[Optional[torch.Tensor]] = [None] * num_layouts
    need_self = (edge_scale is not None and edge_scale != 1.0) or col_scale is not None
    need_adj_t_build = not _USE_ADJ_T_TRANSPOSE and not _USE_ADJ_T_DIRECT
    for layout_idx in range(num_layouts):
        row_idx = row_indices[layout_idx]
        col_idx = col_indices[layout_idx]
        row_count = int(row_idx.numel())
        col_count = int(col_idx.numel())
        if row_count == 0 and col_count == 0:
            continue

        adj_ref = adj_shards[layout_idx][0]
        row_dim = int(adj_ref.size(0))
        col_dim = int(adj_ref.size(1))
        device = adj_ref.device

        col_map_cols, col_tag_cols, col_tag_val_cols = _make_col_map(
            col_dim,
            col_idx,
            device,
            role="col",
            scope=layout_idx,
        )
        col_maps_cols[layout_idx] = col_map_cols
        col_tags_cols[layout_idx] = col_tag_cols
        col_tag_vals_cols[layout_idx] = col_tag_val_cols

        if need_adj_t_build:
            col_map_rows, col_tag_rows, col_tag_val_rows = _make_col_map(
                row_dim,
                row_idx,
                device,
                role="row",
                scope=layout_idx,
            )
            col_maps_rows[layout_idx] = col_map_rows
            col_tags_rows[layout_idx] = col_tag_rows
            col_tag_vals_rows[layout_idx] = col_tag_val_rows

        if need_self:
            delta = int(row_starts[layout_idx]) - int(col_starts[layout_idx])
            if row_count > 0:
                self_cols_rows[layout_idx] = _build_self_col(
                    row_idx,
                    delta,
                    col_dim,
                    col_map_cols,
                    col_tag_cols,
                    col_tag_val_cols,
                )
            if need_adj_t_build and col_count > 0:
                self_cols_cols[layout_idx] = _build_self_col(
                    col_idx,
                    -delta,
                    row_dim,
                    col_maps_rows[layout_idx],
                    col_tags_rows[layout_idx],
                    col_tag_vals_rows[layout_idx],
                )
    return (
        col_maps_cols,
        col_tags_cols,
        col_tag_vals_cols,
        col_maps_rows,
        col_tags_rows,
        col_tag_vals_rows,
        self_cols_rows,
        self_cols_cols,
    )


def _build_compact_shard(
    adj: torch.Tensor,
    adj_t: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    row_start: int,
    col_start: int,
    edge_scale: Optional[float],
    col_scale: Optional[torch.Tensor],
    col_map_cols: Optional[torch.Tensor],
    col_tag_cols: Optional[torch.Tensor],
    col_tag_val_cols: Optional[int],
    col_map_rows: Optional[torch.Tensor],
    col_tag_rows: Optional[torch.Tensor],
    col_tag_val_rows: Optional[int],
    self_cols_rows: Optional[torch.Tensor],
    self_cols_cols: Optional[torch.Tensor],
    enable_timers: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _USE_ADJ_T_DIRECT:
        adj_new, row_ids, col_compact, val_compact = compact_subgraph_csr_native(
            adj,
            row_idx,
            col_idx,
            row_offset=row_start,
            col_offset=col_start,
            edge_scale=edge_scale,
            col_map=col_map_cols,
            col_tag=col_tag_cols,
            col_tag_value=col_tag_val_cols,
            self_col_compact=self_cols_rows,
            return_edges=True,
            enable_timers=enable_timers,
            col_scale=col_scale,
        )
        if enable_timers:
            ax.get_timers().start("compact csr: adjt direct")
        adj_t_new = _build_transpose_csr(
            row_ids,
            col_compact,
            val_compact,
            int(col_idx.numel()),
            int(row_idx.numel()),
        )
        if enable_timers:
            ax.get_timers().stop("compact csr: adjt direct")
        return adj_new, adj_t_new

    adj_new = compact_subgraph_csr_native(
        adj,
        row_idx,
        col_idx,
        row_offset=row_start,
        col_offset=col_start,
        edge_scale=edge_scale,
        col_map=col_map_cols,
        col_tag=col_tag_cols,
        col_tag_value=col_tag_val_cols,
        self_col_compact=self_cols_rows,
        enable_timers=enable_timers,
        col_scale=col_scale,
    )
    if _USE_ADJ_T_TRANSPOSE:
        if enable_timers:
            ax.get_timers().start("compact csr: transpose")
        adj_t_new = adj_new.transpose(0, 1).to_sparse_csr()
        if enable_timers:
            ax.get_timers().stop("compact csr: transpose")
        return adj_new, adj_t_new

    if col_scale is not None:
        raise NotImplementedError(
            "col_scale requires PLEXUS_COMPACT_ADJ_T_DIRECT=1 or "
            "PLEXUS_COMPACT_TRANSPOSE_T=1; the separate adj_t build path would "
            "apply per-column factors to the wrong axis."
        )
    adj_t_new = compact_subgraph_csr_native(
        adj_t,
        col_idx,
        row_idx,
        row_offset=col_start,
        col_offset=row_start,
        edge_scale=edge_scale,
        col_map=col_map_rows,
        col_tag=col_tag_rows,
        col_tag_value=col_tag_val_rows,
        self_col_compact=self_cols_cols,
        enable_timers=enable_timers,
    )
    return adj_new, adj_t_new




def build_compact_adj_shards(
    adj_shards: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    row_indices: Sequence[torch.Tensor],
    col_indices: Sequence[torch.Tensor],
    row_starts: Sequence[int],
    col_starts: Sequence[int],
    edge_scale: Optional[float] = None,
    enable_timers: bool = True,
    col_scale: Optional[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if len(row_indices) == 0:
        return list(adj_shards)
    if enable_timers:
        ax.get_timers().start("compact layout prep")
    (
        col_maps_cols,
        col_tags_cols,
        col_tag_vals_cols,
        col_maps_rows,
        col_tags_rows,
        col_tag_vals_rows,
        self_cols_rows,
        self_cols_cols,
    ) = _prepare_compact_layout(
        adj_shards,
        row_indices,
        col_indices,
        row_starts,
        col_starts,
        edge_scale,
        col_scale=col_scale,
    )
    if enable_timers:
        ax.get_timers().stop("compact layout prep")

    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    num_layouts = len(row_indices)
    for shard_idx, (adj, adj_t) in enumerate(adj_shards):
        layout_idx = shard_idx % num_layouts
        row_idx = row_indices[layout_idx]
        col_idx = col_indices[layout_idx]
        adj_new, adj_t_new = _build_compact_shard(
            adj,
            adj_t,
            row_idx,
            col_idx,
            row_starts[layout_idx],
            col_starts[layout_idx],
            edge_scale,
            col_scale,
            col_maps_cols[layout_idx],
            col_tags_cols[layout_idx],
            col_tag_vals_cols[layout_idx],
            col_maps_rows[layout_idx],
            col_tags_rows[layout_idx],
            col_tag_vals_rows[layout_idx],
            self_cols_rows[layout_idx],
            self_cols_cols[layout_idx],
            enable_timers=enable_timers,
        )
        out.append((adj_new, adj_t_new))
    return out
