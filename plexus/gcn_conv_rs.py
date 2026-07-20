# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

"""
Half-collective ("reduce-scatter / lazy all-gather") 3D parallel GCNConv.

Motivation (see analysis/gcn_parallel_cost.tex): the existing GCNConv keeps
activations replicated over the feature axis and pays four node-proportional
ALL-REDUCEs per layer.  Since AR = RS + AG in cost, replacing every AR by a
reduce-scatter right after the producing matmul and a lazy all-gather right
before the consuming matmul cuts the moved volume by ~25-33% on balanced
grids, removes the Gf-fold redundancy of the element-wise operators between
layers, and shrinks saved activations.

Layout contract
---------------
Grid axes per layer (same rotation as GCNConv, use_3d_linear=True):
    layer l%3==0: (outer=c, inner=f, depth=r) = ("x","z","y")
    layer l%3==1:                              ("y","x","z")
    layer l%3==2:                              ("z","y","x")
where r = A's row axis, c = A's column axis, f = feature axis.

Steady-state activation layout between layers:每个 GPU 持有全局行的一段
连续块 (nested split c > f > r), full feature columns:
    x_local: (N / P) x F        with row block index (rank_c, rank_f, rank_r)

Because the rotation maps (r,c,f)_l -> (c,f,r)_{l+1}, the OUTPUT nesting of
layer l, (r > c > f)_l, is EXACTLY the INPUT nesting (c > f > r)_{l+1} of
layer l+1: the chain reshards for free.  A shards are the same 2D blocks the
existing DataLoader produces (contiguous node ranges), no column permutation
is required.

Per-layer schedule (volumes per GPU in words, see the analysis doc):

  fwd  1. all-to-all over f  (split cols into Gf blocks)   NF(Gf-1)/(P*Gf)
       2. all-gather rows over r (+ block permute)         NF(Gr-1)/P
          -> H: (N/Gc) x (F/Gf)
       3. AGG_p = spmm(A_loc, H)          (N/Gr) x (F/Gf)  partial over c
       4. reduce-scatter rows over c                       NF(Gc-1)/P
       5. OUT_p = AGG @ W_loc             W_loc: (F/Gf) x F_out, partial f
       6. reduce-scatter rows over f                       NF(Gf-1)/P
          -> OUT: (N/P) x F_out   (steady state of layer l+1)
  bwd  1. all-gather rows over f                           NF(Gf-1)/P
       2. grad_W = AGG_s^T @ g            AR over c and r  (O(F^2/Gf))
       3. grad_AGG = g @ W_loc^T          exact, no comm
       4. all-gather rows over c                           NF(Gc-1)/P
       5. gH_p = spmm(A_T_loc, .)         partial over r
       6. reduce-scatter rows over r (+ permute)           NF(Gr-1)/P
       7. all-to-all over f (cols back)                    NF(Gf-1)/(P*Gf)

Requirements: padded N divisible by P (= Gx*Gy*Gz); in/out channels
divisible by Gf (per layer's inner axis).  W is replicated over (r,c) and
split over f on its input dimension; its columns are never split, so the
backward GEMM needs no reduction.
"""

import math
import os

import torch
import torch.distributed as dist
from torch.nn import Parameter
import torch.nn.functional as F
from axonn import axonn as ax
from plexus import plexus as plx
from plexus.utils.matmul_tuning import tuned_matmul
from plexus.utils.general import get_process_groups_info


def _world(group):
    return dist.get_world_size(group) if dist.is_initialized() else 1


def _rank(group):
    return dist.get_rank(group) if dist.is_initialized() else 0


def _wire_dtype():
    """Low-precision wire dtype for the big collectives (reuses the existing
    --allreduce_lowp flags).  None = keep native dtype."""
    if plx.lowp_allreduce:
        return (torch.float16 if plx.lowp_allreduce_dtype == "fp16"
                else torch.bfloat16)
    return None


def _cast(x, dtype):
    if dtype is None or not x.is_floating_point() or x.dtype == dtype:
        return x
    return x.to(dtype)


# ---------------------------------------------------------------------------
# collective helpers (explicit, used inside a single autograd.Function)
# ---------------------------------------------------------------------------

def _all_gather_rows(x, group, wire_dtype=None, async_op=False):
    """(m x n) -> (g*m x n), concatenated in group rank order.

    wire_dtype: optional low-precision dtype used only on the wire; the
    result is cast back to x's dtype.  With async_op returns
    (work, finalize) -- call work.wait() then finalize()."""
    g = _world(group)
    if g == 1:
        assert not async_op
        return x
    orig = x.dtype
    x = _cast(x, wire_dtype).contiguous()
    out = torch.empty(g * x.shape[0], x.shape[1], dtype=x.dtype,
                      device=x.device)
    work = dist.all_gather_into_tensor(out.view(-1), x.view(-1),
                                       group=group, async_op=async_op)

    def _fin():
        return out if out.dtype == orig else out.to(orig)

    return (work, _fin) if async_op else _fin()


def _reduce_scatter_rows(x, group, wire_dtype=None):
    """(g*m x n) partial-summed -> (m x n) exact, my rank's row chunk.
    With wire_dtype the reduction itself runs in low precision (same
    semantics as the existing lowp all-reduce path)."""
    g = _world(group)
    if g == 1:
        return x
    orig = x.dtype
    x = _cast(x, wire_dtype).contiguous()
    out = torch.empty(x.shape[0] // g, x.shape[1], dtype=x.dtype,
                      device=x.device)
    dist.reduce_scatter_tensor(out.view(-1), x.view(-1), group=group)
    return out if out.dtype == orig else out.to(orig)


def _all_to_all_cols_to_rows(x, group, wire_dtype=None, async_op=False):
    """(m x n) full cols -> (g*m x n/g): keep my column block, receive the
    same column block of the other ranks' rows (received in rank order)."""
    g = _world(group)
    if g == 1:
        assert not async_op
        return x
    orig = x.dtype
    x = _cast(x, wire_dtype)
    m, n = x.shape
    # send buffer: rank j's column block, laid out contiguously
    send = x.view(m, g, n // g).permute(1, 0, 2).contiguous()
    recv = torch.empty_like(send)
    work = dist.all_to_all_single(recv.view(-1), send.view(-1), group=group,
                                  async_op=async_op)

    def _fin():
        out = recv.view(g * m, n // g)
        return out if out.dtype == orig else out.to(orig)

    return (work, _fin) if async_op else _fin()


def _all_to_all_rows_to_cols(x, group, wire_dtype=None):
    """Inverse of _all_to_all_cols_to_rows: (g*m x n/g) -> (m x n)."""
    g = _world(group)
    if g == 1:
        return x
    orig = x.dtype
    x = _cast(x, wire_dtype)
    gm, nb = x.shape
    m = gm // g
    send = x.contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv.view(-1), send.view(-1), group=group)
    out = recv.view(g, m, nb).permute(1, 0, 2).reshape(m, g * nb)
    return out if out.dtype == orig else out.to(orig)


def _permute_blocks(x, g_outer, g_inner):
    """View rows as (g_outer, g_inner, m) blocks and swap the two block
    levels: row block order (a, b) -> (b, a)."""
    if g_outer == 1 or g_inner == 1:
        return x
    gm, n = x.shape
    m = gm // (g_outer * g_inner)
    return (x.view(g_outer, g_inner, m, n).permute(1, 0, 2, 3)
            .reshape(gm, n))


# ---------------------------------------------------------------------------
# uneven-block support (minibatch: sampled counts per coarse block differ)
#
# The scattered nesting is purely hierarchical: only the OUTERMOST level of
# each layer's nesting must match a physical blocking (the A shard's coarse
# row/col blocks).  Inner subdivisions are free choices; all ranks use the
# same deterministic near-equal rule (_splits, remainder on the first
# chunks, matching torch.tensor_split).  Every split size is derivable from
# the A shard's (rows, cols) -- which are shared within each communicator
# group -- so no extra metadata and no divergent branches inside a group.
# ---------------------------------------------------------------------------

def _splits(n, g):
    """Near-equal subdivision of n into g chunks (tensor_split semantics)."""
    base, rem = divmod(int(n), g)
    return [base + (1 if i < rem else 0) for i in range(g)]


# Native pad-to-even path for the uneven collectives (PLEXUS_UNEVEN_NATIVE=0
# falls back to the all-to-all emulation for A/B comparison).
_UNEVEN_NATIVE = os.environ.get("PLEXUS_UNEVEN_NATIVE", "1") != "0"


def _ag_rows_uneven_a2a(x, group, recv_counts, wire_dtype=None):
    """All-gather emulated via all-to-all with a tiled send buffer."""
    g = _world(group)
    orig = x.dtype
    x = _cast(x, wire_dtype).contiguous()
    send = x.repeat(g, 1)
    recv = torch.empty(sum(recv_counts), x.shape[1], dtype=x.dtype,
                       device=x.device)
    dist.all_to_all_single(recv, send,
                           output_split_sizes=list(recv_counts),
                           input_split_sizes=[x.shape[0]] * g, group=group)
    return recv if recv.dtype == orig else recv.to(orig)


def _rs_rows_uneven_a2a(x, group, chunk_counts, wire_dtype=None):
    """Reduce-scatter emulated: all-to-all the chunks, sum locally."""
    g = _world(group)
    orig = x.dtype
    x = _cast(x, wire_dtype).contiguous()
    n_me = chunk_counts[_rank(group)]
    recv = torch.empty(n_me * g, x.shape[1], dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv, x,
                           output_split_sizes=[n_me] * g,
                           input_split_sizes=list(chunk_counts), group=group)
    out = recv.view(g, n_me, x.shape[1]).sum(dim=0)
    return out if out.dtype == orig else out.to(orig)


def _ag_rows_uneven(x, group, recv_counts, wire_dtype=None):
    """All-gather rows with per-rank row counts: pad every shard to the max
    count, native all-gather (ring/LL protocols), slice the padding away.
    Pack/unpack uses only slice copies and cat -- no host-device transfers,
    so nothing here synchronizes the stream."""
    g = _world(group)
    if g == 1:
        return x
    if not _UNEVEN_NATIVE:
        return _ag_rows_uneven_a2a(x, group, recv_counts, wire_dtype)
    orig = x.dtype
    x = _cast(x, wire_dtype).contiguous()
    m = max(recv_counts)
    n_me = x.shape[0]
    if n_me == m:
        send = x
    else:
        send = torch.zeros(m, x.shape[1], dtype=x.dtype, device=x.device)
        send[:n_me] = x
    out = torch.empty(g * m, x.shape[1], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out.view(-1), send.view(-1), group=group)
    if min(recv_counts) == m:
        recv = out
    else:
        recv = torch.cat([out[j * m:j * m + recv_counts[j]]
                          for j in range(g)])
    return recv if recv.dtype == orig else recv.to(orig)


def _rs_rows_uneven(x, group, chunk_counts, wire_dtype=None):
    """Reduce-scatter partial rows with per-rank chunk counts: pad every
    destination chunk to the max count (zeros are the additive identity),
    native reduce-scatter, slice.  Sync-free pack (slice copies only)."""
    g = _world(group)
    if g == 1:
        return x
    if not _UNEVEN_NATIVE:
        return _rs_rows_uneven_a2a(x, group, chunk_counts, wire_dtype)
    orig = x.dtype
    x = _cast(x, wire_dtype).contiguous()
    m = max(chunk_counts)
    me = _rank(group)
    if min(chunk_counts) == m:
        send = x
    else:
        send = torch.zeros(g * m, x.shape[1], dtype=x.dtype, device=x.device)
        off = 0
        for j, n in enumerate(chunk_counts):
            send[j * m:j * m + n] = x[off:off + n]
            off += n
    out = torch.empty(m, x.shape[1], dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out.view(-1), send.view(-1), group=group)
    out = out[:chunk_counts[me]]
    return out if out.dtype == orig else out.to(orig)


def _a2a_cols_to_rows_uneven(x, group, recv_counts, wire_dtype=None,
                             async_op=False):
    """(m x n) full cols -> rows from every peer at my column block, where
    peer j contributes recv_counts[j] rows."""
    g = _world(group)
    if g == 1:
        assert not async_op
        return x
    orig = x.dtype
    x = _cast(x, wire_dtype)
    m, n = x.shape
    send = x.view(m, g, n // g).permute(1, 0, 2).reshape(g * m, n // g)
    send = send.contiguous()
    recv = torch.empty(sum(recv_counts), n // g, dtype=x.dtype,
                       device=x.device)
    work = dist.all_to_all_single(recv, send,
                                  output_split_sizes=list(recv_counts),
                                  input_split_sizes=[m] * g, group=group,
                                  async_op=async_op)

    def _fin():
        return recv if recv.dtype == orig else recv.to(orig)

    return (work, _fin) if async_op else _fin()


def _a2a_rows_to_cols_uneven(x, group, in_splits, out_rows, wire_dtype=None):
    """Row-chunk j of x (size in_splits[j]) goes to rank j; every peer sends
    me a chunk of out_rows rows; received chunks are concatenated along the
    column dim: result (out_rows x g*n_cols)."""
    g = _world(group)
    if g == 1:
        return x
    orig = x.dtype
    x = _cast(x, wire_dtype).contiguous()
    nb = x.shape[1]
    recv = torch.empty(g * out_rows, nb, dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv, x,
                           output_split_sizes=[out_rows] * g,
                           input_split_sizes=list(in_splits), group=group)
    out = recv.view(g, out_rows, nb).permute(1, 0, 2).reshape(out_rows, g * nb)
    return out if out.dtype == orig else out.to(orig)


def _reorder_blocks_uneven(x, sizes_kj, transpose=True):
    """x rows are blocks ordered (k-major, j-minor) with sizes sizes_kj[k][j];
    return rows reordered (j-major, k-minor).  Small number of slices."""
    K = len(sizes_kj)
    J = len(sizes_kj[0])
    offs = []
    off = 0
    for k in range(K):
        row = []
        for j in range(J):
            row.append(off)
            off += sizes_kj[k][j]
        offs.append(row)
    pieces = []
    for j in range(J):
        for k in range(K):
            pieces.append(x[offs[k][j]: offs[k][j] + sizes_kj[k][j]])
    return torch.cat(pieces, dim=0)


# ---------------------------------------------------------------------------
# sparse-aware gather (minibatch): only the H rows referenced by nonzeros
# of the local compact A shard move over the r axis.  The shard's column
# space is compacted to its nonempty columns in prep (compact_shard_columns
# + sparse_gather_plan_local, both comm-free); one tiny count/index exchange
# per layout per step (sparse_gather_exchange) completes the plan before
# the forward pass.  Coverage of sampled subgraphs is 1-exp(-d_b/Gr) with
# d_b = in-batch mean degree, so this cuts AG(H)/RS(gH) volume by 3-10x on
# low-degree batches (papers-class); see analysis/dist_spmm_papers_review.md.
# ---------------------------------------------------------------------------

def _a2av_rows(x, group, in_splits, out_splits, wire_dtype=None):
    """Generic uneven row all-to-all: row chunk j of x (in_splits[j] rows)
    goes to rank j; received chunks (out_splits[j] rows from rank j) are
    concatenated in rank order."""
    g = _world(group)
    if g == 1:
        return x
    orig = x.dtype
    x = _cast(x, wire_dtype).contiguous()
    recv = torch.empty(int(sum(out_splits)), x.shape[1], dtype=x.dtype,
                       device=x.device)
    dist.all_to_all_single(recv, x,
                           output_split_sizes=list(out_splits),
                           input_split_sizes=list(in_splits), group=group)
    return recv if recv.dtype == orig else recv.to(orig)


def _select_csr_rows(csr, rows):
    """Row subset of a CSR tensor (rows ascending, sort-free)."""
    crow = csr.crow_indices()
    starts = crow[rows]
    cnt = crow[rows + 1] - starts
    new_crow = torch.zeros(rows.numel() + 1, dtype=crow.dtype,
                           device=crow.device)
    torch.cumsum(cnt, 0, out=new_crow[1:])
    pos = torch.arange(int(new_crow[-1]), device=crow.device)
    idx = (torch.repeat_interleave(starts, cnt) + pos
           - torch.repeat_interleave(new_crow[:-1], cnt))
    return torch.sparse_csr_tensor(new_crow, csr.col_indices()[idx],
                                   csr.values()[idx],
                                   (rows.numel(), csr.shape[1]))


def compact_shard_columns(adj, adj_t):
    """Remap a compact CSR pair to reference only adj's nonempty columns.
    Returns (adj', adj_t', need): need = sorted unique original column ids;
    adj' has n_sel columns, adj_t' has n_sel rows (in need order)."""
    cols = adj.col_indices()
    need = torch.unique(cols)
    new_cols = torch.searchsorted(need, cols)
    adj2 = torch.sparse_csr_tensor(adj.crow_indices(), new_cols,
                                   adj.values(),
                                   (adj.shape[0], int(need.numel())))
    adj_t2 = _select_csr_rows(adj_t, need)
    return adj2, adj_t2, need


def sparse_gather_plan_local(need, Nc, Gf, Gr):
    """Comm-free half of the selective-gather plan for one layout (run in
    prep).  H (block column) order is j-major/k-minor cells of sizes
    s_fr[j][k]; cell (j,k) lives on r-peer k at x1 offset sum_{j'<j}."""
    s_f = _splits(Nc, Gf)
    s_fr = [_splits(v, Gr) for v in s_f]
    dev = need.device
    h_starts, own_off, owners = [], [], []
    off = 0
    x1_off = [0] * Gr
    for j in range(Gf):
        for k in range(Gr):
            h_starts.append(off)
            own_off.append(x1_off[k])
            owners.append(k)
            off += s_fr[j][k]
            x1_off[k] += s_fr[j][k]
    bounds = torch.tensor(h_starts, device=dev)
    seg = torch.searchsorted(bounds, need, right=True) - 1
    local = need - bounds[seg]
    owner_of = torch.tensor(owners, device=dev)[seg]
    owner_idx = torch.tensor(own_off, device=dev)[seg] + local
    order = torch.argsort(owner_of, stable=True)   # grouped pos -> need pos
    req_cnt = torch.bincount(owner_of, minlength=Gr)
    return {"n_sel": int(need.numel()), "Nc": int(Nc),
            "order": order, "req_idx": owner_idx[order],
            "req_cnt": [int(v) for v in req_cnt.tolist()]}


def sparse_gather_exchange(plan, r_group):
    """Runtime half: exchange request counts + owner-local row indices over
    the r group (2 tiny collectives).  Idempotent per step."""
    if "serve_idx" in plan:
        return plan
    g = _world(r_group)
    if g == 1:
        plan["serve_cnt"] = plan["req_cnt"]
        plan["serve_idx"] = plan["req_idx"]
        return plan
    cnt = torch.tensor(plan["req_cnt"], dtype=torch.int64, device="cuda")
    other = torch.empty_like(cnt)
    dist.all_to_all_single(other, cnt, group=r_group)
    serve_cnt = [int(v) for v in other.tolist()]
    serve_idx = torch.empty(sum(serve_cnt), dtype=plan["req_idx"].dtype,
                            device="cuda")
    dist.all_to_all_single(serve_idx, plan["req_idx"].contiguous(),
                           output_split_sizes=serve_cnt,
                           input_split_sizes=plan["req_cnt"], group=r_group)
    plan["serve_cnt"] = serve_cnt
    plan["serve_idx"] = serve_idx
    return plan


# ---------------------------------------------------------------------------
# autograd function
# ---------------------------------------------------------------------------

class GCNConvRSFunction(torch.autograd.Function):
    """Half-collective 3D tensor-parallel GCN layer (fwd + bwd)."""

    @staticmethod
    def forward(ctx, x, edge_index, edge_index_t, weight,
                c_group, f_group, r_group, layer_num, gemm_1d,
                sg_plan=None):
        timers = ax.get_timers()
        timers.start("gcn conv rs fwd")
        Gc, Gf, Gr = _world(c_group), _world(f_group), _world(r_group)

        wire = _wire_dtype()

        # coarse block sizes come straight from the A shard; inner
        # subdivisions follow the shared near-equal rule (_splits).
        # sparse gather: the shard's column space is compacted, so the
        # original block width comes from the plan instead of the shape.
        if sg_plan is not None:
            Nc = sg_plan["r"]["Nc"]   # original block width (cols of A)
            Nr = sg_plan["c"]["Nc"]   # original block height (rows of A)
            assert "serve_idx" in sg_plan["r"] and \
                "serve_idx" in sg_plan["c"], \
                "sg_plan missing exchange -- call sparse_gather_exchange"
        else:
            Nc = edge_index.shape[1]  # my coarse input block (A cols)
            Nr = edge_index.shape[0]  # my coarse output block (A rows)
        even_in = (Nc % (Gf * Gr) == 0) and sg_plan is None
        even_out = (Nr % (Gc * Gf) == 0) and sg_plan is None
        if not (even_in and even_out):
            assert gemm_1d, \
                "uneven (minibatch) blocks are only supported with gemm_1d"
        mf, mr, mc = _rank(f_group), _rank(r_group), _rank(c_group)
        s_f = _splits(Nc, Gf)
        s_fr = [_splits(v, Gr) for v in s_f]

        # 1. cols -> f blocks (all-to-all over f).  x: (n_me x F_in)
        timers.start("rs A2A(x)")
        if even_in:
            x1 = _all_to_all_cols_to_rows(x, f_group, wire)
        else:
            assert x.shape[0] == s_fr[mf][mr], \
                f"steady rows {x.shape[0]} != expected {s_fr[mf][mr]}"
            x1 = _a2a_cols_to_rows_uneven(
                x, f_group, [s_fr[j][mr] for j in range(Gf)], wire)
        timers.stop("rs A2A(x)")

        # 2. gather rows over r; received r-major, need f-major nesting
        timers.start("rs AG(H) r")
        if sg_plan is not None:
            # selective gather: send each r-peer exactly the x1 rows its
            # compacted shard references; received rows land in owner-
            # grouped order and are scattered to need order via `order`.
            pr = sg_plan["r"]
            send = x1.index_select(0, pr["serve_idx"])
            recv = _a2av_rows(send, r_group,
                              pr["serve_cnt"], pr["req_cnt"], wire)
            H = torch.empty(pr["n_sel"], x1.shape[1],
                            dtype=x1.dtype, device=x1.device)
            H.index_copy_(0, pr["order"], recv)
        elif even_in:
            x2 = _all_gather_rows(x1, r_group, wire)
            H = _permute_blocks(x2, Gr, Gf)            # rows = c block
        else:
            recv_r = [sum(s_fr[j][k] for j in range(Gf)) for k in range(Gr)]
            x2 = _ag_rows_uneven(x1, r_group, recv_r, wire)
            H = _reorder_blocks_uneven(
                x2, [[s_fr[j][k] for j in range(Gf)] for k in range(Gr)])
        timers.stop("rs AG(H) r")

        # 3. aggregation: AGG partial over c
        timers.start("rs AGG = A * H")
        agg_p = torch.sparse.mm(edge_index, H)         # (N/Gr x F/Gf)
        timers.stop("rs AGG = A * H")

        # 4. reduce-scatter rows over c
        t = _splits(Nr, Gc)
        timers.start("rs RS(AGG) c")
        if sg_plan is not None:
            # sparse reduce: agg_p already holds only the nonzero rows of
            # my A shard (row-compacted, in rows_nz order = grouped by
            # destination chunk); push them and scatter-add at the owner.
            pc = sg_plan["c"]
            send = agg_p.index_select(0, pc["order"])
            recv = _a2av_rows(send, c_group,
                              pc["req_cnt"], pc["serve_cnt"], wire)
            agg = torch.zeros(t[mc], agg_p.shape[1],
                              dtype=agg_p.dtype, device=agg_p.device)
            agg.index_add_(0, pc["serve_idx"], recv)
        elif even_out:
            agg = _reduce_scatter_rows(agg_p, c_group, wire)
        else:
            agg = _rs_rows_uneven(agg_p, c_group, t, wire)  # (t[mc] x F/Gf)
        timers.stop("rs RS(AGG) c")

        if gemm_1d:
            # 5'. hybrid path (3D SpMM + 1D GEMM): convert AGG's f-split
            # columns into f-split rows with a cheap all-to-all (pure
            # permutation, AGG is exact after the RS), then the GEMM is
            # fully local with W replicated -- no reduction at all.
            u = _splits(t[mc], Gf)
            timers.start("rs A2A(AGG)")
            if even_out:
                agg_rows = _all_to_all_rows_to_cols(agg, f_group, wire)
            else:
                agg_rows = _a2a_rows_to_cols_uneven(
                    agg, f_group, u, u[mf], wire)      # (u[mf] x F)
            timers.stop("rs A2A(AGG)")
            timers.start("rs OUT = AGG * W")
            out = tuned_matmul(agg_rows, weight,
                               "rs AGG * W " + str(layer_num))
            timers.stop("rs OUT = AGG * W")            # (N/P x F_out)
            ctx.save_for_backward(agg_rows, weight, edge_index_t)
        else:
            # 5. combination (contract f); W columns unsplit
            timers.start("rs OUT = AGG * W")
            out_p = tuned_matmul(agg, weight, "rs AGG * W " + str(layer_num))
            timers.stop("rs OUT = AGG * W")

            # 6. reduce-scatter rows over f -> steady state of next layer
            timers.start("rs RS(OUT) f")
            out = _reduce_scatter_rows(out_p, f_group, wire)  # (N/P x F_out)
            timers.stop("rs RS(OUT) f")
            ctx.save_for_backward(agg, weight, edge_index_t)

        ctx.groups = (c_group, f_group, r_group)
        ctx.layer_num = layer_num
        ctx.gemm_1d = gemm_1d
        ctx.splits = (even_in, even_out, mf, mr, mc, s_fr, t,
                      _splits(t[mc], Gf))
        ctx.sg_plan = sg_plan
        ctx.x1_rows = x1.shape[0]
        timers.stop("gcn conv rs fwd")
        return out

    @staticmethod
    def backward(ctx, grad_out):
        timers = ax.get_timers()
        timers.start("gcn conv rs bwd")
        agg, weight, edge_index_t = ctx.saved_tensors
        c_group, f_group, r_group = ctx.groups
        Gc, Gf, Gr = _world(c_group), _world(f_group), _world(r_group)

        wire = _wire_dtype()
        even_in, even_out, mf, mr, mc, s_fr, t, u = ctx.splits

        ar_w_work = None    # deferred AR(grad_W), waited right before return

        if ctx.gemm_1d:
            # 1'. hybrid path: grad_out is already in steady layout
            # (N/P x F_out); GEMM^T and grad_W are fully local, then the
            # all-to-all mirrors the forward one.
            #
            # overlap_bwd_comm (v2, strictly cross-communicator):
            #   - A2A(gAGG) [f comm] is issued async and overlaps the
            #     independent grad_W GEMM (local compute);
            #   - AR(grad_W) [whole intra comm] is issued async and hides
            #     behind AG(c) + SpMM^T + RS(gH) + A2A(gx); it is only
            #     waited at the very end of backward.
            g1 = grad_out.contiguous()

            timers.start("rs GRAD_AGG")
            grad_agg_rows = tuned_matmul(
                g1, weight.t(), "rs g * W.T " + str(ctx.layer_num))
            timers.stop("rs GRAD_AGG")                 # (u[mf] x F)

            _ov = plx.overlap_bwd_comm and dist.is_initialized() \
                and _world(f_group) > 1
            a2a_work = None
            if _ov:
                timers.start("rs A2A(gAGG) launch")
                if even_out:
                    a2a_work, a2a_fin = _all_to_all_cols_to_rows(
                        grad_agg_rows, f_group, wire, async_op=True)
                else:
                    a2a_work, a2a_fin = _a2a_cols_to_rows_uneven(
                        grad_agg_rows, f_group, u, wire, async_op=True)
                timers.stop("rs A2A(gAGG) launch")
            else:
                timers.start("rs A2A(gAGG)")
                if even_out:
                    grad_agg = _all_to_all_cols_to_rows(
                        grad_agg_rows, f_group, wire)
                else:
                    grad_agg = _a2a_cols_to_rows_uneven(
                        grad_agg_rows, f_group, u, wire)
                timers.stop("rs A2A(gAGG)")            # (t[mc] x F/Gf)

            # grad_W GEMM (local; overlaps the async A2A).  W is replicated
            # over ALL P ranks, so one all-reduce over the whole intra-layer
            # group replaces the (mathematically identical) sequence of
            # three per-axis all-reduces.
            timers.start("rs GRAD_W")
            grad_w = tuned_matmul(agg.t(), g1,
                                  "rs AGG.T * g " + str(ctx.layer_num))
            timers.stop("rs GRAD_W")

            if a2a_work is not None:
                timers.start("rs A2A(gAGG) wait")
                a2a_work.wait()
                grad_agg = a2a_fin()
                timers.stop("rs A2A(gAGG) wait")

            timers.start("rs AR(grad_W)")
            whole = ax.comm_handle.intra_layer_group
            if _world(whole) > 1:
                if _ov:
                    ar_w_work = dist.all_reduce(grad_w, group=whole,
                                                async_op=True)
                else:
                    dist.all_reduce(grad_w, group=whole)
            timers.stop("rs AR(grad_W)")

            timers.start("rs AG(gAGG) c")
            if ctx.sg_plan is not None:
                # mirror of the sparse reduce: return to each c-peer the
                # grad rows of the partials it pushed; result lands in
                # rows_nz order, matching edge_index_t's column space.
                pc = ctx.sg_plan["c"]
                back = grad_agg.index_select(0, pc["serve_idx"])
                recv = _a2av_rows(back, c_group,
                                  pc["serve_cnt"], pc["req_cnt"], wire)
                g2 = torch.empty(pc["n_sel"], grad_agg.shape[1],
                                 dtype=grad_agg.dtype,
                                 device=grad_agg.device)
                g2.index_copy_(0, pc["order"], recv)
            elif even_out:
                g2 = _all_gather_rows(grad_agg, c_group, wire)
            else:
                g2 = _ag_rows_uneven(grad_agg, c_group, t, wire)
            timers.stop("rs AG(gAGG) c")
        else:
            # 1. undo fwd RS(OUT): gather rows over f
            timers.start("rs AG(gOUT) f")
            g1 = _all_gather_rows(grad_out.contiguous(), f_group, wire)
            timers.stop("rs AG(gOUT) f")               # (N/(GrGc) x F_out)

            # 2. weight gradient; partial over the (r, c) replica groups
            timers.start("rs GRAD_W")
            grad_w = tuned_matmul(agg.t(), g1,
                                  "rs AGG.T * g " + str(ctx.layer_num))
            timers.stop("rs GRAD_W")
            timers.start("rs AR(grad_W)")
            if _world(c_group) > 1:
                dist.all_reduce(grad_w, group=c_group)
            if _world(r_group) > 1:
                dist.all_reduce(grad_w, group=r_group)
            timers.stop("rs AR(grad_W)")

            # 3. grad wrt AGG: contraction over unsplit F_out, no comm
            timers.start("rs GRAD_AGG")
            grad_agg = tuned_matmul(g1, weight.t(),
                                    "rs g * W.T " + str(ctx.layer_num))
            timers.stop("rs GRAD_AGG")                 # (N/(GrGc) x F/Gf)

            # 4. gather rows over c (assemble my r block)
            timers.start("rs AG(gAGG) c")
            g2 = _all_gather_rows(grad_agg, c_group, wire)  # (N/Gr x F/Gf)
            timers.stop("rs AG(gAGG) c")

        # 5. grad wrt H: partial over r
        timers.start("rs GRAD_H = A.T * g")
        gh_p = torch.sparse.mm(edge_index_t, g2)       # (N/Gc x F/Gf)
        timers.stop("rs GRAD_H = A.T * g")

        # 6. undo fwd AG over r: permute back, reduce-scatter rows over r
        timers.start("rs RS(gH) r")
        if ctx.sg_plan is not None:
            # mirror of the selective gather: regroup grad rows by owner,
            # send back, scatter-add into each owner's x1 row space (rows
            # requested by several peers accumulate -- exact).
            pr = ctx.sg_plan["r"]
            gh_grouped = gh_p.index_select(0, pr["order"])
            recv = _a2av_rows(gh_grouped, r_group,
                              pr["req_cnt"], pr["serve_cnt"], wire)
            g3 = torch.zeros(ctx.x1_rows, gh_p.shape[1],
                             dtype=gh_p.dtype, device=gh_p.device)
            g3.index_add_(0, pr["serve_idx"], recv)
        elif even_in:
            gh_p = _permute_blocks(gh_p, Gf, Gr)       # back to r-major
            g3 = _reduce_scatter_rows(gh_p, r_group, wire)
        else:
            # input rows (j-major, k-minor) sizes s_fr[j][k] -> (k-major)
            gh_kj = _reorder_blocks_uneven(gh_p, s_fr)
            recv_r = [sum(s_fr[j][k] for j in range(Gf)) for k in range(Gr)]
            g3 = _rs_rows_uneven(gh_kj, r_group, recv_r, wire)
        timers.stop("rs RS(gH) r")

        # 7. undo fwd all-to-all: rows -> cols
        timers.start("rs A2A(gx)")
        if even_in:
            grad_x = _all_to_all_rows_to_cols(g3, f_group, wire)
        else:
            grad_x = _a2a_rows_to_cols_uneven(
                g3, f_group, [s_fr[j][mr] for j in range(Gf)],
                s_fr[mf][mr], wire)                    # (n_me x F_in)
        timers.stop("rs A2A(gx)")

        if ar_w_work is not None:
            timers.start("rs AR(grad_W) wait")
            ar_w_work.wait()
            timers.stop("rs AR(grad_W) wait")

        timers.stop("gcn conv rs bwd")
        return (grad_x, None, None, grad_w,
                None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------

class GCNConvRS(torch.nn.Module):
    """
    Half-collective 3D parallel GCNConv layer.

    Drop-in for GCNConv w.r.t. the adjacency shards (same DataLoader), but
    the activation layout between layers is fully scattered:
    (padded_N / P) rows x full feature columns per GPU.  Element-wise ops,
    norms and dropout between layers therefore run with zero redundancy and
    need no communication (norms see full feature rows).
    """

    def __init__(self, in_channels, out_channels, layer_num,
                 gemm_1d: bool = True, fixed_axes: bool = False, **kwargs):
        super().__init__()
        self.layer_num = layer_num
        self.gemm_1d = bool(gemm_1d)
        self.fixed_axes = bool(fixed_axes)

        # identical rotation to GCNConv (use_3d_linear=True path):
        # tuple = (outer=c, inner=f, depth=r)
        # fixed_axes=True ("featpar"): no rotation -- only valid when the
        # node axes are trivial (Gr == Gc == 1, grid (1,1,P)), where layer
        # output layout == layer input layout without rotating.
        if self.fixed_axes or layer_num % 3 == 0:
            groups = ("x", "z", "y")
        elif layer_num % 3 == 1:
            groups = ("y", "x", "z")
        else:
            groups = ("z", "y", "x")

        num_gpus, _, process_groups = get_process_groups_info(groups)
        self.c_group, self.f_group, self.r_group = process_groups
        self.Gc, self.Gf, self.Gr = num_gpus
        if self.fixed_axes:
            assert self.Gc == 1 and self.Gr == 1, \
                "fixed_axes (featpar) requires grid (1,1,P): Gr=Gc=1"

        assert in_channels % self.Gf == 0, \
            f"in_channels={in_channels} must divide Gf={self.Gf}"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_in_channels = in_channels // self.Gf

        # W init: same full matrix on every rank (same seed everywhere).
        # gemm_1d=True (hybrid): W kept fully replicated (F_in x F_out);
        # gemm_1d=False (half):  input dim split over f, replicated (r, c).
        full_weight = torch.empty(in_channels, out_channels, device="cuda")
        torch.nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        if self.gemm_1d:
            self.local_in_channels = in_channels
            self.weight = Parameter(full_weight, requires_grad=True)
        else:
            f_rank = _rank(self.f_group)
            local = full_weight[
                f_rank * self.local_in_channels:
                (f_rank + 1) * self.local_in_channels, :].contiguous()
            del full_weight
            self.weight = Parameter(local, requires_grad=True)

    def forward(self, x, edge_index_shards):
        shard = edge_index_shards[
            self.layer_num % len(edge_index_shards)
        ]
        sg_plan = None
        if len(shard) == 3:                # sparse-gather minibatch path
            edge_index, edge_index_t, sg_plan = shard
        else:
            edge_index, edge_index_t = shard
        if edge_index_t is None:
            edge_index_t = edge_index.transpose(0, 1).to_sparse_csr()
        return GCNConvRSFunction.apply(
            x, edge_index, edge_index_t, self.weight,
            self.c_group, self.f_group, self.r_group, self.layer_num,
            self.gemm_1d, sg_plan,
        )
