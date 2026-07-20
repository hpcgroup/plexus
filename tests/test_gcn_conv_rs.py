#!/usr/bin/env python
"""
Correctness test + microbenchmark for GCNConvRS (half-collective 3D GCN).

Run on a 4-GPU node:
  torchrun --nproc_per_node=4 tests/test_gcn_conv_rs.py --G_intra_r 2 --G_intra_c 2 --G_intra_d 1
  torchrun --nproc_per_node=4 tests/test_gcn_conv_rs.py --G_intra_r 2 --G_intra_c 1 --G_intra_d 2
  torchrun --nproc_per_node=4 tests/test_gcn_conv_rs.py --G_intra_r 1 --G_intra_c 2 --G_intra_d 2
  torchrun --nproc_per_node=4 tests/test_gcn_conv_rs.py --G_intra_r 2 --G_intra_c 2 --G_intra_d 1 \
      --bench --N 480000 --F 256 --avg-degree 50

The correctness test builds a small global graph, runs L=3 GCNConvRS layers
(with ReLU in between), and checks forward output, grad_x and grad_W against
a dense single-process reference, elementwise.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plexus import plexus as plx                       # noqa: E402
from plexus.gcn_conv_rs import (GCNConvRS, compact_shard_columns,  # noqa: E402
    sparse_gather_plan_local, sparse_gather_exchange)
from plexus.gcn_conv import GCNConv                    # noqa: E402
from plexus.utils.general import set_seed, get_process_groups_info  # noqa: E402


def steady_slice(rank_letters, N, use_layer0=True):
    """Global row range of this rank's steady-state chunk.

    Layer-0 input nesting is (c > f > r) = ("x" > "z" > "y"); after a full
    3-layer rotation the output nesting is (r2 > c2 > f2) = ("x" > "z" > "y")
    again, so the same mapping serves both ends when L % 3 == 0.
    """
    (Gc, Gf, Gr), (rc, rf, rr), _ = rank_letters
    P = Gc * Gf * Gr
    n0 = N // P
    start = ((rc * Gf + rf) * Gr + rr) * n0
    return start, start + n0


def _uneven_bounds(N, g):
    """Deterministic, deliberately uneven coarse boundaries (weights 2,3,...)."""
    w = list(range(2, 2 + g))
    tot = sum(w)
    sizes = [N * wi // tot for wi in w]
    sizes[-1] = N - sum(sizes[:-1])
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    return bounds


def _hier_slice(bounds, letters, N):
    """Global row range of this rank's hierarchical cell for input nesting
    (c > f > r) with per-axis coarse `bounds` and _splits inner rule."""
    from plexus.gcn_conv_rs import _splits
    (Gc, Gf, Gr), (rc, rf, rr), _ = get_process_groups_info(letters)
    c_letter = letters[0]
    c0, c1 = bounds[c_letter][rc], bounds[c_letter][rc + 1]
    s_f = _splits(c1 - c0, Gf)
    f0 = c0 + sum(s_f[:rf])
    s_fr = _splits(s_f[rf], Gr)
    r0 = f0 + sum(s_fr[:rr])
    return r0, r0 + s_fr[rr]


def correctness_uneven(args, sparse=False):
    """Uneven coarse blocks (the minibatch case): arbitrary per-axis
    boundaries, hybrid path, vs dense serial reference.  sparse=True
    additionally compacts shard columns + selective-gather plans."""
    N, F, L = 91, args.F_small, 3
    set_seed(0)
    A = (torch.rand(N, N, device="cuda") < 0.08).float() \
        * torch.rand(N, N, device="cuda")
    X = torch.randn(N, F, device="cuda")
    R = torch.randn(N, F, device="cuda")

    tag = "sgather" if sparse else "uneven"
    num_gpus, _, _ = get_process_groups_info(("x", "y", "z"))
    bounds = {a: _uneven_bounds(N, g)
              for a, g in zip(("x", "y", "z"), num_gpus)}

    set_seed(1)
    layers = [GCNConvRS(F, F, l, gemm_1d=True) for l in range(L)]
    full_W = [layer.weight.detach().clone() for layer in layers]

    # A shards with uneven boundaries (layer l rows d1-block, cols d2-block)
    adj_groups = (("y", "x"), ("z", "y"), ("x", "z"))
    shards = []
    for l in range(L):
        d1, d2 = adj_groups[l]
        _, (r1, r2), _ = get_process_groups_info((d1, d2))
        rb = slice(bounds[d1][r1], bounds[d1][r1 + 1])
        cb = slice(bounds[d2][r2], bounds[d2][r2 + 1])
        blk = A[rb, cb]
        shards.append((blk.to_sparse_csr(),
                       blk.t().contiguous().to_sparse_csr()))
    if sparse:
        sg = []
        for l, (adj, adjt) in enumerate(shards):
            adj2, adjt2, need = compact_shard_columns(adj, adjt)
            adjt3, adj3, rows_nz = compact_shard_columns(adjt2, adj2)
            plan = {
                "r": sparse_gather_plan_local(need, adj.shape[1],
                                              layers[l].Gf, layers[l].Gr),
                "c": sparse_gather_plan_local(rows_nz, adj.shape[0],
                                              1, layers[l].Gc),
            }
            sparse_gather_exchange(plan["r"], layers[l].r_group)
            sparse_gather_exchange(plan["c"], layers[l].c_group)
            sg.append((adj3, adjt3, plan))
        shards = sg

    s, e = _hier_slice(bounds, ("x", "z", "y"), N)   # layer-0 input nesting
    x_local = X[s:e].clone().requires_grad_(True)
    h = x_local
    for l in range(L):
        h = layers[l](h, shards)
        if l < L - 1:
            h = torch.relu(h)
    # final nesting (L=3) is ("x","z","y") again with the same x bounds
    (h * R[s:e]).sum().backward()

    Xr = X.clone().requires_grad_(True)
    Wr = [w.clone().requires_grad_(True) for w in full_W]
    hr = Xr
    for l in range(L):
        hr = A @ hr @ Wr[l]
        if l < L - 1:
            hr = torch.relu(hr)
    (hr * R).sum().backward()

    ok = True
    for name, mine, ref in (
        ("uneven forward OUT", h.detach(), hr.detach()[s:e]),
        ("uneven grad_X", x_local.grad, Xr.grad[s:e]),
    ):
        err = (mine - ref).abs().max().item()
        scale = ref.abs().max().item() + 1e-12
        good = err / scale < 2e-4
        ok &= good
        if dist.get_rank() == 0 or not good:
            print(f"[rank {dist.get_rank()}] {'OK ' if good else 'FAIL'} "
                  f"[{tag}] {name}: max_abs_err={err:.3e}")
    for l, layer in enumerate(layers):
        err = (layer.weight.grad - Wr[l].grad).abs().max().item()
        good = err / (Wr[l].grad.abs().max().item() + 1e-12) < 2e-4
        ok &= good
        if dist.get_rank() == 0 or not good:
            print(f"[rank {dist.get_rank()}] {'OK ' if good else 'FAIL'} "
                  f"[{tag}] grad_W[{l}]: max_abs_err={err:.3e}")
    flag = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(flag)
    if dist.get_rank() == 0:
        print(f"[{tag}] ALL CHECKS PASSED" if flag.item() == 0
              else f"[{tag}] FAILURES DETECTED")
    return flag.item() == 0


def build_adj_shards(A_dense, num_layers, fixed=False):
    """Replicate the DataLoader's per-layer 2D blocking of A.

    use_3d_linear adj_groups: layer l rows split by dim1, cols by dim2:
      l%3==0: ("y","x");  l%3==1: ("z","y");  l%3==2: ("x","z")
    fixed=True (featpar): every layer uses the ("y","x") blocking.
    """
    N = A_dense.shape[0]
    shards = []
    adj_groups = (("y", "x"), ("z", "y"), ("x", "z"))
    for l in range(min(3, num_layers)):
        d1, d2 = adj_groups[0] if fixed else adj_groups[l]
        (g1, g2), (r1, r2), _ = get_process_groups_info((d1, d2))
        rb = slice(r1 * (N // g1), (r1 + 1) * (N // g1))
        cb = slice(r2 * (N // g2), (r2 + 1) * (N // g2))
        blk = A_dense[rb, cb]
        shards.append((blk.to_sparse_csr(),
                       blk.t().contiguous().to_sparse_csr()))
    return shards


def correctness(args, gemm_1d=False, fixed_axes=False):
    N, F = args.N_small, args.F_small
    L = 3
    set_seed(0)

    # global data, identical on every rank
    A = (torch.rand(N, N, device="cuda") < 0.05).float() \
        * torch.rand(N, N, device="cuda")
    X = torch.randn(N, F, device="cuda")
    R = torch.randn(N, F, device="cuda")   # random cotangent for the loss

    # layers draw their weights in the same order on every rank
    set_seed(1)
    layers = [GCNConvRS(F, F, l, gemm_1d=gemm_1d, fixed_axes=fixed_axes)
              for l in range(L)]

    # full weights for the reference: reconstruct by all-gathering each
    # layer's f-sharded weight over its f-group (hybrid keeps W full)
    full_W = []
    for l, layer in enumerate(layers):
        w = layer.weight.detach()
        if not gemm_1d and layer.Gf > 1:
            parts = [torch.empty_like(w) for _ in range(layer.Gf)]
            dist.all_gather(parts, w.contiguous(), group=layer.f_group)
            full_W.append(torch.cat(parts, dim=0))
        else:
            full_W.append(w.clone())

    shards = build_adj_shards(A, L, fixed=fixed_axes)

    # local run: slice steady-state input (layer-0 nesting (x > z > y))
    info = get_process_groups_info(("x", "z", "y"))
    s, e = steady_slice(info, N)
    x_local = X[s:e].clone().requires_grad_(True)

    h = x_local
    for l in range(L):
        h = layers[l](h, shards)
        if l < L - 1:
            h = torch.relu(h)
    out_local = h

    # final nesting after L=3 is (x > z > y) again
    (out_local * R[s:e]).sum().backward()

    # dense reference (identical on every rank)
    Xr = X.clone().requires_grad_(True)
    Wr = [w.clone().requires_grad_(True) for w in full_W]
    hr = Xr
    for l in range(L):
        hr = A @ hr @ Wr[l]
        if l < L - 1:
            hr = torch.relu(hr)
    (hr * R).sum().backward()

    tag = ("featpar" if fixed_axes else "hybrid") if gemm_1d else "half"

    def check(name, mine, ref):
        err = (mine - ref).abs().max().item()
        scale = ref.abs().max().item() + 1e-12
        ok = err / scale < 2e-4
        flag = "OK " if ok else "FAIL"
        if dist.get_rank() == 0 or not ok:
            print(f"[rank {dist.get_rank()}] {flag} [{tag}] {name}: "
                  f"max_abs_err={err:.3e} (scale {scale:.3e})")
        return ok

    ok = True
    ok &= check("forward OUT", out_local.detach(), hr.detach()[s:e])
    ok &= check("grad_X", x_local.grad, Xr.grad[s:e])
    for l, layer in enumerate(layers):
        if gemm_1d:
            ref_grad = Wr[l].grad
        else:
            fr = dist.get_rank(layer.f_group) if layer.Gf > 1 else 0
            li = layer.local_in_channels
            ref_grad = Wr[l].grad[fr * li:(fr + 1) * li]
        ok &= check(f"grad_W[{l}]", layer.weight.grad, ref_grad)
    flag = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(flag)
    if dist.get_rank() == 0:
        print("=" * 50)
        print(f"[{tag}] ALL CHECKS PASSED" if flag.item() == 0
              else f"[{tag}] FAILURES DETECTED")
    return flag.item() == 0


def bench(args):
    """Compare per-step time: existing GCNConv vs GCNConvRS, L=3."""
    N, F, L = args.N, args.F, 3
    world = dist.get_world_size()
    set_seed(0)

    # random uniform graph with the requested average degree, built shard-
    # by-shard so it scales (no global dense A)
    nnz_total = int(N * args.avg_degree)

    def rand_csr(rows, cols, nnz):
        idx_r = torch.randint(0, rows, (nnz,), device="cuda")
        idx_c = torch.randint(0, cols, (nnz,), device="cuda")
        val = torch.rand(nnz, device="cuda")
        a = torch.sparse_coo_tensor(torch.stack([idx_r, idx_c]), val,
                                    (rows, cols)).coalesce()
        return a.to_sparse_csr()

    adj_groups = (("y", "x"), ("z", "y"), ("x", "z"))
    shards = []
    for l in range(L):
        d1, d2 = adj_groups[l]
        (g1, g2), _, _ = get_process_groups_info((d1, d2))
        blk = rand_csr(N // g1, N // g2, nnz_total // (g1 * g2))
        blk_t = rand_csr(N // g2, N // g1, nnz_total // (g1 * g2))
        shards.append((blk, blk_t))

    def timeit(fn, iters=10, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters

    # ---- current GCNConv ----
    set_seed(1)
    cur_layers = [GCNConv(F, F, l, shard_features_in_depth=False)
                  for l in range(L)]
    info0 = get_process_groups_info(("x", "z", "y"))
    (Gc, Gf, Gr) = info0[0]
    x_cur = torch.randn(N // Gc, F // Gf, device="cuda", requires_grad=True)

    def step_cur():
        h = x_cur
        for l in range(L):
            h = cur_layers[l](h, shards)
            if l < L - 1:
                h = torch.relu(h)
        h.sum().backward()
        x_cur.grad = None
        for lay in cur_layers:
            lay.weight.grad = None

    def dump_timers(tag):
        if not args.timers:
            return
        from axonn import axonn as ax
        times, counts = ax.get_timers().get_times()
        if dist.get_rank() == 0:
            print(f"  -- {tag} stage totals (ms, all iters incl warmup) --")
            rows = sorted(times.items(), key=lambda kv: -kv[1])
            for key, t in rows[:30]:
                if t > 1.0:
                    print(f"     {t:9.1f}  n={counts[key]:4d}  {'/'.join(key)}")

    t_cur = timeit(step_cur)
    dump_timers("GCNConv")

    # ---- GCNConvRS (half / hybrid / featpar variants) ----
    variants = [("half", dict(gemm_1d=False)), ("hybrid", dict(gemm_1d=True))]
    info0 = get_process_groups_info(("x", "z", "y"))
    if info0[0][0] == 1 and info0[0][2] == 1:      # grid (1,1,P)
        variants.append(("featpar", dict(gemm_1d=True, fixed_axes=True)))

    t_rs = {}
    for tag, kw in variants:
        set_seed(1)
        rs_layers = [GCNConvRS(F, F, l, **kw) for l in range(L)]
        var_shards = shards
        if kw.get("fixed_axes"):
            var_shards = [shards[0]] * L    # full A every layer
        x_rs = torch.randn(N // world, F, device="cuda", requires_grad=True)

        def step_rs():
            h = x_rs
            for l in range(L):
                h = rs_layers[l](h, var_shards)
                if l < L - 1:
                    h = torch.relu(h)
            h.sum().backward()
            x_rs.grad = None
            for lay in rs_layers:
                lay.weight.grad = None

        t_rs[tag] = timeit(step_rs)
        dump_timers(f"GCNConvRS[{tag}]")
        del rs_layers, x_rs

    if dist.get_rank() == 0:
        print(f"\nbench N={N} F={F} L={L} avg_deg={args.avg_degree} "
              f"P={world}")
        print(f"  GCNConv   (current): {t_cur:8.2f} ms/step")
        for tag, t in t_rs.items():
            print(f"  GCNConvRS ({tag:6s}): {t:8.2f} ms/step   "
                  f"speedup {t_cur / t:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G_intra_r", type=int, default=2)
    ap.add_argument("--G_intra_c", type=int, default=2)
    ap.add_argument("--G_intra_d", type=int, default=1)
    ap.add_argument("--N_small", type=int, default=48)
    ap.add_argument("--F_small", type=int, default=16)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--N", type=int, default=480000)
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--avg-degree", type=float, default=50)
    ap.add_argument("--timers", action="store_true",
                    help="print per-stage timer totals in --bench mode")
    ap.add_argument("--overlap_bwd_comm", action="store_true", default=False)
    args = ap.parse_args()

    # initialise torch.distributed from the launcher env (torchrun sets
    # RANK/WORLD_SIZE/LOCAL_RANK; srun + get_rank.sh sets RANK/WORLD_SIZE)
    if not dist.is_initialized():
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        elif "SLURM_LOCALID" in os.environ:
            torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
        dist.init_process_group(backend="nccl")

    plx.init(G_intra_r=args.G_intra_r, G_intra_c=args.G_intra_c,
             G_intra_d=args.G_intra_d,
             enable_internal_timers=bool(args.timers),
             overlap_bwd_comm_flag=args.overlap_bwd_comm)

    if args.bench:
        ok = True
        bench(args)
    else:
        ok = correctness(args, gemm_1d=False)
        ok = correctness(args, gemm_1d=True) and ok
        ok = correctness_uneven(args) and ok
        ok = correctness_uneven(args, sparse=True) and ok
        if args.G_intra_r == 1 and args.G_intra_c == 1:
            # featpar (fixed axes) is only defined on grid (1,1,P)
            ok = correctness(args, gemm_1d=True, fixed_axes=True) and ok
    dist.barrier()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
