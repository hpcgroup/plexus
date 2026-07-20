#!/usr/bin/env python
"""
Replay the production AG(H) / RS(AGG) collectives of GCNConvRS on the REAL
axonn group topology, with the exact per-layer shapes of a training step.
Answers: why do these run at 5-8 GB/s effective when microbench rings with
consecutive ranks reach 40+ GB/s?  (suspicion: strided cross-node groups)

Run:  srun -n64 ./get_rank.sh python -u analysis/agh_bench.py \
          --G_intra_r 8 --G_intra_c 2 --G_intra_d 4 --Nb 2221199
"""

import argparse
import os

import torch
import torch.distributed as dist

from plexus import plexus as plx
from plexus.utils.general import get_process_groups_info
from plexus.gcn_conv_rs import (_ag_rows_uneven, _rs_rows_uneven, _splits,
                                _world)

ROT = [("x", "z", "y"), ("y", "x", "z"), ("z", "y", "x")]   # (c, f, r)
WARMUP, ITERS = 3, 10


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def group_nodes(pg, tasks_per_node=4):
    ranks = dist.get_process_group_ranks(pg)
    return ranks, sorted({r // tasks_per_node for r in ranks})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--G_intra_r", type=int, required=True)
    p.add_argument("--G_intra_c", type=int, required=True)
    p.add_argument("--G_intra_d", type=int, required=True)
    p.add_argument("--Nb", type=int, default=2221199)
    p.add_argument("--F", type=int, default=256)
    args = p.parse_args()

    plx.init(G_intra_r=args.G_intra_r, G_intra_c=args.G_intra_c,
             G_intra_d=args.G_intra_d, gpus_per_node=4)
    rank0 = dist.get_rank() == 0
    dev = torch.device("cuda")

    tot_ag = tot_rs = 0.0
    for l, letters in enumerate(ROT):
        (sizes, _, pgs) = get_process_groups_info(list(letters))
        Gc, Gf, Gr = sizes
        c_group, f_group, r_group = pgs
        if rank0:
            for name, pg in (("c", c_group), ("f", f_group), ("r", r_group)):
                ranks, nodes = group_nodes(pg)
                print(f"L{l} {name}-axis({letters[('c','f','r').index(name)]}"
                      f",g={_world(pg)}) ranks={ranks[:8]}"
                      f"{'...' if len(ranks) > 8 else ''} nodes={nodes}",
                      flush=True)

        F_loc = args.F // Gf
        # AG(H) over r: shard Nb/(Gc*Gr) rows, gathered Nb/Gc rows
        n_blk = args.Nb // Gc
        recv_r = _splits(n_blk, Gr)
        me_r = dist.get_group_rank(r_group, dist.get_rank())
        shard = torch.randn(recv_r[me_r], F_loc, device=dev,
                            dtype=torch.bfloat16)
        t_ag = bench(lambda: _ag_rows_uneven(shard, r_group, recv_r))
        wire_ag = (Gr - 1) / Gr * n_blk * F_loc * 2 / 1e9

        # RS(AGG) over c: partial Nb/Gr rows
        n_r = args.Nb // Gr
        t = _splits(n_r, Gc)
        partial = torch.randn(n_r, F_loc, device=dev, dtype=torch.bfloat16)
        t_rs = bench(lambda: _rs_rows_uneven(partial, c_group, t))
        wire_rs = (Gc - 1) / Gc * n_r * F_loc * 2 / 1e9

        tot_ag += t_ag
        tot_rs += t_rs
        if rank0:
            print(f"L{l} (Gr,Gc,Gf)=({Gr},{Gc},{Gf})  "
                  f"AG {t_ag:7.2f}ms ({wire_ag / t_ag * 1e3:5.1f} GB/s eff, "
                  f"{wire_ag * 1e3:6.1f}MB)   "
                  f"RS {t_rs:7.2f}ms ({wire_rs / t_rs * 1e3:5.1f} GB/s eff, "
                  f"{wire_rs * 1e3:6.1f}MB)", flush=True)

    if rank0:
        print(f"TOTAL per step: AG {tot_ag:.1f}ms  RS {tot_rs:.1f}ms  "
              f"(training measured: AG 24.5, RS 37.2)", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
