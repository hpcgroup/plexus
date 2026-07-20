#!/usr/bin/env python
"""
Collective microbenchmark: native NCCL AG/RS vs the uneven-emulated path
(gcn_conv_rs._ag_rows_uneven / _rs_rows_uneven, i.e. what minibatch uses),
swept over message size x group size (intra-node and cross-node).

Separates the two slowdown hypotheses:
  - latency floor: both paths flatten at small messages; alpha_eff = flat value
  - protocol/bandwidth: gap between emulated and native at LARGE messages

Run (16 GPUs / 4 nodes):
  srun -n16 ./get_rank.sh python -u analysis/collective_microbench.py

Prints a rank-0 table and writes analysis/collective_curve.json.
"""

import json
import os

import torch
import torch.distributed as dist

from plexus.gcn_conv_rs import _ag_rows_uneven, _rs_rows_uneven

F = 256                       # columns, matches training
SIZES_MB = [0.0625, 0.25, 1.0, 4.0, 16.0, 64.0]   # gathered-result size
GROUP_SIZES = [2, 4, 8, 16]
WARMUP, ITERS = 5, 20


def bench(fn, *args, **kw):
    for _ in range(WARMUP):
        fn(*args, **kw)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(*args, **kw)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / ITERS


def main():
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    results = []
    for g in GROUP_SIZES:
        if g > world:
            continue
        # consecutive ranks: g<=4 stays intra-node (4 GPUs/node), g>4 crosses
        groups = [dist.new_group(list(range(i, i + g)))
                  for i in range(0, world, g)]
        group = groups[rank // g]
        span = "intra" if g <= 4 else f"{g // 4}nodes"

        for mb in SIZES_MB:
            m_rows = max(g, int(mb * 1e6 / (F * 2)))    # bf16 gathered rows
            m_rows -= m_rows % g
            shard = torch.randn(m_rows // g, F, device="cuda",
                                dtype=torch.bfloat16)
            full = torch.randn(m_rows, F, device="cuda",
                               dtype=torch.bfloat16)
            counts = [m_rows // g] * g
            out_ag = torch.empty(m_rows, F, device="cuda",
                                 dtype=torch.bfloat16)
            out_rs = torch.empty(m_rows // g, F, device="cuda",
                                 dtype=torch.bfloat16)

            t_ag_nat = bench(lambda: dist.all_gather_into_tensor(
                out_ag.view(-1), shard.contiguous().view(-1), group=group))
            t_ag_emu = bench(lambda: _ag_rows_uneven(shard, group, counts))
            t_rs_nat = bench(lambda: dist.reduce_scatter_tensor(
                out_rs.view(-1), full.contiguous().view(-1), group=group))
            t_rs_emu = bench(lambda: _rs_rows_uneven(full, group, counts))

            wire = (g - 1) / g * m_rows * F * 2 / 1e9    # GB on the wire
            row = dict(g=g, span=span, mb=mb,
                       ag_native_ms=t_ag_nat, ag_emul_ms=t_ag_emu,
                       rs_native_ms=t_rs_nat, rs_emul_ms=t_rs_emu,
                       ag_native_gbs=wire / t_ag_nat * 1e3,
                       ag_emul_gbs=wire / t_ag_emu * 1e3,
                       rs_native_gbs=wire / t_rs_nat * 1e3,
                       rs_emul_gbs=wire / t_rs_emu * 1e3)
            results.append(row)
            if rank == 0:
                print(f"g={g:2d} {span:7s} {mb:7.3f}MB | "
                      f"AG nat {t_ag_nat:7.3f}ms ({row['ag_native_gbs']:5.1f}GB/s) "
                      f"emu {t_ag_emu:7.3f}ms ({row['ag_emul_gbs']:5.1f}GB/s) "
                      f"x{t_ag_emu / t_ag_nat:4.1f} | "
                      f"RS nat {t_rs_nat:7.3f}ms "
                      f"emu {t_rs_emu:7.3f}ms x{t_rs_emu / t_rs_nat:4.1f}",
                      flush=True)
        for gr in groups:
            dist.destroy_process_group(gr)

    if rank == 0:
        with open("analysis/collective_curve.json", "w") as f:
            json.dump(results, f, indent=1)
        print("wrote analysis/collective_curve.json")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
