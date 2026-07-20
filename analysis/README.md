# analysis/ — cost model and microbenchmarks

Communication/compute cost model for the rs parallel method (3D SpMM + 1D
GEMM hybrid) plus the measurement tools that calibrate it.  Typical
workflow: run the microbenchmarks once to characterize the machine → use
the model to pick a grid → train.

## 1. Cost model: gcn_comm_model.py

Predicts per-step / per-epoch time and peak memory for a given (workload,
GPU count, grid, scheme) and recommends configurations.

```bash
# recommend a configuration (scans all feasible grids + schemes,
# including a memory-feasibility gate)
python analysis/gcn_comm_model.py --recommend --workload papers_full --gpus 64

# replace the analytic narrow-column penalty with the measured kernel
# curve (theory-first principle: constants come from independent
# measurement, not end-to-end fitting)
python analysis/gcn_comm_model.py --recommend --workload products_mini --gpus 32 \
    --kernel-curve analysis/kernel_curve.json

# re-run the calibration against recorded measurements
python analysis/gcn_comm_model.py --calibrate
```

Model structure (the parts validated against measurements):
- Communication: analytic flow cost, per layer
  [2(Gr−1)+2(Gc−1)+4(Gf−1)/Gf]·u with u = N·F/P; hierarchical
  intra-node (NVLink) / inter-node (NIC) bandwidth chosen by the physical
  placement of each grid axis;
- SpMM: bandwidth roofline nnz/(Gr·Gc)·(c0+c1·F_loc)/HBM_BW, with the
  narrow-column penalty looked up from kernel_curve.json;
- Memory: static adjacency (24 B/edge × Σ layouts) + conversion peak
  (20 B/edge) + features + activations.
- Known accuracy: full-graph absolute error 1–10% (measured on
  protein/papers); minibatch is systematically ~2× optimistic while the
  ranking stays mostly correct.  End-to-end prediction additionally needs
  the transport tier (native ring vs a2av), the α × collective count, and
  straggler exposure — a pure-volume model is blind to all three.

## 2. Microbenchmarks (machine characterization)

### spmm_microbench.py — narrow-column SpMM curve (1 GPU, ~30 s)
```bash
python analysis/spmm_microbench.py --out analysis/kernel_curve.json
```
Measures bytes/nnz vs F_loc ∈ {4..256} in two L2-residency regimes; the
output is loaded directly by the model via `--kernel-curve`.  Measured on
A100-40G: c0 ≈ 76 B, c1 ≈ 3.7 B/column; large-H regime penalties
F=32: +53 B, F=16: +101 B.

### collective_microbench.py — native vs a2av collectives (16 GPUs / 4 nodes)
```bash
srun -N4 -n16 --ntasks-per-node=4 --gpus-per-node=4 ./get_rank.sh \
    python -u analysis/collective_microbench.py   # writes analysis/collective_curve.json
```
Sweeps message size (64 KB–64 MB) × group size (2–16; intra-node / 2-node
/ 4-node span), comparing native AG/RS against the a2av-emulated path.
Key results (Perlmutter): α native 0.05–0.08 ms vs a2av 0.13–0.14 ms;
large-message cross-node bandwidth native 42–65 GB/s vs a2av 15–25 GB/s;
intra-node NVLink shows almost no gap between the two.

### agh_bench.py — production-topology replay (P=64 / 16 nodes)
```bash
srun -N16 -n64 --ntasks-per-node=4 --gpus-per-node=4 ./get_rank.sh \
    python -u analysis/agh_bench.py --G_intra_r 8 --G_intra_c 2 --G_intra_d 4 \
    --Nb 2221199 --F 256
```
Replays AG(H)/RS(AGG) with the real axonn groups (including the physical
rank-to-node placement) under the training layer rotation; prints each
axis group's node span and effective bandwidth.  Use it to attribute
"isolated optimum vs in-training measurement" gaps (contention vs
topology) and to compare the axis placement quality of candidate grids.
Measured axis bandwidths: y axis intra-node 100+ GB/s, x axis strided
cross-node 30–50, z axis fully cross-node 13–16 GB/s.

## 3. rs training quick reference

```bash
# full graph: train.py with --conv rs (the original gcn path is untouched)
srun ... python -u examples/train.py --conv rs --allreduce_lowp \
    --G_intra_r 4 --G_intra_c 4 --G_intra_d 4 ...

# minibatch: standalone entry point train_mini_rs.py (epoch_perm sampler
# is the default)
srun ... python -u examples/train_mini_rs.py --allreduce_lowp \
    --G_intra_r 2 --G_intra_c 8 --G_intra_d 4 --minibatch_ratio 0.02 \
    --minibatch_unbiased ...
```

Environment switches:
- `PLEXUS_SPARSE_GATHER=1`: sparsity-aware collectives (move only the
  rows referenced by the local shard's nonzeros; default off).  Pays off
  when the in-batch mean degree is well below F and volume is the
  bottleneck (e.g. large ratio / large batches);
- `PLEXUS_UNEVEN_NATIVE=0`: fall back to the a2av-emulated uneven
  collectives (default 1 = pad-to-even native NCCL path);
- `PLEXUS_SYNC_PREP=1`: run minibatch prep serially (diagnostic: the
  "prefetch wait" timer then measures the pure prep cost).

Grid rules of thumb (from measurements): minibatch on small graphs →
featpar (1,1,P) (fewest collectives); papers-class large graphs →
balanced grids ((2,8,4)/(4,4,4) tier — mind the 80 GB memory need);
full graph → featpar for P≤4, balanced 3D for P≥16;
enable --overlap_bwd_comm only for P≤8.
