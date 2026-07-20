#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Communication/compute cost model for distributed GCN training (SpMM + GEMM).

Context
-------
plexus parallelizes each GCN layer as two chained matmuls on a 3D process
grid (Gx, Gy, Gz) = (G_intra_r, G_intra_c, G_intra_d):

    AGG = A * H      (SpMM,  A: Nv x Nv sparse,  H: Nv x F dense)
    OUT = AGG * W    (GEMM,  W: F x F dense)

For layer l the three grid axes are assigned (by rotation) to three roles:

    r  = node-row axis   (A's rows split across it; "depth" group in code)
    c  = node-col axis   (A's cols / H's rows split across it; "outer" group)
    f  = feature axis    (H's cols split across it; "inner" group)

use_3d_linear=True rotation (see plexus/gcn_conv.py + utils/dataloader.py):
    layer l%3==0: (r,c,f) = (y,x,z)
    layer l%3==1: (r,c,f) = (z,y,x)
    layer l%3==2: (r,c,f) = (x,z,y)

Current per-layer collectives (all node-proportional ones are ALL-REDUCEs):
    fwd:  AR(AGG)      over c   msg (Nb/Gr)(F/Gf)     <- "allreduce H"
          AG(W)        over r   msg F^2/(Gf*Gc)       <- "Allgather W"
          AR(OUT)      over f   msg (Nb/Gr)(F/Gc)     <- "allreduce Q"
    bwd:  AR(grad_agg) over c   msg (Nb/Gr)(F/Gf)     <- "gcn conv bwd/all-reduce"
          RS(grad_W)   over r   msg F^2/(Gf*Gc)       <- "ReduceScatter grad_weight"
          AR(grad_x)   over r   msg (Nb/Gc)(F/Gf)     <- "allreduce grad_x"

This file models alternative parallelizations of the same chain, counting
ONLY data that must flow every iteration (activations/gradients and weight
collectives).  Static tensors (the adjacency A) are placed once and never
communicated -- unlike the end-to-end distributed-GEMM I/O analysis of
Kwasniewski et al. (arXiv:1908.09606), whose per-op costs include fetching
both inputs and storing the output.  Our per-op cost keeps only:
  (1) the reduction of partial sums over the contraction-split axis, and
  (2) the resharding of the *flowing* tensor between consecutive ops.

Schemes
-------
  current    : plexus today.  Full all-reduces; activations replicated over
               the f axis between layers.  Per-layer moved words/GPU:
                   (2*Nb*F/P) * (2(Gc-1) + (Gf-1) + (Gr-1))
  current_sp : minimal change.  Same collectives, but AR(OUT) is split into
               RS ... elementwise ... AG (Megatron sequence-parallel style):
               identical comm cost, but norm/relu/dropout run on 1/Gf of the
               data and activation memory drops.
  half       : proposed.  Same A placement & rotation, but only "half"
               collectives (RS after each contraction, lazy AG before each
               consumer); activations live fully scattered (Nb*F/P per GPU)
               between layers; W replicated over (r,c), f-sharded.
               Per-layer moved words/GPU:
                   (Nb*F/P) * (2Gr + 2Gc + 2Gf - 4 - 1/Gf - 1/Gc)
  cosma2d    : per-op optimal ("COSMA-degenerate") layouts + explicit
               reshard, no axis rotation:  SpMM on a fixed 2D (Gr x Gc) grid
               (optimal cuboid for M=K>>N is a x 1 x c), GEMM pure
               row-parallel (optimal cuboid for M>>N=K is p x 1 x 1,
               comm-free).  The "reshard" between them is the AG/RS pair.
               Per-layer moved words/GPU: (2*Nb*F/P) * (Gr + Gc - 2)
  oned       : CAGNET-style 1D vertex partition (A row-split across all P,
               H row-split, full feature dim).  fwd all-gathers H, bwd
               reduce-scatters grad_H.  No locality assumed (random perm).

All costs are per *training step* (fwd+bwd), per GPU, summed over the L GCN
layers (input/output linear + loss excluded; they are grid-invariant here).

Ring collective model (g = group size, m = words of the FULL per-GPU message,
B = bytes/word, bw = per-GPU bandwidth of the slowest link in the group):
    all-reduce      t = 2*(g-1)/g * m*B/bw + 2*(g-1)*alpha
    reduce-scatter  t =   (g-1)/g * m*B/bw +   (g-1)*alpha
    all-gather      t =   (g-1)/g * m*B/bw +   (g-1)*alpha
Reshard (gather over a composite axis set S from a finer sharding):
    time = volume*B/bw_worst + (prod g_S - 1)*alpha,  volume = target - have.

Axis -> physical mapping follows AxoNN rank order (y fastest, then x, then z):
    rank = z*(Gx*Gy) + x*Gy + y
so a group along an axis is intra-node iff its rank span fits in a node.
Defaults are calibrated against 4.19/opt/breakdown.json (Perlmutter A100,
grid 2x2x2, ogbn-products minibatch): see --calibrate.

Usage
-----
  python analysis/gcn_comm_model.py --calibrate 4.19/opt/breakdown.json
  python analysis/gcn_comm_model.py --workload products_mini --P 8 16 32 64
  python analysis/gcn_comm_model.py --workload papers_full --P 64 512 --dtype-comm 2
  python analysis/gcn_comm_model.py --workload products_mini --P 8 --grids
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# machine model
# --------------------------------------------------------------------------

@dataclass
class Machine:
    gpus_per_node: int = 4
    # calibrated on Perlmutter: 4xA100, pairwise NVLink3 (4 links = 100GB/s
    # per direction), 1 Slingshot-11 NIC per GPU (~13 GB/s effective for
    # tens-of-MB NCCL rings crossing nodes)
    bw_intra: float = 100e9    # bytes/s per GPU
    bw_inter: float = 13e9     # bytes/s per GPU
    alpha_intra: float = 8e-6  # seconds per ring step
    alpha_inter: float = 15e-6
    # effective compute rates (calibrated, fp32)
    gemm_tflops: float = 16.0    # cuBLAS fp32, tall-skinny
    elem_bw: float = 900e9       # effective HBM bw for elementwise chains
    elem_passes: float = 12.0    # r/w passes over activations per layer
                                 # (rmsnorm fwd+bwd, relu, dropout, ...)
    # SpMM bandwidth-roofline model:
    #   t = nnz_local * (spmm_c0 + spmm_c1 * F_loc) / hbm_bw
    # nnz_local = nnz / (Gr*Gc): A is replicated along f, so splitting the
    # feature axis does NOT shrink the per-GPU edge count -- this is the
    # narrow-column penalty that a flat-tflops model misses (featpar's
    # hidden cost).  c0 = per-nonzero index/value traffic + short-row
    # overhead; c1 = per-column traffic of the H row and output.
    # Calibrated on the 2026-07 products full-graph sweeps (4/8/16 GPU).
    spmm_c0: float = 20.0        # bytes per nonzero, F-independent
    spmm_c1: float = 8.0         # bytes per nonzero per local column
    # narrow-column kernel inefficiency: extra c2 / F_loc^c2_exp bytes/nnz
    # (short rows underutilize warps; fitted to the measured featpar
    # penalties at F_loc=32 (+72B) and F_loc=16 (+173B), vanishing >=64)
    spmm_c2: float = 5700.0
    spmm_c2_exp: float = 1.26
    hbm_bw: float = 1.3e12       # effective HBM bandwidth
    l2_bytes: float = 30e6       # if the dense H fits in L2 the narrow-
                                 # column penalty vanishes (compact
                                 # minibatch subgraphs are cache-resident)
    mini_event_us: float = 150.0  # fixed per-collective overhead on the
                                  # minibatch path (a2a-emulated uneven
                                  # collectives + block-reorder kernels)
    gpu_mem_budget: float = 34e9  # usable HBM for static A + activations
    adj_bytes_per_edge: float = 24.0   # A + A^T, int64 idx + fp32 val
    # measured kernel curve (analysis/spmm_microbench.py --out ...):
    # {f_loc: extra bytes/nnz} tables replacing the analytic c2 term
    kernel_penalty_large: dict = None
    kernel_penalty_small: dict = None

    def load_kernel_curve(self, path):
        with open(path) as f:
            d = json.load(f)
        self.spmm_c0 = d["c0"]
        self.spmm_c1 = d["c1"]
        self.kernel_penalty_large = {int(k): max(0.0, v)
                                     for k, v in d["penalty_large"].items()}
        self.kernel_penalty_small = {int(k): max(0.0, v)
                                     for k, v in d["penalty_small"].items()}

    def bw_alpha(self, internode: bool):
        if internode:
            return self.bw_inter, self.alpha_inter
        return self.bw_intra, self.alpha_intra


# --------------------------------------------------------------------------
# grid & axis->hardware mapping
# --------------------------------------------------------------------------

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Grid:
    gx: int
    gy: int
    gz: int

    @property
    def P(self):
        return self.gx * self.gy * self.gz

    def size(self, axis):
        return {"x": self.gx, "y": self.gy, "z": self.gz}[axis]

    def stride(self, axis):
        # rank = z*(gx*gy) + x*gy + y   (AxoNN: y fastest, then x, then z)
        return {"y": 1, "x": self.gy, "z": self.gx * self.gy}[axis]

    def axis_internode(self, axis, gpus_per_node):
        """True if a process group along `axis` spans more than one node."""
        g = self.size(axis)
        if g == 1:
            return False
        span = self.stride(axis) * g          # rank span of the group
        if span > gpus_per_node:
            return True
        return gpus_per_node % span != 0

    def axes_internode(self, axes, gpus_per_node):
        return any(self.axis_internode(a, gpus_per_node) for a in axes)


def layer_axes(l, use_3d_linear=True):
    """(r, c, f) grid-axis letters for GCN layer l (matches plexus code)."""
    if use_3d_linear:
        rot = [("y", "x", "z"), ("z", "y", "x"), ("x", "z", "y")]
    else:
        rot = [("z", "x", "y"), ("y", "z", "x"), ("x", "y", "z")]
    return rot[l % 3]


# --------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------

@dataclass
class Workload:
    name: str
    Nb: float          # nodes per (mini)batch step
    nnz: float         # edges per (mini)batch step (incl. self loops)
    F: float           # hidden width
    L: int = 3         # number of GCN layers
    F_in: float = 100  # raw input features (input_linear)
    C: float = 47      # classes (output_linear + loss)
    steps: int = 1     # steps per epoch (for epoch-time reporting)
    static_nnz: float = None   # full-graph edges kept resident (minibatch
                               # compacts FROM the full shards); defaults
                               # to nnz


WORKLOADS = {
    # ogbn-products minibatch (ratio=0.05 of 2.449M nodes, 123.7M edges)
    "products_mini": Workload("products_mini", Nb=0.05 * 2449029,
                              nnz=0.05 * 123718280, F=256, L=3,
                              F_in=100, C=47, steps=21),
    # ogbn-products full graph
    "products_full": Workload("products_full", Nb=2449029, nnz=123718280,
                              F=256, L=3, F_in=100, C=47, steps=1),
    # ogbn-papers100M full graph (symmetrised)
    "papers_full": Workload("papers_full", Nb=111059956, nnz=3231371744,
                            F=256, L=3, F_in=128, C=172, steps=1),
    # protein_8m (8.75M nodes, ~1.3B edges, 32 classes)
    "protein_8m_full": Workload("protein_8m_full", Nb=8745542, nnz=1.3e9,
                                F=256, L=3, F_in=128, C=32, steps=1),
    "protein_8m_mini": Workload("protein_8m_mini", Nb=0.05 * 8745542,
                                nnz=0.05 * 0.05 * 1.3e9, F=256, L=3,
                                F_in=128, C=32, steps=20,
                                static_nnz=1.3e9),
}


def adjacency_bytes(scheme, grid, wl, mach):
    """Static per-GPU adjacency memory for feasibility checks: steady
    (A + A^T for all layouts) PLUS the load-time conversion peak of the
    largest layout (COO->CSR holds a ~20B/edge copy alive).  Validated
    against 6 OOM/fit observations on protein_8m (2026-07-09).
    Minibatch keeps the FULL shards resident (compaction source)."""
    nnz = wl.static_nnz if wl.static_nnz is not None else wl.nnz
    conv = 20.0                                      # bytes/edge transient
    if scheme == "featpar":
        return nnz * (mach.adj_bytes_per_edge + conv)
    steady, biggest = 0.0, 0.0
    for l in range(min(3, wl.L)):
        Gr, Gc, _ = _layer_rcf_sizes(grid, scheme, l)
        share = nnz / (Gr * Gc)
        steady += share * mach.adj_bytes_per_edge
        biggest = max(biggest, share)
    return steady + biggest * conv


# --------------------------------------------------------------------------
# collective events
# --------------------------------------------------------------------------

@dataclass
class Event:
    name: str          # e.g. "AR(AGG)"
    kind: str          # "ar" | "rs" | "ag" | "reshard"
    axes: tuple        # grid-axis letters the collective runs over
    words: float       # full message size in words ("reshard": moved volume)
    phase: str         # "fwd" | "bwd"

    def group_size(self, grid):
        g = 1
        for a in self.axes:
            g *= grid.size(a)
        return g

    def _stages(self, grid, gpus_per_node):
        """Decompose the group into (factor, internode) stages.

        A mixed axis (group partly inside a node, partly across nodes) is
        split into an intra-node factor and an inter-node factor, modelling
        a hierarchical implementation (NCCL trees / explicit 2-level rings).
        """
        stages = []
        for a in self.axes:
            g = grid.size(a)
            if g == 1:
                continue
            span = grid.stride(a) * g
            if not grid.axis_internode(a, gpus_per_node):
                stages.append((g, False))
            elif grid.stride(a) >= gpus_per_node:
                stages.append((g, True))      # one rank per node
            else:
                gi = max(1, gpus_per_node // grid.stride(a))  # intra part
                gi = math.gcd(gi, g)
                if gi > 1:
                    stages.append((gi, False))
                if g // gi > 1:
                    stages.append((g // gi, True))
        return stages

    def moved_words(self, grid):
        """Words moved per GPU (hierarchical ring model)."""
        g = self.group_size(grid)
        if g == 1 or self.words <= 0:
            return 0.0
        if self.kind == "ar":
            return 2.0 * (g - 1) / g * self.words
        if self.kind in ("rs", "ag"):
            return (g - 1) / g * self.words
        if self.kind == "reshard":
            return self.words        # already a volume (target - have)
        raise ValueError(self.kind)

    def _staged_time(self, stages, m_start, mach, bpw, gather: bool):
        """Time of a staged AG (gather=True, message grows, inter-node
        stages first) or RS (message shrinks, intra-node stages first).
        `m_start` is the initial per-GPU shard size in words."""
        # AG: inter-node factors first (message still small);
        # RS: intra-node factors first (message shrinks before the NIC)
        order = sorted(stages, key=lambda s: (not s[1]) if gather else s[1])
        t = 0.0
        m = m_start
        for gfac, inter in order:
            bw, alpha = mach.bw_alpha(inter)
            if gather:
                t += m * (gfac - 1) * bpw / bw + (gfac - 1) * alpha
                m *= gfac
            else:
                t += m * (gfac - 1) / gfac * bpw / bw + (gfac - 1) * alpha
                m /= gfac
        return t

    def time(self, grid, mach, bytes_per_word):
        g = self.group_size(grid)
        if g == 1 or self.words <= 0:
            return 0.0
        stages = self._stages(grid, mach.gpus_per_node)
        if not stages:
            return 0.0
        if self.kind in ("ag", "reshard"):
            # AG of full message m: start from shard m/g, grow to m.
            # reshard: words is the received volume; equivalent AG has
            # full size words*g/(g-1) -> start shard words/(g-1).
            if self.kind == "ag":
                m0 = self.words / g
            else:
                m0 = self.words / (g - 1)
            return self._staged_time(stages, m0, mach, bytes_per_word, True)
        if self.kind == "rs":
            return self._staged_time(stages, self.words, mach,
                                     bytes_per_word, False)
        if self.kind == "ar":
            # hierarchical AR = staged RS (intra first) + staged AG (inter
            # first); on a pure group this equals the classic ring formula
            t = self._staged_time(stages, self.words, mach, bytes_per_word,
                                  False)
            t += self._staged_time(stages, self.words / g, mach,
                                   bytes_per_word, True)
            return t
        raise ValueError(self.kind)


# --------------------------------------------------------------------------
# schemes: build the per-step event list for the L GCN layers
# --------------------------------------------------------------------------

def scheme_current(wl, grid, use_3d_linear=True):
    """plexus today: 4 big ARs per layer, activations replicated over f."""
    ev = []
    N, F = wl.Nb, wl.F
    for l in range(wl.L):
        r, c, f = layer_axes(l, use_3d_linear)
        Gr, Gc, Gf = grid.size(r), grid.size(c), grid.size(f)
        m_agg = (N / Gr) * (F / Gf)
        m_out = (N / Gr) * (F / Gc)
        m_h   = (N / Gc) * (F / Gf)
        m_w   = F * F / (Gf * Gc)
        ev += [
            Event("AR(AGG)",      "ar", (c,), m_agg, "fwd"),
            Event("AG(W)",        "ag", (r,), m_w,   "fwd"),
            Event("AR(OUT)",      "ar", (f,), m_out, "fwd"),
            Event("AR(grad_agg)", "ar", (c,), m_agg, "bwd"),
            Event("RS(grad_W)",   "rs", (r,), m_w,   "bwd"),
            Event("AR(grad_x)",   "ar", (r,), m_h,   "bwd"),
        ]
    return ev


def scheme_current_sp(wl, grid, use_3d_linear=True):
    """current + sequence-parallel elementwise: identical comm events
    (AR(OUT) becomes RS+AG around the elementwise = same cost), but the
    elementwise redundancy over f disappears."""
    return scheme_current(wl, grid, use_3d_linear)


def scheme_half(wl, grid, use_3d_linear=True):
    """
    Proposed: same A placement & rotation, but only "half" collectives.
    Steady-state activation layout between layers: fully row-scattered,
    N/P rows x full F cols per GPU.  Derivation of volumes: each AG moves
    (target - have) words, where "have" is the reusable local sub-block:

      fwd:  AG(H): rows c-block, cols f-block, repl r.
                target N*F/(Gc*Gf), have N*F/(P*Gf)
                -> vol (N*F/P)(Gr - 1/Gf), group axes {r,f}
            SpMM -> partial AGG (N/Gr x F/Gf); RS(AGG) over c
                -> vol (N*F/P)(Gc-1); rows split (r,c), cols f
            AG(W) over (r,c) (ZeRO-sharded W, full F^2/Gf) -- small
            GEMM (contract f, W cols unsplit) -> partial OUT (N/(GrGc) x F)
            RS(OUT) over f -> vol (N*F/P)(Gf-1); rows split (r,c,f)=N/P
      bwd:  AG(grad_out): rows (r,c)-block, full cols.
                target N*F*Gf/P, have N*F/(P*Gc)
                -> vol (N*F/P)(Gf - 1/Gc), group axes {c,f}
            GEMM^T (contract unsplit F_out) -> grad_AGG exact, no comm
            RS(grad_W) over (r,c) -- small
            AG(grad_agg) over c -> vol (N*F/P)(Gc-1)   [exact reuse]
            SpMM^T -> partial grad_H (N/Gc x F/Gf); RS(grad_H) over r
                -> vol (N*F/P)(Gr-1); rows split (c,r), cols f
    """
    ev = []
    N, F, P = wl.Nb, wl.F, grid.P
    for l in range(wl.L):
        r, c, f = layer_axes(l, use_3d_linear)
        Gr, Gc, Gf = grid.size(r), grid.size(c), grid.size(f)
        m_agg = (N / Gr) * (F / Gf)          # partial AGG on each GPU
        m_out_rs = (N / (Gr * Gc)) * F       # partial OUT before RS over f
        m_w = F * F / Gf                     # full W needed per GPU
        ev += [
            Event("AG(H)",        "reshard", (r, f),
                  (N * F / P) * (Gr - 1.0 / Gf), "fwd"),
            Event("RS(AGG)",      "rs", (c,), m_agg, "fwd"),
            Event("AG(W)",        "ag", (r, c), m_w, "fwd"),
            Event("RS(OUT)",      "rs", (f,), m_out_rs, "fwd"),
            Event("AG(grad_out)", "reshard", (c, f),
                  (N * F / P) * (Gf - 1.0 / Gc), "bwd"),
            Event("RS(grad_W)",   "rs", (r, c), m_w, "bwd"),
            Event("AG(grad_agg)", "reshard", (c,),
                  (N * F / P) * (Gc - 1.0), "bwd"),
            Event("RS(grad_H)",   "rs", (r,), (N / Gc) * (F / Gf), "bwd"),
        ]
    return ev


def scheme_hybrid(wl, grid, use_3d_linear=True):
    """
    3D SpMM + 1D (row-parallel) GEMM, glued by all-to-alls over f.

    Same A placement & rotation as `half`, but after RS(AGG) over c the
    exact (not partial!) AGG is converted from col-split-f to row-split-f
    by an all-to-all (a pure permutation: every element moves at most once,
    volume (Gf-1)/Gf * u).  The GEMM then runs row-parallel with W fully
    replicated: contraction over the full F is local, output needs no
    reduction, and the backward GEMM^T is local too.  Weight gradients need
    an O(F^2) all-reduce over all P.

      fwd:  A2A_in(f), AG(H) over r, SpMM, RS(AGG) over c, A2A_mid(f),
            GEMM (local)
      bwd:  GEMM^T (local), A2A_mid_back(f), AG(g_AGG) over c, SpMM^T,
            RS(g_H) over r, A2A_in_back(f), AR(grad_W) over all axes

    Per-layer moved words/GPU:
        (Nb*F/P) * (2(Gr-1) + 2(Gc-1) + 4(Gf-1)/Gf)  + O(F^2)
    The f-axis cost is bounded by a constant 4u, so enlarging Gf (up to the
    F/32 kernel limit) shrinks the node axes almost for free.
    """
    ev = []
    N, F, P = wl.Nb, wl.F, grid.P
    all_axes = tuple(a for a in AXES if grid.size(a) > 1) or ("x",)
    for l in range(wl.L):
        r, c, f = layer_axes(l, use_3d_linear)
        Gr, Gc, Gf = grid.size(r), grid.size(c), grid.size(f)
        u = N * F / P
        a2a = u * (Gf - 1.0) / Gf
        ev += [
            Event("A2A(x)",       "reshard", (f,), a2a, "fwd"),
            Event("AG(H)",        "ag", (r,), (N / Gc) * (F / Gf), "fwd"),
            Event("RS(AGG)",      "rs", (c,), (N / Gr) * (F / Gf), "fwd"),
            Event("A2A(AGG)",     "reshard", (f,), a2a, "fwd"),
            Event("A2A(gAGG)",    "reshard", (f,), a2a, "bwd"),
            Event("AR(grad_W)",   "ar", all_axes, F * F, "bwd"),
            Event("AG(gAGG)",     "reshard", (c,), u * (Gc - 1.0), "bwd"),
            Event("RS(gH)",       "rs", (r,), (N / Gc) * (F / Gf), "bwd"),
            Event("A2A(gx)",      "reshard", (f,), a2a, "bwd"),
        ]
    return ev


def scheme_featpar(wl, grid, use_3d_linear=True):
    """
    Feature-parallel SpMM + 1D GEMM, A fully replicated ("featpar").

    The Gr=Gc=1 member of the hybrid family, with FIXED axes (no rotation
    needed: with no node-dim split there is no layout mismatch between
    layers).  A is replicated on every GPU; H is split by feature columns
    (F/P each) for the SpMM, which is then fully local and exact; two
    all-to-alls per direction flip between column-split (SpMM) and
    row-split (GEMM/element-wise) layouts.

    Per-layer moved words/GPU:  4 * (P-1)/P * u  + O(F^2)   -- constant!
    Constraints: F/P >= 32 (SpMM kernel width), A must fit replicated.
    """
    if grid.gx != 1 or grid.gy != 1:
        return None
    ev = []
    N, F, P = wl.Nb, wl.F, grid.P
    axes = ("z",) if grid.gz > 1 else ("x",)
    u = N * F / P
    a2a = u * (P - 1.0) / P
    for l in range(wl.L):
        ev += [
            Event("A2A(x)",     "reshard", axes, a2a, "fwd"),
            Event("A2A(AGG)",   "reshard", axes, a2a, "fwd"),
            Event("A2A(gAGG)",  "reshard", axes, a2a, "bwd"),
            Event("AR(grad_W)", "ar", axes, F * F, "bwd"),
            Event("A2A(gx)",    "reshard", axes, a2a, "bwd"),
        ]
    return ev


def scheme_cosma2d(wl, grid, use_3d_linear=True):
    """
    Per-op optimal layouts + explicit reshard, no rotation (fixed axes).
    Requires gz == 1: A stationary on (r=x, c=y) 2D grid every layer,
    W fully replicated, GEMM row-parallel (comm-free).
      fwd:  AG(H) over r, RS(AGG) over c
      bwd:  AG(grad_agg) over c, RS(grad_H) over r, AR(grad_W) over all
    """
    if grid.gz != 1:
        return None
    ev = []
    N, F, P = wl.Nb, wl.F, grid.P
    r, c = "x", "y"
    Gr, Gc = grid.size(r), grid.size(c)
    all_axes = tuple(a for a in AXES if grid.size(a) > 1) or ("x",)
    for l in range(wl.L):
        ev += [
            Event("AG(H)",        "ag", (r,), (N / Gc) * F, "fwd"),
            Event("RS(AGG)",      "rs", (c,), (N / Gr) * F, "fwd"),
            Event("AG(grad_agg)", "ag", (c,), (N / Gr) * F, "bwd"),
            Event("RS(grad_H)",   "rs", (r,), (N / Gc) * F, "bwd"),
            Event("AR(grad_W)",   "ar", all_axes, F * F, "bwd"),
        ]
    return ev


def scheme_oned(wl, grid, use_3d_linear=True):
    """CAGNET-style 1D: A and H row-split across all P, no locality."""
    ev = []
    N, F = wl.Nb, wl.F
    axes = tuple(a for a in AXES if grid.size(a) > 1) or ("x",)
    for l in range(wl.L):
        ev += [
            Event("AG(H)",      "ag", axes, N * F, "fwd"),
            Event("RS(grad_H)", "rs", axes, N * F, "bwd"),
            Event("AR(grad_W)", "ar", axes, F * F, "bwd"),
        ]
    return ev


SCHEMES = {
    "current": scheme_current,
    "current_sp": scheme_current_sp,
    "half": scheme_half,
    "hybrid": scheme_hybrid,
    "featpar": scheme_featpar,
    "cosma2d": scheme_cosma2d,
    "oned": scheme_oned,
}

# does the scheme use every axis as feature axis at some layer (rotation)?
ROTATING = {"current": True, "current_sp": True, "half": True,
            "hybrid": True, "featpar": False, "cosma2d": False,
            "oned": False}
# elementwise redundancy factor over the f axis
ELEM_REDUNDANT = {"current": True, "current_sp": False, "half": False,
                  "hybrid": False, "featpar": False, "cosma2d": False,
                  "oned": False}


def adjacency_replication(scheme, grid):
    """Copies of each stored A shard across the cluster (memory factor).

    A is stationary on its (r, c) grid and replicated over the remaining
    axes: factor = P / (Gr * Gc).
    """
    if scheme == "oned":
        return 1
    if scheme == "featpar":
        return grid.P
    if scheme == "cosma2d":
        return 1
    # rotating 3D family: per layer the f axis replicates A
    reps = []
    for l in range(3):
        _, _, f = layer_axes(l)
        reps.append(grid.size(f))
    return max(reps)


# --------------------------------------------------------------------------
# non-communication costs
# --------------------------------------------------------------------------

def _layer_rcf_sizes(grid, scheme, l):
    """(Gr, Gc, Gf) axis sizes of layer l under the scheme's axis policy."""
    if scheme == "oned":
        return grid.P, 1, 1
    if scheme == "cosma2d":
        return grid.size("x"), grid.size("y"), 1
    if scheme == "featpar":
        return 1, 1, grid.P
    r, c, f = layer_axes(l)
    return grid.size(r), grid.size(c), grid.size(f)


def compute_times(wl, grid, mach, scheme, bytes_act=4):
    """(spmm, gemm, elementwise) seconds per step for the L GCN layers."""
    P = grid.P
    # SpMM: bandwidth roofline, per layer (axes rotate per layer).
    # fwd (A) + bwd (A^T) -> factor 2.
    spmm = 0.0
    for l in range(wl.L):
        Gr, Gc, Gf = _layer_rcf_sizes(grid, scheme, l)
        nnz_loc = wl.nnz / (Gr * Gc)
        f_loc = wl.F / Gf
        bytes_per_nnz = mach.spmm_c0 + mach.spmm_c1 * f_loc
        h_in_l2 = wl.Nb / Gc * f_loc * 4 <= mach.l2_bytes
        table = (mach.kernel_penalty_small if h_in_l2
                 else mach.kernel_penalty_large)
        if table is not None:
            # measured kernel curve (microbench): nearest-F lookup
            fk = min(table, key=lambda k: abs(k - f_loc))
            bytes_per_nnz += table.get(fk, 0.0) if f_loc < 64 else 0.0
        elif f_loc < 64 and not h_in_l2:
            # analytic fallback fitted on the products sweeps
            bytes_per_nnz += mach.spmm_c2 / (f_loc ** mach.spmm_c2_exp)
        spmm += 2 * nnz_loc * bytes_per_nnz / mach.hbm_bw
    # GEMM: OUT, grad_AGG, grad_W per layer
    gemm_flops = 2 * wl.Nb * wl.F * wl.F / P * 3
    gemm = wl.L * gemm_flops / (mach.gemm_tflops * 1e12)
    elem_words = wl.Nb * wl.F / P
    if ELEM_REDUNDANT[scheme]:
        red = 0.0
        for l in range(wl.L):
            _, _, f = layer_axes(l)
            red += grid.size(f)
        elem_words *= red / wl.L
    elem = wl.L * mach.elem_passes * elem_words * bytes_act / mach.elem_bw
    return spmm, gemm, elem


def activation_words(wl, grid, scheme):
    """Rough per-GPU activation memory (saved-for-backward), words/layer."""
    N, F, P = wl.Nb, wl.F, grid.P
    if scheme in ("half", "cosma2d", "oned"):
        return N * F / P * 2          # AGG + OUT, fully scattered
    # current: AGG (N/Gr x F/Gf) + OUT replicated over f
    l0 = layer_axes(0)
    Gr, Gc = grid.size(l0[0]), grid.size(l0[1])
    return N * F / (Gr * grid.size(l0[2])) + N * F / (Gr * Gc)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@dataclass
class Result:
    scheme: str
    grid: Grid
    comm_s: float
    comm_words: float
    spmm_s: float
    gemm_s: float
    elem_s: float
    events: list = field(default_factory=list)

    @property
    def total_s(self):
        return self.comm_s + self.spmm_s + self.gemm_s + self.elem_s


def evaluate(wl, grid, mach, scheme_name, bytes_comm=4, bytes_act=4):
    events = SCHEMES[scheme_name](wl, grid)
    if events is None:
        return None
    comm_s = sum(e.time(grid, mach, bytes_comm) for e in events)
    if wl.steps > 1:
        # minibatch path: fixed per-collective overhead of the a2a-emulated
        # uneven collectives and block-reorder kernels
        n_active = sum(1 for e in events if e.group_size(grid) > 1)
        comm_s += n_active * mach.mini_event_us * 1e-6
    comm_words = sum(e.moved_words(grid) for e in events)
    spmm, gemm, elem = compute_times(wl, grid, mach, scheme_name, bytes_act)
    return Result(scheme_name, grid, comm_s, comm_words, spmm, gemm, elem,
                  events)


def factor_grids(P):
    """All (gx, gy, gz) with gx*gy*gz == P."""
    out = []
    for gx in range(1, P + 1):
        if P % gx:
            continue
        rest = P // gx
        for gy in range(1, rest + 1):
            if rest % gy:
                continue
            out.append(Grid(gx, gy, rest // gy))
    return out


def best_grid(wl, P, mach, scheme_name, bytes_comm=4, bytes_act=4,
              min_local_f=32):
    """Search all grid factorizations of P for the fastest one."""
    best = None
    for g in factor_grids(P):
        if scheme_name in ("oned", "featpar") and (g.gx != 1 or g.gy != 1):
            continue
        if scheme_name == "featpar" and wl.F % P != 0:
            continue   # feature dim must divide P (weight/A2A splits)
        n_static = (wl.static_nnz and wl.Nb / (wl.nnz / wl.static_nnz) ** 0.5
                    or wl.Nb)   # full node count (mini samples ratio*N)
        static_feats = n_static / g.gx * wl.F_in * 4   # x-block features
        acts = wl.Nb * wl.F * 4 * 3 / P
        if adjacency_bytes(scheme_name, g, wl, mach) + static_feats + acts \
                > mach.gpu_mem_budget:
            continue   # static A + features + activations exceed HBM
        if ROTATING[scheme_name]:
            # with rotation every axis serves as the feature axis in some
            # layer; keep local feature width sane on all of them
            if wl.F / max(g.gx, g.gy, g.gz) < min_local_f:
                continue
        r = evaluate(wl, g, mach, scheme_name, bytes_comm, bytes_act)
        if r is None:
            continue
        if best is None or r.total_s < best.total_s:
            best = r
    return best


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def fmt_bytes(words, bytes_per_word=4):
    b = words * bytes_per_word
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}PB"


def print_events(res, mach, bytes_comm):
    n_layer = max(1, len(res.events) // 3)
    print(f"    layer-0 events (grid {res.grid.gx}x{res.grid.gy}x{res.grid.gz}):")
    for e in res.events[:n_layer]:
        g = e.group_size(res.grid)
        inter = res.grid.axes_internode(e.axes, mach.gpus_per_node)
        loc = "inter" if inter else "intra"
        print(f"      {e.phase} {e.name:14s} axes={'/'.join(e.axes):4s} "
              f"g={g:3d} {loc} "
              f"moved={fmt_bytes(e.moved_words(res.grid), bytes_comm):>9s} "
              f"t={e.time(res.grid, mach, bytes_comm)*1e3:7.3f}ms")


def compare(wl, P_list, mach, bytes_comm, bytes_act, show_events=False,
            schemes=("current", "current_sp", "half", "hybrid", "featpar",
                     "cosma2d", "oned")):
    print(f"\n=== workload {wl.name}: Nb={wl.Nb:.3g} nnz={wl.nnz:.3g} "
          f"F={wl.F:g} L={wl.L}  (comm dtype {bytes_comm}B) ===")
    hdr = (f"{'P':>4s} {'scheme':>10s} {'grid':>10s} {'comm':>9s} "
           f"{'spmm':>7s} {'gemm':>7s} {'elem':>7s} {'total':>8s} "
           f"{'moved/GPU':>10s} {'A-repl':>6s} {'speedup':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for P in P_list:
        base_t = None
        for s in schemes:
            r = best_grid(wl, P, mach, s, bytes_comm, bytes_act)
            if r is None:
                print(f"{P:4d} {s:>10s}   (no valid grid)")
                continue
            if s == "current":
                base_t = r.total_s
            rel = f"{base_t / r.total_s:7.2f}x" if base_t else "      -"
            g = r.grid
            print(f"{P:4d} {s:>10s} {g.gx:3d}x{g.gy}x{g.gz:<3d} "
                  f"{r.comm_s*1e3:8.2f}m {r.spmm_s*1e3:6.2f}m "
                  f"{r.gemm_s*1e3:6.2f}m {r.elem_s*1e3:6.2f}m "
                  f"{r.total_s*1e3:7.2f}m "
                  f"{fmt_bytes(r.comm_words, bytes_comm):>10s} "
                  f"x{adjacency_replication(s, g):<4d} {rel}")
            if show_events:
                print_events(r, mach, bytes_comm)
        print()


def grids_table(wl, P, mach, bytes_comm, bytes_act,
                schemes=("current", "half")):
    print(f"\n=== all grid factorizations, P={P}, workload {wl.name} ===")
    for s in schemes:
        print(f"-- scheme {s}")
        rows = [evaluate(wl, g, mach, s, bytes_comm, bytes_act)
                for g in factor_grids(P)]
        rows = [r for r in rows if r is not None]
        rows.sort(key=lambda r: r.total_s)
        for r in rows[:10]:
            g = r.grid
            print(f"   {g.gx:3d}x{g.gy}x{g.gz:<3d} comm={r.comm_s*1e3:8.2f}ms "
                  f"total={r.total_s*1e3:8.2f}ms "
                  f"moved={fmt_bytes(r.comm_words, bytes_comm)}")


def recommend(wl, P, mach, bytes_comm=2):
    """Predict the best (scheme, grid) for this workload at P GPUs and
    print the launch flags, including the empirical overlap rule."""
    best = None
    for s in ("hybrid", "featpar"):
        r = best_grid(wl, P, mach, s, bytes_comm, 4)
        if r is not None and (best is None or r.total_s < best.total_s):
            best = r
    if best is None:
        print(f"\n=== recommendation: {wl.name}, P={P} ===")
        print("  NO feasible config: static adjacency (+load peak) exceeds "
              "the HBM budget on every grid.")
        print("  Options: more GPUs, --int32_indices (A memory x2/3), or "
              "minibatch instead of full-graph.")
        return None
    g = best.grid
    kind = "featpar" if best.scheme == "featpar" else "hybrid"
    print(f"\n=== recommendation: {wl.name}, P={P} "
          f"(wire {bytes_comm}B) ===")
    print(f"  scheme: {kind}  grid: --G_intra_r {g.gx} "
          f"--G_intra_c {g.gy} --G_intra_d {g.gz}")
    print(f"  predicted: total {best.total_s*1e3:.1f}ms/step "
          f"(comm {best.comm_s*1e3:.1f}, spmm {best.spmm_s*1e3:.1f})")
    flags = "--allreduce_lowp"
    # empirical rule (products 4/8/16-GPU A/B): async bwd overlap helps on
    # NVLink-dominated groups, hurts once the NIC is the shared bottleneck
    if P <= 8:
        flags += " --overlap_bwd_comm"
    print(f"  flags: {flags}"
          + ("" if P <= 8 else "   (overlap OFF: NIC contention at P>=16)"))
    runner = ("examples/train_mini_rs.py" if wl.steps > 1
              else "examples/train.py --conv rs")
    print(f"  runner: {runner}")
    return best


# --------------------------------------------------------------------------
# calibration against a measured breakdown.json
# --------------------------------------------------------------------------

def calibrate(path, wl, grid, mach, bytes_comm=4):
    """Compare model vs measured per-epoch breakdown (products_mini, 2x2x2).

    Note: the breakdown parser puts two dense GEMMs (GRAD_AGG = GRAD_OUT*W^T
    and GRAD_W = AGG^T*GRAD_OUT) into its "SPMM" bucket; we regroup here.
    """
    with open(path) as f:
        data = json.load(f)
    run = data.get("G1_sampopt") or next(iter(data.values()))
    steps = wl.steps
    det = run["breakdown_details_ms"]
    meas_ar = dict(det["Allreduce"])
    spmm_d = dict(det["SPMM"])
    gemm_d = dict(det["GEMM"])
    ev = scheme_current(wl, grid)
    groups = {
        "allreduce H": "AR(AGG)",
        "allreduce Q": "AR(OUT)",
        "gcn conv bwd/all-reduce": "AR(grad_agg)",
        "allreduce grad_x": "AR(grad_x)",
    }
    print(f"\n=== calibration vs {os.path.basename(path)} "
          f"[{run['variant']}, grid {grid.gx}x{grid.gy}x{grid.gz}, "
          f"{steps} steps/epoch] ===")
    print(f"{'item':>26s} {'measured':>10s} {'model':>10s} {'meas/model':>10s}")
    for meas_name, ev_name in groups.items():
        t_model = sum(e.time(grid, mach, bytes_comm)
                      for e in ev if e.name == ev_name) * steps * 1e3
        t_meas = meas_ar.get(meas_name, float("nan"))
        print(f"{meas_name:>26s} {t_meas:9.1f}m {t_model:9.1f}m "
              f"{t_meas / t_model if t_model else float('nan'):10.2f}")
    spmm, gemm, elem = compute_times(wl, grid, mach, "current")
    spmm_meas = spmm_d.get("AGG = A * H", 0) + spmm_d.get(
        "GRAD_H = A.T * GRAD_AGG", 0)
    gemm_meas = (spmm_d.get("GRAD_AGG = GRAD_OUT * W.T", 0)
                 + spmm_d.get("GRAD_W = AGG.T * GRAD_OUT", 0)
                 + gemm_d.get("OUT = AGG * W", 0))
    print(f"{'SpMM (true: A*H + A^T*g)':>26s} {spmm_meas:9.1f}m "
          f"{spmm*steps*1e3:9.1f}m {spmm_meas/(spmm*steps*1e3):10.2f}")
    print(f"{'GCN GEMMs (regrouped)':>26s} {gemm_meas:9.1f}m "
          f"{gemm*steps*1e3:9.1f}m {gemm_meas/(gemm*steps*1e3):10.2f}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", default="products_mini",
                    choices=sorted(WORKLOADS))
    ap.add_argument("--P", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--dtype-comm", type=int, default=4,
                    help="bytes per word on the wire (4=fp32, 2=bf16)")
    ap.add_argument("--dtype-act", type=int, default=4)
    ap.add_argument("--grids", action="store_true",
                    help="print top grid factorizations for each P")
    ap.add_argument("--events", action="store_true",
                    help="print per-collective detail for the best grid")
    ap.add_argument("--calibrate", metavar="BREAKDOWN_JSON",
                    help="compare model vs measured breakdown")
    ap.add_argument("--recommend", action="store_true",
                    help="print the predicted best config per P")
    ap.add_argument("--bw-intra", type=float, default=None)
    ap.add_argument("--bw-inter", type=float, default=None)
    ap.add_argument("--min-local-f", type=int, default=32)
    args = ap.parse_args()

    mach = Machine()
    if args.bw_intra:
        mach.bw_intra = args.bw_intra
    if args.bw_inter:
        mach.bw_inter = args.bw_inter
    wl = WORKLOADS[args.workload]

    if args.calibrate:
        calibrate(args.calibrate, WORKLOADS["products_mini"], Grid(2, 2, 2),
                  mach, args.dtype_comm)
        return

    if args.recommend:
        for P in args.P:
            recommend(wl, P, mach, args.dtype_comm)
        return

    compare(wl, args.P, mach, args.dtype_comm, args.dtype_act,
            show_events=args.events)
    if args.grids:
        for P in args.P:
            grids_table(wl, P, mach, args.dtype_comm, args.dtype_act)


if __name__ == "__main__":
    main()
