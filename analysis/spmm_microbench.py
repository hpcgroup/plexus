#!/usr/bin/env python
"""
Single-GPU SpMM microbenchmark: measure torch.sparse.mm effective cost
(bytes moved per nonzero) as a function of the dense-matrix width F_loc.

Purpose: turn the cost model's narrow-column kernel term into an
independently *measured* machine parameter instead of a fit against
end-to-end results (analytic communication model x measured kernel
efficiency curve -- no circularity).

Two regimes are measured, matching the model's L2 gate:
  - "large": N chosen so the dense H spills L2 (full-graph regime)
  - "small": N chosen so H is L2-resident (compact-minibatch regime)

Output: a table of bytes/nnz vs F_loc per regime, plus a fit of the model
constants (c0, c1) on the wide-F points and the residual narrow-column
penalty at F_loc < 64.  Optionally writes JSON consumable by
gcn_comm_model.Machine (--out kernel_curve.json).

Run (any single GPU):
  python analysis/spmm_microbench.py --out analysis/kernel_curve.json
"""

import argparse
import json
import time

import torch


def make_csr(n_rows, n_cols, nnz, device, seed=0, powerlaw=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if powerlaw:
        # degree ~ Zipf-ish: sample rows with prob ~ 1/rank
        w = 1.0 / torch.arange(1, n_rows + 1, dtype=torch.float64)
        rows = torch.multinomial(w, nnz, replacement=True, generator=g)
    else:
        rows = torch.randint(0, n_rows, (nnz,), generator=g)
    cols = torch.randint(0, n_cols, (nnz,), generator=g)
    vals = torch.rand(nnz, generator=g)
    a = torch.sparse_coo_tensor(
        torch.stack([rows, cols]).to(device), vals.to(device),
        (n_rows, n_cols)).coalesce()
    return a.to_sparse_csr()


def bench_spmm(A, H, iters=20, warmup=5):
    for _ in range(warmup):
        torch.sparse.mm(A, H)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.sparse.mm(A, H)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run_regime(name, n, nnz, f_list, device, powerlaw):
    A = make_csr(n, n, nnz, device, powerlaw=powerlaw)
    nnz_real = A.values().numel()
    rows = []
    print(f"\n-- regime {name}: N={n:,} nnz={nnz_real:,} "
          f"({'powerlaw' if powerlaw else 'uniform'} degrees)")
    print(f"   {'F_loc':>6} {'ms':>9} {'bytes/nnz':>10} {'GB/s eff':>9}")
    for f in f_list:
        H = torch.randn(n, f, device=device)
        t = bench_spmm(A, H)
        bpn = t * BW_REF / nnz_real          # provisional, for display only
        bytes_per_nnz = t / nnz_real         # seconds/nnz; scaled later
        eff_bw = nnz_real * (12 + 8 * f) / t / 1e9   # vs naive traffic
        rows.append({"f": f, "ms": t * 1e3, "s_per_nnz": t / nnz_real,
                     "naive_gbs": eff_bw})
        print(f"   {f:6d} {t*1e3:9.3f} "
              f"{t/nnz_real*BW_REF:10.1f} {eff_bw:9.1f}")
    return {"name": name, "N": n, "nnz": nnz_real, "rows": rows}


BW_REF = 1.3e12   # reference bandwidth for the bytes/nnz display


def fit_constants(regime):
    """c0 + c1*F fitted on F>=64 points; narrow penalty = residual."""
    import numpy as np
    pts = [(r["f"], r["s_per_nnz"] * BW_REF) for r in regime["rows"]]
    wide = [(f, b) for f, b in pts if f >= 64]
    Fs = np.array([f for f, _ in wide]); Bs = np.array([b for _, b in wide])
    c1, c0 = np.polyfit(Fs, Bs, 1)
    penalty = {f: b - (c0 + c1 * f) for f, b in pts if f < 64}
    return float(c0), float(c1), {int(k): float(v)
                                  for k, v in penalty.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write kernel curve JSON")
    ap.add_argument("--large-n", type=int, default=2400000)
    ap.add_argument("--large-nnz", type=int, default=120000000)
    ap.add_argument("--small-n", type=int, default=120000)
    ap.add_argument("--small-nnz", type=int, default=6000000)
    ap.add_argument("--powerlaw", action="store_true", default=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.cuda.init()
    print(f"GPU: {torch.cuda.get_device_name()}")

    f_list = [4, 8, 16, 32, 64, 128, 256]
    large = run_regime("large(H>L2)", args.large_n, args.large_nnz,
                       f_list, device, args.powerlaw)
    small = run_regime("small(H<L2)", args.small_n, args.small_nnz,
                       f_list, device, args.powerlaw)

    c0, c1, penalty = fit_constants(large)
    _, _, penalty_small = fit_constants(small)
    print(f"\n== fitted on wide-F (>=64) of the LARGE regime ==")
    print(f"   c0 = {c0:.1f} bytes/nnz, c1 = {c1:.2f} bytes/nnz/col")
    print(f"   narrow-column penalty (large regime, bytes/nnz): {penalty}")
    print(f"   narrow-column penalty (small/L2 regime):        {penalty_small}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"bw_ref": BW_REF, "c0": c0, "c1": c1,
                       "penalty_large": penalty,
                       "penalty_small": penalty_small,
                       "regimes": [large, small]}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
