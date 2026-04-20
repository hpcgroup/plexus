#!/usr/bin/env python3
"""
Parse Plexus strong scaling logs and extract timing breakdowns.

Categories:
  sampling   = prefetch launch + prefetch wait
  GEMM       = input_linear X*W + OUT=AGG*W + output_linear X*W
               + output_linear GRAD_X + output_linear GRAD_W + input_linear GRAD_W
  SPMM       = AGG=A*H + GRAD_AGG=GRAD_OUT*W.T + GRAD_W=AGG.T*GRAD_OUT + GRAD_H=A.T*GRAD_AGG
  Allreduce  = all-reduce (under input_linear/output_linear/linear_3d_bwd)
               + async AR(AGG) launch + allreduce H + allreduce Q
               + async AR(grad_x) launch + async AR(grad_W+bias) launch
               + wait AR(grad_x) + wait AR(grad_W)
               + async AR(grad_agg) launch + wait AR(grad_agg) + allreduce grad_x
               (allreduce Q / allreduce grad_x use parent value;
                sub-items allreduce lowp comm dtype / allreduce lowp / allreduce copy back excluded)
  elementwise= OUT+BIAS (all instances) + activation compute (= activation - activation/all-reduce)
               + activation_bwd (estimated from train_step gap: norm+relu+dropout backward is
                 handled by torch autograd and has no explicit timer; we estimate it as
                 train_step - sum(timed direct children of train_step))
  DP         = sync_data_parallel_gradients (full timer, including dp grad sync sub-item)
  others     = epoch_time - sum(above)

Usage:
    python parse_strong_scaling.py <log_dir> [--epoch EPOCH] [-o OUTPUT]

Examples:
    python parse_strong_scaling.py ./products_14m/ --epoch 1
    python parse_strong_scaling.py ./products_14m/ --epoch 2 -o my_breakdown.json

Reading the output in Python:
    import json
    with open('breakdown.json') as f:
        data = json.load(f)
    for dp_key, info in data.items():
        print(dp_key, info['epoch_time_ms'], info['breakdown_ms'])
"""

import re
import sys
import json
import argparse
from pathlib import Path

CONTAINERS = {
    'train step', 'input_linear', 'gcn conv fwd', 'output_linear',
    'linear_3d_bwd', 'gcn conv bwd',
}

PARENT_LEVEL_ENTRIES = {
    'allreduce Q': 'Allreduce',
    'allreduce grad_x': 'Allreduce',
    'sync_data_parallel_gradients': 'DP',
}

LEAF_MAP = {
    'prefetch launch': 'sampling',
    'prefetch wait': 'sampling',
    'input_linear X * W': 'GEMM',
    'OUT = AGG * W': 'GEMM',
    'output_linear X * W': 'GEMM',
    'output_linear GRAD_X': 'GEMM',
    'output_linear GRAD_W': 'GEMM',
    'input_linear GRAD_W': 'GEMM',
    'AGG = A * H': 'SPMM',
    'GRAD_AGG = GRAD_OUT * W.T': 'SPMM',
    'GRAD_W = AGG.T * GRAD_OUT': 'SPMM',
    'GRAD_H = A.T * GRAD_AGG': 'SPMM',
    'async AR(AGG) launch': 'Allreduce',
    'allreduce H': 'Allreduce',
    'async AR(grad_x) launch': 'Allreduce',
    'async AR(grad_W+bias) launch': 'Allreduce',
    'wait AR(grad_x)': 'Allreduce',
    'wait AR(grad_W)': 'Allreduce',
    'async AR(grad_agg) launch': 'Allreduce',
    'wait AR(grad_agg)': 'Allreduce',
    'OUT + BIAS': 'elementwise',
}

ALLREDUCE_PARENTS = {'input_linear', 'output_linear', 'linear_3d_bwd', 'activation'}

CAT_ORDER = ['sampling', 'GEMM', 'SPMM', 'Allreduce', 'elementwise', 'DP', 'others']


def clean_ansi(text):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    text = re.sub(r'\[\d+m', '', text)
    return text


def parse_timer_line(line):
    depth = line.count('\t')
    clean = clean_ansi(line)
    m = re.search(
        r"'(.+?)\s*\|\s*Max Time:\s*([\d.]+)\s*ms\s*\|\s*Avg Time:\s*([\d.]+)\s*ms",
        clean,
    )
    if m:
        return depth, m.group(1).strip(), float(m.group(3))
    return None


def extract_epoch_timers(filepath, epoch_num):
    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()

    entries = []
    active = False
    ep_re = re.compile(rf"'epoch {epoch_num}\s*\|")
    any_ep_re = re.compile(r"'epoch \d+\s*\|")

    for line in lines:
        if not active:
            if ep_re.search(line):
                active = True
                r = parse_timer_line(line)
                if r:
                    entries.append(r)
            continue

        if 'Max Time:' in line and 'Avg Time:' in line:
            if any_ep_re.search(line) and not ep_re.search(line):
                break
            r = parse_timer_line(line)
            if r:
                entries.append(r)
        elif line.strip() == '':
            continue
        else:
            if not line.strip().startswith('|'):
                break

    depth_map = {}
    timers = []
    for depth, name, avg in entries:
        depth_map = {d: n for d, n in depth_map.items() if d < depth}
        depth_map[depth] = name
        parent = depth_map.get(depth - 1)
        timers.append({
            'depth': depth, 'name': name, 'avg': avg, 'parent': parent,
        })
    return timers


def categorize(timers):
    epoch_time = None
    cats = {c: 0.0 for c in CAT_ORDER if c != 'others'}
    details = {c: [] for c in cats}
    uncategorized = []
    skip_depth = None

    train_step_avg = None
    train_step_depth = None
    train_step_children_sum = 0.0

    activation_fwd_total = 0.0
    activation_fwd_ar = 0.0

    for t in timers:
        d, name, avg, par = t['depth'], t['name'], t['avg'], t['parent']

        if skip_depth is not None:
            if d > skip_depth:
                continue
            skip_depth = None

        if name.startswith('epoch'):
            epoch_time = avg
            continue

        if name in CONTAINERS:
            if name == 'train step':
                train_step_avg = avg
                train_step_depth = d
            elif train_step_depth is not None and par == 'train step':
                train_step_children_sum += avg
            continue

        if name == 'activation' and par == 'train step':
            activation_fwd_total = avg
            if train_step_depth is not None:
                train_step_children_sum += avg
            continue

        if par == 'train step' and train_step_depth is not None:
            train_step_children_sum += avg

        if name in PARENT_LEVEL_ENTRIES:
            c = PARENT_LEVEL_ENTRIES[name]
            cats[c] += avg
            details[c].append((name, avg))
            skip_depth = d
            continue

        if name == 'all-reduce':
            if par in ALLREDUCE_PARENTS:
                cats['Allreduce'] += avg
                details['Allreduce'].append((f'{par}/all-reduce', avg))
                if par == 'activation':
                    activation_fwd_ar = avg
            continue

        if name in LEAF_MAP:
            c = LEAF_MAP[name]
            cats[c] += avg
            details[c].append((name, avg))
            continue

        uncategorized.append((name, avg, par))

    act_compute = activation_fwd_total - activation_fwd_ar
    if act_compute > 0:
        cats['elementwise'] += act_compute
        details['elementwise'].append(('activation (compute)', act_compute))

    activation_bwd = 0.0
    if train_step_avg is not None:
        activation_bwd = train_step_avg - train_step_children_sum
        if activation_bwd > 0:
            cats['elementwise'] += activation_bwd
            details['elementwise'].append(('activation_bwd (estimated)', activation_bwd))

    cats['others'] = (epoch_time or 0) - sum(cats.values())
    return epoch_time, cats, details, uncategorized, activation_bwd


def get_dp(filename):
    m = re.search(r'G_DATA(\d+)', filename)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(
        description='Parse Plexus strong scaling logs into timing breakdowns',
    )
    ap.add_argument('log_dir', help='Directory containing .log files')
    ap.add_argument('--epoch', type=int, default=1,
                    help='Epoch number to extract (default: 1)')
    ap.add_argument('-o', '--output', default=None,
                    help='Output JSON path (default: <log_dir>/breakdown.json)')
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    files = sorted(log_dir.glob('*.log'))
    if not files:
        sys.exit(f'No .log files found in {log_dir}')

    results = {}
    for fp in files:
        dp = get_dp(fp.name)
        if dp is None:
            print(f'[WARN] skip {fp.name}: no G_DATA in filename', file=sys.stderr)
            continue

        timers = extract_epoch_timers(str(fp), args.epoch)
        if not timers:
            print(f'[WARN] no epoch {args.epoch} in {fp.name}', file=sys.stderr)
            continue

        ep, cats, det, unc, act_bwd = categorize(timers)
        results[f'DP{dp}'] = {
            'filename': fp.name,
            'G_DATA': dp,
            'epoch': args.epoch,
            'epoch_time_ms': round(ep, 3),
            'activation_bwd_estimate_ms': round(act_bwd, 3),
            'breakdown_ms': {k: round(v, 3) for k, v in cats.items()},
            'breakdown_pct': {
                k: round(v / ep * 100, 2) for k, v in cats.items()
            } if ep else {},
            'breakdown_details_ms': {
                k: [(n, round(t, 3)) for n, t in items]
                for k, items in det.items()
            },
            'others_details_ms': [
                (n, round(t, 3), p) for n, t, p in unc
            ],
        }

    results = dict(sorted(results.items(), key=lambda x: x[1]['G_DATA']))

    out = args.output or str(log_dir / 'breakdown.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to: {out}', file=sys.stderr)

    # ── Summary table (ms) ──
    hdr = f"{'DP':>4} {'epoch':>10}"
    for c in CAT_ORDER:
        hdr += f' {c:>12}'
    print('\n' + hdr)
    print('-' * len(hdr))
    for d in results.values():
        row = f"{d['G_DATA']:>4} {d['epoch_time_ms']:>10.3f}"
        for c in CAT_ORDER:
            row += f" {d['breakdown_ms'].get(c, 0):>12.3f}"
        print(row)

    # ── Summary table (%) ──
    print()
    hdr2 = f"{'DP':>4} {'epoch':>10}"
    for c in CAT_ORDER:
        hdr2 += f' {c:>12}'
    print(hdr2)
    print('-' * len(hdr2))
    for d in results.values():
        row = f"{d['G_DATA']:>4} {d['epoch_time_ms']:>10.3f}"
        for c in CAT_ORDER:
            row += f" {d['breakdown_pct'].get(c, 0):>11.2f}%"
        print(row)

    # ── activation detail ──
    print('\n--- activation detail ---')
    print('  activation fwd is split: compute -> elementwise, all-reduce -> Allreduce')
    print('  activation bwd (no explicit timer) estimated as:')
    print('    train_step - sum(timed direct children of train_step)')
    print()
    for d in results.values():
        dp = d['G_DATA']
        act_comp = 0.0
        act_ar = 0.0
        for n, t in d['breakdown_details_ms'].get('elementwise', []):
            if n == 'activation (compute)':
                act_comp = t
        for n, t in d['breakdown_details_ms'].get('Allreduce', []):
            if n == 'activation/all-reduce':
                act_ar = t
        act_bwd = d['activation_bwd_estimate_ms']
        print(f'  DP={dp:>2}  fwd_compute={act_comp:8.3f}  fwd_AR={act_ar:8.3f}  '
              f'bwd(est)={act_bwd:8.3f} ms')

    # ── "others" detail ──
    print('\n--- "others" breakdown ---')
    for d in results.values():
        dp = d['G_DATA']
        ot = d['breakdown_ms']['others']
        items = d['others_details_ms']
        esum = sum(t for _, t, _ in items)
        gap = ot - esum
        print(f"\nDP={dp}  others={ot:.3f} ms:")
        for n, t, p in items:
            print(f'  {n:40s} {t:8.3f} ms  (parent: {p})')
        print(f'  {"[parent-child timing gaps]":40s} {gap:8.3f} ms')
        print(f'  (gaps exclude train_step gap which is now in elementwise as activation_bwd)')


if __name__ == '__main__':
    main()
