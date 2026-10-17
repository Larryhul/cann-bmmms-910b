"""Measure chip task time for a shape grid, to see which dispatch path each
shape class lands on and where the time actually goes.

Shapes come from a JSON list on the command line, so the grid can change
without re-uploading this file.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, '/data/cann_bmm_20261015_v04/batchmatmulmaxsum_v04/.python_deps')
import ml_dtypes

ROOT = Path('/data/bmmms_opt_20261015')


def build_case(B, M, N, K, dtype, ta, tb, index):
    name = f'{B}_{M}_{N}_{K}_d{dtype}_t{ta}{tb}_scan'
    folder = ROOT / 'cases' / name
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / 'golden.bin').exists():
        rng = np.random.default_rng(31000 + index)
        typ = np.float16 if dtype == 1 else ml_dtypes.bfloat16
        a = rng.uniform(-1, 1, (B, M, K)).astype(typ)
        b = rng.uniform(-1, 1, (B, K, N)).astype(typ)
        (a.swapaxes(-1, -2) if ta else a).copy().tofile(folder / 'a.bin')
        (b.swapaxes(-1, -2) if tb else b).copy().tofile(folder / 'b.bin')
        golden = (a.astype(np.float64) @ b.astype(np.float64)).max(-1).sum(-1)
        golden.astype(np.float32).tofile(folder / 'golden.bin')
    return name, folder


def chip_time(variant, name, folder, B, M, N, K, dtype, ta, tb):
    output = ROOT / 'scans' / variant / name
    application = [str(ROOT / f'build_{variant}' / 'probe'),
                   *map(str, (B, M, N, K, dtype, ta, tb)), str(folder), '1', '0']
    command = ['msprof', 'op', '--application=' + ' '.join(application),
               '--aic-metrics=BasicInfo', '--launch-count=5', '--warm-up=3',
               '--output=' + str(output)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=600)
    best = None
    kernel = ''
    for path in sorted(output.rglob('OpBasicInfo.csv')):
        for line in path.read_text(errors='replace').splitlines()[1:]:
            parts = line.split(',')
            if len(parts) > 3 and parts[2]:
                try:
                    value = float(parts[2])
                except ValueError:
                    continue
                if best is None or value < best:
                    best = value
                    kernel = parts[0][:48]
    return best, kernel, run.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='v13')
    parser.add_argument('--shapes', required=True,
                        help='JSON list of [B,M,N,K] or [B,M,N,K,dtype,ta,tb]')
    parser.add_argument('--out', default='shape_scan.json')
    args = parser.parse_args()

    shapes = json.loads(args.shapes)
    results = []
    for index, entry in enumerate(shapes):
        B, M, N, K = entry[:4]
        dtype = entry[4] if len(entry) > 4 else 1
        ta = entry[5] if len(entry) > 5 else 0
        tb = entry[6] if len(entry) > 6 else 0
        name, folder = build_case(B, M, N, K, dtype, ta, tb, index)
        duration, kernel, code = chip_time(args.variant, name, folder,
                                           B, M, N, K, dtype, ta, tb)
        row = {'shape': [B, M, N, K], 'dtype': dtype, 'ta': ta, 'tb': tb,
               'work': B * M * N * K, 'us': duration, 'kernel': kernel,
               'exit_code': code}
        results.append(row)
        (ROOT / args.out).write_text(json.dumps(results, indent=2))
        print('SCAN %-28s work=%-10d %9s  %s' % (
            name, row['work'],
            'n/a' if duration is None else ('%.2f' % duration),
            kernel), flush=True)


if __name__ == '__main__':
    main()
