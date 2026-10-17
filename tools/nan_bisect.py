"""Find where a variant starts returning a wrong value, by running the probe
over a shape ladder and comparing against a float64 reference.

Correctness only: no msprof, so a ladder of a dozen shapes takes seconds.
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
    name = f'{B}_{M}_{N}_{K}_d{dtype}_t{ta}{tb}_bis'
    folder = ROOT / 'cases' / name
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / 'golden.bin').exists():
        rng = np.random.default_rng(77000 + index)
        typ = np.float16 if dtype == 1 else ml_dtypes.bfloat16
        a = rng.uniform(-1, 1, (B, M, K)).astype(typ)
        b = rng.uniform(-1, 1, (B, K, N)).astype(typ)
        (a.swapaxes(-1, -2) if ta else a).copy().tofile(folder / 'a.bin')
        (b.swapaxes(-1, -2) if tb else b).copy().tofile(folder / 'b.bin')
        golden = (a.astype(np.float64) @ b.astype(np.float64)).max(-1).sum(-1)
        golden.astype(np.float32).tofile(folder / 'golden.bin')
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--shapes', required=True)
    args = parser.parse_args()

    probe = ROOT / f'build_{args.variant}' / 'probe'
    for index, entry in enumerate(json.loads(args.shapes)):
        B, M, N, K = entry[:4]
        dtype = entry[4] if len(entry) > 4 else 1
        ta = entry[5] if len(entry) > 5 else 0
        tb = entry[6] if len(entry) > 6 else 0
        folder = build_case(B, M, N, K, dtype, ta, tb, index)
        command = [str(probe), *map(str, (B, M, N, K, dtype, ta, tb)),
                   str(folder), '1', '0']
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=300)
            code = run.returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
        actual = np.fromfile(folder / 'actual_0.bin', dtype=np.float32)
        golden = np.fromfile(folder / 'golden.bin', dtype=np.float32)
        if actual.size != golden.size:
            verdict = f'size {actual.size} vs {golden.size}'
        else:
            scale = max(float(np.abs(golden).max()), 1.0)
            error = float(np.abs(actual - golden).max()) / scale
            verdict = 'nan' if not np.isfinite(error) else f'{error:.2e}'
        print(f'BIS {B}x{M}x{N}x{K} d{dtype} t{ta}{tb} rc={code} rel={verdict}',
              flush=True)


if __name__ == '__main__':
    main()
