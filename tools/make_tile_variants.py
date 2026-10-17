"""Generate Cube tiling variants of a base kernel for a single-variable sweep.

Only TILE_M, TILE_N and the C queue depth change between variants; every other
line is the base kernel, so a timing difference is attributable to the tiling.
"""

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

COMBINATIONS = [
    # name, TILE_M, TILE_N, queue depth
    ('t64x192d1', 64, 192, 1),
    ('t64x192d2', 64, 192, 2),
    ('t128x128d1', 128, 128, 1),
    ('t128x192d1', 128, 192, 1),
    ('t32x256d1', 32, 256, 1),
    ('t64x256d1', 64, 256, 1),
]


def build(base_text, tile_m, tile_n, depth, tag):
    text = base_text
    text = re.sub(r'constexpr int32_t TILE_M = \d+;',
                  f'constexpr int32_t TILE_M = {tile_m};', text, count=1)
    text = re.sub(r'constexpr int32_t TILE_N = \d+;',
                  f'constexpr int32_t TILE_N = {tile_n};', text, count=1)
    # The C queue and its buffer must agree on depth.
    text = re.sub(r'AscendC::TQue<AscendC::TPosition::VECIN, \d+> cQueue;',
                  f'AscendC::TQue<AscendC::TPosition::VECIN, {depth}> cQueue;', text)
    text = re.sub(r'pipe\.InitBuffer\(cQueue, \d+,',
                  f'pipe.InitBuffer(cQueue, {depth},', text)
    text = re.sub(r'SUBMISSION_VERSION: \S+',
                  f'SUBMISSION_VERSION: SWEEP_{tag.upper()}', text, count=1)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='solution_v12/kernel.asc')
    parser.add_argument('--prefix', default='sweep')
    args = parser.parse_args()

    base_path = ROOT / args.base
    base_text = base_path.read_text(encoding='utf-8')
    if 'constexpr int32_t TILE_M' not in base_text:
        raise SystemExit('base kernel does not define TILE_M')

    made = []
    for tag, tile_m, tile_n, depth in COMBINATIONS:
        name = f'{args.prefix}_{tag}'
        folder = ROOT / f'solution_{name}'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'kernel.asc').write_text(
            build(base_text, tile_m, tile_n, depth, tag), encoding='utf-8')
        made.append(name)
        print(f'{name}: TILE_M={tile_m} TILE_N={tile_n} depth={depth}')
    print('VARIANTS ' + ','.join(made))


if __name__ == '__main__':
    main()
