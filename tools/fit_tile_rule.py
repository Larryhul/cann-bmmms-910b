"""Turn a tile sweep into a selection rule, and show what the rule would cost.

The rule may only read K, M, N, B and the core count: those are general shape
properties. Anything keyed to a particular measured shape would be a per-case
implementation, which the contest rules forbid.
"""
import argparse
import json
import math
from pathlib import Path

CORES = 24
L0C_FLOATS = 32768


def parse_pair(text):
    bm, bn = text.split('x')
    return int(bm), int(bn)


def rule_blocks(B, M, N, K, cores=CORES):
    """Pick (baseM, baseN) from shape alone.

    Parallelism comes only from (batch, row block), so the row block is sized
    to put at least one chunk on every core; the column block then takes
    whatever L0C has left, because a wider block means fewer Matmul tiles per
    chunk.
    """
    # Tallest row block that still gives every core a chunk.
    target = max(1, (M * B + cores - 1) // cores)
    bm = 16
    for cand in (512, 384, 256, 192, 128, 96, 64, 48, 32, 16):
        if cand <= target or cand <= 16:
            bm = cand
            break
    bm = min(bm, ((M + 15) // 16) * 16)
    bm = max(bm, 16)
    # Column block: as wide as L0C allows, capped by N.
    bn = min(L0C_FLOATS // bm, ((N + 15) // 16) * 16)
    bn = (bn // 16) * 16
    bn = max(bn, 16)
    return bm, bn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', default='research/tile_sweep.json')
    args = parser.parse_args()
    rows = json.loads(Path(args.sweep).read_text(encoding='utf-8'))

    print('%-18s %5s %9s %9s %9s %9s  %s' % (
        'shape', 'K', 'default', 'best', 'rule', 'rule/best', 'best pair'))
    total_default = total_best = total_rule = 0.0
    for r in rows:
        B, M, N, K = r['shape']
        points = r['points']
        if not points:
            continue
        best_pair, best_us = min(points.items(), key=lambda kv: kv[1])
        bm, bn = rule_blocks(B, M, N, K)
        rule_us = points.get(f'{bm}x{bn}')
        default_us = r.get('default_us') or float('nan')
        total_default += default_us
        total_best += best_us
        total_rule += rule_us if rule_us else best_us
        print('%-18s %5d %9.2f %9.2f %9s %9s  %s' % (
            'x'.join(map(str, r['shape'])), K, default_us, best_us,
            ('%.2f' % rule_us) if rule_us else 'n/a',
            ('%.2f' % (rule_us / best_us)) if rule_us else '-', best_pair))
    print()
    print('sum default %.1f   sum rule %.1f   sum best %.1f' % (
        total_default, total_rule, total_best))
    print('rule vs default: %+.1f%%' % ((total_rule - total_default) / total_default * 100))

    print()
    print('top pairs per shape:')
    for r in rows:
        ranked = sorted(r['points'].items(), key=lambda kv: kv[1])[:5]
        print('  %-18s %s' % ('x'.join(map(str, r['shape'])), ranked))


if __name__ == '__main__':
    main()
