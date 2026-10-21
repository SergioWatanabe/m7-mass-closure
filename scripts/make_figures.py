"""Aggregate the per-seed closure results into the README table and figure.

Reads results/scores_nn_closure_<tag>.json for every NN seed present, and writes
  results/closure_table.md   mean (min-max) over seeds, per split and closure
  results/closure_nmse.png   per-channel NMSE, val and test side by side
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from mass_constrained import NAMES, DERIVED  # noqa: E402

TAGS = ['nn_base', 'nn_base_seed1', 'nn_base_seed2']
VARIANTS = ['unconstrained', 'completion', 'projection']
COLORS = {'unconstrained': '#2a78d6', 'completion': '#eb6834', 'projection': '#1baf7a'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load():
    runs = []
    for t in TAGS:
        p = os.path.join(ROOT, 'results', f'scores_nn_closure_{t}.json')
        if os.path.exists(p):
            with open(p) as f:
                runs.append(json.load(f))
    if not runs:
        sys.exit('no results/scores_nn_closure_*.json found; run nn_closure.py first')
    return runs


def stat(runs, split, variant, key):
    v = np.array([r[split][variant][key] for r in runs], dtype=float)
    return v.mean(0), v.min(0), v.max(0)


def fmt(m, lo, hi, f):
    return f'{m:{f}}' if lo == hi else f'{m:{f}} ({lo:{f}}–{hi:{f}})'


def table(runs):
    bc = NAMES.index('BC_cs')
    rows = [f'Mean over {len(runs)} seed(s); range in parentheses.\n',
            '| split | closure | mean R² | median R² | mean NMSE | BC_cs R² | BC_cs NMSE | max rel. mass residual |',
            '|---|---|---|---|---|---|---|---|']
    for split in ('val', 'test'):
        for v in VARIANTS:
            r2 = stat(runs, split, v, 'R2')
            med = stat(runs, split, v, 'median_R2')
            nm = np.array([np.mean(r[split][v]['per_channel_NMSE']) for r in runs])
            pr2 = np.array([r[split][v]['per_channel_R2'][bc] for r in runs])
            pnm = np.array([r[split][v]['per_channel_NMSE'][bc] for r in runs])
            res = stat(runs, split, v, 'max_rel_mass_residual')
            rows.append(f'| {split} | {v} | {fmt(*r2, ".4f")} | {fmt(*med, ".4f")} | '
                        f'{fmt(nm.mean(), nm.min(), nm.max(), ".4f")} | '
                        f'{fmt(pr2.mean(), pr2.min(), pr2.max(), ".3f")} | '
                        f'{fmt(pnm.mean(), pnm.min(), pnm.max(), ".3f")} | '
                        f'{fmt(*res, ".1e")} |')
    return '\n'.join(rows) + '\n'


def figure(runs, path):
    derived = set(DERIVED.values())
    x = np.arange(len(NAMES))
    w = 0.27
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for ax, split in zip(axes, ('val', 'test')):
        ax.set_facecolor(SURFACE)
        for k, v in enumerate(VARIANTS):
            m, lo, hi = stat(runs, split, v, 'per_channel_NMSE')
            xs = x + (k - 1) * w
            ax.bar(xs, m, w - 0.04, color=COLORS[v], label=v, zorder=3)
            if len(runs) > 1:
                ax.vlines(xs, lo, hi, color=INK2, lw=1, zorder=4)
        ax.set_yscale('log')
        ax.axhline(1, color=INK2, lw=1, ls='--', zorder=2)
        ax.text(len(NAMES) - 0.4, 1.15, 'NMSE = 1 (predicting the train mean)',
                ha='right', va='bottom', fontsize=8, color=INK2)
        ax.set_ylabel(f'{split} NMSE (log)', color=INK)
        ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n + (' *' if i in derived else '') for i, n in enumerate(NAMES)],
                            rotation=60, ha='right')
    axes[0].legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0),
                   fontsize=9)
    fig.suptitle('Per-channel NMSE of the base NN under each mass closure '
                 f'({len(runs)} seed(s); whiskers = min–max)', color=INK, fontsize=11)
    fig.text(0.01, 0.005, '* = mode the completion layer solves for. Lower is better.',
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    runs = load()
    md = table(runs)
    with open(os.path.join(ROOT, 'results', 'closure_table.md'), 'w') as f:
        f.write(md)
    print(md)
    figure(runs, os.path.join(ROOT, 'results', 'closure_nmse.png'))
    print('wrote results/closure_table.md, results/closure_nmse.png')


if __name__ == '__main__':
    main()
