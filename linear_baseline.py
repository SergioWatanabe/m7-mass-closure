"""Linear baselines for the M7 aerosol microphysics emulation problem.

Mirrors the preprocessing and metrics of paulaharder/aerosol-microphysics-emulation
(main.py + utils.py) so the numbers are directly comparable to Table 1 of the paper.

Everything is done with streaming normal equations (X^T X, X^T y accumulated in
chunks), so the 5.7M x 32 training set never has to be held in memory at once.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
from numpy.linalg import solve

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aerosol_emulation_data')
CHUNK = 250_000
EPS = 1e-8


# --------------------------------------------------------------------------- data
def load(split):
    """Memory-mapped X and y for a split, with y already turned into tendencies."""
    X = np.load(f'{DATA}/X_{split}.npy', mmap_mode='r')
    y = np.load(f'{DATA}/y_{split}.npy', mmap_mode='r')
    return X, y


def chunks(n, size=CHUNK):
    for i in range(0, n, size):
        yield i, min(i + size, n)


def get_xy(X, y, a, b):
    """Chunk [a:b) in original units, y converted to tendency (as main.py does)."""
    xc = np.asarray(X[a:b], dtype=np.float64)
    yc = np.asarray(y[a:b], dtype=np.float64).copy()
    yc[:, :24] -= xc[:, 8:]
    return xc, yc


# --------------------------------------------------------------------------- stats
def calculate_stats(verbose=True):
    """Same quantities as utils.calculate_stats, accumulated in a streaming fashion."""
    Xtr, ytr = load('train')
    n = Xtr.shape[0]
    sx = np.zeros(32); sx2 = np.zeros(32)
    sy = np.zeros(28); sy2 = np.zeros(28)
    slx = np.zeros(32); slx2 = np.zeros(32)
    sly = np.zeros(28); sly2 = np.zeros(28)
    for a, b in chunks(n):
        xc, yc = get_xy(Xtr, ytr, a, b)
        sx += xc.sum(0); sx2 += (xc ** 2).sum(0)
        sy += yc.sum(0); sy2 += (yc ** 2).sum(0)
        lx = np.log(np.abs(xc) + EPS); ly = np.log(np.abs(yc) + EPS)
        slx += lx.sum(0); slx2 += (lx ** 2).sum(0)
        sly += ly.sum(0); sly2 += (ly ** 2).sum(0)

    def ms(s, s2):
        m = s / n
        return m, np.sqrt(np.maximum(s2 / n - m ** 2, 0.0))

    xm, xs = ms(sx, sx2); ym, ys = ms(sy, sy2)
    lxm, lxs = ms(slx, slx2); lym, lys = ms(sly, sly2)

    # per-species absolute-scale means, from the training inputs (utils.calculate_stats)
    so4 = xm[8:13].mean(); bc = xm[13:17].mean()
    oc = xm[17:21].mean(); du = xm[[21, 24, 22, 23]].mean()

    # stats['means']: mean of the *untransformed* y[:, :24] over the test split
    Xte, yte = load('test')
    sm = np.zeros(24)
    for a, b in chunks(Xte.shape[0]):
        sm += np.asarray(yte[a:b, :24], dtype=np.float64).sum(0)
    means = sm / Xte.shape[0]

    stats = dict(xtrain_mean=xm, xtrain_std=xs, ytrain_mean=ym, ytrain_std=ys,
                 X_log_eps_mean=lxm, X_log_eps_std=lxs,
                 y_log_eps_mean=lym, y_log_eps_std=lys,
                 so4_mean=so4, bc_mean=bc, oc_mean=oc, du_mean=du, means=means)
    if verbose:
        print(f'stats over {n:,} training rows done')
    return stats


# --------------------------------------------------------------------------- features
def features(xc, stats, kind):
    """Build the design matrix (without intercept) for a chunk of raw inputs."""
    z = (xc - stats['xtrain_mean']) / stats['xtrain_std']
    if kind == 'z':
        return z
    lg = (np.log(np.abs(xc) + EPS) - stats['X_log_eps_mean']) / stats['X_log_eps_std']
    if kind == 'log':
        return lg
    if kind == 'z+log':
        return np.hstack([z, lg])
    if kind in ('poly2', 'poly2log'):
        base = z if kind == 'poly2' else lg
        d = base.shape[1]
        idx = np.array(list(itertools.combinations_with_replacement(range(d), 2)))
        quad = base[:, idx[:, 0]] * base[:, idx[:, 1]]
        return np.hstack([base, quad])
    if kind == 'poly2_zlog':
        base = np.hstack([z, lg])
        d = base.shape[1]
        idx = np.array(list(itertools.combinations_with_replacement(range(d), 2)))
        quad = base[:, idx[:, 0]] * base[:, idx[:, 1]]
        return np.hstack([base, quad])
    raise ValueError(kind)


def n_features(kind):
    d = {'z': 32, 'log': 32, 'z+log': 64}
    if kind in d:
        return d[kind]
    b = 32 if kind in ('poly2', 'poly2log') else 64
    return b + b * (b + 1) // 2


# --------------------------------------------------------------------------- fitting
def normal_equations(stats, kind, split='train', chunk=CHUNK):
    """Accumulate [Z 1]^T [Z 1] and [Z 1]^T Yz over a split."""
    X, y = load(split)
    n = X.shape[0]
    p = n_features(kind) + 1
    A = np.zeros((p, p)); B = np.zeros((p, 28))
    for a, b in chunks(n, chunk):
        xc, yc = get_xy(X, y, a, b)
        Z = np.empty((b - a, p))
        Z[:, :-1] = features(xc, stats, kind)
        Z[:, -1] = 1.0
        yz = (yc - stats['ytrain_mean']) / stats['ytrain_std']
        A += Z.T @ Z
        B += Z.T @ yz
    return A, B, n


def solve_ridge(A, B, alpha, n):
    p = A.shape[0]
    reg = np.eye(p) * alpha * n
    reg[-1, -1] = 0.0          # never penalize the intercept
    return solve(A + reg, B)


# --------------------------------------------------------------------------- metrics
class Scorer:
    """Streaming version of utils.get_scores (original-scale tendencies)."""

    def __init__(self, stats):
        self.s = stats
        self.n = 0
        # sums of (true - shift) and (true - shift)^2; the shift is set from the
        # first chunk's mean so that ss_tot = S2 - n*mean_c^2 stays well conditioned
        # (log-transformed targets have means ~20x their spread otherwise).
        self.shift = None; self.lshift = None
        self.ss_res = np.zeros(28); self.s_t = np.zeros(28); self.s_t2 = np.zeros(28)
        self.lss_res = np.zeros(28); self.ls_t = np.zeros(28); self.ls_t2 = np.zeros(28)
        self.se_norm = 0.0
        self.mass_sum = np.zeros(4); self.mass_sq = np.zeros(4)
        self.neg_count = 0
        self.neg_sum = np.zeros(24)

    def update(self, true, pred, xc):
        lt = np.log(np.abs(true) + EPS); lp = np.log(np.abs(pred) + EPS)
        if self.shift is None:
            self.shift = true.mean(0); self.lshift = lt.mean(0)
        self.n += true.shape[0]
        d = true - pred
        self.ss_res += (d ** 2).sum(0)
        c = true - self.shift
        self.s_t += c.sum(0); self.s_t2 += (c ** 2).sum(0)
        self.lss_res += ((lt - lp) ** 2).sum(0)
        lc = lt - self.lshift
        self.ls_t += lc.sum(0); self.ls_t2 += (lc ** 2).sum(0)
        self.se_norm += ((d / self.s['ytrain_std']) ** 2).sum()
        m = np.stack([pred[:, 0:5].sum(1), pred[:, 5:9].sum(1),
                      pred[:, 9:13].sum(1), pred[:, 13:17].sum(1)], axis=1)
        self.mass_sum += m.sum(0); self.mass_sq += (m ** 2).sum(0)
        full = pred.copy()
        full[:, :24] += xc[:, 8:]
        self.neg_count += int((full < 0).sum())
        neg = np.where(full[:, :24] < 0, full[:, :24], 0.0)
        self.neg_sum += neg.sum(0)

    def result(self):
        n = self.n
        ss_tot = self.s_t2 - self.s_t ** 2 / n
        r2 = np.mean(1 - self.ss_res / ss_tot)
        lss_tot = self.ls_t2 - self.ls_t ** 2 / n
        r2_log = np.mean(1 - self.lss_res / lss_tot)
        rmse = np.sqrt(self.se_norm / (n * 28))
        mm = np.array([self.s['so4_mean'], self.s['bc_mean'],
                       self.s['oc_mean'], self.s['du_mean']])
        mass_bias = (self.mass_sum / n) / mm
        mass_rmse = float(np.mean(np.sqrt(self.mass_sq / n) / mm))
        return {'R2': float(r2), 'R2 log': float(r2_log), 'RMSE': float(rmse),
                'MSE(norm)': float(self.se_norm / (n * 28)),
                'Mass Bias': mass_bias.tolist(), 'Mass RMSE': mass_rmse,
                'Negative fraction': self.neg_count / (n * 28),
                'Negative mean': float(np.mean((self.neg_sum / n) / self.s['means']))}


def evaluate(W, stats, kind, split, chunk=CHUNK, constant=None):
    X, y = load(split)
    sc = Scorer(stats)
    for a, b in chunks(X.shape[0], chunk):
        xc, yc = get_xy(X, y, a, b)
        if constant is not None:
            pred = np.tile(constant, (b - a, 1))
        else:
            p = W.shape[0]
            Z = np.empty((b - a, p))
            Z[:, :-1] = features(xc, stats, kind)
            Z[:, -1] = 1.0
            pred = (Z @ W) * stats['ytrain_std'] + stats['ytrain_mean']
        sc.update(yc, pred, xc)
    return sc.result()


# --------------------------------------------------------------------------- main
def fmt(name, s):
    return (f'{name:<16} R2={s["R2"]:+.4f}  R2log={s["R2 log"]:+.4f}  '
            f'MSE={s["MSE(norm)"]:.4f}  massRMSE={s["Mass RMSE"]:.2e}  '
            f'neg_frac={s["Negative fraction"]:.4f}  neg_mean={s["Negative mean"]:.2e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='z',
                    choices=['z', 'log', 'z+log', 'poly2', 'poly2log', 'poly2_zlog'])
    ap.add_argument('--alphas', default='0,1e-8,1e-6,1e-4,1e-2')
    ap.add_argument('--trivial', action='store_true',
                    help='also score the zero / train-mean predictors')
    ap.add_argument('--chunk', type=int, default=CHUNK)
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stats_path = f'{args.out}/stats.npz'
    if os.path.exists(stats_path):
        stats = dict(np.load(stats_path))
        for k in ('so4_mean', 'bc_mean', 'oc_mean', 'du_mean'):
            stats[k] = float(stats[k])
        print('loaded cached stats')
    else:
        t = time.time(); stats = calculate_stats(); print(f'  ({time.time()-t:.0f}s)')
        np.savez(stats_path, **stats)

    results = {}
    if args.trivial:
        for name, const in [('zero', np.zeros(28)), ('train-mean', stats['ytrain_mean'])]:
            s = evaluate(None, stats, None, 'test', args.chunk, constant=const)
            results[name] = s
            print(fmt(name, s))

    kind = args.features
    print(f'\naccumulating normal equations [{kind}, p={n_features(kind)+1}] ...')
    t = time.time()
    A, B, n = normal_equations(stats, kind, 'train', args.chunk)
    print(f'  ({time.time()-t:.0f}s)')

    alphas = [float(a) for a in args.alphas.split(',')]
    best, best_a, best_W = None, None, None
    for alpha in alphas:
        W = solve_ridge(A, B, alpha, n)
        s = evaluate(W, stats, kind, 'val', args.chunk)
        print(fmt(f'{kind} a={alpha:g} [val]', s))
        if best is None or s['MSE(norm)'] < best:
            best, best_a, best_W = s['MSE(norm)'], alpha, W

    s = evaluate(best_W, stats, kind, 'test', args.chunk)
    print('\n' + fmt(f'{kind} a={best_a:g} [TEST]', s))
    results[f'{kind}'] = dict(s, alpha=best_a)
    np.save(f'{args.out}/W_{kind}.npy', best_W)
    with open(f'{args.out}/scores_{kind}.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
