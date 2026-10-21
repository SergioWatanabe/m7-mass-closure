"""Hard per-species mass conservation for the linear baselines.

The tendencies satisfy sum_{j in modes(s)} y_j = 0 exactly for each species s.
For a linear model that constraint is linear in the coefficients, so it can be
imposed in closed form on an already-fitted W -- no retraining.

With Z the design matrix (last column = intercept), predictions in original units
are (Z W) * sigma + mu, so the constraint for species s reads

    W a_s = d_s,    a_s = sigma restricted to modes(s),
                    d_s = -(sum_{j in modes(s)} mu_j) * e_intercept

Two ways to satisfy it:
  completion  -- solve for one designated mode, the repo's CompletionLayer choice
  projection  -- constrained least squares; Lagrange multipliers give a rank-one
                 correction that spreads the fix over all modes and minimizes the
                 induced error. Independent of Z^T Z, so it holds for any ridge alpha.
"""
import argparse
import json

import numpy as np

import linear_baseline as lb

SPECIES = {'SO4': [0, 1, 2, 3, 4], 'BC': [5, 6, 7, 8],
           'OC': [9, 10, 11, 12], 'DU': [13, 14, 15, 16]}
# the mode each variant solves for, matching models.CompletionLayer
DERIVED = {'SO4': 4, 'BC': 7, 'OC': 11, 'DU': 13}

NAMES = (['SO4_ns', 'SO4_ks', 'SO4_as', 'SO4_cs', 'SO4_c4']
         + [f'BC_{m}' for m in ['ks', 'as', 'cs', 'ki']]
         + [f'OC_{m}' for m in ['ks', 'as', 'cs', 'ki']]
         + [f'DU_{m}' for m in ['as', 'cs', 'ai', 'ci']]
         + [f'NUM_{i}' for i in range(7)]
         + [f'H2O_{i}' for i in range(4)])


def _target(stats, idx, p):
    d = np.zeros(p)
    d[-1] = -stats['ytrain_mean'][idx].sum()
    return d


def complete(W, stats):
    W = W.copy()
    ystd = stats['ytrain_std']
    for s, idx in SPECIES.items():
        j = DERIVED[s]
        others = [k for k in idx if k != j]
        d = _target(stats, idx, W.shape[0])
        W[:, j] = (d - W[:, others] @ ystd[others]) / ystd[j]
    return W


def project(W, stats):
    W = W.copy()
    ystd = stats['ytrain_std']
    for s, idx in SPECIES.items():
        a = ystd[idx]
        r = W[:, idx] @ a - _target(stats, idx, W.shape[0])
        W[:, idx] -= np.outer(r, a) / (a ** 2).sum()
    return W


def evaluate_all(variants, stats, kind, split, chunk=100_000):
    """One pass over the split, scoring every variant on the same chunks."""
    X, y = lb.load(split)
    scorers = {k: lb.Scorer(stats) for k in variants}
    resid = {k: 0.0 for k in variants}          # max |sum of species tendencies|
    for a, b in lb.chunks(X.shape[0], chunk):
        xc, yc = lb.get_xy(X, y, a, b)
        p = next(iter(variants.values())).shape[0]
        Z = np.empty((b - a, p))
        Z[:, :-1] = lb.features(xc, stats, kind)
        Z[:, -1] = 1.0
        for name, W in variants.items():
            pred = (Z @ W) * stats['ytrain_std'] + stats['ytrain_mean']
            scorers[name].update(yc, pred, xc)
            for idx in SPECIES.values():
                scale = np.abs(pred[:, idx]).sum(1).mean()
                resid[name] = max(resid[name],
                                  np.abs(pred[:, idx].sum(1)).max() / scale)
    out = {}
    for name, sc in scorers.items():
        n = sc.n
        r = sc.result()
        r['per_channel_R2'] = (1 - sc.ss_res / (sc.s_t2 - sc.s_t ** 2 / n)).tolist()
        r['max_rel_mass_residual'] = resid[name]
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='poly2')
    ap.add_argument('--chunk', type=int, default=100_000)
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    stats = dict(np.load(f'{args.out}/stats.npz'))
    for k in ('so4_mean', 'bc_mean', 'oc_mean', 'du_mean'):
        stats[k] = float(stats[k])

    W = np.load(f'{args.out}/W_{args.features}.npy')
    variants = {'unconstrained': W, 'completion': complete(W, stats),
                'projection': project(W, stats)}

    res = {}
    for split in ['val', 'test']:
        print(f'scoring {split} ...', flush=True)
        res[split] = evaluate_all(variants, stats, args.features, split, args.chunk)

    hdr = f'{"channel":<9}' + ''.join(f'{v[:9]:>11}' for v in variants) * 1
    print(f'\n{"":<9}{"--- val R2 ---":^33}   {"--- test R2 ---":^33}')
    print(f'{"channel":<9}' + ''.join(f'{v[:10]:>11}' for v in variants)
          + '   ' + ''.join(f'{v[:10]:>11}' for v in variants))
    for i, nm in enumerate(NAMES):
        row = f'{nm:<9}'
        for split in ['val', 'test']:
            for v in variants:
                row += f'{res[split][v]["per_channel_R2"][i]:11.3f}'
            row += '   ' if split == 'val' else ''
        print(row)
    print(f'{"MEAN":<9}' + ''.join(f'{res["val"][v]["R2"]:11.3f}' for v in variants)
          + '   ' + ''.join(f'{res["test"][v]["R2"]:11.3f}' for v in variants))

    print(f'\n{"metric":<22}' + ''.join(f'{v[:14]:>16}' for v in variants))
    for key, fmt in [('R2', '16.4f'), ('MSE(norm)', '16.4f'),
                     ('Mass RMSE', '16.3e'), ('Negative fraction', '16.4f'),
                     ('max_rel_mass_residual', '16.3e')]:
        print(f'{key+" [test]":<22}' + ''.join(
            f'{res["test"][v][key]:{fmt}}' for v in variants))

    with open(f'{args.out}/scores_mass_{args.features}.json', 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
