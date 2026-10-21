"""Does hard mass conservation transfer from the linear models to the NN?

`mass_constrained.py` imposes the per-species conservation law in closed form on a
fitted linear W. The NN has no such structure -- 28 independent heads -- so it sits
at Mass RMSE 3.7e-4. But the closure is arithmetic on the output vector, not a
property of the model that produced it, so it should apply just as well to the NN's
predictions.

This checks that, and in particular whether it costs any accuracy.

Constraint (original units): sum_{j in modes(s)} y_j = 0 for each species s.

  completion  -- solve for one designated mode (models.CompletionLayer's choice)
  projection  -- spread the fix across the species' modes, minimizing the induced
                 error measured in z-space. The linear version corrects W by
                 -outer(r, a)/||a||^2 with a = ystd; pushed through pred = z*std+mean
                 that is pred_j -= R * std_j^2 / sum(std^2), R the mass residual.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from m7.model import Base  # noqa: E402

import linear_baseline as lb  # noqa: E402
from mass_constrained import SPECIES, DERIVED, NAMES  # noqa: E402


def complete(pred, stats):
    """Set each species' designated mode from the other modes. Exact by construction."""
    pred = pred.copy()
    for s, idx in SPECIES.items():
        j = DERIVED[s]
        others = [k for k in idx if k != j]
        pred[:, j] = -pred[:, others].sum(1)
    return pred


def project(pred, stats):
    """Minimum-z-norm correction spread over each species' modes."""
    pred = pred.copy()
    ystd = stats['ytrain_std']
    for s, idx in SPECIES.items():
        w = ystd[idx] ** 2
        r = pred[:, idx].sum(1)
        pred[:, idx] -= np.outer(r, w) / w.sum()
    return pred


def evaluate_all(model, stats, split, device, chunk=100_000):
    """One pass over the split, scoring every closure variant on the same chunks."""
    X, y = lb.load(split)
    variants = {'unconstrained': lambda p: p, 'completion': complete,
                'projection': project}
    scorers = {k: lb.Scorer(stats) for k in variants}
    resid = {k: 0.0 for k in variants}      # max |sum over modes| / mean |modes|, as
                                            # in mass_constrained.evaluate_all
    model.eval()
    with torch.no_grad():
        for a, b in lb.chunks(X.shape[0], chunk):
            xc, yc = lb.get_xy(X, y, a, b)
            xz = ((xc - stats['xtrain_mean']) / stats['xtrain_std']).astype(np.float32)
            out = model(torch.from_numpy(xz).to(device)).cpu().numpy().astype(np.float64)
            base = out * stats['ytrain_std'] + stats['ytrain_mean']
            for name, fn in variants.items():
                pred = fn(base, stats) if name != 'unconstrained' else base
                scorers[name].update(yc, pred, xc)
                for idx in SPECIES.values():
                    scale = np.abs(pred[:, idx]).sum(1).mean()
                    resid[name] = max(resid[name],
                                      np.abs(pred[:, idx].sum(1)).max() / scale)
    out = {}
    for name, sc in scorers.items():
        r = sc.result()
        r['per_channel_R2'] = (1 - sc.ss_res / (sc.s_t2 - sc.s_t ** 2 / sc.n)).tolist()
        r['median_R2'] = float(np.median(r['per_channel_R2']))
        # NMSE against the *train* variance: unlike R2 it does not swing with the
        # evaluation split's own variance, which differs up to 20x between val and test
        r['per_channel_NMSE'] = (sc.ss_res / (sc.n * stats['ytrain_std'] ** 2)).tolist()
        r['max_rel_mass_residual'] = resid[name]
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/nn_base.pth')
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--chunk', type=int, default=100_000)
    ap.add_argument('--splits', default='val,test')
    ap.add_argument('--out', default='results')
    ap.add_argument('--tag', default=None,
                    help='output suffix; defaults to the checkpoint file name')
    args = ap.parse_args()
    tag = args.tag or os.path.splitext(os.path.basename(args.ckpt))[0]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    stats = dict(np.load(f'{args.out}/stats.npz'))
    for k in ('so4_mean', 'bc_mean', 'oc_mean', 'du_mean'):
        stats[k] = float(stats[k])

    model = Base(32, 28, args.width).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['state_dict'])

    variants = ['unconstrained', 'completion', 'projection']
    res = {}
    for split in args.splits.split(','):
        print(f'scoring {split} ...', flush=True)
        res[split] = evaluate_all(model, stats, split, device, args.chunk)

    derived = set(DERIVED.values())
    for split in res:
        print(f'\n--- {split} per-channel R2 ---')
        print(f'{"channel":<9}' + ''.join(f'{v[:11]:>13}' for v in variants))
        for i, nm in enumerate(NAMES):
            mark = ' *' if i in derived else '  '
            print(f'{nm:<7}{mark}' + ''.join(
                f'{res[split][v]["per_channel_R2"][i]:13.4f}' for v in variants))
        print(f'{"MEAN":<9}' + ''.join(f'{res[split][v]["R2"]:13.4f}' for v in variants))
        print(f'{"MEDIAN":<9}' + ''.join(
            f'{res[split][v]["median_R2"]:13.4f}' for v in variants))
        print('  (* = mode the completion variant solves for)')

    print(f'\n{"metric":<26}' + ''.join(f'{v[:14]:>16}' for v in variants))
    for split in res:
        for key, fmt in [('R2', '16.4f'), ('MSE(norm)', '16.4f'),
                         ('Mass RMSE', '16.3e'), ('Negative fraction', '16.4f'),
                         ('max_rel_mass_residual', '16.3e')]:
            print(f'{key+" ["+split+"]":<26}' + ''.join(
                f'{res[split][v][key]:{fmt}}' for v in variants))

    path = f'{args.out}/scores_nn_closure_{tag}.json'
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
