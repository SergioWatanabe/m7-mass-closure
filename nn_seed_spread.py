"""Retrain the base NN on an 80% subsample of train/val, over several seeds.

How much does a correct NN vary from run to run? `torch.manual_seed` makes one run
reproducible; it says nothing about what a *different* seed scores. Training several
seeds on the same data gives the per-channel spread, which is the yardstick for
whether a difference between two models is real or seed noise.

Stats (means/stds, species scales) are recomputed on the subsample so nothing about
the dropped rows leaks in. Test is untouched and used only for scoring.

Metric is per-channel NMSE = mean((yhat_j - y_j)^2) / var(y_train_j), lower better.
Its mean over channels is the repo's MSE(norm), and it is also exactly the training
loss, so the objective and the score are the same quantity.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from m7.model import Base  # noqa: E402

import linear_baseline as lb  # noqa: E402
from mass_constrained import SPECIES, DERIVED, NAMES  # noqa: E402
from nn_closure import complete, project  # noqa: E402


def subsample_mask(n, drop, seed):
    """Deterministic keep-mask dropping `drop` of n rows."""
    rng = np.random.default_rng(seed)
    return rng.random(n) >= drop


def load_subset(split, mask, chunk=250_000):
    """Kept rows of a split, in original units, y already a tendency."""
    X, y = lb.load(split)
    xs, ys = [], []
    for a, b in lb.chunks(X.shape[0], chunk):
        m = mask[a:b]
        if not m.any():
            continue
        xc, yc = lb.get_xy(X, y, a, b)
        xs.append(xc[m]); ys.append(yc[m])
    return np.concatenate(xs), np.concatenate(ys)


def subset_stats(Xtr, ytr, Xte_means):
    """utils.calculate_stats over the subsample."""
    s = dict(xtrain_mean=Xtr.mean(0), xtrain_std=Xtr.std(0),
             ytrain_mean=ytr.mean(0), ytrain_std=ytr.std(0))
    s['so4_mean'] = float(Xtr[:, 8:13].mean())
    s['bc_mean'] = float(Xtr[:, 13:17].mean())
    s['oc_mean'] = float(Xtr[:, 17:21].mean())
    s['du_mean'] = float(Xtr[:, [21, 24, 22, 23]].mean())
    s['means'] = Xte_means
    return s


def test_means(chunk=250_000):
    """stats['means']: mean untransformed y[:, :24] over test (unchanged by subsetting)."""
    X, y = lb.load('test')
    n = X.shape[0]
    acc = np.zeros(24)
    for a, b in lb.chunks(n, chunk):
        acc += np.asarray(y[a:b, :24], dtype=np.float64).sum(0)
    return acc / n


def score_test(model, stats, device, chunk=100_000):
    """Per-channel NMSE + mass residual for the unconstrained and projected model."""
    X, y = lb.load('test')
    variants = {'unconstrained': None, 'completion': complete, 'projection': project}
    sc = {k: lb.Scorer(stats) for k in variants}
    resid = {k: 0.0 for k in variants}
    model.eval()
    with torch.no_grad():
        for a, b in lb.chunks(X.shape[0], chunk):
            xc, yc = lb.get_xy(X, y, a, b)
            xz = ((xc - stats['xtrain_mean']) / stats['xtrain_std']).astype(np.float32)
            out = model(torch.from_numpy(xz).to(device)).cpu().numpy().astype(np.float64)
            base = out * stats['ytrain_std'] + stats['ytrain_mean']
            for name, fn in variants.items():
                pred = base if fn is None else fn(base, stats)
                sc[name].update(yc, pred, xc)
                for idx in SPECIES.values():
                    scale = np.abs(pred[:, idx]).sum(1).mean()
                    resid[name] = max(resid[name],
                                      np.abs(pred[:, idx].sum(1)).max() / scale)
    model.train()
    var_tr = stats['ytrain_std'] ** 2
    out = {}
    for name, s in variants.items():
        r = sc[name].result()
        r['per_channel_NMSE'] = (sc[name].ss_res / (sc[name].n * var_tr)).tolist()
        r['per_channel_R2'] = (1 - sc[name].ss_res /
                               (sc[name].s_t2 - sc[name].s_t ** 2 / sc[name].n)).tolist()
        r['max_rel_mass_residual'] = resid[name]
        out[name] = r
    return out


def val_nmse(model, Xv, yv, stats, device, chunk=250_000):
    """Mean NMSE on the held-out subsample val -- the checkpoint criterion."""
    var_tr = stats['ytrain_std'] ** 2
    ss = np.zeros(28)
    model.eval()
    with torch.no_grad():
        for a in range(0, Xv.shape[0], chunk):
            xz = ((Xv[a:a + chunk] - stats['xtrain_mean'])
                  / stats['xtrain_std']).astype(np.float32)
            out = model(torch.from_numpy(xz).to(device)).cpu().numpy().astype(np.float64)
            pred = out * stats['ytrain_std'] + stats['ytrain_mean']
            ss += ((yv[a:a + chunk] - pred) ** 2).sum(0)
    model.train()
    return float(np.mean(ss / (Xv.shape[0] * var_tr)))


def train_one(seed, Xz, yz, Xv, yv, stats, args, device):
    torch.manual_seed(seed)
    model = Base(32, 28, args.width).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss()
    ckpt = f'{args.out}/nn_sub_seed{seed}.pth'
    n = Xz.shape[0]

    best, curve, start = np.inf, [], time.time()
    for ep in range(args.epochs):
        t = time.time()
        perm = torch.randperm(n)
        run, steps = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = crit(model(Xz[idx].to(device)), yz[idx].to(device))
            loss.backward()
            opt.step()
            run += loss.item(); steps += 1
        v = val_nmse(model, Xv, yv, stats, device)
        curve.append({'epoch': ep + 1, 'train_loss': run / steps, 'val_NMSE': v})
        flag = ''
        if v < best:
            best = v
            torch.save({'state_dict': model.state_dict()}, ckpt)
            flag = ' *'
        el = time.time() - start
        print(f'  seed {seed} epoch {ep+1:3d}  train={run/steps:.4f}  '
              f'val NMSE={v:.4f}{flag}  ({time.time()-t:.0f}s, '
              f'eta {el/(ep+1)*(args.epochs-ep-1)/60:.0f}m)', flush=True)
        with open(f'{args.out}/nn_sub_seed{seed}_curve.json', 'w') as f:
            json.dump(curve, f, indent=2)

    model.load_state_dict(torch.load(ckpt)['state_dict'])
    return model, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop', type=float, default=0.20)
    ap.add_argument('--subset_seed', type=int, default=1234)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-9)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  threads={torch.get_num_threads()}', flush=True)

    t0 = time.time()
    ntr = lb.load('train')[0].shape[0]
    nva = lb.load('val')[0].shape[0]
    mtr = subsample_mask(ntr, args.drop, args.subset_seed)
    mva = subsample_mask(nva, args.drop, args.subset_seed + 1)
    Xtr, ytr = load_subset('train', mtr)
    Xv, yv = load_subset('val', mva)
    print(f'train {ntr:,} -> {Xtr.shape[0]:,}   val {nva:,} -> {Xv.shape[0]:,}  '
          f'({time.time()-t0:.0f}s)', flush=True)

    stats = subset_stats(Xtr, ytr, test_means())
    np.savez(f'{args.out}/stats_sub.npz', **stats)

    Xz = torch.from_numpy(((Xtr - stats['xtrain_mean'])
                           / stats['xtrain_std']).astype(np.float32))
    yz = torch.from_numpy(((ytr - stats['ytrain_mean'])
                           / stats['ytrain_std']).astype(np.float32))
    del Xtr, ytr

    res = {}
    for seed in [int(s) for s in args.seeds.split(',')]:
        print(f'\n=== seed {seed} ===', flush=True)
        model, curve = train_one(seed, Xz, yz, Xv, yv, stats, args, device)
        res[str(seed)] = {'test': score_test(model, stats, device), 'curve': curve}
        with open(f'{args.out}/scores_nn_subset.json', 'w') as f:
            json.dump({'drop': args.drop, 'subset_seed': args.subset_seed,
                       'n_train': int(Xz.shape[0]), 'n_val': int(Xv.shape[0]),
                       'seeds': res}, f, indent=2)
        p = res[str(seed)]['test']['projection']
        print(f'  seed {seed} test mean NMSE (projection) = '
              f'{np.mean(p["per_channel_NMSE"]):.4f}', flush=True)

    seeds = list(res)
    M = np.array([res[s]['test']['projection']['per_channel_NMSE'] for s in seeds])
    print(f'\n{"channel":<9}' + ''.join(f'{"seed "+s:>10}' for s in seeds)
          + f'{"worst":>10}')
    for i, nm in enumerate(NAMES):
        print(f'{nm:<9}' + ''.join(f'{M[j, i]:10.4f}' for j in range(len(seeds)))
              + f'{M[:, i].max():10.4f}')
    print(f'{"MEAN":<9}' + ''.join(f'{M[j].mean():10.4f}' for j in range(len(seeds))))
    print(f'\nmax per-channel spread across seeds: {(M.max(0)-M.min(0)).max():.4f}')
    print(f'wrote {args.out}/scores_nn_subset.json')


if __name__ == '__main__':
    main()
