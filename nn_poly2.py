"""Does feeding the poly2 linear model into the NN help?

Two arms, both against the plain NN control (nn_reference.py, test R2 0.7746):

  concat    input = [z-scored x (32), poly2 predictions (28)] -> Base(60, 28)
            The quadratic terms are real added content: a ReLU MLP cannot form
            products in one layer, so this is not a reparameterization.

  residual  y_hat = poly2_pred + NN(x), i.e. the network fits only what poly2
            misses. Note this also reweights the implicit loss: channels poly2
            already nails contribute smaller residuals and so less gradient.

Both use the repo's hyperparameters, the same seed, and the same 100 epochs as
the control, so the only change is the input/parameterization.
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
from nn_reference import NAMES, load_split_z  # noqa: E402

device = torch.device('cpu')


def poly2_pred_z(stats, W, split, kind='poly2', chunk=100_000):
    """poly2 predictions in z-scored target space, i.e. Z @ W."""
    X, y = lb.load(split)
    n = X.shape[0]
    p = W.shape[0]
    out = np.empty((n, 28), dtype=np.float32)
    for a, b in lb.chunks(n, chunk):
        xc = np.asarray(X[a:b], dtype=np.float64)
        Z = np.empty((b - a, p))
        Z[:, :-1] = lb.features(xc, stats, kind)
        Z[:, -1] = 1.0
        out[a:b] = (Z @ W).astype(np.float32)
    return out


def score(model, stats, split, pz, arm, chunk=250_000):
    X, y = lb.load(split)
    sc = lb.Scorer(stats)
    model.eval()
    with torch.no_grad():
        for a, b in lb.chunks(X.shape[0], chunk):
            xc, yc = lb.get_xy(X, y, a, b)
            xz = ((xc - stats['xtrain_mean']) / stats['xtrain_std']).astype(np.float32)
            pzc = pz[a:b]
            inp = np.hstack([xz, pzc]) if arm == 'concat' else xz
            out = model(torch.from_numpy(inp)).numpy()
            predz = out if arm == 'concat' else pzc + out
            pred = predz.astype(np.float64) * stats['ytrain_std'] + stats['ytrain_mean']
            sc.update(yc, pred, xc)
    model.train()
    res = sc.result()
    n = sc.n
    res['per_channel_R2'] = (1 - sc.ss_res / (sc.s_t2 - sc.s_t ** 2 / n)).tolist()
    return res


def train_arm(arm, data, stats, args):
    Xtr, ytr, ptr, pval, ptest = data
    n = Xtr.shape[0]

    if arm == 'concat':
        inputs = torch.cat([Xtr, ptr], dim=1)
        targets = ytr
        in_dim = 60
    else:
        inputs = Xtr
        targets = ytr - ptr          # residual target
        in_dim = 32

    torch.manual_seed(0)             # same init draw as the control
    model = Base(in_dim, 28, args.width).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss()
    ckpt = f'{args.out}/nn_poly2_{arm}.pth'

    best, curve = np.inf, []
    start = time.time()
    for ep in range(args.epochs):
        t = time.time()
        perm = torch.randperm(n)
        running, steps = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = crit(model(inputs[idx]), targets[idx])
            loss.backward()
            opt.step()
            running += loss.item(); steps += 1

        s = score(model, stats, 'val', pval, arm)
        curve.append({'epoch': ep + 1, 'train_loss': running / steps,
                      'val_R2': s['R2'], 'val_MSE': s['MSE(norm)']})
        flag = ''
        if s['MSE(norm)'] < best:
            best = s['MSE(norm)']
            torch.save({'state_dict': model.state_dict()}, ckpt)
            flag = ' *'
        el = time.time() - start
        print(f'[{arm}] epoch {ep+1:3d}  train={running/steps:.4f}  '
              f'val R2={s["R2"]:+.4f}  val MSE={s["MSE(norm)"]:.4f}{flag}  '
              f'({time.time()-t:.0f}s, eta {el/(ep+1)*(args.epochs-ep-1)/60:.0f}m)',
              flush=True)

    model.load_state_dict(torch.load(ckpt)['state_dict'])
    val = score(model, stats, 'val', pval, arm)
    test = score(model, stats, 'test', ptest, arm)
    print(f'\n[{arm}] total {(time.time()-start)/60:.1f} min')
    print('[' + arm + '] ' + lb.fmt('[TEST]', test), flush=True)
    return {'val': val, 'test': test, 'curve': curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-9)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--arms', default='concat,residual')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    stats = dict(np.load(f'{args.out}/stats.npz'))
    for k in ('so4_mean', 'bc_mean', 'oc_mean', 'du_mean'):
        stats[k] = float(stats[k])
    W = np.load(f'{args.out}/W_poly2.npy')

    t = time.time()
    Xtr, ytr = load_split_z('train', stats)
    print(f'train loaded ({time.time()-t:.0f}s)', flush=True)
    t = time.time()
    ptr = torch.from_numpy(poly2_pred_z(stats, W, 'train'))
    pval = poly2_pred_z(stats, W, 'val')
    ptest = poly2_pred_z(stats, W, 'test')
    print(f'poly2 predictions cached ({time.time()-t:.0f}s)', flush=True)

    data = (Xtr, ytr, ptr, pval, ptest)
    results = {}
    for arm in args.arms.split(','):
        results[arm] = train_arm(arm, data, stats, args)
        with open(f'{args.out}/scores_nn_poly2.json', 'w') as f:
            json.dump(results, f, indent=2)

    # final comparison against the plain-NN control
    ctrl = json.load(open(f'{args.out}/scores_nn_base.json'))
    cols = [('NN control', ctrl), *[(a, results[a]) for a in results]]
    print(f'\n{"channel":<9}' + ''.join(f'{c[0][:11]:>12}' for c in cols))
    for i, nm in enumerate(NAMES):
        print(f'{nm:<9}' + ''.join(f'{c[1]["test"]["per_channel_R2"][i]:12.3f}' for c in cols))
    print(f'{"MEAN":<9}' + ''.join(f'{c[1]["test"]["R2"]:12.3f}' for c in cols))
    print(f'{"MSE":<9}' + ''.join(f'{c[1]["test"]["MSE(norm)"]:12.4f}' for c in cols))


if __name__ == '__main__':
    main()
