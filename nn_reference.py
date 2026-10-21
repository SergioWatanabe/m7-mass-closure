"""Reference run of the repo's Base NN, scored with the same harness as the
linear baselines, so the linear-vs-NN comparison is apples-to-apples rather than
a cross-reference to the paper's table.

Settings match `python main.py --model_id standard` exactly: models.Base (2 layers,
width 128), Adam lr 1e-3 / weight_decay 1e-9, batch 256, MSE on z-scored targets,
100 epochs, no early stopping, checkpoint on best validation loss.

Runs on CPU deliberately -- the model is ~25k parameters, so MPS is ~4x slower at
this batch size (kernel launch overhead dominates).
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

NAMES = (['SO4_ns', 'SO4_ks', 'SO4_as', 'SO4_cs', 'SO4_c4']
         + [f'BC_{m}' for m in ['ks', 'as', 'cs', 'ki']]
         + [f'OC_{m}' for m in ['ks', 'as', 'cs', 'ki']]
         + [f'DU_{m}' for m in ['as', 'cs', 'ai', 'ci']]
         + [f'NUM_{i}' for i in range(7)]
         + [f'H2O_{i}' for i in range(4)])


def load_split_z(split, stats):
    """Whole split, z-scored, as float32 tensors."""
    X, y = lb.load(split)
    n = X.shape[0]
    Xz = np.empty((n, 32), dtype=np.float32)
    yz = np.empty((n, 28), dtype=np.float32)
    for a, b in lb.chunks(n):
        xc, yc = lb.get_xy(X, y, a, b)
        Xz[a:b] = (xc - stats['xtrain_mean']) / stats['xtrain_std']
        yz[a:b] = (yc - stats['ytrain_mean']) / stats['ytrain_std']
    return torch.from_numpy(Xz), torch.from_numpy(yz)


def score(model, stats, split, chunk=250_000):
    """Full metric suite + per-channel R2. MSE(norm) equals the repo's val MSE loss."""
    X, y = lb.load(split)
    sc = lb.Scorer(stats)
    model.eval()
    with torch.no_grad():
        for a, b in lb.chunks(X.shape[0], chunk):
            xc, yc = lb.get_xy(X, y, a, b)
            xz = ((xc - stats['xtrain_mean']) / stats['xtrain_std']).astype(np.float32)
            out = model(torch.from_numpy(xz).to(device)).cpu().numpy().astype(np.float64)
            pred = out * stats['ytrain_std'] + stats['ytrain_mean']
            sc.update(yc, pred, xc)
    model.train()
    res = sc.result()
    n = sc.n
    res['per_channel_R2'] = (1 - sc.ss_res / (sc.s_t2 - sc.s_t ** 2 / n)).tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-9)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--val_every', type=int, default=1)
    ap.add_argument('--out', default='results')
    ap.add_argument('--tag', default='nn_base')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    stats = dict(np.load(f'{args.out}/stats.npz'))
    for k in ('so4_mean', 'bc_mean', 'oc_mean', 'du_mean'):
        stats[k] = float(stats[k])

    print(f'device={device}  threads={torch.get_num_threads()}', flush=True)
    t0 = time.time()
    Xtr, ytr = load_split_z('train', stats)
    n = Xtr.shape[0]
    print(f'loaded {n:,} train rows ({time.time()-t0:.0f}s)', flush=True)

    model = Base(32, 28, args.width).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss()
    ckpt = f'{args.out}/{args.tag}.pth'

    best = np.inf
    curve = []
    start = time.time()
    for ep in range(args.epochs):
        t = time.time()
        perm = torch.randperm(n)
        running, steps = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = crit(model(Xtr[idx].to(device)), ytr[idx].to(device))
            loss.backward()
            opt.step()
            running += loss.item(); steps += 1
        train_loss = running / steps

        if (ep + 1) % args.val_every == 0 or ep == args.epochs - 1:
            s = score(model, stats, 'val')
            curve.append({'epoch': ep + 1, 'train_loss': train_loss,
                          'val_R2': s['R2'], 'val_MSE': s['MSE(norm)']})
            flag = ''
            if s['MSE(norm)'] < best:          # repo checkpoints on best val loss
                best = s['MSE(norm)']
                torch.save({'state_dict': model.state_dict()}, ckpt)
                flag = ' *'
            el = time.time() - start
            eta = el / (ep + 1) * (args.epochs - ep - 1)
            print(f'epoch {ep+1:3d}  train={train_loss:.4f}  val R2={s["R2"]:+.4f}  '
                  f'val MSE={s["MSE(norm)"]:.4f}{flag}  '
                  f'({time.time()-t:.0f}s, eta {eta/60:.0f}m)', flush=True)
            with open(f'{args.out}/{args.tag}_curve.json', 'w') as f:
                json.dump(curve, f, indent=2)

    model.load_state_dict(torch.load(ckpt)['state_dict'])   # best-val model
    model.to(device)
    val = score(model, stats, 'val')
    test = score(model, stats, 'test')

    print(f'\ntotal {(time.time()-start)/60:.1f} min')
    print('\n' + lb.fmt('NN base [TEST]', test))
    print(f'\n{"channel":<9}{"val R2":>10}{"test R2":>10}')
    for i, nm in enumerate(NAMES):
        print(f'{nm:<9}{val["per_channel_R2"][i]:10.3f}{test["per_channel_R2"][i]:10.3f}')
    print(f'{"MEAN":<9}{val["R2"]:10.3f}{test["R2"]:10.3f}')
    print(f'{"MEDIAN":<9}{np.median(val["per_channel_R2"]):10.3f}'
          f'{np.median(test["per_channel_R2"]):10.3f}')

    with open(f'{args.out}/scores_{args.tag}.json', 'w') as f:
        json.dump({'val': val, 'test': test, 'epochs': args.epochs,
                   'curve': curve}, f, indent=2)


if __name__ == '__main__':
    main()
