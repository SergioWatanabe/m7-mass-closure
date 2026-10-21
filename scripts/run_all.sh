#!/bin/sh
# Reproduce every table in the README. Runs strictly one job at a time: each NN run
# holds the full train split in memory (~1.4 GB), so this fits on an 8 GB laptop.
# Timings are from an M1 (CPU, 4 threads): ~35 min per 100-epoch NN run.
set -e
cd "$(dirname "$0")/.."
sh scripts/download_data.sh

# Linear baselines (minutes each). The first run also writes results/stats.npz.
for f in z log z+log poly2 poly2_zlog; do python linear_baseline.py --features $f; done
python mass_constrained.py --features poly2

# Base NN, 3 seeds on the full data (~35 min each)
python nn_reference.py --seed 0 --tag nn_base
python nn_reference.py --seed 1 --tag nn_base_seed1
python nn_reference.py --seed 2 --tag nn_base_seed2

# Mass closure on each checkpoint (the main result)
for t in nn_base nn_base_seed1 nn_base_seed2; do python nn_closure.py --ckpt results/$t.pth; done

# Supporting experiments: stacking poly2 under shift (~70 min), seed spread on an 80% subsample (~90 min)
python nn_poly2.py
python nn_seed_spread.py

python scripts/make_figures.py
