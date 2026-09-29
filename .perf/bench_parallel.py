"""Can the independent LightGBM fits be run concurrently without changing results?

exp06b fits 4 systems x 3 folds x 5 seeds = 60 models, sequentially, each with
n_jobs=-1.  Thread scaling measured only ~2.3x from 1->12 threads, so a
12-thread fit leaves the CPU mostly idle.  This checks whether running the
independent fits CONCURRENTLY (processes x few threads each) is faster AND
bit-identical.

The fits are genuinely independent: each (system, fold) pair is a separate fit
on its own train/val indices with its own random_state=42, writing to
oof[key][va].  No shared mutable state, no leakage path.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cache_matrices as CM  # noqa: E402
import lightgbm as lgb       # noqa: E402

PARAMS = dict(objective="multiclass", num_class=7, learning_rate=0.05,
              num_leaves=63, min_child_samples=20, subsample=0.9,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=42, verbose=-1)


def _one(job):
    key, tr, va, nj, X, y = job
    m = lgb.LGBMClassifier(n_jobs=nj, **PARAMS)
    m.fit(X[key][tr], y[tr])
    return np.asarray(m.predict(X[key][va]))


def main():
    mats, y = CM.load()
    z = np.load(CM.OUT)
    keys = list(mats.keys())
    print("systems:", keys)

    base_jobs = []
    for f in (0, 1, 2):
        tr, va = CM.split(z, 0, f)
        for k in keys:
            base_jobs.append((k, tr, va))

    from joblib import Parallel, delayed

    print("\n%-32s %9s %8s  %s" % ("config", "wall_s", "speedup", "parity"))
    print("-" * 70)
    ref = None
    base = None
    for label, workers, nj in [
            ("sequential  1 proc x 12 thr", 1, -1),
            ("parallel    2 proc x  6 thr", 2, 6),
            ("parallel    3 proc x  4 thr", 3, 4),
            ("parallel    4 proc x  3 thr", 4, 3),
            ("parallel    6 proc x  2 thr", 6, 2),
            ("parallel    8 proc x  1 thr", 8, 1),
            ("parallel   12 proc x  1 thr", 12, 1)]:
        jobs = [(k, tr, va, nj, mats, y) for (k, tr, va) in base_jobs]
        t0 = time.perf_counter()
        out = Parallel(n_jobs=workers, backend="loky", verbose=0)(
            delayed(_one)(j) for j in jobs)
        wall = time.perf_counter() - t0
        if base is None:
            base = wall
        if ref is None:
            ref = out
            parity = "(reference)"
        else:
            bad = sum(int((a != b).sum()) for a, b in zip(out, ref))
            parity = "class-mismatches=%d" % bad
        print("%-32s %8.1fs %7.2fx  %s" % (label, wall, base / wall, parity))


if __name__ == "__main__":
    main()
