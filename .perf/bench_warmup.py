"""Resolve the per-fit cost discrepancy: is one LightGBM fit ~4s or ~20s?

Runs the identical config repeatedly in ONE process, then the full 12-fit batch,
timing every fit individually.  Separates warmup from real cost.
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

KEY = "full_minus_age"


def timed(X, y, tr, va, nj, label):
    m = lgb.LGBMClassifier(n_jobs=nj, **PARAMS)
    t0 = time.perf_counter()
    m.fit(X[tr], y[tr])
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.predict_proba(X[va])
    tp = time.perf_counter() - t0
    print("  %-38s fit %7.2fs  pred %6.2fs" % (label, tf, tp))
    return tf


def main():
    mats, y = CM.load()
    z = np.load(CM.OUT)
    tr, va = CM.split(z, 0, 0)
    X = mats[KEY]
    print("X", X.shape, "train", len(tr), "\n")

    print("A. same fit repeated 3x, n_jobs=-1 (warmup check)")
    for i in range(3):
        timed(X, y, tr, va, -1, "full_minus_age fold0 run%d" % i)

    print("\nB. thread sweep")
    for nj in (1, 12):
        timed(X, y, tr, va, nj, "n_jobs=%d" % nj)

    print("\nC. all 12 (4 systems x 3 folds) sequentially, n_jobs=-1")
    t0 = time.perf_counter()
    tot = 0.0
    for f in (0, 1, 2):
        tr2, va2 = CM.split(z, 0, f)
        for k in mats.keys():
            tot += timed(mats[k], y, tr2, va2, -1, "%s fold%d" % (k, f))
    wall = time.perf_counter() - t0
    print("  TOTAL wall %.1fs   sum of fits %.1fs" % (wall, tot))

    print("\nD. matrix shapes / dtypes")
    for k, v in mats.items():
        print("  %-32s %s %s  %.1f MB" % (k, v.shape, v.dtype, v.nbytes / 1e6))


if __name__ == "__main__":
    main()
