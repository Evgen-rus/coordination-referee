"""Decisive A/B: force_col_wise, and the 4.1s vs 19s per-fit discrepancy."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cache_matrices as CM  # noqa: E402
import lightgbm as lgb       # noqa: E402

BASE = dict(objective="multiclass", num_class=7, learning_rate=0.05,
            num_leaves=63, min_child_samples=20, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, verbose=-1)
KEY = "full_minus_age"


def one(X, y, tr, va, **extra):
    m = lgb.LGBMClassifier(n_jobs=-1, **BASE, **extra)
    t0 = time.perf_counter()
    m.fit(X[tr], y[tr])
    tf = time.perf_counter() - t0
    p = m.predict_proba(X[va])
    return p, tf


def main():
    mats, y = CM.load()
    z = np.load(CM.OUT)
    tr, va = CM.split(z, 0, 0)
    X = mats[KEY]

    arms = [("force_col_wise=True", dict(force_col_wise=True)),
            ("force_col_wise=False", dict(force_col_wise=False)),
            ("auto (no kwarg)", dict())]
    print("interleaved, 3 reps each, n_jobs=-1, numpy input")
    res = {k: [] for k, _ in arms}
    ref = None
    for rep in range(3):
        for label, extra in arms:
            p, tf = one(X, y, tr, va, **extra)
            res[label].append(tf)
            if label == "force_col_wise=True" and rep == 0:
                ref = p
    for label, _ in arms:
        v = res[label]
        print("%-26s %s  mean %6.2fs"
              % (label, " ".join("%5.2f" % t for t in v), np.mean(v)))

    p_auto, _ = one(X, y, tr, va)
    print("\nmax|dP| auto vs force_col_wise=True = %.3e"
          % float(np.abs(p_auto - ref).max()))

    print("\n=== the exact 12-fit batch the real pipeline runs ===")
    t0 = time.perf_counter()
    tot = []
    for f in (0, 1, 2):
        tr2, va2 = CM.split(z, 0, f)
        for k in mats.keys():
            _, tf = one(mats[k], y, tr2, va2)
            tot.append(tf)
    wall = time.perf_counter() - t0
    print("  12 fits: wall %.1fs, mean fit %.2fs, sum %.1fs"
          % (wall, np.mean(tot), sum(tot)))
    print("  => extrapolated 60-fit study: %.0fs" % (wall * 5))


if __name__ == "__main__":
    main()
