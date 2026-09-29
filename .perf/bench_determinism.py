"""Does LightGBM num_threads change the fitted model / predictions?

This decides whether threading is a SAFE optimization (bit-identical results)
or a behaviour-changing one.  Uses cached matrices so timing is clean.
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

KEY = "full_minus_age"


def lgb_n(nj):
    return lgb.LGBMClassifier(objective="multiclass", num_class=7,
                              n_estimators=600, learning_rate=0.05,
                              num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0,
                              n_jobs=nj, random_state=42, verbose=-1,
                              force_col_wise=True)


def fit_pred(X, y, tr, va, nj):
    m = lgb_n(nj)
    t0 = time.perf_counter()
    m.fit(X[tr], y[tr])
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = m.predict_proba(X[va])
    tp = time.perf_counter() - t0
    return p, tf, tp


def main():
    mats, y = CM.load()
    z = np.load(CM.OUT)
    tr, va = CM.split(z, 0, 0)
    X = mats[KEY]
    print("X", X.shape, "train", len(tr), "val", len(va))

    ref, tf, tp = fit_pred(X, y, tr, va, -1)
    print("\nreference n_jobs=-1: fit %.2fs predict %.2fs" % (tf, tp))
    print("\nthreads |  fit_s  pred_s | max|dP|  argmax_diff")
    print("-" * 62)
    for nj in (1, 2, 3, 4, 6, 8, 12, -1):
        p, f, q = fit_pred(X, y, tr, va, nj)
        d = float(np.abs(p - ref).max())
        am = int((p.argmax(1) != ref.argmax(1)).sum())
        print("%7d | %6.2f  %7.2f | %.3e  %8d" % (nj, f, q, d, am))

    print("\n--- repeatability of the SAME config (n_jobs=-1 twice) ---")
    a, _, _ = fit_pred(X, y, tr, va, -1)
    b, _, _ = fit_pred(X, y, tr, va, -1)
    print("max|dP| = %.3e   argmax diff = %d"
          % (np.abs(a - b).max(), int((a.argmax(1) != b.argmax(1)).sum())))

    print("\n--- does force_col_wise change results? ---")
    for fcw in (True, False):
        m = lgb.LGBMClassifier(objective="multiclass", num_class=7,
                               n_estimators=600, learning_rate=0.05,
                               num_leaves=63, min_child_samples=20,
                               subsample=0.9, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0,
                               n_jobs=-1, random_state=42, verbose=-1,
                               force_col_wise=fcw)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])
        print("force_col_wise=%-5s fit %.2fs  max|dP| vs ref %.3e"
              % (fcw, time.perf_counter() - t0, float(np.abs(p - ref).max())))


if __name__ == "__main__":
    main()
