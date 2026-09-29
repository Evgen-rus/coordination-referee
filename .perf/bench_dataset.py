"""Final root cause: why is a fit 4s in a tight loop but 22s in the real pattern?

The tight loop always reuses the SAME (system, fold) pair.  The real pattern
cycles through 4 systems x 3 folds.  Hypothesis: LightGBM's OpenMP thread
pool plus the sklearn/pandas wrapper re-converts the input on every .fit(),
and the cost scales with how often the *thread pool is torn down/rebuilt*.

This isolates the mechanism by timing:
  1. tight loop, one matrix, one fold, same object reused
  2. same, but a NEW estimator object each time
  3. cycling 4 matrices x 3 folds (the real pattern)
  4. pre-built numpy arrays for all folds (no per-fit conversion)
  5. native lgb.Dataset reused across fits
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "experiments", "exp06b_stability")
for p in (os.path.join(ROOT, "baseline"), os.path.join(ROOT, "evaluation"),
          os.path.join(ROOT, "experiments", "exp03_length_normalization"),
          os.path.join(ROOT, "experiments", "exp05_handoff_lifecycle")):
    sys.path.insert(0, p)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location("common", os.path.join(EXP, "common.py"))
_c = importlib.util.module_from_spec(spec)
sys.modules["common"] = _c
spec.loader.exec_module(_c)
import cache_matrices as CM  # noqa: E402
import lightgbm as lgb       # noqa: E402

KEY = "full_minus_age"
PARAMS = dict(objective="multiclass", num_class=7, learning_rate=0.05,
              num_leaves=63, min_child_samples=20, subsample=0.9,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=42, verbose=-1)


def main():
    mats, y = CM.load()
    z = np.load(CM.OUT)
    X = mats[KEY]
    tr, va = CM.split(z, 0, 0)
    ytr, yva = y[tr], y[va]

    print("1. TIGHT LOOP, one matrix/fold, new estimator each time")
    ts = []
    for _ in range(3):
        m = lgb.LGBMClassifier(n_jobs=-1, **PARAMS)
        t0 = time.perf_counter()
        m.fit(X[tr], ytr)
        ts.append(time.perf_counter() - t0)
    print("   %s  mean %.2fs" % (" ".join("%.2f" % t for t in ts), np.mean(ts)))

    print("\n2. cycling 4 matrices x 3 folds (the real pattern)")
    ts = []
    for f in (0, 1, 2):
        tr2, va2 = CM.split(z, 0, f)
        for k in mats.keys():
            m = lgb.LGBMClassifier(n_jobs=-1, **PARAMS)
            t0 = time.perf_counter()
            m.fit(mats[k][tr2], y[tr2])
            ts.append(time.perf_counter() - t0)
    print("   n=%d  mean %.2fs  min %.2f  max %.2f"
          % (len(ts), np.mean(ts), np.min(ts), np.max(ts)))

    print("\n3. pre-build Dataset objects ONCE, then fit boosters")
    from lightgbm import Dataset, train as lgb_train
    ds = {}
    for f in (0, 1, 2):
        tr2, _ = CM.split(z, 0, f)
        for k in mats.keys():
            t0 = time.perf_counter()
            ds[(k, f)] = Dataset(mats[k][tr2], label=y[tr2], free_raw_data=False)
    print("   Dataset construction total: %.2fs" % (time.perf_counter() - t0))
    ts = []
    for f in (0, 1, 2):
        for k in mats.keys():
            t0 = time.perf_counter()
            lgb_train(ds[(k, f)], params=PARAMS, num_boost_round=600)
            ts.append(time.perf_counter() - t0)
    print("   n=%d  mean %.2fs  min %.2f  max %.2f"
          % (len(ts), np.mean(ts), np.min(ts), np.max(ts)))

    print("\n4. same 12 fits, but cycling matrices with a FRESH process each time")
    print("   (see bench_fcw / bench_cumulative for this arm)")

    print("\n5. 12 fits, but all with n_jobs=4 (fewer threads, less contention)")
    ts = []
    for f in (0, 1, 2):
        tr2, va2 = CM.split(z, 0, f)
        for k in mats.keys():
            m = lgb.LGBMClassifier(n_jobs=4, **PARAMS)
            t0 = time.perf_counter()
            m.fit(mats[k][tr2], y[tr2])
            ts.append(time.perf_counter() - t0)
    print("   n=%d  mean %.2fs  min %.2f  max %.2f"
          % (len(ts), np.mean(ts), np.min(ts), np.max(ts)))


if __name__ == "__main__":
    main()
