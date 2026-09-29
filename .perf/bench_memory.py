"""Is the pipeline memory-bound?

The identical LightGBM fit is measured in three states:
  A. clean process, only the cached matrices resident
  B. after build(): train df + 10k parsed runs + all matrices alive
  C. same, but `runs` (10k parsed run dicts) dropped

Reports wall time and peak RSS to separate a compute limit from a RAM limit.
"""
from __future__ import annotations

import gc
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


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return -1.0


def peak_mb():
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e6
    except Exception:
        return -1.0


def fit3(X, y, tr, va, label):
    ts = []
    for _ in range(3):
        m = lgb.LGBMClassifier(n_jobs=-1, **PARAMS)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        ts.append(time.perf_counter() - t0)
        del m
    print("  %-36s %s | rss=%.0fMB peak=%.0fMB"
          % (label, " ".join("%.2fs" % t for t in ts), rss_mb(), peak_mb()))
    return ts


def main():
    print("=== A. clean process, matrices only ===")
    mats, y = CM.load()
    z = np.load(CM.OUT)
    tr, va = CM.split(z, 0, 0)
    a = fit3(mats[KEY], y, tr, va, "matrices only (numpy)")

    print("\n=== B. after build(): runs + train + matrices alive ===")
    import profile_exp06b as P
    c = P.do_build(verbose=True)
    Xb = c["mats"][KEY].to_numpy(dtype=np.float64)
    y2 = c["yi"]
    b = fit3(Xb, y2, tr, va, "after build(), all alive")

    print("\n=== C. same, but `runs` dropped ===")
    n_runs = len(c["runs"])
    c["runs"] = None
    gc.collect()
    cc = fit3(Xb, y2, tr, va, "runs dropped (was %d)" % n_runs)

    print("\n=== D. pandas .iloc[tr] slicing, as the pipeline does ===")
    m = c["mats"][KEY]
    ts = []
    for _ in range(3):
        mm = lgb.LGBMClassifier(n_jobs=-1, **PARAMS)
        t0 = time.perf_counter()
        mm.fit(m.iloc[tr], y2[tr])
        ts.append(time.perf_counter() - t0)
        del mm
    print("  pandas .iloc[tr] fits: %s" % " ".join("%.2fs" % t for t in ts))

    print("\nSUMMARY A=%.2fs B=%.2fs C=%.2fs" % (np.mean(a), np.mean(b), np.mean(cc)))
    print("dropping `runs` gives %.2fx; build() residency costs %.2fx"
          % (np.mean(b) / np.mean(cc), np.mean(b) / np.mean(a)))


if __name__ == "__main__":
    main()
