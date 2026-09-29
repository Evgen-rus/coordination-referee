"""Is the 4s-vs-22s per-fit swing caused by machine state rather than code?

Repeats the IDENTICAL 12-fit batch several times in one process, and reports
each round's mean.  A code-level cause would give a stable number; a machine
(thermal/power/contention) cause shows up as drift across rounds.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cache_matrices as CM  # noqa: E402
import lightgbm as lgb       # noqa: E402

PARAMS = dict(objective="multiclass", num_class=7, learning_rate=0.05,
              num_leaves=63, min_child_samples=20, subsample=0.9,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=42, verbose=-1)


def load_avg():
    """1-minute load average, a proxy for external CPU contention."""
    try:
        return subprocess.run(["wmic", "cpu", "get", "loadpercentage"],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return "?"


def round_once(mats, y, z, label):
    ts = []
    for f in (0, 1, 2):
        tr, va = CM.split(z, 0, f)
        for k in mats.keys():
            m = lgb.LGBMClassifier(n_jobs=-1, **PARAMS)
            t0 = time.perf_counter()
            m.fit(mats[k][tr], y[tr])
            ts.append(time.perf_counter() - t0)
            del m
    print("  %-28s mean %6.2fs  min %5.2f  max %6.2f  | cpu load: %s"
          % (label, np.mean(ts), np.min(ts), np.max(ts),
             " ".join(load_avg().split())[:4]))
    return np.mean(ts)


def main():
    mats, y = CM.load()
    z = np.load(CM.OUT)
    res = []
    for i in range(5):
        res.append(round_once(mats, y, z, "round %d" % i))
    print("\nmeans:", ["%.2f" % r for r in res])
    print("spread: %.1f%% of the slowest" %
          (100 * (max(res) - min(res)) / max(res)))


if __name__ == "__main__":
    main()
