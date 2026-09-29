"""Settle the 4x per-fit discrepancy (4.3s vs 19s for the SAME fit).

Runs one identical fit many times in a single process, so machine state (power
plan / thermal throttling) is visible alongside fit timings.
"""
from __future__ import annotations

import os
import subprocess
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


def cpu_mhz():
    try:
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty CurrentClockSpeed)"],
            text=True).strip()
    except Exception as e:
        return "n/a"


def main():
    print("logical CPUs:", os.cpu_count(),
          "| OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", "<unset>"))
    print("current clock MHz:", cpu_mhz())
    print()

    mats, y = CM.load()
    z = np.load(CM.OUT)
    tr, va = CM.split(z, 0, 0)
    X = mats[KEY]

    print("%-4s %-8s %10s %10s %10s" % ("run", "n_jobs", "fit_s", "pred_s", "MHz"))
    print("-" * 48)
    for i in range(10):
        m = lgb.LGBMClassifier(n_jobs=-1, **PARAMS)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        tf = time.perf_counter() - t0
        t0 = time.perf_counter()
        m.predict(X[va])
        tp = time.perf_counter() - t0
        print("%-4d %-8s %9.2fs %9.2fs %10s" % (i, -1, tf, tp, cpu_mhz()))

    print("\nsingle-thread reference:")
    for i in range(3):
        m = lgb.LGBMClassifier(n_jobs=1, **PARAMS)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        print("  run%d fit %.2fs  MHz=%s" % (i, time.perf_counter() - t0, cpu_mhz()))


if __name__ == "__main__":
    main()
