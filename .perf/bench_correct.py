"""CORRECTED benchmark: the real 600-tree config, straight from common.lgb_label().

The earlier micro-benchmarks accidentally used sklearn's default
n_estimators=100 instead of the pipeline's 600, which is why they read ~4.5s
per fit while the real pipeline pays ~22s.  Everything here uses the exact
hyper-parameters of ``exp06b_stability.common.lgb_label()``.

Measures:
  1. thread scaling and bit-identical parity across num_threads
  2. process-level parallelism over the 12 independent fits
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

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
# exactly common.lgb_label(), with n_jobs overridable
PARAMS = dict(objective="multiclass", num_class=7, n_estimators=600,
              learning_rate=0.05, num_leaves=63, min_child_samples=20,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=42, verbose=-1)


def fit(X, y, tr, nj):
    m = lgb.LGBMClassifier(n_jobs=nj, **PARAMS)
    t0 = time.perf_counter()
    m.fit(X[tr], y[tr])
    return m, time.perf_counter() - t0


def part_threads():
    print("=== 1. thread scaling + parity, real 600-tree config ===")
    mats, y = CM.load()
    z = np.load(CM.OUT)
    tr, va = CM.split(z, 0, 0)
    X = mats[KEY]
    m, t = fit(X, y, tr, -1)
    ref = m.predict_proba(X[va])
    print("  n_jobs=-1  fit %6.2fs  (reference)" % t)
    print("  %-9s %9s %14s %s" % ("n_jobs", "fit_s", "max|dP|", "argmax diff"))
    for nj in (1, 2, 4, 6, 8, 12):
        m2, t2 = fit(X, y, tr, nj)
        p = m2.predict_proba(X[va])
        print("  %-9d %8.2fs %14.3e %d"
              % (nj, t2, float(np.abs(p - ref).max()),
                 int((p.argmax(1) != ref.argmax(1)).sum())))
    return mats, y, z


def part_parallel(mats, y, z):
    print("\n=== 2. process parallelism over the 12 independent fits ===")
    jobs = []
    for f in (0, 1, 2):
        tr, va = CM.split(z, 0, f)
        for k in mats.keys():
            jobs.append((k, tr, va))

    def one(job):
        k, tr, va, nj = job
        m = lgb.LGBMClassifier(n_jobs=nj, **PARAMS)
        m.fit(mats[k][tr], y[tr])
        return np.asarray(m.predict(mats[k][va]))

    from joblib import Parallel, delayed
    ref, base = None, None
    print("  %-30s %9s %8s  %s" % ("config", "wall_s", "speedup", "parity"))
    for label, workers, nj in [
            ("sequential  1 proc x 12 thr", 1, -1),
            ("parallel    2 proc x  6 thr", 2, 6),
            ("parallel    3 proc x  4 thr", 3, 4),
            ("parallel    4 proc x  3 thr", 4, 3),
            ("parallel    6 proc x  2 thr", 6, 2)]:
        t0 = time.perf_counter()
        out = Parallel(n_jobs=workers, backend="loky")(
            delayed(one)((k, tr, va, nj)) for (k, tr, va) in jobs)
        wall = time.perf_counter() - t0
        if base is None:
            base, ref = wall, out
        bad = sum(int((a != b).sum()) for a, b in zip(out, ref))
        print("  %-30s %8.1fs %7.2fx  class-mismatches=%d"
              % (label, wall, base / wall, bad))


def main():
    mats, y, z = part_threads()
    part_parallel(mats, y, z)


if __name__ == "__main__":
    main()
