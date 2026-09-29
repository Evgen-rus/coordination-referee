"""Pick the best (processes x threads) split for the 60 real 600-tree fits.

Measurements on this machine carry ~30% noise, so every arm is repeated and
the MEDIAN is reported.  All arms use the exact exp06b hyper-parameters, and
every arm is checked for bit-identical predictions against the sequential
n_jobs=-1 reference.
"""
from __future__ import annotations

import importlib.util
import os
import statistics
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

PARAMS = dict(objective="multiclass", num_class=7, n_estimators=600,
              learning_rate=0.05, num_leaves=63, min_child_samples=20,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=42, verbose=-1)

MATS, Y = None, None


def one(job):
    k, tr, va, nj = job
    m = lgb.LGBMClassifier(n_jobs=nj, **PARAMS)
    m.fit(MATS[k][tr], Y[tr])
    return np.asarray(m.predict(MATS[k][va]))


def main():
    global MATS, Y
    MATS, Y = CM.load()
    z = np.load(CM.OUT)
    # one fold only: 4 systems, to keep each round affordable
    tr, va = CM.split(z, 0, 0)
    jobs = [(k, tr, va) for k in MATS.keys()]
    from joblib import Parallel, delayed

    arms = [("1 proc  x 12 thr (n_jobs=-1)", 1, -1),
            ("1 proc  x  8 thr", 1, 8),
            ("1 proc  x  6 thr", 1, 6),
            ("2 procs x  6 thr", 2, 6),
            ("3 procs x  4 thr", 3, 4),
            ("4 procs x  3 thr", 4, 3)]
    reps = 3
    ref = None
    print("%-28s %s   %-8s %s" % ("config", "  ".join("rep%d" % i for i in range(reps)),
                                  "median", "parity"))
    print("-" * 76)
    for label, workers, nj in arms:
        walls = []
        mism = 0
        for r in range(reps):
            t0 = time.perf_counter()
            out = Parallel(n_jobs=workers, backend="loky")(
                delayed(one)((k, tr, va, nj)) for (k, tr, va) in jobs)
            walls.append(time.perf_counter() - t0)
            if ref is None:
                ref = out
            else:
                mism = sum(int((a != b).sum()) for a, b in zip(out, ref))
        print("%-28s %s  %6.1fs  mismatches=%d"
              % (label, " ".join("%6.1f" % w for w in walls),
                 statistics.median(walls), mism))
    print("\n(reference = first arm; 0 mismatches everywhere => bit-identical)")


if __name__ == "__main__":
    main()
