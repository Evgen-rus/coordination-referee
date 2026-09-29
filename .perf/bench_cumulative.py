"""Cumulative degradation: does each successive fit get slower?

Times all 12 fits (4 systems x 3 folds) in the exact order the real
run_validation.py runs them, printing each one.  If fit #1 is ~4s and
fit #12 is ~26s, the cause is cumulative state inside the process (LightGBM
thread pool / memory fragmentation / GC pressure), not the model or the data.
"""
from __future__ import annotations

import gc
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

import profile_exp06b as P  # noqa: E402
import cache_matrices as CM  # noqa: E402


def main():
    print("=== CASE 1: all 12 fits, matrices ONLY from cache (no build) ===")
    mats, y = CM.load()
    z = np.load(CM.OUT)
    tot = 0.0
    for f in (0, 1, 2):
        tr, va = CM.split(z, 0, f)
        for k in mats.keys():
            t0 = time.perf_counter()
            m = _c.lgb_label().fit(mats[k][tr], y[tr])
            tf = time.perf_counter() - t0
            tot += tf
            print("  fit      %-32s fold%d  %6.2fs" % (k, f, tf))
            del m
        gc.collect()
    print("  SUM of 12 fits: %.1fs\n" % tot)

    print("=== CASE 2: same 12 fits, but AFTER a real build() ===")
    c = P.do_build(verbose=False)
    mats2, y2 = c["mats"], c["yi"]
    tot2 = 0.0
    for f in (0, 1, 2):
        tr, va = CM.split(np.load(CM.OUT), 0, f)
        for k in mats2.keys():
            t0 = time.perf_counter()
            m = _c.lgb_label().fit(mats2[k].iloc[tr], y2[tr])
            tf = time.perf_counter() - t0
            tot2 += tf
            print("  fit      %-32s fold%d  %6.2fs" % (k, f, tf))
            del m
        gc.collect()
    print("  SUM of 12 fits: %.1fs" % tot2)
    print("\nCASE1=%.1fs  CASE2=%.1fs  delta=%.1fs" % (tot, tot2, tot2 - tot))


if __name__ == "__main__":
    main()
