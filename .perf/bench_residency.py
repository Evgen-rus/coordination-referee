"""Why does the REAL pipeline pay ~26s/fit when an identical isolated fit is ~4s?

Runs the real build(), then times ONE fit exactly as run_validation does
(pandas .iloc[tr]) while progressively releasing the objects that build()
left alive: the parsed `runs`, the raw `train` DataFrame, the per-system
DataFrames, and the X6/XA intermediates.

Isolates whether the slowdown is (a) pandas, (b) memory pressure, or
(c) numpy array duplication/copying, and which release actually recovers speed.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "experiments", "exp06b_stability")
sys.path.insert(0, HERE)
for p in (os.path.join(ROOT, "baseline"), os.path.join(ROOT, "evaluation"),
          os.path.join(ROOT, "experiments", "exp03_length_normalization"),
          os.path.join(ROOT, "experiments", "exp05_handoff_lifecycle")):
    sys.path.insert(0, p)
sys.path.insert(0, EXP)

_c = _load = None
spec = importlib.util.spec_from_file_location("common", os.path.join(EXP, "common.py"))
_c = importlib.util.module_from_spec(spec)
sys.modules["common"] = _c
spec.loader.exec_module(_c)
import lightgbm as lgb  # noqa: E402
import profile_exp06b as P  # noqa: E402

KEY = "full_minus_age"
PARAMS = dict(objective="multiclass", num_class=7, learning_rate=0.05,
              num_leaves=63, min_child_samples=20, subsample=0.9,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=42, verbose=-1)


def rss():
    try:
        import ctypes, ctypes.wintypes as w

        class PMC(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 1e6, c.PeakWorkingSetSize / 1e6
    except Exception as e:
        return -1, -1


def timed_fit(mat, y, tr, va, label, n=2):
    ts = []
    for _ in range(n):
        m = lgb.LGBMClassifier(n_jobs=-1, **PARAMS)
        t0 = time.perf_counter()
        m.fit(mat.iloc[tr], y[tr])
        ts.append(time.perf_counter() - t0)
        del m
    a, p = rss()
    print("  %-44s %s  mean %6.2fs  rss=%.0fMB peak=%.0fMB"
          % (label, " ".join("%5.2f" % t for t in ts), np.mean(ts), a, p))
    return np.mean(ts)


def main():
    print("=== STEP 1: fresh build, then fit exactly as the pipeline does ===")
    c = P.do_build(verbose=True)
    yi = c["yi"]
    tr, va = P.CM.split(np.load(P.CM.OUT), 0, 0) if hasattr(P, "CM") else (None, None)
    from sklearn.model_selection import StratifiedKFold
    tr, va = list(StratifiedKFold(3, shuffle=True, random_state=0)
                  .split(np.zeros(c["n"]), yi))[0]
    mat = c["mats"][KEY]
    t0 = timed_fit(mat, yi, tr, va, "A. all build() objects alive (pandas)")

    print("\n=== STEP 2: same fit, but convert that one matrix to numpy ===")
    Xn = mat.to_numpy(dtype=np.float64)
    t0n = timed_fit_n = 0
    ts = []
    for _ in range(2):
        m = lgb.LGBMClassifier(n_jobs=-1, **PARAMS)
        t0 = time.perf_counter()
        m.fit(Xn[tr], yi[tr])
        ts.append(time.perf_counter() - t0)
        del m
    print("  %-44s %s  mean %6.2fs" % ("B. numpy fancy-index, all alive",
                                       " ".join("%5.2f" % t for t in ts), np.mean(ts)))

    print("\n=== STEP 3: drop `runs` (10k parsed run dicts) ===")
    c["runs"] = None
    gc.collect()
    t1 = timed_fit(mat, yi, tr, va, "C. runs dropped (pandas)")

    print("\n=== STEP 4: drop the other 3 system matrices too ===")
    keep = c["mats"][KEY]
    c["mats"] = {KEY: keep}
    gc.collect()
    t2 = timed_fit(mat, yi, tr, va, "D. only 1 matrix alive (pandas)")

    print("\n=== STEP 5: also drop the raw train DataFrame ===")
    c["train"] = None
    gc.collect()
    t3 = timed_fit(mat, yi, tr, va, "E. 1 matrix, no train (pandas)")

    print("\nSUMMARY  A=%.2f  C=%.2f  D=%.2f  E=%.2f" % (t0, t1, t2, t3))


if __name__ == "__main__":
    main()
