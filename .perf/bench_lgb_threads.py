"""LightGBM thread-scaling benchmark for the exp06b workload.

Answers three questions at once:
  1. how wall time scales with num_threads (is the 12-thread fit efficient?);
  2. whether num_threads changes the model (max |dP| vs n_jobs=-1);
  3. whether a fit is CPU-saturated (so concurrent fits can overlap).

Uses the real exp06b matrices/fold so timings are representative.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "experiments", "exp06b_stability")
sys.path.insert(0, os.path.join(ROOT, ".perf"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_c = _load("common", os.path.join(EXP, "common.py"))
COLS, N_FEATURES, LABELS = _c.COLS, _c.N_FEATURES, _c.LABELS
import profile_exp06b as P  # noqa: E402

import lightgbm as lgb  # noqa: E402


def lgb_n(nj):
    return lgb.LGBMClassifier(objective="multiclass", num_class=7,
                              n_estimators=600, learning_rate=0.05,
                              num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0,
                              n_jobs=nj, random_state=42, verbose=-1)


def main():
    c = P.do_build(verbose=False)
    mats, yi = c["mats"], c["yi"]
    n = c["n"]
    tr, va = list(StratifiedKFold(3, shuffle=True, random_state=0)
                  .split(np.zeros(n), yi))[0]
    key = "full_minus_age"
    Xtr, ytr = mats[key].iloc[tr], yi[tr]
    Xva = mats[key].iloc[va]
    Xtrv = Xtr.to_numpy(dtype=np.float64, copy=False)
    ytrv = np.asarray(ytr)
    Xvav = Xva.to_numpy(dtype=np.float64, copy=False)

    print("threads |  fit_s  predict_s | max|dP| vs n_jobs=-1")
    print("-" * 62)
    ref = None
    out = {}
    for nj in (1, 2, 3, 4, 6, 8, 12, -1):
        m = lgb_n(nj)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        tf = time.perf_counter() - t0
        t0 = time.perf_counter()
        P_ = m.predict_proba(Xva)
        tp = time.perf_counter() - t0
        if nj == -1:
            ref = P_.copy()
        d = float(np.abs(P_ - ref).max()) if ref is not None else float("nan")
        print("%7d | %6.2f  %9.2f | %.3e" % (nj, tf, tp, d))
        out[nj] = tf
        del m

    a = lgb_n(-1).fit(Xtr, ytr).predict_proba(Xva)
    b = lgb_n(-1).fit(Xtr, ytr).predict_proba(Xva)
    print("\nsame-config repeat max|dP| = %.3e" % np.abs(a - b).max())

    for name, (XX, yy) in (("pandas", (Xtr, ytr)), ("numpy", (Xtrv, ytrv))):
        m = lgb_n(-1)
        t0 = time.perf_counter()
        m.fit(XX, yy)
        print("%-8s fit with n_jobs=-1: %.2fs" % (name, time.perf_counter() - t0))

    mp = lgb_n(-1).fit(Xtr, ytr).predict_proba(Xva)
    mn = lgb_n(-1).fit(Xtrv, ytrv).predict_proba(Xvav)
    print("pandas vs numpy input max|dP| = %.3e" % np.abs(mp - mn).max())

    with open(os.path.join(HERE, "lgb_threads.json"), "w") as f:
        json.dump({str(k): v for k, v in out.items()}, f, indent=2)


if __name__ == "__main__":
    main()
