#!/usr/bin/env python3
"""Honest shift comparison: within-slice CV vs cross-slice transfer.

The in-sample reference in shift_probes.py is 1.0, so degraded-copy deltas
there are optimistic. Here every number is out-of-sample:

  * reference  = 3-fold CV *inside* the target slice (model may see the slice)
  * transfer   = model trained only on the complementary slice, scored on target

transfer << reference  =>  the baseline depends on the training distribution.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from features import extract_features, parse_run   # noqa: E402
from metrics import macro_f1                       # noqa: E402

_T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - _T0, m), flush=True)


def mk():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def _mf1(a, b):
    assert len(a) == len(b)
    return macro_f1(list(a), list(b))


def cv_within(X, y, idx, folds=3):
    from sklearn.model_selection import StratifiedKFold
    oof = np.empty(len(idx), dtype=object)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(idx, y[idx]):
        m = mk().fit(X.iloc[idx[tr]], y[idx[tr]])
        oof[te] = m.predict(X.iloc[idx[te]])
    return _mf1(y[idx], oof)


def main():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    runs = [parse_run(r) for r in train.to_dict("records")]
    X = pd.DataFrame([extract_features(r) for r in runs]).fillna(0.0)
    y = train["label"].values
    nmsg, nagt = X["n_messages"].values, X["n_agents"].values
    med, med_agt = np.median(nmsg), np.median(nagt)
    allidx = np.arange(len(y))

    slices = {
        "long_runs": np.where(nmsg > med)[0],
        "many_agents": np.where(nagt > med_agt)[0],
        "rare_topo": np.where((X["topo_mesh"] + X["topo_blackboard"]).values > 0)[0],
        "no_intent_telemetry": np.where(X["n_intents"].values <= 1)[0],
        "blackboard_only": np.where(X["topo_blackboard"].values > 0)[0],
    }
    res = {}
    print("\n=== WITHIN-SLICE CV (reference) vs CROSS-SLICE TRANSFER (test-time shift) ===")
    print("%-24s %6s %10s %10s %9s" % ("slice", "n", "ref_CV", "transfer", "penalty"))
    for name, idx in slices.items():
        comp = np.setdiff1d(allidx, idx)
        ref = cv_within(X, y, idx)
        m = mk().fit(X.iloc[comp], y[comp])
        tr_f1 = _mf1(y[idx], m.predict(X.iloc[idx]))
        pen = tr_f1 - ref
        res[name] = {"n": len(idx), "ref_cv": ref, "transfer": tr_f1, "penalty": pen}
        log("  %-22s %6d %10.4f %10.4f %+9.4f" % (name, len(idx), ref, tr_f1, pen))

    # per-class transfer degradation on the hardest slices
    print("\n=== PER-CLASS F1: within-slice CV vs cross-slice transfer ===")
    LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
              "conflict", "goal_drift", "runaway_loop"]
    from metrics import f1_per_class
    from sklearn.model_selection import StratifiedKFold
    for name in ("long_runs", "no_intent_telemetry"):
        idx = slices[name]
        oof = np.empty(len(idx), dtype=object)
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(idx, y[idx]):
            oof[te] = mk().fit(X.iloc[idx[tr]], y[idx[tr]]).predict(X.iloc[idx[te]])
        comp = np.setdiff1d(allidx, idx)
        p = mk().fit(X.iloc[comp], y[comp]).predict(X.iloc[idx])
        a = f1_per_class(list(y[idx]), list(oof), LABELS)
        b = f1_per_class(list(y[idx]), list(p), LABELS)
        print("\n  slice: %s" % name)
        print("    %-18s %9s %9s %9s" % ("class", "ref_CV", "transfer", "delta"))
        for c in LABELS:
            print("    %-18s %9.4f %9.4f %+9.4f" % (c, a[c], b[c], b[c] - a[c]))
        res["per_class_" + name] = {"ref": a, "transfer": b}

    with open(os.path.join(HERE, "shift_honest.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    log("wrote experiments/shift_honest.json")


if __name__ == "__main__":
    main()
