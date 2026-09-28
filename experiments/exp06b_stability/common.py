"""Shared setup for Exp06b: the four pre-registered systems, nothing else.

This module deliberately rebuilds only the four feature matrices that Exp06b is
allowed to compare.  No feature is added, removed or renamed relative to
Exp05/Exp06, and the LightGBM hyper-parameters are byte-identical to
``exp06_temporal_dynamics.common.lgb_label``.

The only thing that varies across the study is the CV split
(``StratifiedKFold(3, shuffle=True, random_state=seed)``).  LightGBM keeps
``random_state=42`` throughout, so the study measures *split* variance and not
model-seed noise - which is exactly the axis on which a post-hoc pick like
``full - age`` can be selection noise.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("baseline", "evaluation",
          os.path.join(HERE, "..", "exp03_length_normalization"),
          os.path.join(HERE, "..", "exp05_handoff_lifecycle")):
    sys.path.insert(0, os.path.join(ROOT, p))

from features import extract_features, parse_run           # noqa: E402
from metrics import f1_per_class, macro_f1                # noqa: E402
import new_features as nf                                 # noqa: E402
import lifecycle as lc                                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp06_features", os.path.join(
        HERE, "..", "exp06_temporal_dynamics", "features.py"))
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
DH = "dropped_handoff"

# Feature counts asserted below are the ones quoted in the study design.
N_FEATURES = {"exp05_B": 226, "exp06_ALL": 255,
              "full_minus_age": 249, "full_minus_age_minus_reassign": 245}

_B1 = ft.BLOCKS["1_age"]
_B4 = ft.BLOCKS["4_reassign"]
_ALL = ft.ALL_NAMES

# The four systems, fixed in advance.  Membership is defined by reference to the
# Exp06 blocks, never re-derived from these seeds' results.
COLS = {
    "exp05_B": [],
    "exp06_ALL": list(_ALL),
    "full_minus_age": [c for c in _ALL if c not in _B1],
    "full_minus_age_minus_reassign": [c for c in _ALL
                                      if c not in _B1 and c not in _B4],
}
REFERENCE = "exp05_B"
CANDIDATES = ["full_minus_age", "full_minus_age_minus_reassign"]
SEEDS = [0, 7, 21, 42, 99]
N_FOLDS = 3

T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def lgb_label():
    """Identical to Exp05/Exp06.  Do not tune: this study is about features."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def prf(y_true, y_pred, cls, mask=None):
    yt = y_true[mask] if mask is not None else y_true
    yp = y_pred[mask] if mask is not None else y_pred
    tp = int(((yt == cls) & (yp == cls)).sum())
    fp = int(((yt != cls) & (yp == cls)).sum())
    fn = int(((yt == cls) & (yp != cls)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if (p + r) else 0.0, "n": int(len(yt))}


def build():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    ys = train["label"].values.astype(object)
    yi = np.array([LABELS.index(l) for l in ys], dtype=int)
    runs = [parse_run(r) for r in train.to_dict("records")]

    log("parsing runs + official 122 features")
    base_rows = [extract_features(r) for r in runs]
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    log("building exp03 normalised + exp05 lifecycle + exp06 temporal")
    Xn = nf.build_matrix(runs, base_rows)
    del base_rows
    XA = pd.concat([Xb, Xn, lc.build_matrix(runs)[lc.LIFECYCLE_NAMES]], axis=1)
    X6 = ft.build_matrix(runs)

    mats = {k: (pd.concat([XA, X6[c]], axis=1) if c else XA)
            for k, c in COLS.items()}
    for k, mat in mats.items():
        assert mat.shape[1] == N_FEATURES[k], \
            "%s: %d cols, design says %d" % (k, mat.shape[1], N_FEATURES[k])
    log("  " + "  ".join("%s=%d" % (k, m.shape[1]) for k, m in mats.items()))

    # Success F1 stays pinned to the Exp03/Exp05 B value: no Exp05/Exp06 feature
    # targets `success`, so re-fitting it would add noise without informing this
    # comparison.  Same convention as Exp06.
    e3 = pd.read_csv(os.path.join(HERE, "..", "exp03_length_normalization",
                                  "oof_B_baseline_plus_norm.csv"))
    assert (e3["run_id"].values == train["run_id"].values).all()
    from metrics import binary_f1
    success_f1 = float(binary_f1(list(train["success"].values),
                                 list(e3["success_pred"].values)))
    log("Success F1 held fixed from Exp03/Exp05 B = %.4f" % success_f1)

    nmsg = Xb["n_messages"].values
    q33, q66 = np.quantile(nmsg, 1 / 3), np.quantile(nmsg, 2 / 3)
    SL = {"hard/robustness": (nmsg > np.median(nmsg)) &
          ((Xb["topo_mesh"].values > 0) | (Xb["topo_blackboard"].values > 0)),
          "topo_mesh": Xb["topo_mesh"].values > 0}
    DH_SL = {"long(>p66)": (ys == DH) & (nmsg > q66)}
    return {"train": train, "runs": runs, "mats": mats, "yi": yi, "ys": ys,
            "n": len(yi), "success_f1": success_f1, "fturn": train["fault_turn"].values,
            "nmsg": nmsg, "SL": SL, "DH_SL": DH_SL}
