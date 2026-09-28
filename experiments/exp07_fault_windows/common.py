"""Shared setup for Exp07: data, foundation matrices, metrics, slices.

The FOUNDATION is the Exp06b candidate, imported from the experiment that
validated it.  Nothing here modifies it: the 249-column ``full - age`` matrix is
rebuilt by calling the same code path, and a parity assert guards against drift.
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
sys.path.insert(0, HERE)

from features import extract_features, parse_run           # noqa: E402
from metrics import composite, f1_per_class, macro_f1       # noqa: E402
from metrics import binary_f1, fault_turn_hit_at_k         # noqa: E402
from localize import localize                              # noqa: E402
import new_features as nf                                  # noqa: E402
import lifecycle as lc                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp06_features", os.path.join(ROOT, "experiments",
                                   "exp06_temporal_dynamics", "features.py"))
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

import window_features as wf                               # noqa: E402
import window_dataset as wd                                # noqa: E402
import aggregate as ag                                     # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
DH = "dropped_handoff"

N_FOUNDATION = 249
N_SUCCESS_FEATS = 185
B1 = set(ft.BLOCKS["1_age"])

T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def lgb_label():
    """Byte-identical to the validated Exp05/Exp06/Exp06b configuration."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def lgb_window():
    """Fixed, reasonable, never tuned.  Identical in every fold.

    Imbalance is handled with an explicit sample_weight vector (the pre-declared
    class weight of the window's true class) rather than sklearn's
    ``class_weight``, so the weighting is visible at the call site and is
    trivially auditable.
    """
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass",
                              num_class=wd.N_WINDOW_CLASSES, n_estimators=300,
                              learning_rate=0.08, num_leaves=63, min_child_samples=50,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def fit_window(X, y, **kw):
    """Fit the window model with the fixed, pre-declared class weights."""
    m = lgb_window()
    m.fit(X, y, sample_weight=wd.class_weight_vector()[y])
    return m


def build_all():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    ys = train["label"].values.astype(object)
    yi = np.array([L2I[l] for l in ys], dtype=int)
    runs = [parse_run(r) for r in train.to_dict("records")]

    log("parsing runs + official 122 features")
    base_rows = [extract_features(r) for r in runs]
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    log("building exp03 norm + exp05 lifecycle + exp06 temporal (foundation)")
    Xn = nf.build_matrix(runs, base_rows)
    XA = pd.concat([Xb, Xn, lc.build_matrix(runs)[lc.LIFECYCLE_NAMES]], axis=1)
    X6 = ft.build_matrix(runs)
    X6 = X6[[c for c in X6.columns if c not in B1]]
    foundation = pd.concat([XA, X6], axis=1)
    assert foundation.shape[1] == N_FOUNDATION, \
        "foundation is %d cols, expected %d" % (foundation.shape[1], N_FOUNDATION)
    success = pd.concat([Xb, Xn], axis=1)
    assert success.shape[1] == N_SUCCESS_FEATS, \
        "success matrix is %d cols, expected %d" % (success.shape[1], N_SUCCESS_FEATS)
    log("  foundation=%d  success=%d" % (foundation.shape[1], success.shape[1]))

    # Success F1 is PINNED to the validated Exp03 B value, exactly as in
    # Exp05/Exp06/Exp06b.  No Exp03..Exp07 feature targets `success`, so
    # refitting it would add noise without informing the comparison.
    e3 = pd.read_csv(os.path.join(ROOT, "experiments", "exp03_length_normalization",
                                  "oof_B_baseline_plus_norm.csv"))
    assert (e3["run_id"].values == train["run_id"].values).all()
    success_f1 = float(binary_f1(list(train["success"].values),
                                 list(e3["success_pred"].values)))
    log("Success F1 pinned from Exp03 B = %.4f" % success_f1)

    nmsg = Xb["n_messages"].values
    q66 = np.quantile(nmsg, 2 / 3)
    SL = {
        "long(>p66)": nmsg > q66,
        "no_intent": Xb["n_intents"].values <= 1,
        "topo_mesh": Xb["topo_mesh"].values > 0,
        "topo_blackboard": Xb["topo_blackboard"].values > 0,
        "hard/robustness": (nmsg > np.median(nmsg)) &
        ((Xb["topo_mesh"].values > 0) | (Xb["topo_blackboard"].values > 0)),
    }
    return {"train": train, "runs": runs, "foundation": foundation,
            "success": success, "yi": yi, "ys": ys, "n": len(yi),
            "success_f1": success_f1, "fturn": train["fault_turn"].values,
            "nmsg": nmsg, "SL": SL}


def build_windows(runs, ys, fturn, tag=""):
    log("building window dataset %s" % tag)
    d = wd.build_window_dataset(runs, ys, fturn)
    log(wd.summarise(d, "windows" + (" " + tag if tag else "")))
    return d


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


def sl_macro(ys, pred, mask, min_n=30):
    idx = np.where(mask)[0]
    if len(idx) < min_n:
        return None
    return macro_f1([ys[i] for i in idx], [pred[i] for i in idx])
