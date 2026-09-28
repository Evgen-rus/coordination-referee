"""Shared setup for Exp06: data, feature matrices, masks, helpers."""

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

from features import extract_features, parse_run          # noqa: E402
from metrics import binary_f1, composite, f1_per_class     # noqa: E402
from metrics import fault_turn_hit_at_k, macro_f1         # noqa: E402
from localize import localize                             # noqa: E402
import new_features as nf                                 # noqa: E402
import lifecycle as lc                                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp06_features", os.path.join(HERE, "features.py"))
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
DH = "dropped_handoff"

B1, B2, B3 = ft.BLOCKS["1_age"], ft.BLOCKS["2_backlog"], ft.BLOCKS["3_receiver"]
B4, B5 = ft.BLOCKS["4_reassign"], ft.BLOCKS["5_tail"]
ALL6 = ft.ALL_NAMES

VARIANTS = [
    ("0_B_exp05_B", []),
    ("1_B_age", B1),
    ("2_B_backlog", B2),
    ("3_B_receiver", B3),
    ("4_B_reassign", B4),
    ("5_B_tail", B5),
    ("6_B_ALL", ALL6),
    ("7_full_minus_age", [c for c in ALL6 if c not in B1]),
    ("8_full_minus_backlog", [c for c in ALL6 if c not in B2]),
    ("9_full_minus_receiver", [c for c in ALL6 if c not in B3]),
    ("10_full_minus_reassign", [c for c in ALL6 if c not in B4]),
    ("11_full_minus_tail", [c for c in ALL6 if c not in B5]),
]
INCR = [v[0] for v in VARIANTS[:7]]
DROPS = [v[0] for v in VARIANTS[7:]]

T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def mf1(t, p):
    return macro_f1(list(t), list(p))


def lgb_label():
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


def build_all():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    yi = train["label"].map(L2I).values
    ys = np.array([LABELS[i] for i in yi], dtype=object)
    runs = [parse_run(r) for r in train.to_dict("records")]
    log("parsing runs + official 122 features")
    base_rows = [extract_features(r) for r in runs]
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    log("building exp03 normalised + exp05 lifecycle + exp06 temporal")
    Xn = nf.build_matrix(runs, base_rows)
    del base_rows
    XA = pd.concat([Xb, Xn, lc.build_matrix(runs)[lc.LIFECYCLE_NAMES]], axis=1)
    X6 = ft.build_matrix(runs)
    log("  foundation=%d  exp06=%d" % (XA.shape[1], X6.shape[1]))
    if XA.shape[1] != 226:
        log("  WARNING: foundation %d cols, expected 226" % XA.shape[1])
    mats = {name: (pd.concat([XA, X6[cols]], axis=1) if cols else XA)
            for name, cols in VARIANTS}

    # Success F1 is HELD FIXED at the Exp03 B value, which is what Exp05 B also
    # used: no Exp05/Exp06 feature targets `success`, so re-fitting it would add
    # noise without informing the comparison.  The saved Exp03 B OOF is the only
    # artifact that carries success_pred.
    e3 = pd.read_csv(os.path.join(HERE, "..", "exp03_length_normalization",
                                  "oof_B_baseline_plus_norm.csv"))
    assert (e3["run_id"].values == train["run_id"].values).all()
    success_f1 = float(binary_f1(list(train["success"].values),
                                 list(e3["success_pred"].values)))
    log("Success F1 held fixed from Exp03/Exp05 B = %.4f" % success_f1)

    nmsg = Xb["n_messages"].values
    q33, q66, q80 = (np.quantile(nmsg, 1 / 3), np.quantile(nmsg, 2 / 3),
                     np.quantile(nmsg, 0.8))
    hard = (nmsg > np.median(nmsg)) & ((Xb["topo_mesh"].values > 0) |
                                       (Xb["topo_blackboard"].values > 0))
    no_intent = Xb["n_intents"].values <= 1
    dhm = ys == DH
    SL = {"no_intent": no_intent, "long(>p66)": nmsg > q66,
          "longest_20%": nmsg >= q80, "hard/robustness": hard,
          "topo_mesh": Xb["topo_mesh"].values > 0,
          "topo_blackboard": Xb["topo_blackboard"].values > 0}
    DH_SL = {"all dropped_handoff": dhm,
             "short(<=p33)": dhm & (nmsg <= q33),
             "medium(p33-p66)": dhm & (nmsg > q33) & (nmsg <= q66),
             "long(>p66)": dhm & (nmsg > q66),
             "longest 20%": dhm & (nmsg >= q80),
             "no_intent": dhm & no_intent,
             "mesh": dhm & (Xb["topo_mesh"].values > 0),
             "blackboard": dhm & (Xb["topo_blackboard"].values > 0)}
    return {"train": train, "runs": runs, "XA": XA, "X6": X6, "mats": mats,
            "yi": yi, "ys": ys, "fturn": train["fault_turn"].values,
            "n": len(yi), "success_f1": success_f1, "nmsg": nmsg,
            "SL": SL, "DH_SL": DH_SL}
