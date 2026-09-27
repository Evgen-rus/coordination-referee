"""Shared pieces for experiment 02 (clean vs fault hierarchy)."""

from __future__ import annotations

import numpy as np

from metrics import macro_f1

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
FAULTY = LABELS[1:]

BASE = {
    "macro_f1": 0.7572580862071641,
    "robustness_f1": 0.7153778055125197,
    "success_f1": 0.8568961795812648,
    "fault_turn_hit2": 0.4073611111111111,
    "clean_recall": 0.8917857142857143,
    "clean_precision": 0.7725866336633663,
}
BASE_COMPOSITE = (0.50 * BASE["macro_f1"] + 0.25 * BASE["robustness_f1"]
                  + 0.15 * BASE["success_f1"] + 0.10 * BASE["fault_turn_hit2"])

VARIANTS = ["baseline_flat", "A_hier_thr0.5", "B_hier_thr_tuned",
            "C1_hier_bal_thr0.5", "C2_hier_bal_thr_tuned"]


def mf1(a, b):
    """Order-safe macro F1 (evaluation.metrics.f1_per_class zips positionally)."""
    assert len(a) == len(b), (len(a), len(b))
    return macro_f1(list(a), list(b))


def make_binary(weighted=False):
    import lightgbm as lgb
    kw = {"class_weight": "balanced"} if weighted else {}
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                              num_leaves=63, min_child_samples=20, subsample=0.9,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              n_jobs=-1, random_state=42, verbose=-1, **kw)


def make_type():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=6, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def make_flat():
    """The official flat 7-class baseline model, for a same-folds reference."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def tune_threshold(X, y, ftr, tr, weighted):
    """Best P(fault) threshold by 3-fold CV strictly inside the train part.

    Returns (best_thr, inner_macro_at_best, inner_macro_at_0.5).
    """
    from sklearn.model_selection import StratifiedKFold
    itr = np.arange(len(tr))
    p = np.zeros(len(tr))
    tl = np.empty(len(tr), dtype=object)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(itr, y[tr]):
        gate = make_binary(weighted).fit(X.iloc[tr[a]], ftr[a])
        p[b] = gate.predict_proba(X.iloc[tr[b]])[:, 1]
        fr = tr[a][ftr[a] == 1]
        tm = make_type().fit(X.iloc[fr], y[fr])
        # IMPORTANT: the class order is tm.classes_ (alphabetical for string
        # labels), NOT the FAULTY constant order.  Using FAULTY here permutes
        # every predicted class.
        tl[b] = tm.classes_[np.argmax(tm.predict_proba(X.iloc[tr[b]]), axis=1)]
    yt = y[tr]
    best, bf = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        f1 = mf1(yt, np.where(p >= thr, tl, "clean"))
        if f1 > bf:
            bf, best = f1, float(thr)
    return best, bf, mf1(yt, np.where(p >= 0.5, tl, "clean"))
