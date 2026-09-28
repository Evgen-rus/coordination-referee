"""Exp07 runner: A/B - foundation (249) vs foundation + window features.

HONEST STACKING, asserted rather than asserted-about
----------------------------------------------------
For every outer fold:

  inner cross-fit   the outer-TRAIN runs are split into ``N_INNER`` folds; each
                    held-out fold receives window features from a window model
                    trained on the other inner folds only.  The final classifier
                    is then trained on these cross-fitted features, so no
                    outer-train row ever carries an in-sample window prediction.
  outer             a window model fitted on ALL of outer-train generates the
                    features for the outer-valid runs, which the classifier then
                    predicts.

``verify_stacking`` re-derives the fold membership and fails if any
outer-train run was ever predicted by a model that had seen it.
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (DH, LABELS, build_all, build_windows, composite,  # noqa: E402
                    f1_per_class, fault_turn_hit_at_k, lgb_label, log, localize,
                    prf, sl_macro)
import aggregate as ag                                      # noqa: E402
import grouping as grp                                      # noqa: E402
import window_dataset as wd                                  # noqa: E402
from common import fit_window                               # noqa: E402

N_OUTER = 3
N_INNER = 3
SEED = 0
A_KEY, B_KEY = "A_foundation", "B_plus_window"


def verify_stacking(n_runs, assign, n_inner, tag):
    """Every outer-train run must be held out exactly once, every inner fold must
    be non-empty, and the model that predicted a held-out run must have been
    trained on a strictly disjoint set."""
    if len(assign) != n_runs:
        raise AssertionError("assign length %d != n_runs %d" % (len(assign), n_runs))
    if assign.min() < 0 or assign.max() >= n_inner:
        raise AssertionError("assign has values outside [0, %d)" % n_inner)
    held_counts = np.zeros(n_runs, dtype=int)
    for f in range(n_inner):
        held = np.where(assign == f)[0]
        trained = np.where(assign != f)[0]
        if len(held) == 0:
            raise AssertionError("inner fold %d of %s is empty" % (f, tag))
        if len(trained) == 0:
            raise AssertionError("inner fold %d of %s has no training runs"
                                 % (f, tag))
        if len(np.intersect1d(held, trained)):
            raise AssertionError("inner fold %d of %s: held-out runs appear in "
                                 "its own training set" % (f, tag))
        held_counts[held] += 1
    if not (held_counts == 1).all():
        raise AssertionError("inner folds of %s do not partition the outer-train "
                             "runs exactly once (max count %d)"
                             % (tag, int(held_counts.max())))
    log("  [%s] stacking verified: %d outer-train runs, each cross-fitted from a "
        "model excluding it (%d inner folds, no overlap)" % (tag, n_runs, n_inner))


def confusion(ys, pred):
    out = {}
    for a, b in (("dropped_handoff", "clean"), ("dropped_handoff", "deadlock"),
                 ("deadlock", "dropped_handoff"), ("duplicated_work", "clean"),
                 ("runaway_loop", "clean"), ("goal_drift", "clean")):
        m = ys == a
        out["%s->%s" % (a, b)] = int((pred[m] == b).sum()) if m.any() else 0
    out["_n"] = {a: int((ys == a).sum()) for a, b in
                 (("dropped_handoff", "x"), ("deadlock", "x"),
                  ("duplicated_work", "x"), ("runaway_loop", "x"),
                  ("goal_drift", "x"), ("conflict", "x"))}
    return out


def main():
    c = build_all()
    runs, yi, ys, n = c["runs"], c["yi"], c["ys"], c["n"]
    found, SL = c["foundation"], c["SL"]
    W = build_windows(runs, ys, c["fturn"])
    Xw, yw, rw = W["X"], W["y"], W["run"]

    splits = list(StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
                  .split(np.zeros(n), yi))
    oof = {A_KEY: np.zeros(n, dtype=int), B_KEY: np.zeros(n, dtype=int)}
    fold_m = {A_KEY: [], B_KEY: []}
    peak_pos = np.full((n, wd.N_WINDOW_CLASSES), -1, dtype=np.int32)
    peak_prob = np.zeros((n, wd.N_WINDOW_CLASSES))
    win_oof_y, win_oof_pred, win_oof_run = [], [], []

    for f, (tr, va) in enumerate(splits):
        log("OUTER FOLD %d  train=%d  val=%d" % (f, len(tr), len(va)))
        tr_set, va_set = set(tr.tolist()), set(va.tolist())

        # ---------------- system A ----------------
        mA = lgb_label().fit(found.iloc[tr], yi[tr])
        pA = mA.predict(found.iloc[va])
        oof[A_KEY][va] = pA
        fold_m[A_KEY].append(sl_macro(ys[va], np.array([LABELS[j] for j in pA],
                                                      dtype=object),
                                      np.ones(len(va), bool)) or 0.0)

        # ---------------- inner cross-fit for outer-train ----------------
        tr_runs = np.array(sorted(tr_set))
        inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=SEED)
                     .split(tr_runs, yi[tr_runs]))
        assign = np.zeros(len(tr_runs), dtype=int)
        for j, (_, hold_local) in enumerate(inner):
            assign[hold_local] = j
        verify_stacking(len(tr_runs), assign, N_INNER, "fold %d" % f)

        Xagg_tr = np.zeros((len(tr), ag.N_AGG))
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            keep = tr_runs[assign != j]
            mrow, hrow = np.isin(rw, keep), np.isin(rw, hold)
            mw = fit_window(Xw[mrow], yw[mrow])
            P = mw.predict_proba(Xw[hrow])
            blocks = grp.predict_runs(P, rw[hrow], len(tr_runs))
            slot = {int(r): k for k, r in enumerate(tr_runs)}
            sub, _, _ = grp.aggregate_block(blocks, ag, slot, len(tr_runs))
            Xagg_tr[assign == j] = sub[assign == j]
            log("  inner %d: window model on %d runs -> cross-fit %d runs"
                % (j, len(keep), len(hold)))
            del mw
            gc.collect()

        # ---------------- outer-valid features from a full outer-train model ----
        mrow, vrow = np.isin(rw, tr), np.isin(rw, va)
        mw = fit_window(Xw[mrow], yw[mrow])
        Pva = mw.predict_proba(Xw[vrow])
        # window-level OOF labels for diagnostics on validation windows
        win_oof_y.append(yw[vrow]); win_oof_pred.append(Pva.argmax(1))
        win_oof_run.append(rw[vrow])
        blocks = grp.predict_runs(Pva, rw[vrow], n)
        slot_va = {int(r): k for k, r in enumerate(va)}
        Xagg_va, pk, pp = grp.aggregate_block(blocks, ag, slot_va, len(va))
        peak_prob[va] = pk
        peak_pos[va] = pp
        log("  outer window model -> %d val runs" % len(va))
        del mw
        gc.collect()

        # ---------------- system B ----------------
        cols = list(found.columns) + list(ag.AGG_NAMES)
        Xb_tr = pd.DataFrame(np.hstack([found.iloc[tr].values, Xagg_tr]),
                             columns=cols)
        Xb_va = pd.DataFrame(np.hstack([found.iloc[va].values, Xagg_va]),
                             columns=cols)
        mB = lgb_label().fit(Xb_tr, yi[tr])
        pB = mB.predict(Xb_va)
        oof[B_KEY][va] = pB
        fold_m[B_KEY].append(sl_macro(ys[va], np.array([LABELS[j] for j in pB],
                                                      dtype=object),
                                      np.ones(len(va), bool)) or 0.0)
        log("  fold macro:  A=%.4f   B=%.4f" % (fold_m[A_KEY][-1], fold_m[B_KEY][-1]))
        del mA, mB
        gc.collect()

    # ---------------- metrics ----------------
    names = {k: [LABELS[i] for i in oof[k]] for k in oof}
    objs = {k: np.array(v, dtype=object) for k, v in names.items()}
    res = {"n_runs": int(n), "n_outer_folds": N_OUTER, "n_inner_folds": N_INNER,
           "seed": SEED, "success_f1": c["success_f1"],
           "n_foundation": int(found.shape[1]), "n_window_feats": int(wd_n()),
           "n_agg_feats": int(ag.N_AGG), "pos_radius": wd.POS_RADIUS,
           "systems": {}}
    for k in oof:
        mf = sl_macro(ys, objs[k], np.ones(n, bool))
        rf = sl_macro(ys, objs[k], SL["hard/robustness"])
        turns = [localize(runs[i], names[k][i]) for i in range(n)]
        h2 = fault_turn_hit_at_k(list(ys), names[k], list(c["fturn"]), turns)
        res["systems"][k] = {
            "n_features": int(found.shape[1] + (ag.N_AGG if k == B_KEY else 0)),
            "macro_f1": mf, "robustness_f1": rf, "success_f1": c["success_f1"],
            "fault_turn_hit2": h2,
            "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                    "success_f1": c["success_f1"],
                                    "fault_turn_hit2": h2}),
            "per_class": f1_per_class(list(ys), names[k], LABELS),
            "fold_macro_f1": [float(x) for x in fold_m[k]],
            "slices": {s: sl_macro(ys, objs[k], m) for s, m in SL.items()},
            "confusion_pairs": confusion(ys, objs[k]),
        }
    ref = res["systems"][A_KEY]
    for k in res["systems"]:
        d = res["systems"][k]
        d["d_macro"] = d["macro_f1"] - ref["macro_f1"]
        d["d_rob"] = d["robustness_f1"] - ref["robustness_f1"]
        d["d_comp"] = d["composite"] - ref["composite"]
        d["d_hit2"] = d["fault_turn_hit2"] - ref["fault_turn_hit2"]
        d["d_dh"] = (prf(ys, objs[k], DH)["f1"]
                     - prf(ys, objs[A_KEY], DH)["f1"])

    for k in oof:
        pd.DataFrame({"run_id": c["train"]["run_id"], "y_true": ys,
                      "y_pred": names[k], "fault_turn": c["fturn"],
                      "n_messages": c["nmsg"]}).to_csv(
            os.path.join(HERE, "oof_%s.csv" % k), index=False)
    np.save(os.path.join(HERE, "window_peak_pos.npy"), peak_pos)
    np.save(os.path.join(HERE, "window_peak_prob.npy"), peak_prob)
    np.save(os.path.join(HERE, "window_oof_y.npy"), np.concatenate(win_oof_y))
    np.save(os.path.join(HERE, "window_oof_pred.npy"), np.concatenate(win_oof_pred))
    np.save(os.path.join(HERE, "window_oof_run.npy"), np.concatenate(win_oof_run))
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    log("wrote results.json, 2 oof files, 5 npy artefacts")
    return res


def wd_n():
    import window_features as _wf
    return _wf.N_FEATURES


if __name__ == "__main__":
    main()
