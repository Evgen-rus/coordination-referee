#!/usr/bin/env python3
"""Experiment 01: can a learned per-turn ranker replace the rule-based localiser?

Protocol
--------
Same 3 StratifiedKFold splits as ``scripts/local_cv.py`` (shuffle, seed 0) so
the comparison is apples-to-apples.

For every fold, with ``tr`` = train part and ``va`` = validation part:

  1. OOF-predicted ``label`` for the validation part comes from a LightGBM
     multiclass classifier fitted on ``tr`` only, with the baseline's 122
     run-level features.  This is the realistic case: at test time the true
     label is unavailable and only the predicted one can drive the localiser.
  2. The turn ranker is fitted on ``tr`` only.  Training rows come from faulty
     runs whose *true* label is in ``tr``; the class-conditional decision rule
     uses the true training label, so no predicted (noisy) label is needed at
     fit time.
  3. Ranker features use the predicted label of the *run the turn belongs to*
     as a one-hot.  Features are otherwise label-free.  Validation labels and
     validation ``fault_turn`` are never read.
  4. Ranker is applied to validation turns, argmax per run, ``-1`` for ``clean``.

Leakage guards
--------------
  * run-level features are the official ``features.extract_features`` (unchanged);
  * turn features are causal / windowed and never see the run's label or target;
  * the ranker's decision rule is class-conditional, so a misclassified run
    still gets a class-appropriate turn;
  * folds are saved to ``folds.json`` so the split can be verified.

Oracle variant (clearly separated, diagnostic only) feeds the *true* label of
validation runs into the same ranker; it is an upper bound, not a result.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from features import extract_features, parse_run          # noqa: E402
from localize import localize                             # noqa: E402
from metrics import binary_f1, composite, f1_per_class, fault_turn_hit_at_k, macro_f1  # noqa: E402
import turn_features as turnfeat                          # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
FAULTY = LABELS[1:]
L2I = {l: i for i, l in enumerate(LABELS)}

_RANKER_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                      min_child_samples=50, subsample=0.9, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1,
                      random_state=42, verbose=-1)

_T0 = time.time()
RESULTS = {"folds": [], "per_class": {}, "config": {}}


def log(m):
    print("[%7.1fs] %s" % (time.time() - _T0, m), flush=True)


def make_label_clf():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def make_suc_clf():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05,
                              num_leaves=63, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                              random_state=42, verbose=-1)


def make_ranker():
    """Per-class binary 'is this the fault turn?' model (7 independent LGBMs)."""
    import lightgbm as lgb
    return {c: lgb.LGBMClassifier(objective="binary", **_RANKER_PARAMS) for c in FAULTY}


def label_onehot(labels, used_classes):
    out = np.zeros((len(labels), len(used_classes)), dtype=np.float32)
    for i, l in enumerate(labels):
        if l in used_classes:
            out[i, used_classes.index(l)] = 1.0
    return out


def score_by_class(rankers, X, run_labels, used_classes, n_rows):
    """P(class is fault_turn) for every turn, using the model of its run's class.

    Turn-level labels are constant within a run, so the per-run assignment
    vector selects whole blocks; scoring is batched, not per turn.
    """
    out = np.full(n_rows, -1.0, dtype=np.float32)
    for c in used_classes:
        sel = np.where(run_labels == c)[0]
        if not len(sel):
            continue
        out[sel] = rankers[c].predict_proba(X[sel])[:, 1]
    return out


def main():
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    y = train["label"].values
    s = train["success"].values
    ft_true = train["fault_turn"].values

    log("parsing %d runs" % len(train))
    runs = [parse_run(r) for r in train.to_dict("records")]

    log("run-level features (official baseline, unchanged)")
    X = pd.DataFrame([extract_features(r) for r in runs]).fillna(0.0)

    log("per-turn features")
    t0 = time.time()
    turn_X, T_names = [], []
    for r in runs:
        M, _ = turnfeat.build_turn_matrix(r)
        turn_X.append(M)
    turn_feat_time = time.time() - t0
    lens = np.array([m.shape[0] for m in turn_X])
    log("turn features: %d turns, %d dims, %.1fs (%.1f turns/run)"
        % (int(lens.sum()), turn_X[0].shape[1], turn_feat_time, lens.mean()))

    # Flatten into one matrix.  turn_start[i] is the first row of run i, and
    # turn_start is reused for every split, so the SAME indexing helper is
    # used for train and validation rows (an earlier version indexed the
    # global matrix with within-split offsets, which silently mismatched
    # features to labels).
    run_of_turn = np.repeat(np.arange(len(runs)), lens)
    T = np.vstack(turn_X) if lens.sum() else np.zeros((0, len(turnfeat.feature_names())))
    T_names_full = turnfeat.feature_names()
    assert T.shape[1] == len(T_names_full)
    turn_start = np.zeros(len(runs) + 1, dtype=np.int64)
    np.cumsum(lens, out=turn_start[1:])

    def rows_for(run_ids):
        """Global turn-matrix row indices belonging to the given run ids.

        rows_for([r0, r1, ...]) == [turn_start[r0] ... turn_start[r0]+len(r0)-1,
                                      turn_start[r1] ..., ...]
        """
        run_ids = np.asarray(run_ids, dtype=np.int64)
        if not len(run_ids):
            return np.zeros(0, dtype=np.int64)
        counts = lens[run_ids]
        starts = turn_start[run_ids]
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        # expand each run into its `count` consecutive turn rows
        starts_rep = np.repeat(starts, counts)
        # offset of each turn inside its own run: 0,1,...,count-1 per run
        offs = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64), counts)
        return starts_rep + offs

    fold_id = np.empty(len(train), dtype=int)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    for k, (tr, va) in enumerate(splits):
        fold_id[va] = k

    oof_lab = np.empty(len(train), dtype=object)
    oof_suc = np.zeros(len(train), dtype=int)
    turns_rule = np.full(len(train), -1, dtype=int)
    turns_new = np.full(len(train), -1, dtype=int)
    turns_oracle = np.full(len(train), -1, dtype=int)

    for k, (tr, va) in enumerate(splits):
        log("=" * 70)
        log("FOLD %d: train=%d val=%d" % (k, len(tr), len(va)))

        # ---- (1) OOF label + success from train part only ----
        clf = make_label_clf().fit(X.iloc[tr], y[tr])
        pred_va = clf.predict(X.iloc[va])
        oof_lab[va] = pred_va
        oof_suc[va] = make_suc_clf().fit(X.iloc[tr], s[tr]).predict(X.iloc[va])

        # ---- (2) rule-based baseline localiser (uses predicted label) ----
        for i, p in zip(va, pred_va):
            turns_rule[i] = localize(runs[i], p)

        # ---- (3) build ranker training data from the TRAIN part only ----
        used_classes = sorted(set(y[tr]) & set(FAULTY))
        sel_runs = np.array([i for i in tr if y[i] != "clean" and ft_true[i] >= 0])
        sel_lens = lens[sel_runs]
        sel_idx = np.repeat(np.arange(len(sel_runs)), sel_lens)
        sel_run_id = np.repeat(sel_runs, sel_lens)
        sel_turn_id = np.concatenate([np.arange(l) for l in sel_lens])
        target = np.zeros(int(lens[sel_runs].sum()), dtype=int)
        tgt_turn = ft_true[sel_runs]
        mask = sel_turn_id == np.repeat(tgt_turn, sel_lens)
        target[mask] = 1

        # --- self-check: the assembled rows must be the right turns ---
        chk = rows_for(sel_runs)
        assert len(chk) == len(target) == len(sel_idx)
        # every positive row must belong to its own run and sit at fault_turn
        for j in np.where(target == 1)[0][:50]:
            gi = sel_run_id[j]
            assert rows_for(np.array([gi]))[0] <= chk[j] <= rows_for(np.array([gi]))[-1]
            assert (chk[j] - turn_start[gi]) == int(ft_true[gi])
            assert run_of_turn[chk[j]] == gi
        # and the negative rows must sit at a DIFFERENT turn than the target
        neg = np.where(target == 0)[0][:50]
        for j in neg:
            gi = sel_run_id[j]
            assert (chk[j] - turn_start[gi]) != int(ft_true[gi])

        # ranker rows: turn features + one-hot of the run's class.
        # For TRAIN rows we may use the true label (it is inside the fold).
        base = T[rows_for(sel_runs)]
        oh = label_onehot(list(y[sel_runs][sel_idx]), used_classes)
        base = np.hstack([base, oh])
        cls_arr = np.array([y[i] for i in sel_run_id])

        rankers = make_ranker()
        for c in used_classes:
            m = cls_arr == c
            rankers[c].fit(base[m], target[m])
        log("  ranker trained on %d turns / %d runs, %d positives (%.3f%%)"
            % (base.shape[0], len(sel_runs), int(target.sum()),
               100.0 * target.mean()))

        # ---- (4) apply to validation part ----
        val_lens = lens[va]
        val_idx = np.repeat(np.arange(len(va)), val_lens)
        val_run_id = np.repeat(va, val_lens)
        vbase = T[rows_for(va)]
        pred_lab_va = oof_lab[va]
        # per-TURN label one-hot (labels repeat once per turn, not once per run)
        oh_pred = label_onehot(list(pred_lab_va[val_idx]), used_classes)
        vbase = np.hstack([vbase, oh_pred])

        # apply each class model to its own runs, argmax over turns
        offs = np.cumsum(np.concatenate([[0], val_lens]))
        turn_scores = score_by_class(rankers, vbase, pred_lab_va, used_classes, int(val_lens.sum()))

        # decode: argmax per run
        for j, gi in enumerate(va):
            if pred_lab_va[j] == "clean":
                turns_new[gi] = -1
                continue
            lo, hi = offs[j], offs[j + 1]
            if hi <= lo:
                turns_new[gi] = -1
                continue
            turns_new[gi] = int(np.argmax(turn_scores[lo:hi]))

        # ---- oracle variant (diagnostic upper bound, NOT a result) ----
        true_lab_va = y[va]
        ohs = label_onehot(list(true_lab_va[val_idx]), used_classes)
        vbase_or = np.hstack([T[rows_for(va)], ohs])
        turn_scores_or = score_by_class(rankers, vbase_or, true_lab_va, used_classes,
                                        int(val_lens.sum()))
        for j, gi in enumerate(va):
            if true_lab_va[j] == "clean":
                turns_oracle[gi] = -1
                continue
            lo, hi = offs[j], offs[j + 1]
            if hi <= lo:
                turns_oracle[gi] = -1
                continue
            turns_oracle[gi] = int(np.argmax(turn_scores_or[lo:hi]))

        h_rule = fault_turn_hit_at_k(list(y[va]), list(oof_lab[va]),
                                     list(ft_true[va]), list(turns_rule[va]))
        h_new = fault_turn_hit_at_k(list(y[va]), list(oof_lab[va]),
                                    list(ft_true[va]), list(turns_new[va]))
        h_or = fault_turn_hit_at_k(list(y[va]), list(oof_lab[va]),
                                   list(ft_true[va]), list(turns_oracle[va]))
        mf = macro_f1(list(y[va]), list(oof_lab[va]))
        RESULTS["folds"].append({
            "fold": k, "n_val": len(va),
            "macro_f1": mf,
            "hit2_rule": h_rule, "hit2_new": h_new, "hit2_oracle": h_or,
            "delta_hit2": h_new - h_rule,
        })
        log("  fold %d: macro_f1=%.4f  hit@2 rule=%.4f new=%.4f (d=%+.4f) oracle=%.4f"
            % (k, mf, h_rule, h_new, h_new - h_rule, h_or))

    # ---------------- aggregate ----------------
    log("=" * 70)
    def agg(key):
        return float(np.mean([f[key] for f in RESULTS["folds"]]))

    macro = macro_f1(list(y), list(oof_lab))
    sf1 = binary_f1(list(s), list(oof_suc))
    hard = ((X["n_messages"] > X["n_messages"].median())
            & (X["topo_mesh"] + X["topo_blackboard"] > 0)).values
    rob = macro_f1(list(y[hard]), list(oof_lab[hard]))
    h_rule = fault_turn_hit_at_k(list(y), list(oof_lab), list(ft_true), list(turns_rule))
    h_new = fault_turn_hit_at_k(list(y), list(oof_lab), list(ft_true), list(turns_new))
    h_or = fault_turn_hit_at_k(list(y), list(oof_lab), list(ft_true), list(turns_oracle))
    comp_rule = composite({"macro_f1": macro, "robustness_f1": rob,
                           "success_f1": sf1, "fault_turn_hit2": h_rule})
    comp_new = composite({"macro_f1": macro, "robustness_f1": rob,
                          "success_f1": sf1, "fault_turn_hit2": h_new})

    print("\n" + "=" * 70)
    print("OVERALL (OOF, 3 folds)")
    print("=" * 70)
    print("  macro_f1 (unchanged)      = %.4f" % macro)
    print("  robustness_f1 (unchanged) = %.4f" % rob)
    print("  success_f1 (unchanged)    = %.4f" % sf1)
    print("  hit@2 rule-based          = %.4f" % h_rule)
    print("  hit@2 learned ranker      = %.4f" % h_new)
    print("  hit@2 learned + ORACLE lbl = %.4f   (upper bound only)" % h_or)
    print("  delta hit@2               = %+.4f" % (h_new - h_rule))
    print("  composite rule            = %.4f" % comp_rule)
    print("  composite new             = %.4f" % comp_new)
    print("  DELTA composite (=0.10*dh) = %+.5f" % (comp_new - comp_rule))

    RESULTS["overall"] = {
        "macro_f1": macro, "robustness_f1": rob, "success_f1": sf1,
        "hit2_rule": h_rule, "hit2_new": h_new, "hit2_oracle": h_or,
        "delta_hit2": h_new - h_rule,
        "composite_rule": comp_rule, "composite_new": comp_new,
        "delta_composite": comp_new - comp_rule,
        "turn_feature_time_s": turn_feat_time,
        "n_turn_features": int(T.shape[1]),
    }

    # ---------------- per-class detail ----------------
    print("\n" + "=" * 70)
    print("PER-CLASS hit@2 (denominator = all true runs of that class)")
    print("=" * 70)
    print("%-18s %6s %8s %8s %8s %9s %9s"
          % ("class", "n", "rule", "new", "oracle", "d(hit2)", "d(comp)"))
    for c in FAULTY:
        idx = np.where(y == c)[0]
        hr = fault_turn_hit_at_k(list(y[idx]), list(oof_lab[idx]), list(ft_true[idx]),
                                 list(turns_rule[idx]))
        hn = fault_turn_hit_at_k(list(y[idx]), list(oof_lab[idx]), list(ft_true[idx]),
                                 list(turns_new[idx]))
        ho = fault_turn_hit_at_k(list(y[idx]), list(oof_lab[idx]), list(ft_true[idx]),
                                 list(turns_oracle[idx]))
        RESULTS["per_class"][c] = {"n": len(idx), "rule": hr, "new": hn, "oracle": ho}
        # hit@2 is computed over ALL faulty runs; class c holds n_c of them,
        # so its contribution to the global hit@2 delta is (hn-hr) * n_c / n_faulty.
        share = len(idx) / float((y != "clean").sum())
        d_comp = 0.10 * (hn - hr) * share
        print("%-18s %6d %8.4f %8.4f %8.4f %+9.4f %+9.5f"
              % (c, len(idx), hr, hn, ho, hn - hr, d_comp))

    # ---------------- error statistics ----------------
    print("\n" + "=" * 70)
    print("TURN ERROR STATISTICS (faulty runs with correctly predicted class only)")
    print("=" * 70)
    for name, turns in (("rule", turns_rule), ("new", turns_new), ("oracle", turns_oracle)):
        sel = (y != "clean") & (oof_lab == y) & (turns >= 0)
        err = turns[sel].astype(float) - ft_true[sel].astype(float)
        ae = np.abs(err)
        if len(ae) == 0:
            continue
        print("  %-7s n=%4d  medAE=%6.1f  meanAE=%7.2f  exact=%.3f  early=%d  late=%d  pred -1: %d"
              % (name, len(ae), np.median(ae), ae.mean(), (ae == 0).mean(),
                 (err < 0).sum(), (err > 0).sum(),
                 int(((y != "clean") & (oof_lab == y) & (turns < 0)).sum())))
        RESULTS.setdefault("err", {})[name] = {
            "n": int(len(ae)), "medAE": float(np.median(ae)), "meanAE": float(ae.mean()),
            "exact": float((ae == 0).mean()), "early": int((err < 0).sum()),
            "late": int((err > 0).sum())}

    RESULTS["config"] = {
        "folds": 3, "splitter": "StratifiedKFold(shuffle=True, random_state=0)",
        "n_turn_features": int(T.shape[1]),
        "n_turn_features_plus_label_onehot": int(T.shape[1] + 6),
        "ranker": "LightGBM binary, per fault class, %s" % _RANKER_PARAMS,
        "decision_rule": "argmax over turns within run, model selected by OOF-predicted label",
        "turn_feature_time_s": turn_feat_time,
        "total_runtime_s": time.time() - _T0,
    }
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    with open(os.path.join(HERE, "folds.json"), "w", encoding="utf-8") as f:
        json.dump([list(map(int, tr)) for tr, _ in splits], f)
    log("wrote results.json / folds.json; total %.1fs" % (time.time() - _T0))


if __name__ == "__main__":
    main()
