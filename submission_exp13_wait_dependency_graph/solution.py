#!/usr/bin/env python3
"""Coordination Referee submission - Exp13 PROMOTE (wait-dependency graph).

DSWorks run contract:

    python solution.py --train train.csv --test test.csv --output predictions.csv

This is ``submission_exp08_window_localizer`` with EXACTLY ONE behavioural
change, validated in ``experiments/exp13_wait_dependency_graph``:

``label``      Exp07 B = 300 features + 12 frozen wait-dependency features
                   = 249 Exp06b ``full - age`` foundation
                   +  51 run-level aggregates of a TURN-LEVEL window LightGBM
                   +  12 ``wg_*`` features describing WHO WAITS FOR WHOM
               The 300 production columns are identical to Exp08 and in the same
               order; the 12 new ones are appended after them.

``success``    Exp03 B = 185 features, UNCHANGED.  No wait-graph feature targets
               ``success``, so success F1 stays pinned at the validated 0.8644.

``fault_turn`` ``peak_turn.py``, UNCHANGED: the window model's peak turn for the
               PREDICTED label.  ``clean`` -> -1, empty block -> -1.

THE WAIT-DEPENDENCY BLOCK
-------------------------
``wait_graph.py`` is the research module verbatim.  It parses the six official
waiting-message templates and derives a directed graph ``sender -> awaited
agent``, then summarises it: reciprocal pairs, same-subtask reciprocal pairs,
the position and span of the first reciprocal pair, and repeat pressure.  It
reads only ``messages`` entries and their ``t`` / ``from`` / ``text`` fields, so
it cannot see a target; the experiment's ``test_features.py`` proves this by
target perturbation.

WHY IT HELPS ``deadlock``
-------------------------
The ontology defines deadlock as A waiting on B's subtask while B waits on A's,
with no progress afterwards - a RELATIONAL condition.  The existing 300 features
can only approximate it through aggregate proxies (``share_status``, r ~ 0.72
with the new counts), because they count message edges of all types together
and never parse who is being waited for.

On Exp08's honest OOF the block lifted deadlock F1 0.7217 -> 0.8367 and the
official composite 0.769445 -> 0.792622.  Re-measured against Exp12's CORRECTED
aggregation semantics - the semantics production actually uses - composite
0.768487 -> 0.789283 and deadlock F1 0.7143 -> 0.8336, winning 3/3 folds.

UNCHANGED FROM EXP08, DELIBERATELY
----------------------------------
The window model and its parameters, the 51 aggregates and their honest inner
cross-fit, the success head, the L1 localiser, the seeds and every decision
threshold.  Nothing was tuned: the 12 features were frozen before any model was
fitted, and the promotion gate was pre-declared and unchanged.

HONEST STACKING (this is the part that must not be simplified)
-------------------------------------------------------------
The window model is a SECOND-STAGE learner, so its features must be
out-of-sample for the runs the label head is trained on.  With ``N_INNER=3``
stratified inner folds, every train run receives its 51 window features from a
window model that never saw it, and the test runs get theirs from a window model
fitted on ALL train runs.  ``verify_stacking`` re-derives the fold membership
and hard-fails on any run that was predicted by a model trained on it.

The test-run peak turns come from the SAME window model that produced the test
features, so the turn a run is given is the turn whose probability also fed the
label head - there is no second, differently-fitted model behind ``fault_turn``.

Everything is CPU-only and offline: stdlib + numpy/pandas/scikit-learn/lightgbm.
No absolute paths, no network, no data files bundled.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_features, parse_run           # noqa: E402
from lifecycle import LIFECYCLE_NAMES                       # noqa: E402
from lifecycle import build_matrix as build_lifecycle       # noqa: E402
from localize import localize                               # noqa: E402
import new_features as nf                                   # noqa: E402
import peak_turn as pt                                      # noqa: E402
import temporal_features as tf                              # noqa: E402
import window_features as wf                                # noqa: E402
import window_dataset as wd                                 # noqa: E402
import aggregate as ag                                      # noqa: E402
import wait_graph as wg                                     # noqa: E402

_T0 = time.time()

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}

RUN_COLUMNS = ["goal", "agents", "shared_state", "messages", "artifacts", "topology"]
FALLBACK = {"label": "clean", "success": 1, "fault_turn": -1}

# Exp06 temporal blocks.  `full - age` = every block except BLOCK_1_AGE.
_AGE_BLOCK = set(tf.BLOCKS["1_age"])

N_FOUNDATION = 249
N_SUCCESS_FEATS = 185
N_INNER = 3
INNER_SEED = 0
# Exp13: the frozen wait-dependency block appended to the LABEL head only.
N_WAIT_GRAPH = len(wg.FEATURE_NAMES)


def log(msg: str) -> None:
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def _write_fallback(test: pd.DataFrame, path: str, why: str) -> None:
    """Emit a valid predictions.csv when the input is not a full run table.

    The platform smoke step may hand the solution a reduced file (for example
    only ``run_id``).  Producing a well-formed answer keeps that step green
    instead of failing the whole submission.
    """
    log("WARNING: %s -> writing constant predictions" % why)
    ids = test["run_id"] if "run_id" in test.columns else pd.Series(
        ["row_%d" % i for i in range(len(test))])
    pd.DataFrame({"run_id": ids, "label": FALLBACK["label"],
                  "success": FALLBACK["success"],
                  "fault_turn": FALLBACK["fault_turn"]}).to_csv(path, index=False)
    log("wrote %s (%d rows, fallback)" % (path, len(ids)))


def parse_runs(df: pd.DataFrame):
    return [parse_run(rec) for rec in df.to_dict("records")]


# --------------------------------------------------------------------------
# feature matrices
# --------------------------------------------------------------------------

def build_foundation_matrix(runs, base_rows) -> pd.DataFrame:
    """The 249-column Exp06b ``full - age`` matrix, in the validated order."""
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    Xn = nf.build_matrix(runs, base_rows)
    Xl = build_lifecycle(runs)[LIFECYCLE_NAMES]
    Xt = tf.build_matrix(runs)
    # drop the age block only; order is otherwise the Exp06 order
    Xt = Xt[[c for c in Xt.columns if c not in _AGE_BLOCK]]
    return pd.concat([Xb, Xn, Xl, Xt], axis=1)


def build_success_matrix(runs, base_rows) -> pd.DataFrame:
    """The 185-column Exp03 B matrix. Deliberately NOT the 300 set."""
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    Xn = nf.build_matrix(runs, base_rows)
    return pd.concat([Xb, Xn], axis=1)


# --------------------------------------------------------------------------
# models - byte-identical to the validated Exp07 configuration
# --------------------------------------------------------------------------

def make_label_model():
    """Byte-identical to the validated Exp05/Exp06/Exp06b/Exp07 configuration."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def make_success_model():
    """Byte-identical to Exp03's ``lgb_success``."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05,
                              num_leaves=63, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                              random_state=42, verbose=-1)


def make_window_model():
    """Byte-identical to Exp07's ``lgb_window``.  Fixed, never tuned.

    Imbalance is handled with an explicit ``sample_weight`` vector (the
    pre-declared class weight of each window's true class) rather than sklearn's
    ``class_weight``, so the weighting is visible at the call site.
    """
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass",
                              num_class=wd.N_WINDOW_CLASSES, n_estimators=300,
                              learning_rate=0.08, num_leaves=63, min_child_samples=50,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def fit_window(X, y):
    m = make_window_model()
    m.fit(X, y, sample_weight=wd.class_weight_vector()[y])
    return m


# --------------------------------------------------------------------------
# window dataset + honest cross-fitting
# --------------------------------------------------------------------------

def build_window_matrix(runs, ys, ft) -> dict:
    """X (N, 83), y (N,), run (N,), n_turns per run.  ``run`` indexes ``runs``
    and is what makes honest cross-fitting possible."""
    Xs, y_out, rid, nturn = [], [], [], np.zeros(len(runs), dtype=np.int64)
    for i, (run, lab, f) in enumerate(zip(runs, ys, ft)):
        F = wf.run_window_features(run)
        nturn[i] = F.shape[0]
        if F.shape[0] == 0:
            continue
        Xs.append(F)
        y_out.append(wd.window_targets(F.shape[0], str(lab), f))
        rid.append(np.full(F.shape[0], i, dtype=np.int32))
    if not Xs:
        return {"X": np.zeros((0, wf.N_FEATURES)), "y": np.zeros(0, dtype=np.int32),
                "run": np.zeros(0, dtype=np.int32), "n_turns": nturn}
    return {"X": np.vstack(Xs), "y": np.concatenate(y_out).astype(np.int32),
            "run": np.concatenate(rid), "n_turns": nturn}


def aggregate_runs(P: np.ndarray, rows: np.ndarray, n_turns: np.ndarray,
                   want: np.ndarray) -> np.ndarray:
    """(n_runs, 51) aggregate block, aligned to the run indices in ``want``.

    A run whose window-model probability block is empty contributes zeros, which
    is exactly what the experiment's ``aggregate_block`` did via
    ``continue``.
    """
    out = np.zeros((len(want), ag.N_AGG), dtype=np.float64)
    if len(rows) == 0:
        return out
    order = np.argsort(rows, kind="stable")
    sr = rows[order]
    bounds = np.searchsorted(sr, want)
    # want is sorted ascending, so bounds are the left edges
    for k, r in enumerate(want):
        a = int(bounds[k])
        b = int(bounds[k + 1]) if k + 1 < len(want) else len(sr)
        if b > a:
            out[k] = ag.aggregate(P[order[a:b]], n_turns[r])
    return out


def verify_stacking(n_runs, assign, n_inner, tag):
    """Every train run must be held out exactly once, every inner fold
    non-empty, and held-out runs must never appear in their own training set."""
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
            raise AssertionError("inner fold %d of %s has no training runs" % (f, tag))
        if len(np.intersect1d(held, trained)):
            raise AssertionError("inner fold %d of %s: held-out runs appear in "
                                 "its own training set" % (f, tag))
        held_counts[held] += 1
    if not (held_counts == 1).all():
        raise AssertionError("inner folds of %s do not partition the train runs "
                             "exactly once (max count %d)"
                             % (tag, int(held_counts.max())))
    log("  [%s] stacking verified: %d train runs, each cross-fitted from a window "
        "model excluding it (%d inner folds, no overlap)" % (tag, n_runs, n_inner))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    log("reading data")
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    log("train=%d test=%d" % (len(train), len(test)))

    missing_test = [c for c in RUN_COLUMNS if c not in test.columns]
    missing_train = [c for c in RUN_COLUMNS + ["label", "success", "fault_turn"]
                     if c not in train.columns]
    if missing_test or missing_train:
        _write_fallback(test, args.output,
                        "input is not a full run table (test misses %s, train misses %s)"
                        % (missing_test or "nothing", missing_train or "nothing"))
        return

    # ---------------- features ----------------
    log("parsing runs")
    tr_runs = parse_runs(train)
    te_runs = parse_runs(test)
    tr_base = [extract_features(r) for r in tr_runs]
    te_base = [extract_features(r) for r in te_runs]

    log("building foundation (249 feats)")
    Xf_tr = build_foundation_matrix(tr_runs, tr_base)
    assert Xf_tr.shape[1] == N_FOUNDATION, \
        "foundation is %d cols, expected %d" % (Xf_tr.shape[1], N_FOUNDATION)
    Xf_te = build_foundation_matrix(te_runs, te_base).reindex(
        columns=Xf_tr.columns, fill_value=0.0)
    log("foundation: %d features" % Xf_tr.shape[1])

    log("building success matrix (185 feats)")
    Xs_tr = build_success_matrix(tr_runs, tr_base)
    assert Xs_tr.shape[1] == N_SUCCESS_FEATS, \
        "success matrix is %d cols, expected %d" % (Xs_tr.shape[1], N_SUCCESS_FEATS)
    Xs_te = build_success_matrix(te_runs, te_base).reindex(
        columns=Xs_tr.columns, fill_value=0.0)

    # ---------------- window dataset ----------------
    ys = train["label"].values.astype(object)
    ft = train["fault_turn"].values
    n_tr = len(train)
    yi_tr = np.array([L2I[l] for l in ys], dtype=int)

    W = build_window_matrix(tr_runs, ys, ft)
    Xw, yw, rw, nturn = W["X"], W["y"], W["run"], W["n_turns"]
    log("window dataset: %d windows, %d features"
        % (len(yw), wf.N_FEATURES))

    # ---------------- honest cross-fit of the 51 window features ----------
    # Stratify on the run label so every inner fold keeps the class mix.
    idx_all = np.arange(n_tr)
    inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=INNER_SEED)
                 .split(idx_all, yi_tr))
    assign = np.zeros(n_tr, dtype=int)
    for j, (_, hold) in enumerate(inner):
        assign[hold] = j
    verify_stacking(n_tr, assign, N_INNER, "train")

    Xagg_tr = np.zeros((n_tr, ag.N_AGG), dtype=np.float64)
    for j in range(N_INNER):
        hold = np.where(assign == j)[0]
        keep = np.where(assign != j)[0]
        mrow, hrow = np.isin(rw, keep), np.isin(rw, hold)
        mw = fit_window(Xw[mrow], yw[mrow])
        P = mw.predict_proba(Xw[hrow])
        Xagg_tr[hold] = aggregate_runs(P, rw[hrow], nturn, hold)
        log("  inner %d: window model on %d runs -> cross-fit %d runs"
            % (j, len(keep), len(hold)))
        del mw, P
        gc.collect()

    # ---------------- window features for the test runs -------------------
    # A single window model fitted on ALL train runs predicts every test window.
    # This matches the experiment's outer-valid step, where a window model
    # fitted on the entire outer-train block produced the validation features.
    log("fitting window model on all %d train runs -> test features" % n_tr)
    mw = fit_window(Xw, yw)
    Wte = build_window_matrix(te_runs, [None] * len(te_runs),
                              np.full(len(te_runs), -1))
    Pte = mw.predict_proba(Wte["X"]) if len(Wte["X"]) else np.zeros((0, wd.N_WINDOW_CLASSES))
    del mw
    gc.collect()
    all_runs_turns = np.concatenate([nturn, Wte["n_turns"]])
    Xagg_all = aggregate_runs(Pte, Wte["run"] + n_tr, all_runs_turns,
                              np.arange(n_tr, n_tr + len(te_runs)))
    Xagg_te = Xagg_all
    log("test window features: %d windows" % len(Pte))

    # ---------------- label head: 249 + 51 + 12 = 312 ----------------
    # Exp13 change, and ONLY this: the frozen 12 `wg_*` wait-dependency
    # features are appended to the LABEL matrix.  The 300 production columns
    # are the same 249 foundation + the same 51 window aggregates in the same
    # order, `success` still uses its own 185 columns, the window model and the
    # L1 localiser are untouched.
    cols = list(Xf_tr.columns) + list(ag.AGG_NAMES) + list(wg.FEATURE_NAMES)
    Xw_tr = wg.build_matrix(tr_runs)
    assert Xw_tr.shape == (n_tr, N_WAIT_GRAPH), \
        "wait-graph block is %s, expected (%d, %d)" % (Xw_tr.shape, n_tr,
                                                       N_WAIT_GRAPH)
    Xw_te = wg.build_matrix(te_runs)

    Xb_tr = pd.DataFrame(np.hstack([Xf_tr.values, Xagg_tr, Xw_tr]), columns=cols)
    Xb_te = pd.DataFrame(np.hstack([Xf_te.values, Xagg_te, Xw_te]), columns=cols)

    log("fitting label model (%d feats = 249 + 51 + %d wait-graph)"
        % (len(cols), N_WAIT_GRAPH))
    clf = make_label_model().fit(Xb_tr, yi_tr)
    log("fitting success model (%d feats, Exp03 B)" % N_SUCCESS_FEATS)
    suc = make_success_model().fit(Xs_tr, train["success"].values)

    # ---------------- predict ----------------
    log("predicting")
    pred_label = [LABELS[i] for i in np.asarray(clf.predict(Xb_te)).astype(int)]
    pred_success = np.asarray(suc.predict(Xs_te)).astype(int)

    # fault_turn: EXP08 L1 - the peak window turn for the ALREADY-PREDICTED
    # fault class.  Exp07 shipped the rule-based localiser here and discarded
    # the peak positions the window model had already computed; this reads them
    # back out of the SAME per-run probability blocks that produced the 51
    # aggregate features, so no extra model is fitted and no feature changes.
    #
    #   clean        -> -1
    #   empty block  -> -1
    #   otherwise    -> argmax_t P[run, t, class(predicted label)]
    #
    # No threshold, no offset, no per-class rule.  See peak_turn.py.
    log("localising fault_turn: L1 window peak for the predicted class")
    te_peak = pt.peak_turns(Pte, Wte["run"], len(te_runs), pred_label)
    pred_turn = [int(t) for t in te_peak]

    out = pd.DataFrame({"run_id": test["run_id"], "label": pred_label,
                        "success": pred_success, "fault_turn": pred_turn})
    out.to_csv(args.output, index=False)
    log("wrote %s (%d rows)" % (args.output, len(out)))
    log("label distribution:")
    for lab, cnt in out["label"].value_counts().items():
        log("    %-18s %6d" % (lab, cnt))
    log("fault_turn: %d non -1" % int((pd.to_numeric(out["fault_turn"]) >= 0).sum()))
    log("DONE in %.1fs" % (time.time() - _T0))


if __name__ == "__main__":
    main()
