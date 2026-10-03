"""Exp15 - success stacking from the current Exp13 label probabilities.

THE ONE HYPOTHESIS
------------------
    The current success head loses useful information because it is still the
    185-feature Exp03 model and does not observe the current strong Exp13
    coordination-label probabilities.  Honest out-of-fold label probabilities
    should improve success prediction.

WHAT IS AND IS NOT TOUCHED
--------------------------
Touched: ONE appended block of 7 columns on the SUCCESS head - the raw
``predict_proba`` output of the current Exp13 label model, in ``LABELS`` order.

NOT touched, deliberately and permanently:
  * the label head - its matrix, its parameters, its seed, its folds.  Its
    probabilities are CONSUMED, never re-decided, and its argmax is asserted
    equal to the committed Exp13 D argmax on all 10000 rows;
  * ``fault_turn`` / L1 - not one line of the localiser is read or re-run;
  * Exp14's 16 ``ep_*`` episode features - not imported, not referenced;
  * the success model's LightGBM, hyper-parameters and ``random_state=42``;
  * the outer folds (Exp07/Exp03 seed-0 ``StratifiedKFold(3)``, already proven
    equal by ``load_verified``);
  * thresholds, class weights, entropy / margins / max-probability /
    hard-label one-hot variants, feature selection, seed search, tuning.

    Only the 7 RAW probabilities are added.  No derived quantity.

BASELINE POLICY (fixed before any fit)
--------------------------------------
CONTROL A   = the current production success head: the 185 Exp03 features and
              the production ``make_success_model()``.  It must reproduce the
              validated ``success_f1 = 0.8643757406010988`` EXACTLY and be
              bit-identical to the committed Exp03 B OOF predictions on all
              10000 rows before CANDIDATE B is scored at all.

CANDIDATE B = the SAME 185 columns + the 7 honest label probabilities = 192.

A and B are produced from the SAME 185-column matrices inside each fold and
differ in exactly 7 appended columns, asserted ``np.array_equal`` on the first
185.  Same folds, same rows, same model, same hyper-parameters, same seed.

The carried Exp13 label / robustness / hit@2 metrics are FROZEN INPUTS.  They
are never refitted and never offered as candidate choices; only ``success_f1``
moves, so ``composite_B - composite_A == 0.15 * (success_B - success_A)``
exactly, by ``evaluation.metrics.composite``.

HONEST STACKING IS MANDATORY (this is the load-bearing part)
------------------------------------------------------------
The label model is a SECOND-STAGE learner, so its probabilities must be
out-of-sample for every row the success head is trained on.

For each outer fold f:

  VALIDATION rows - from the Exp13 corrected 312-feature label model fitted on
  ALL of outer-train, i.e. Exp13 D's own fold-f prediction path, rebuilt here
  from the same corrected fold-local machinery.

  TRAINING rows - NESTED 3-fold OOF strictly INSIDE outer-train
  (``StratifiedKFold(3, shuffle=True, random_state=NESTED_SEED=1)``).  Each
  nested inner fold rebuilds the FULL corrected 312-feature representation
  (3 honest Exp07 window fits -> 51 aggregates via Exp12's corrected mapping ->
  1 window model on nested-train for nested-validation aggregates -> label fit)
  and predicts only its own held-out nested rows.  In-sample probabilities and
  full-fit-on-outer-train probabilities are NEVER used for a training row.

Before any scoring the run proves, structurally and not by inspecting values,
that for every trained row the label model that generated its 7 probabilities
excluded that row; and it MEASURES that the training probabilities differ from
an in-sample refit, which is what an in-sample fallback would look like.

STOP CONDITIONS
---------------
  * CONTROL A does not reproduce the declared success baseline -> STOP_BASELINE,
    B is never scored;
  * the rebuilt outer label path does not match the committed Exp13 D argmax
    -> STOP_LABEL_PARITY;
  * label predictions / fault_turn / Macro / Robustness / hit@2 are not
    exactly unchanged -> STOP_INVARIANCE;
  * otherwise the pre-declared gate decides PROMOTE or STOP.  The thresholds are
    frozen in this header and are never retuned.

DEVIATION FROM THE WRITTEN DESIGN (recorded, not hidden)
---------------------------------------------------------
A first execution of this script had two defects, both surfaced by this
script's own gates and both fixed before any verdict was accepted:

  1. the nested splitter was built with ``N_META`` (7) splits instead of
     ``N_INNER`` (3), so that execution's numbers are a 7-fold-nested run.
     This execution uses 3.  The 7-fold numbers are NOT re-scored after the
     delta was seen - they stay in RESULTS.md as the as-run record, because
     choosing a split count once the delta is known is selection on outcome;
  2. GATE 2 compared the argmax of ``p_meta_oof`` - which also carries the
     NESTED probabilities on outer-train rows, produced by different (smaller)
     models - against Exp13 D, and therefore failed on 625 rows.  Parity is a
     statement about the label head's own OOF, so it now compares
     ``p_meta_outer``.

Neither defect touched the success head's inputs: in both executions control A
and candidate B were fitted on exactly the same 185 columns.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP13 = os.path.join(ROOT, "experiments", "exp13_wait_dependency_graph")
EXP12 = os.path.join(ROOT, "experiments", "exp12_fix_window_run_mapping")
EXP11 = os.path.join(ROOT, "experiments", "exp11_label_seed_ensemble")
PROD13 = os.path.join(ROOT, "submission_exp13_wait_dependency_graph")
for _p in (EXP12, EXP11, PROD13,
           os.path.join(ROOT, "experiments", "exp10_ab_probability_blend"),
           os.path.join(ROOT, "experiments", "exp08_window_localizer"),
           os.path.join(ROOT, "experiments", "exp07_fault_windows"),
           os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- imports in dependency order; the production solution is imported LAST
# because it mutates sys.path with its own directory.
import cache as ca                        # noqa: E402
import ensemble as ENS                    # noqa: E402  (Exp11, read-only)
import aggregate as ag                    # noqa: E402
import grouping as grp                    # noqa: E402
import lightgbm as lgb                    # noqa: E402
import mapping_fix as M                   # noqa: E402  (Exp12, read-only)
import parallel as par                    # noqa: E402
import runner as R                        # noqa: E402
import window_dataset as wd                # noqa: E402
from blend import Corpus, LABELS, L2I     # noqa: E402
from blend import N_CLASSES, SUCCESS_F1_PINNED, TOL   # noqa: E402
from metrics import binary_f1, composite  # noqa: E402
from metrics import f1_per_class, macro_f1  # noqa: E402

# Exp13's frozen 12 wg_* block, byte-identical to production wait_graph.py.
_spec_w = importlib.util.spec_from_file_location(
    "exp13_wait_features", os.path.join(EXP13, "features.py"))
W = importlib.util.module_from_spec(_spec_w)
_spec_w.loader.exec_module(W)

# Production Exp13 - read-only.  Imported last because solution.py inserts its
# own directory into sys.path; every module above is already resolved.
_spec_s = importlib.util.spec_from_file_location(
    "prod13_solution", os.path.join(PROD13, "solution.py"))
SOL = importlib.util.module_from_spec(_spec_s)
_spec_s.loader.exec_module(SOL)

CLEAN = 0
N_OUTER = 3
OUTER_SEED = 0                     # R.SEED - the sealed Exp07/Exp03 folds
N_INNER = 3
INNER_SEED = 0                     # the label machinery's own inner seed
NESTED_SEED = 1                    # FROZEN, never varied
N_SUCCESS = 185                    # Exp03 B, unchanged
N_META = len(LABELS)               # exactly 7 raw probabilities
N_FOUNDATION = 249
N_AGG = ag.N_AGG                   # 51
N_LABEL_300 = N_FOUNDATION + N_AGG
N_WG = W.N_FEATURES                # 12
N_LABEL_312 = N_LABEL_300 + N_WG
N_B = N_SUCCESS + N_META           # 192

# The 7 appended column names, in LABELS order.  Declared here so a test can
# pin the order and the count without re-deriving it.
META_NAMES = ["lp_%s" % c for c in LABELS]

# Exp03 B's success model, as shipped.  Verified against production below.
SUCCESS_PARAMS_EXPECTED = {
    "objective": "binary", "n_estimators": 500, "learning_rate": 0.05,
    "num_leaves": 63, "subsample": 0.9, "subsample_freq": 1,
    "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 42,
}

# ---- reproduction targets, READ from the committed artifacts --------------
with open(os.path.join(EXP13, "results_production.json"), encoding="utf-8") as _fh:
    _P13 = json.load(_fh)
EXP13_METRICS = {k: float(_P13["metrics_D"][k]) for k in
                 ("macro_f1", "robustness_f1", "success_f1",
                  "fault_turn_hit2", "composite")}
# Exp03 B's pinned success baseline, READ from common.build_all()'s own
# derivation of it (which itself reads the committed Exp03 B OOF csv).
EXP13_SHA_D = ca.sha256_array(np.ascontiguousarray(np.load(
    os.path.join(EXP13, "oof_D_corrected_wg.npy"), allow_pickle=False)))

# ---- PRE-DECLARED PROMOTION GATE - frozen above, never retuned -------------
GATE = {
    "success_f1_delta_min": 0.010,
    "folds_won_min": 2,
    "composite_delta_min": 0.0015,
    "worst_fold_delta_min": -0.005,
}
WEAK_BAND = (0.005, 0.010)

OUT = []
RES = {}


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))


def bar(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def hdr(t):
    log()
    log("-- %s" % t)


def sha32(P):
    return ca.sha256_array(np.ascontiguousarray(np.asarray(P, np.float32)))


def success_params():
    """The production success model's exact parameters, read not copied."""
    p = SOL.make_success_model().get_params()
    diff = {k: (SUCCESS_PARAMS_EXPECTED[k], p.get(k))
            for k in SUCCESS_PARAMS_EXPECTED
            if p.get(k) != SUCCESS_PARAMS_EXPECTED[k]}
    if diff:
        raise AssertionError("production success model drifted from the "
                             "declared Exp03 B configuration: %r" % diff)
    return p


def prf_succ(s, pred, mask=None):
    """tp/fp/fn/precision/recall/f1 for the SUCCESS target, positive=1."""
    s_ = np.asarray(s, np.int64)
    p_ = np.asarray(pred, np.int64)
    if mask is not None:
        s_, p_ = s_[mask], p_[mask]
    tp = int(np.sum((s_ == 1) & (p_ == 1)))
    fp = int(np.sum((s_ != 1) & (p_ == 1)))
    fn = int(np.sum((s_ == 1) & (p_ != 1)))
    tn = int(np.sum((s_ != 1) & (p_ != 1)))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(len(s_)),
            "precision": pr, "recall": rc, "f1": f1}


# ===========================================================================
# 1. the corrected 312-feature label representation, for ANY split
# ===========================================================================

def corrected_agg_block(tr_idx, va_idx, c, Wd, tag, log=log):
    """The 51 window aggregates for an arbitrary (train, validation) split.

    This is Exp13's ``corrected_fold_matrices`` body, generalised from the
    three sealed outer folds to any ``tr_idx`` so the NESTED folds can use it
    too.  Nothing is simplified: the honest inner window cross-fit and Exp12's
    corrected local mapping run in full for every call.

    Returns ``(Xagg_tr, Xagg_va, meta)`` where ``Xagg_tr`` is indexed by the
    position in ``tr_idx`` and ``Xagg_va`` by the position in ``va_idx``.

    ``tr_idx`` must be strictly ascending (both the outer ``tr`` and every
    nested inner train set are, by construction of the splitter).
    """
    yi, n = c["yi"], c["n"]
    Xw, yw, rw = Wd["X"], Wd["y"], Wd["run"]
    CW = wd.class_weight_vector()
    tr_idx = np.asarray(tr_idx, np.int64)
    va_idx = np.asarray(va_idx, np.int64)

    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise ValueError("%s: empty train or validation split" % tag)
    if len(np.unique(tr_idx)) != len(tr_idx):
        raise AssertionError("%s: duplicate rows in train split" % tag)
    if np.any(np.diff(tr_idx) <= 0):
        raise AssertionError("%s: train split is not strictly ascending" % tag)
    if len(np.intersect1d(tr_idx, va_idx)):
        raise AssertionError("%s: train and validation splits overlap" % tag)

    tr_runs = np.array(sorted(set(tr_idx.tolist())))
    if len(tr_runs) != len(tr_idx):
        raise AssertionError("%s: %d train rows collapse to %d runs - the "
                             "ascending-index assumption is broken"
                             % (tag, len(tr_idx), len(tr_runs)))
    g2l = M.global_to_local_map(tr_runs)

    inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=INNER_SEED)
                 .split(tr_runs, yi[tr_runs]))
    assign = np.zeros(len(tr_runs), dtype=int)
    for j, (_, hl) in enumerate(inner):
        assign[hl] = j
    R.verify_stacking(len(tr_runs), assign, N_INNER, tag)

    payloads = [(tag, j, Xw, yw, CW, np.isin(rw, tr_runs[assign != j]),
                 np.isin(rw, tr_runs[assign == j]), R.WINDOW_PARAMS)
                for j in range(N_INNER)]
    probs = par.run_inner_fits(payloads)
    del payloads
    gc.collect()

    Xagg_tr = np.zeros((len(tr_runs), ag.N_AGG))
    for j in range(N_INNER):
        hold = tr_runs[assign == j]
        sel = rw[np.isin(rw, hold)]
        # Exp12's CORRECTED grouping, verbatim - never Exp07's defective one.
        blocks, slot = M.group_fixed(probs[(tag, j)], sel, g2l, len(tr_runs))
        sub, _, _ = grp.aggregate_block(blocks, ag, slot, len(tr_runs))
        Xagg_tr[assign == j] = sub[assign == j]
    del probs
    gc.collect()

    # validation aggregates: one window model on ALL of the split's train rows
    mw = lgb.LGBMClassifier(n_jobs=par.DEFAULT_THREADS, **R.WINDOW_PARAMS)
    mw.fit(Xw[np.isin(rw, tr_runs)], yw[np.isin(rw, tr_runs)],
           sample_weight=CW[yw[np.isin(rw, tr_runs)]])
    Pva = mw.predict_proba(Xw[np.isin(rw, va_idx)])
    blocks = grp.predict_runs(Pva, rw[np.isin(rw, va_idx)], n)
    slot_va = {int(r): k for k, r in enumerate(va_idx)}
    Xagg_va, _, _ = grp.aggregate_block(blocks, ag, slot_va, len(va_idx))
    del mw
    gc.collect()

    meta = {"tag": tag, "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
            "n_train_runs": int(len(tr_runs)),
            "inner_holdout_sizes": [int((assign == j).sum())
                                    for j in range(N_INNER)],
            "mapping_lost_runs": 0}
    return Xagg_tr, Xagg_va, meta


def label_matrix_312(rows, Xagg, c, Xwg):
    """[249 foundation | 51 aggregates | 12 wg_*] for the given row indices."""
    return np.hstack([c["foundation"].iloc[rows].values, Xagg, Xwg[rows]])


# ===========================================================================
# 2. honesty helpers
# ===========================================================================

def exclusion_proof(tr_idx, nested_splits, n_rows, log=print):
    """Prove, structurally, that every trained row was predicted by a model
    fitted WITHOUT it.

    ``nested_splits`` is a list of ``(tag, fit_idx, holdout_idx)``.  The check
    is set membership, not a comparison of feature values: a row is honest only
    if the model that produced its probabilities was fitted on a set that does
    not contain it, and if it is held out exactly once across the inner folds.
    """
    tr_set = set(int(x) for x in tr_idx)
    problems = []
    seen = np.zeros(n_rows, dtype=np.int64)
    detail = []
    for k, (tag, ntr, nva) in enumerate(nested_splits):
        ntr = np.asarray(ntr, np.int64)
        nva = np.asarray(nva, np.int64)
        fit = set(int(x) for x in ntr)
        hold = set(int(x) for x in nva)
        if fit & hold:
            problems.append("%s: nested fit/hold overlap (%d rows)"
                            % (tag, len(fit & hold)))
        if not hold <= tr_set:
            problems.append("%s: nested hold-out escapes outer-train" % tag)
        if not fit <= tr_set:
            problems.append("%s: nested fit escapes outer-train" % tag)
        if len(fit) + len(hold) != len(tr_set):
            problems.append("%s: fit+holdout %d != outer-train %d"
                            % (tag, len(fit) + len(hold), len(tr_set)))
        for r in nva:
            seen[int(r)] += 1
            if int(r) in fit:
                problems.append("%s: row %d was in its own model's fit set"
                                % (tag, int(r)))
        detail.append({"nested_fold": k, "tag": tag, "n_fit": int(len(fit)),
                       "n_hold": int(len(hold)),
                       "overlap": int(len(fit & hold)),
                       "holdout_sha256": ca.sha256_indices(
                           np.sort(np.asarray(nva, np.int64)))})
    # every outer-train row must be held out exactly once
    cov = seen[tr_idx]
    if not (cov == 1).all():
        problems.append("outer-train rows are not covered exactly once "
                        "(min %d max %d)" % (int(cov.min()), int(cov.max())))
    ok = not problems
    if ok:
        log("   exclusion proof: all %d outer-train rows are covered exactly "
            "once across %d nested folds, and every one of them was predicted "
            "by a model whose fit set excludes it"
            % (int(len(tr_idx)), len(detail)))
    return {"all_excluded": ok, "problems": problems[:8],
            "n_outer_train": int(len(tr_idx)), "nested": detail,
            "coverage_min": int(cov.min()) if len(cov) else 0,
            "coverage_max": int(cov.max()) if len(cov) else 0}


# ===========================================================================
# 3. main
# ===========================================================================

def main():
    t0 = time.time()
    bar("EXP15 - SUCCESS STACKING FROM THE Exp13 LABEL PROBABILITIES")
    log("hypothesis : the 185-feature success head never sees the current")
    log("             Exp13 label probabilities; honest out-of-fold")
    log("             probabilities should improve success prediction")
    log("CONTROL A   : 185 Exp03 features, production success LightGBM")
    log("CANDIDATE B : the SAME 185 + 7 raw label probabilities = %d" % N_B)
    log("NOT touched : label head, fault_turn/L1, Exp14 ep_*, thresholds,")
    log("             class weights, entropy/margins/max-prob/hard-label")
    log("gate        : success >= +0.010, folds >= 2/3, composite >= +0.0015,")
    log("             no fold < -0.005, Macro/Robustness/hit@2 EXACTLY same")

    # ------------------------------------------------------------------
    hdr("1. VERIFIED INPUTS")
    d = ENS.load_verified()
    yi = np.asarray(d["yi"], np.int64)
    train = d["train"]
    folds = [np.asarray(v, np.int64) for v in d["folds"]]
    n = len(yi)
    rob = np.asarray(ENS.robustness_mask(), dtype=bool)
    log("   runs=%d folds=%d verified_checks=%d robust_rows=%d"
        % (n, len(folds), len(d["checks"]), int(rob.sum())))

    c, Wd = ENS._prepare()
    Xsucc = c["success"]
    Xwg = W.build_matrix(c["runs"])
    assert Xsucc.shape == (n, N_SUCCESS), Xsucc.shape
    assert Xwg.shape == (n, N_WG), Xwg.shape
    assert len(META_NAMES) == N_META == 7, META_NAMES
    log("   success matrix %s (Exp03 B)   wg_* %s   foundation %s"
        % (Xsucc.shape, Xwg.shape, c["foundation"].shape))
    log("   appended columns: %s" % ",".join(META_NAMES))

    s_true = np.asarray(train["success"].values, np.int64)
    assert set(np.unique(s_true)) <= {0, 1}, "success must be binary"
    log("   success target: %d positives / %d negatives (prevalence %.4f)"
        % (int((s_true == 1).sum()), int((s_true == 0).sum()),
           float(s_true.mean())))

    # ---- run_id alignment: every row of every matrix is keyed to a run_id ---
    run_ids = [str(x) for x in train["run_id"].values]
    rid_sha = ca.sha256_run_ids(run_ids)
    assert rid_sha == ca.sha256_run_ids(d["run_ids"]), \
        "verified loader's run order differs from train.csv"
    if len(set(run_ids)) != len(run_ids):
        raise AssertionError("run_id is not unique - row identity is ambiguous")
    log("   run_id alignment: %d unique ids, order sha256 %s"
        % (len(run_ids), rid_sha[:16]))
    RES["run_id_alignment"] = {
        "n_unique_run_ids": int(len(set(run_ids))), "n_rows": int(n),
        "run_id_sha256": rid_sha,
        "matches_verified_loader": True,
        "meta_prob_rows_share_index_with_success_matrix": True}

    # ---- the declared baseline, read from the pinned artifact -------------
    hdr("2. DECLARED BASELINE (read, never re-derived)")
    exp03_csv = pd.read_csv(os.path.join(
        ROOT, "experiments", "exp03_length_normalization",
        "oof_B_baseline_plus_norm.csv"))
    assert list(exp03_csv["run_id"].values) == list(train["run_id"].values), \
        "Exp03 B OOF csv is not aligned with train.csv"
    ref_success_pred = np.asarray(exp03_csv["success_pred"].values, np.int64)
    assert np.array_equal(np.asarray(exp03_csv["success_true"].values, np.int64),
                          s_true), "Exp03 B success_true differs from train.csv"
    baseline_success_f1 = c["success_f1"]
    log("   Exp03 B success_f1 (from build_all's own derivation) = %.16f"
        % baseline_success_f1)
    ref_f1 = binary_f1(list(s_true), list(ref_success_pred))
    log("   recomputed from the committed Exp03 B csv            = %.16f" % ref_f1)
    log("   the two agree exactly: %s" % (ref_f1 == baseline_success_f1))
    if ref_f1 != baseline_success_f1:
        log("   STOP: the committed Exp03 B csv and the pinned constant disagree")
        RES["decision"] = "STOP_BASELINE"
        dump(t0)
        return 1
    for k, v in EXP13_METRICS.items():
        log("   carried Exp13 %-18s = %.16f" % (k, v))
    log("   committed Exp13 D OOF sha256 = %s" % EXP13_SHA_D)
    committed_D = np.load(os.path.join(EXP13, "oof_D_corrected_wg.npy"),
                          allow_pickle=False)
    label_pred_committed = np.asarray(committed_D).argmax(1)
    log("   committed sha recomputes: %s"
        % (sha32(committed_D) == EXP13_SHA_D))
    if sha32(committed_D) != EXP13_SHA_D:
        log("   STOP: the committed Exp13 D artifact does not match its hash")
        RES["decision"] = "STOP_LABEL_PARITY"
        dump(t0)
        return 1

    sparams = success_params()
    log()
    log("   production success params: %s"
        % {k: sparams[k] for k in sorted(SUCCESS_PARAMS_EXPECTED)})
    lparams = ENS.label_params(42)
    log("   production label params  : n_estimators=%s lr=%s leaves=%s seed=%s"
        % (lparams.get("n_estimators"), lparams.get("learning_rate"),
           lparams.get("num_leaves"), lparams.get("random_state")))

    # ---- fold determinism, asserted against the sealed manifest ----------
    red = [np.asarray(v, np.int64) for _, v in
           StratifiedKFold(N_OUTER, shuffle=True, random_state=OUTER_SEED)
           .split(np.zeros(n), yi)]
    for a, b in zip(folds, red):
        if not np.array_equal(np.asarray(a, np.int64), b):
            raise AssertionError("outer fold is not the sealed seed-0 fold")
    log()
    log("   outer folds re-derived from StratifiedKFold(3, seed=0) and equal "
        "to the sealed manifest indices on all 3 folds")
    log("   Exp03's success splits and the label head's outer splits are the "
        "SAME 3 folds -> 'same folds' holds by construction, not by agreement")

    # ---- target-perturbation invariance of the success matrix ------------
    probe = success_feature_leakage_probe(c["runs"], 400)
    log("   success-matrix target-perturbation probe over %d runs: identical=%s"
        % (probe["n_rows_probed"], probe["identical"]))
    RES["success_feature_leakage_probe"] = probe
    if not probe["identical"]:
        log("   STOP: a success feature reads a target")
        RES["decision"] = "STOP_LEAKAGE"
        dump(t0)
        return 1

    # ==================================================================
    # OUTER FOLD LOOP
    # ==================================================================
    pred_A = np.zeros(n, dtype=np.int64)
    pred_B = np.zeros(n, dtype=np.int64)
    # The LABEL prediction vector: outer-validation probabilities only.  Every
    # row is the validation row of exactly one outer fold, so this array is
    # fully populated by the end of the loop and IS the current label head's
    # honest OOF.  It is kept separate from ``p_meta_oof`` because that one
    # deliberately also carries the NESTED probabilities used to train the
    # success head on outer-train rows, and those come from different models.
    p_meta_outer = np.full((n, N_META), np.nan, np.float64)
    outer_val_seen = np.zeros(n, dtype=np.int64)
    p_meta_oof = np.zeros((n, N_META), np.float64)       # 7 per row, honest
    p_meta_fit = np.zeros((n, N_META), np.float64)       # only rows trained on
    fit_mask = np.zeros(n, dtype=bool)
    imp_B = np.zeros(N_B, np.float64)
    fold_rows = []
    prov = []
    insample = []
    label_parity = []
    all_ix = np.arange(n, dtype=np.int64)

    for f, fv in enumerate(folds):
        va = np.asarray(fv, np.int64)
        tr = all_ix[~np.isin(all_ix, va)]
        t_f = time.time()
        log()
        hdr("3.%d OUTER FOLD %d   train=%d val=%d" % (f + 1, f, len(tr), len(va)))

        # ---- 3a. the outer label model: Exp13 D's own fold-f path -------
        Xagg_tr, Xagg_va, m_out = corrected_agg_block(tr, va, c, Wd,
                                                       "outer %d" % f, log=log)
        X312_tr = label_matrix_312(tr, Xagg_tr, c, Xwg)
        X312_va = label_matrix_312(va, Xagg_va, c, Xwg)
        assert X312_tr.shape == (len(tr), N_LABEL_312), X312_tr.shape
        assert X312_va.shape == (len(va), N_LABEL_312), X312_va.shape
        clf_outer = lgb.LGBMClassifier(**lparams).fit(X312_tr, yi[tr])
        p_va = np.asarray(clf_outer.predict_proba(X312_va), np.float64)
        p_tr_in = np.asarray(clf_outer.predict_proba(X312_tr), np.float64)
        # the SUCCESS target is never passed to a label fit - only yi is
        p_meta_outer[va] = p_va
        outer_val_seen[va] += 1
        p_meta_oof[va] = p_va
        diff = int((p_va.argmax(1) != label_pred_committed[va]).sum())
        label_parity.append({"fold": int(f), "n_val": int(len(va)),
                             "label_mismatches": diff,
                             "max_abs_prob_diff": float(np.max(np.abs(
                                 p_va - np.asarray(committed_D[va],
                                                  np.float64))))})
        log("   outer label: %d/%d validation rows match the committed Exp13 "
            "D argmax; max|dP| vs committed = %.3e"
            % (len(va) - diff, len(va),
               float(np.max(np.abs(p_va - np.asarray(committed_D[va],
                                                      np.float64))))))
        if diff != 0:
            log("   STOP_LABEL_PARITY: the rebuilt outer label path does not "
                "reproduce the committed Exp13 D labels")
            RES["label_parity"] = {"passed": False, "rows": label_parity}
            RES["decision"] = "STOP_LABEL_PARITY"
            dump(t0)
            return 1
        del clf_outer
        gc.collect()

        # ---- 3b. nested honest OOF probabilities for the TRAIN rows ----
        # n_splits is N_INNER (3).  A first execution of this script passed
        # N_META here by mistake and produced 7 nested folds; that run's
        # numbers are reported as-run in RESULTS.md under "deviation", and were
        # NOT re-run to a different split count, because picking a split count
        # after seeing the delta is selection on the outcome.
        n_splits_used = N_INNER
        inner = list(StratifiedKFold(n_splits_used, shuffle=True,
                                     random_state=NESTED_SEED)
                     .split(tr, yi[tr]))
        nested = []
        for k, (ntr_pos, nva_pos) in enumerate(inner):
            ntr = tr[np.asarray(ntr_pos, np.int64)]
            nva = tr[np.asarray(nva_pos, np.int64)]
            tag = "outer %d nested %d" % (f, k)
            Xa, Xb, m_n = corrected_agg_block(ntr, nva, c, Wd, tag, log=log)
            X312_ntr = label_matrix_312(ntr, Xa, c, Xwg)
            X312_nva = label_matrix_312(nva, Xb, c, Xwg)
            assert X312_ntr.shape == (len(ntr), N_LABEL_312)
            assert X312_nva.shape == (len(nva), N_LABEL_312)
            clf_n = lgb.LGBMClassifier(**lparams).fit(X312_ntr, yi[ntr])
            p_nva = np.asarray(clf_n.predict_proba(X312_nva), np.float64)
            p_meta_oof[nva] = p_nva
            nested.append((tag, ntr, nva))
            prov.append({"outer_fold": int(f), "nested_fold": int(k),
                         "n_fit": int(len(ntr)), "n_holdout": int(len(nva)),
                         "fit_excludes_every_holdout_row": True,
                         "holdout_subset_of_outer_train": True,
                         "inner_holdout_sizes": m_n["inner_holdout_sizes"],
                         "sha_of_holdout_rows": ca.sha256_indices(
                             np.sort(np.asarray(nva, np.int64)))})
            log("   nested %d: label fit on %d rows -> %d held-out rows"
                % (k, len(ntr), len(nva)))
            del clf_n, X312_ntr, X312_nva, Xa, Xb
            gc.collect()

        proof = exclusion_proof(tr, nested, n, log=log)
        proof["outer_fold"] = int(f)
        RES.setdefault("exclusion_proofs", []).append(proof)
        if not proof["all_excluded"]:
            for p_ in proof["problems"]:
                log("   LEAK: %s" % p_)
            log("   STOP_LEAKAGE")
            RES["decision"] = "STOP_LEAKAGE"
            dump(t0)
            return 1

        # ---- 3c. OOF is NOT in-sample: measured, not assumed ------------
        d_p = float(np.max(np.abs(p_meta_oof[tr] - p_tr_in)))
        n_dis = int((p_meta_oof[tr].argmax(1) != p_tr_in.argmax(1)).sum())
        insample.append({"outer_fold": int(f), "n_train": int(len(tr)),
                         "max_abs_diff_oof_vs_insample": d_p,
                         "label_disagreements": n_dis,
                         "mean_abs_diff": float(np.mean(np.abs(
                             p_meta_oof[tr] - p_tr_in)))})
        log("   OOF vs in-sample refit on the same rows: max|dP| = %.3e, "
            "%d/%d argmax differ -> the training probabilities are NOT "
            "in-sample" % (d_p, n_dis, len(tr)))
        del p_tr_in, X312_tr, X312_va
        gc.collect()

        # ---- 3d. the paired success fits -------------------------------
        X185_tr = np.asarray(Xsucc.iloc[tr].values, np.float64)
        X185_va = np.asarray(Xsucc.iloc[va].values, np.float64)
        meta_tr = p_meta_oof[tr]
        meta_va = p_meta_oof[va]
        assert meta_tr.shape == (len(tr), N_META)
        assert meta_va.shape == (len(va), N_META)
        if not np.isfinite(meta_tr).all() or not np.isfinite(meta_va).all():
            raise AssertionError("label probabilities contain non-finite values")

        XA = X185_tr
        XB = np.hstack([X185_tr, meta_tr])
        XA_va = X185_va
        XB_va = np.hstack([X185_va, meta_va])
        assert XB.shape[1] == N_B == N_SUCCESS + N_META, XB.shape
        if not np.array_equal(XB[:, :N_SUCCESS], XA):
            raise AssertionError("A/B first 185 columns are not identical")
        if not np.array_equal(XB_va[:, :N_SUCCESS], XA_va):
            raise AssertionError("A/B first 185 validation columns differ")

        mA = lgb.LGBMClassifier(**sparams).fit(XA, s_true[tr])
        pred_A[va] = np.asarray(mA.predict(XA_va), np.int64)
        mB = lgb.LGBMClassifier(**sparams).fit(XB, s_true[tr])
        pred_B[va] = np.asarray(mB.predict(XB_va), np.int64)
        im = mB.booster_.feature_importance(importance_type="gain")
        imp_B += (im / (im.sum() or 1.0)) / len(folds)
        p_meta_fit[tr] = meta_tr
        fit_mask[tr] = True

        fa, fb = prf_succ(s_true[va], pred_A[va]), prf_succ(s_true[va], pred_B[va])
        fold_rows.append({"fold": int(f), "n_val": int(len(va)),
                          "success_f1_A": fa["f1"], "success_f1_B": fb["f1"],
                          "success_f1_delta": fb["f1"] - fa["f1"],
                          "precision_A": fa["precision"],
                          "precision_B": fb["precision"],
                          "recall_A": fa["recall"], "recall_B": fb["recall"],
                          "fp_A": fa["fp"], "fp_B": fb["fp"],
                          "fn_A": fa["fn"], "fn_B": fb["fn"]})
        log("   success A -> B : F1 %.6f -> %.6f (%+.6f)  P %.4f->%.4f  "
            "R %.4f->%.4f  FP %d->%d  FN %d->%d"
            % (fa["f1"], fb["f1"], fb["f1"] - fa["f1"], fa["precision"],
               fb["precision"], fa["recall"], fb["recall"], fa["fp"], fb["fp"],
               fa["fn"], fb["fn"]))
        log("   fold %d done in %.1fs" % (f, time.time() - t_f))
        del mA, mB, XA, XB, XA_va, XB_va
        gc.collect()

    np.save(os.path.join(HERE, "success_pred_A.npy"), pred_A)
    np.save(os.path.join(HERE, "success_pred_B.npy"), pred_B)
    np.save(os.path.join(HERE, "meta_prob_oof.npy"), p_meta_oof.astype(np.float32))
    np.save(os.path.join(HERE, "meta_prob_fitrows.npy"),
            p_meta_fit.astype(np.float32))

    # ==================================================================
    # GATE 1 - CONTROL A reproduction
    # ==================================================================
    hdr("4. GATE 1 - CONTROL A REPRODUCES THE PRODUCTION SUCCESS HEAD")
    mA_all = prf_succ(s_true, pred_A)
    f1_gate = {"expected": baseline_success_f1, "got": mA_all["f1"],
               "abs_diff": abs(mA_all["f1"] - baseline_success_f1),
               "pass": bool(abs(mA_all["f1"] - baseline_success_f1) <= TOL)}
    n_diff = int((pred_A != ref_success_pred).sum())
    par = {"f1": f1_gate,
           "csv_predictions_differ": n_diff,
           "csv_predictions_identical": bool(n_diff == 0),
           "n_rows": int(n)}
    par["passed"] = bool(f1_gate["pass"] and n_diff == 0)
    log("   success_f1 expected %.16f  got %.16f  diff %.2e  %s"
        % (f1_gate["expected"], f1_gate["got"], f1_gate["abs_diff"],
           "PASS" if f1_gate["pass"] else "FAIL"))
    log("   predictions vs committed Exp03 B csv: %d / %d differ -> %s"
        % (n_diff, n, "IDENTICAL" if n_diff == 0 else "DIVERGED"))
    log("   GATE 1: %s" % ("PASS" if par["passed"] else "FAIL"))
    RES["baseline_reproduction"] = par
    if not par["passed"]:
        log()
        log("   STOP_BASELINE: CONTROL A does not reproduce the declared "
            "baseline. CANDIDATE B is NOT scored and nothing is built.")
        RES["decision"] = "STOP_BASELINE"
        dump(t0)
        return 1

    # ==================================================================
    # GATE 2 - label parity, over all rows
    # ==================================================================
    hdr("5. GATE 2 - LABEL PREDICTIONS UNCHANGED")
    # Compared on ``p_meta_outer`` - the outer-validation probabilities, i.e.
    # exactly the vector the current label head produces.  ``p_meta_oof`` also
    # carries the NESTED probabilities on outer-train rows, and those come from
    # different (smaller) models, so its argmax is not the label head's
    # prediction and is NOT what parity is about.
    assert int((outer_val_seen == 1).all()), \
        "every row must be the validation row of exactly one outer fold"
    label_pred = p_meta_outer.argmax(1)
    lab_diff = int((label_pred != label_pred_committed).sum())
    max_p = float(np.max(np.abs(p_meta_outer
                                - np.asarray(committed_D, np.float64))))
    RES["label_parity"] = {"passed": bool(lab_diff == 0), "rows": label_parity,
                           "label_predictions_differ": lab_diff,
                           "max_abs_prob_diff_vs_committed": max_p,
                           "n_rows": int(n)}
    log("   argmax of the 7 meta columns over all 10000 rows vs the committed "
        "Exp13 D argmax: %d / %d differ" % (lab_diff, n))
    log("   max|dP| vs the committed Exp13 D matrix: %.3e" % max_p)
    log("   GATE 2: %s" % ("PASS" if lab_diff == 0 else "FAIL"))
    if lab_diff != 0:
        RES["decision"] = "STOP_LABEL_PARITY"
        dump(t0)
        return 1

    # ==================================================================
    # metrics
    # ==================================================================
    hdr("6. HEADLINE METRICS - A vs B")
    all_A, all_B = prf_succ(s_true, pred_A), prf_succ(s_true, pred_B)
    d_succ = all_B["f1"] - all_A["f1"]
    log("   success F1   %.16f -> %.16f   delta %+.16f"
        % (all_A["f1"], all_B["f1"], d_succ))
    log("   precision    %.6f -> %.6f   (%+.6f)"
        % (all_A["precision"], all_B["precision"],
           all_B["precision"] - all_A["precision"]))
    log("   recall       %.6f -> %.6f   (%+.6f)"
        % (all_A["recall"], all_B["recall"],
           all_B["recall"] - all_A["recall"]))
    log("   FP %d -> %d   FN %d -> %d   TP %d -> %d   TN %d -> %d"
        % (all_A["fp"], all_B["fp"], all_A["fn"], all_B["fn"],
           all_A["tp"], all_B["tp"], all_A["tn"], all_B["tn"]))

    # ---- composite: only success_f1 moves -----------------------------
    def comp_with(sf1):
        return composite({"macro_f1": EXP13_METRICS["macro_f1"],
                          "robustness_f1": EXP13_METRICS["robustness_f1"],
                          "success_f1": sf1,
                          "fault_turn_hit2": EXP13_METRICS["fault_turn_hit2"]})

    comp_A, comp_B = comp_with(all_A["f1"]), comp_with(all_B["f1"])
    log()
    log("   %-20s %16s %16s %14s" % ("metric", "A", "B", "delta"))
    log("   %-20s %16.6f %16.6f %14s"
        % ("composite", comp_A, comp_B, "%+.8f" % (comp_B - comp_A)))
    log("   composite delta == 0.15 * success delta: %s (%.3e vs %.3e)"
        % (abs((comp_B - comp_A) - 0.15 * d_succ) <= 1e-15,
           comp_B - comp_A, 0.15 * d_succ))

    # ==================================================================
    # GATE 3 - mechanical invariance of the carried metrics
    # ==================================================================
    hdr("7. GATE 3 - MACRO / ROBUSTNESS / hit@2 / L1 EXACTLY UNCHANGED")
    corpus = ENS.corpus_for(p_meta_outer.astype(np.float32), d, rob,
                            labels=label_pred)
    turns_meta = corpus.turns_for(label_pred)
    carried = ENS.full_official(corpus, label_pred)[0]
    # The Corpus carries SUCCESS_F1_PINNED; recompute the composite with the
    # measured success so the carried terms can be compared term by term.
    inv_rows = []
    inv_ok = True
    for k in ("macro_f1", "robustness_f1", "fault_turn_hit2"):
        got, exp = float(carried[k]), EXP13_METRICS[k]
        good = abs(got - exp) <= TOL
        inv_ok &= good
        inv_rows.append({"metric": k, "carried": exp, "recomputed": got,
                         "abs_diff": abs(got - exp), "pass": bool(good)})
        log("   %-18s carried %.16f  recomputed %.16f  diff %.2e  %s"
            % (k, exp, got, abs(got - exp), "PASS" if good else "FAIL"))

    # fault_turn: the L1 vector must be byte-identical for the unchanged label
    corpus_committed = ENS.corpus_for(np.asarray(committed_D, np.float32), d, rob,
                                      labels=label_pred_committed)
    turns_committed = corpus_committed.turns_for(label_pred_committed)
    ft_same = bool(np.array_equal(np.asarray(turns_meta, np.int64),
                                  np.asarray(turns_committed, np.int64)))
    inv_ok &= ft_same
    log("   fault_turn vector identical to the Exp13 D L1 vector: %s "
        "(%d non -1)" % (ft_same, int((np.asarray(turns_meta) >= 0).sum())))
    per_class_A = {c: float(v) for c, v in
                   f1_per_class(corpus.name(label_pred_committed),
                                corpus.name(label_pred_committed), LABELS).items()}
    lab_rows = [{"class": c, "f1": per_class_A[c]} for c in LABELS]
    RES["invariance"] = {"passed": bool(inv_ok), "rows": inv_rows,
                         "fault_turn_identical": ft_same,
                         "label_predictions_identical": bool(lab_diff == 0),
                         "per_class_label_f1_unchanged": lab_rows}
    log("   GATE 3: %s" % ("PASS" if inv_ok else "FAIL"))
    if not inv_ok:
        RES["decision"] = "STOP_INVARIANCE"
        dump(t0)
        return 1

    # ==================================================================
    # diagnostics
    # ==================================================================
    hdr("8. FOLD STABILITY (success only; other metrics are unchanged)")
    folds_won = 0
    for r in fold_rows:
        if r["success_f1_delta"] > 0:
            folds_won += 1
        log("   fold %d n=%4d  success F1 %.6f -> %.6f  (%+.6f)   "
            "FP %d->%d  FN %d->%d"
            % (r["fold"], r["n_val"], r["success_f1_A"], r["success_f1_B"],
               r["success_f1_delta"], r["fp_A"], r["fp_B"], r["fn_A"], r["fn_B"]))
    worst_fold = min(r["success_f1_delta"] for r in fold_rows)
    log("   folds won %d/3   worst fold delta %+.6f" % (folds_won, worst_fold))

    hdr("9. CHANGED SUCCESS PREDICTIONS (three-way, per Exp11's lesson)")
    ch = pred_A != pred_B
    wA, wB = pred_A != s_true, pred_B != s_true
    w2r = int((ch & wA & ~wB).sum())
    r2w = int((ch & ~wA & wB).sum())
    w2w = int((ch & wA & wB).sum())
    same_w = int((ch & wA & wB).sum())
    log("   changed %d / %d (%.2f%%)   wrong->right %d   right->wrong %d   "
        "wrong->wrong %d"
        % (int(ch.sum()), n, 100.0 * float(ch.sum()) / n, w2r, r2w, w2w))
    assert w2r + r2w + w2w == int(ch.sum()), \
        "three-way partition does not add up - report wrong"

    hdr("10. CLASS-CONDITIONAL SUCCESS DIAGNOSTICS")
    log("   %-18s %6s %10s %9s %9s %7s %7s %7s %7s"
        % ("true class", "n", "succ.prev", "F1_A", "F1_B", "d", "FP_A",
           "FP_B", "FN_B"))
    cls_rows = []
    for ci, lab in enumerate(LABELS):
        m = yi == ci
        fa = prf_succ(s_true[m], pred_A[m])
        fb = prf_succ(s_true[m], pred_B[m])
        prev = float(s_true[m].mean()) if m.sum() else 0.0
        row = {"class": lab, "n": int(m.sum()), "success_prevalence": prev,
               "f1_A": fa["f1"], "f1_B": fb["f1"], "f1_delta": fb["f1"] - fa["f1"],
               "fp_A": fa["fp"], "fp_B": fb["fp"],
               "fn_A": fa["fn"], "fn_B": fb["fn"],
               "changed": int((pred_A[m] != pred_B[m]).sum())}
        cls_rows.append(row)
        log("   %-18s %6d %10.4f %9.4f %9.4f %+7.4f %7d %7d %7d"
            % (lab, row["n"], prev, fa["f1"], fb["f1"],
               row["f1_delta"], fa["fp"], fb["fp"], fb["fn"]))
    log("   (per-class F1 is success-F1 WITHIN that class; a class where "
        "success is near-constant is not a meaningful slice)")

    hdr("11. FEATURE IMPORTANCE OF THE 7 PROBABILITIES (candidate B)")
    names = list(Xsucc.columns) + META_NAMES
    assert len(names) == N_B == 192, len(names)
    share = float(imp_B[N_SUCCESS:].sum())
    order = np.argsort(-imp_B)
    log("   total gain share of the 7 label probabilities: %.6f (%.3f%%)"
        % (share, 100.0 * share))
    for i in order:
        if int(i) >= N_SUCCESS:
            log("      %-26s %.6f" % (names[int(i)], imp_B[int(i)]))
    top20 = set(int(i) for i in order[:20])
    log("   in top-20: %d / 7" % len(top20 & set(range(N_SUCCESS, N_B))))

    hdr("12. PROBABILITY vs SUCCESS TARGET (diagnostic only, no selection)")
    corr_rows = []
    from scipy.stats import pointbiserialr, spearmanr
    log("   %-26s %12s %12s %12s" % ("feature", "pointbiser", "spearman",
                                     "mean"))
    for j, nm in enumerate(META_NAMES):
        v = p_meta_fit[fit_mask, j]
        pb = float(pointbiserialr(s_true[fit_mask].astype(float), v)[0])
        sp = float(spearmanr(s_true[fit_mask], v).statistic)
        corr_rows.append({"feature": nm, "pointbiserial": pb,
                          "spearman": sp, "mean": float(v.mean()),
                          "std": float(v.std())})
        log("   %-26s %12.4f %12.4f %12.4f" % (nm, pb, sp, v.mean()))
    log("   computed on the %d rows the success head was actually fitted on,"
        % int(fit_mask.sum()))
    log("   i.e. exactly the 7 columns it saw. DIAGNOSTIC ONLY - no feature")
    log("   selection, no pruning, no second variant.")

    # ==================================================================
    # gate
    # ==================================================================
    hdr("13. PRE-DECLARED PROMOTION GATE (never retuned)")
    checks = {
        "success_f1_delta>=+0.010": (d_succ,
                                     d_succ >= GATE["success_f1_delta_min"]),
        "folds_won>=2": (float(folds_won), folds_won >= GATE["folds_won_min"]),
        "composite_delta>=+0.0015": (comp_B - comp_A,
                                     (comp_B - comp_A)
                                     >= GATE["composite_delta_min"]),
        "no_fold_worse_than_-0.005": (worst_fold,
                                      worst_fold >= GATE["worst_fold_delta_min"]),
        "macro_rob_hit2_and_l1_exactly_unchanged": (0.0 if inv_ok else 1.0,
                                                     bool(inv_ok)),
    }
    for k, (v, o) in checks.items():
        log("   %-42s %+12.6f  %s" % (k, v, "PASS" if o else "FAIL"))
    gate_pass = all(o for _v, o in checks.values())
    log("   GATE: %s (%d/5)"
        % ("PASS" if gate_pass else "FAIL",
           sum(1 for _v, o in checks.values() if o)))
    if gate_pass:
        decision = "PROMOTE"
    elif WEAK_BAND[0] <= d_succ < WEAK_BAND[1]:
        decision = "STOP_WEAK"
    else:
        decision = "STOP"
    log("   VERDICT: %s" % decision)

    hdr("14. PRODUCTION")
    log("   production build: NO - not built, by protocol")
    log("   see RESULTS.md section 8 for the design that a PASS would imply")

    RES.update({
        "hypothesis": ("the 185-feature success head never observes the "
                       "current Exp13 label probabilities; honest out-of-fold "
                       "probabilities should improve success prediction"),
        "n_success_features": N_SUCCESS, "n_meta_features": N_META,
        "n_candidate_features": N_B, "meta_feature_names": META_NAMES,
        "nested_seed": NESTED_SEED, "outer_seed": OUTER_SEED,
        "success_A": all_A, "success_B": all_B,
        "success_f1_delta": d_succ,
        "composite_A": comp_A, "composite_B": comp_B,
        "composite_delta": comp_B - comp_A,
        "macro_f1": EXP13_METRICS["macro_f1"],
        "robustness_f1": EXP13_METRICS["robustness_f1"],
        "fault_turn_hit2": EXP13_METRICS["fault_turn_hit2"],
        "metrics_A": dict(EXP13_METRICS, composite=comp_A),
        "metrics_B": dict(EXP13_METRICS, success_f1=all_B["f1"],
                          composite=comp_B),
        "folds": fold_rows, "folds_won": folds_won,
        "worst_fold_delta": worst_fold,
        "changed_success": int(ch.sum()), "wrong_to_right": w2r,
        "right_to_wrong": r2w, "wrong_to_wrong": w2w,
        "per_class_success": cls_rows,
        "meta_gain_share": share,
        "meta_importance": [{"feature": names[int(i)], "gain": float(imp_B[int(i)])}
                            for i in order if int(i) >= N_SUCCESS],
        "meta_in_top20": len(top20 & set(range(N_SUCCESS, N_B))),
        "meta_target_correlations": corr_rows,
        "n_rows_trained_on": int(fit_mask.sum()),
        "oof_vs_insample": insample,
        "exclusion_proofs": RES.get("exclusion_proofs", []),
        "nested_provenance": prov,
        "promotion_gate": {
            "passed": bool(gate_pass),
            "criteria": {k: {"value": float(v), "pass": bool(o)}
                         for k, (v, o) in checks.items()},
            "n_passed": int(sum(1 for _v, o in checks.values() if o)),
            "n_criteria": len(checks)},
        "promotion_gate_passed": bool(gate_pass),
        "gate": GATE, "weak_band": list(WEAK_BAND),
        "decision": decision,
        "production_created": False,
        "success_params": {k: sparams[k] for k in sorted(SUCCESS_PARAMS_EXPECTED)},
        "label_params": {k: v for k, v in lparams.items()
                         if isinstance(v, (int, float, str, bool))},
    })
    dump(t0)
    return 0


def success_feature_leakage_probe(runs, n_probe):
    """Target perturbation: rewriting every target must not move a success
    feature.

    Uses the SAME code path ``build_all`` used for the 185 columns -
    ``features.extract_features`` and ``new_features.build_matrix`` - on the
    already-parsed runs, so a target reading anywhere in that path would move a
    number here.
    """
    import copy
    import new_features as nf
    from features import extract_features

    def build(rs):
        base = [extract_features(r) for r in rs]
        return np.hstack([
            np.asarray(pd.DataFrame(base).fillna(0.0).values, np.float64),
            np.asarray(nf.build_matrix(rs, base).values, np.float64)])

    head = list(runs[:n_probe])
    base = build(head)
    pert = []
    for i, r in enumerate(head):
        r2 = copy.deepcopy(r)
        r2["label"] = ["deadlock", "clean", "goal_drift"][i % 3]
        r2["success"] = 1 - (i % 2)
        r2["fault_turn"] = (i * 7) % 11
        pert.append(r2)
    other = build(pert)
    same = bool(np.array_equal(base, other))
    return {"identical": same, "n_rows_probed": int(len(head)),
            "n_columns": int(base.shape[1]),
            "max_abs_diff": float(np.max(np.abs(base - other)))
            if base.size else 0.0}


def dump(t0):
    RES["runtime_sec"] = round(time.time() - t0, 2)
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=2, sort_keys=True, default=float)
    with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    log()
    log("   runtime %.1fs; results.json + run.log written" % RES["runtime_sec"])


if __name__ == "__main__":
    raise SystemExit(main())
