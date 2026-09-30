"""Exp11 - fixed 3-seed probability ensemble of the FINAL label head.  Reuse only.

THE ONE HYPOTHESIS
------------------
Exp08's official composite is built on ONE label head: Exp07's system B (300
features) fitted by ``common.lgb_label()``, which pins ``random_state=42``.
LightGBM is stochastic in three places at once - feature subsampling
(``colsample_bytree=0.8``), row subsampling (``subsample=0.9`` with
``subsample_freq=1``) and the histogram binning tie-breaks - so a different
``random_state`` gives a genuinely different model on the SAME rows and the
SAME features.  Exp11 asks one question: does averaging the probabilities of
three such models beat the single incumbent?

    P_ensemble = (P_seed42 + P_seed137 + P_seed2026) / 3
    pred       = argmax(P_ensemble)

The seed set was FIXED BEFORE ANY NUMBER WAS SEEN and is not searched: not
other seeds, not 2 or 4 models, not weights, not a chosen best seed.  The
headline is the plain average.

EVERYTHING ELSE IS FROZEN
-------------------------
The three models differ in EXACTLY ONE attribute - ``random_state``.  Same
outer folds, same outer-train rows, same 300 columns in the same order, same
hyper-parameters, same objective, same ``num_class``.  The window model, its
own ``random_state=42``, the 51 aggregates, the sealed window peaks, the Exp08
L1 localiser and the pinned success F1 are all untouched, and this module
REFUSES to build a model whose parameter dict differs in anything else.

WHY THE 300-FEATURE MATRIX IS REBUILT HERE
-----------------------------------------
There is no stored 300-feature matrix, only the sealed OOF probabilities.
Rebuilding it is permitted once, by the standard Exp07 mechanism, and only
after parity is proven.  The rebuild is a LITERAL transcription of
``runner.run()``'s fold loop - same inner StratifiedKFold(3, seed=0), same
``run_inner_fits`` call, same ``predict_runs`` / ``aggregate_block`` calls with
the same (deliberately odd) arguments - because Exp07's model path is
hash-sealed and any deviation would be a different experiment.

    50|INCLUDING AN EXP07 QUIRK THAT IS DELIBERATELY REPRODUCED
----------------------------------------------------------------
``runner.py:229`` calls
``grp.predict_runs(P, rw[np.isin(rw, hold)], len(tr_runs))``: ``predict_runs``
iterates ``for r in range(n_runs)`` while ``rw`` carries GLOBAL run indices up
to 9999, and ``len(tr_runs)`` is about 6666.  Every outer-train run whose
global index is >= ``len(tr_runs)`` therefore never gets a block, and its 51
window features stay ZERO in Exp07's own training matrix - about a third of
outer-train rows per fold.

    60|That is a real defect in Exp07, and this experiment does NOT fix it.  Fixing it
would change the 300-feature matrix, which would mean testing a different
hypothesis with a different feature set, and the bit-exactness of the
reproduction gate depends on the quirk being present.  It is reported as a
finding and left for a separate experiment; here it is reproduced exactly,
which ``build_features`` proves by reproducing the sealed probabilities
bit-for-bit.

FLOAT PRECISION
---------------
Exp07 stores ``oof_proba_B_plus_window.npy`` as float32 and the seal is
computed over that cast.  Every seed's probabilities are therefore cast to
float32 first and the average is taken in float64.  This makes the seed-42
member BIT-IDENTICAL to the sealed baseline - so ``seed42 alone == current
best`` is an identity rather than a tolerance - and it makes the ensemble the
average of exactly the arrays the incumbent is made of.

SCORING IS NOT REIMPLEMENTED
----------------------------
The scoring stack is Exp10's, reused unchanged: ``blend.Corpus`` and
``blend.Objective`` with ``proba_A is proba_B = P``, for which ``blend(1.0)`` is
``1.0 * P + 0.0 * P == P`` bit-for-bit, and ``Objective.official`` calls
``evaluation.metrics`` directly.  ``evaluation.metrics.composite`` is the only
definition of the objective.  The fast path is cross-checked against the
official one on the real candidates before anything is reported.
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
EXP08 = os.path.join(ROOT, "experiments", "exp08_window_localizer")
EXP10 = os.path.join(ROOT, "experiments", "exp10_ab_probability_blend")
for _p in (HERE, EXP08, EXP10, EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aggregate as ag                       # noqa: E402
import cache as ca                           # noqa: E402
import grouping as grp                       # noqa: E402
import load_exp07 as L                       # noqa: E402
import parallel as par                       # noqa: E402
import runner as R                           # noqa: E402
import window_dataset as wd                   # noqa: E402
from blend import Corpus, Objective, LABELS, L2I, W2I   # noqa: E402
from blend import N_CLASSES, N_RUNS, SUCCESS_F1_PINNED, TOL  # noqa: E402
from common import build_all, lgb_label      # noqa: E402

CLEAN = 0
SEEDS = (42, 137, 2026)                 # FROZEN before any number was seen
BASE_SEED = 42
N_INNER = 3
INNER_SEED = 0
N_AGG_FEATS = 51
N_LABEL_FEATS = 300

# The pre-declared promotion gate, written down before the first fit.
GATE = {"composite_delta_min": 0.002, "folds_won_min": 2,
        "robustness_drop_max": 0.003, "class_f1_drop_max": 0.02}
# Composite delta in [0.001, 0.002) is declared WEAK/INCONCLUSIVE up front.
WEAK_BAND = (0.001, 0.002)


# ---------------------------------------------------------------------------
# 1. strict, fail-closed inputs
# ---------------------------------------------------------------------------

def load_verified(train_csv=None):
    """Verified Exp07 B OOF + sealed peaks, through Exp08's fail-closed loader.

    ``ArtifactRejected`` / ``PeakRejected`` propagate.  There is no lenient path:
    an unverified array must never reach a fit, because every number below it
    would be a guess dressed as a measurement.
    """
    train_csv = train_csv or os.path.join(ROOT, "data", "train.csv")
    train = pd.read_csv(train_csv)
    ys = train["label"].values.astype(object)
    yi = np.array([L2I[l] for l in ys], dtype=int)
    rids = [str(x) for x in train["run_id"]]
    t0 = time.time()
    art = L.load(system="B_plus_window", yi=yi, train_run_ids=rids)  # raises
    folds = L.reuse.fold_assignments(yi)
    got = [L.ca.sha256_indices(np.asarray(v, dtype=np.int64)) for v in folds]
    want = list(art["manifest"]["fold_val_sha256"])
    if got != want:
        raise L.reuse.ArtifactRejected(
            "re-derived outer fold indices differ from the sealed manifest")
    return {"train": train, "ys": ys, "yi": yi, "run_ids": rids,
            "proba_B": art["proba"], "labels_B": art["labels"],
            "peak_pos": art["peak_pos"], "peak_prob": art["peak_prob"],
            "manifest": art["manifest"], "checks": art["verified_checks"],
            "folds": folds, "fold_sha": got,
            "load_sec": round(time.time() - t0, 2)}


def robustness_mask():
    """Exp07's ``hard/robustness`` mask - Exp10's cached, fingerprinted build.

    It is a property of the DATA, not of any prediction, so it is identical for
    the baseline and for every seed.
    """
    import blend as K
    return np.asarray(K.robustness_mask(), dtype=bool)


# ---------------------------------------------------------------------------
# 2. the label head - one attribute may differ, and it is checked
# ---------------------------------------------------------------------------

def label_params(seed):
    """``common.lgb_label()``'s exact parameters with ``random_state`` replaced.

    Built from ``get_params()`` rather than from a hand-copied dict so the
    construction cannot drift away from the validated configuration: whatever
    ``lgb_label`` says it is, these are.
    """
    if int(seed) not in SEEDS:
        raise ValueError("seed %r is outside the frozen set %r" % (seed, SEEDS))
    p = dict(lgb_label().get_params())
    p["random_state"] = int(seed)
    return p


def assert_params_differ_only_in_random_state():
    """The three parameter dicts must be identical except ``random_state``.

    This is the mechanical form of "same rows, same features, same
    hyper-parameters": if someone adds a parameter to one of the three fits,
    this raises instead of quietly testing a different hypothesis.
    """
    base = label_params(BASE_SEED)
    if base != lgb_label().get_params():
        raise AssertionError("label_params(42) is not lgb_label()'s parameters")
    out = {}
    for s in SEEDS:
        p = label_params(s)
        diff = {k: (base[k], p[k]) for k in set(base) | set(p)
                if base.get(k) != p.get(k)}
        # seed 42 IS the reference, so its diff is empty by construction; every
        # other seed must differ in random_state and in nothing else.
        want = set() if s == BASE_SEED else {"random_state"}
        if set(diff) != want:
            raise AssertionError("seed %d differs from seed 42 in %r, expected "
                                 "exactly %r" % (s, diff, want))
        if s != BASE_SEED and diff["random_state"][0] == diff["random_state"][1]:
            raise AssertionError("seed %d did not change random_state" % s)
        out[s] = p
    for key in ("objective", "num_class", "n_estimators", "learning_rate",
                "num_leaves", "min_child_samples", "subsample",
                "subsample_freq", "colsample_bytree", "reg_lambda", "n_jobs"):
        vals = {s: out[s][key] for s in SEEDS}
        if len(set(vals.values())) != 1:
            raise AssertionError("%s varies across seeds: %r" % (key, vals))
    return out


# ---------------------------------------------------------------------------
# 3. the 300-feature matrix - Exp07's own mechanism, transcribed
# ---------------------------------------------------------------------------

def _prepare():
    ca.set_enabled(True)
    c = ca.get_or_build("build_all", build_all)
    W = ca.get_or_build(
        "windows", lambda: wd.build_window_dataset(c["runs"], c["ys"], c["fturn"]))
    return c, W


def build_features(seed, folds, c, W, threads=par.DEFAULT_THREADS,
                   workers=1, log=print):
    """Fit ONE label head per outer fold and return honest OOF probabilities.

    Returns ``(proba, cols)`` with ``proba`` float32-cast, exactly as Exp07
    stores it, and ``cols`` the 300 column names in the fitted order.

    Everything that is not the label model's ``random_state`` is Exp07's
    ``runner.run()`` verbatim: the inner StratifiedKFold(3, seed=0) over
    ``sorted(set(tr))``, ``run_inner_fits`` with the same payloads, and the
    ``predict_runs`` / ``aggregate_block`` calls with the same arguments
    (including the index-range quirk documented in the module docstring).
    """
    import lightgbm as lgb

    yi = c["yi"]
    n = c["n"]
    found = c["foundation"]
    Xw, yw, rw = W["X"], W["y"], W["run"]
    CW = wd.class_weight_vector()
    params = label_params(seed)
    oof = np.zeros((n, N_CLASSES))
    cols = None
    seen_cols = []

    # The outer folds come from the caller, already asserted equal to the
    # sealed manifest's index hashes in ``load_verified``.  They are
    # re-derived here anyway and compared, so a fold can never be substituted
    # between the verification and the fit.
    rederived = [np.asarray(v, dtype=np.int64) for _, v in
                 StratifiedKFold(R.N_OUTER, shuffle=True, random_state=R.SEED)
                 .split(np.zeros(n), yi)]
    for a, b in zip(folds, rederived):
        if not np.array_equal(np.asarray(a, dtype=np.int64), b):
            raise AssertionError("outer fold supplied by the caller is not the "
                                 "seed-0 StratifiedKFold(3) fold")
    all_ix = np.arange(n, dtype=np.int64)
    fidx = [(all_ix[~np.isin(all_ix, fi)], np.asarray(fi, dtype=np.int64))
            for fi in folds]

    for f in range(R.N_OUTER):
        tr, va = fidx[f]
        t_f = time.time()
        tr_runs = np.array(sorted(set(tr.tolist())))
        inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=INNER_SEED)
                     .split(tr_runs, yi[tr_runs]))
        assign = np.zeros(len(tr_runs), dtype=int)
        for j, (_, hl) in enumerate(inner):
            assign[hl] = j
        R.verify_stacking(len(tr_runs), assign, N_INNER, "fold %d" % f)

        payloads = []
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            keep = tr_runs[assign != j]
            payloads.append((f, j, Xw, yw, CW, np.isin(rw, keep),
                             np.isin(rw, hold), R.WINDOW_PARAMS))
        probs = par.run_inner_fits(payloads, threads=threads, workers=workers)
        del payloads
        gc.collect()

        Xagg_tr = np.zeros((len(tr), ag.N_AGG))
        slot = {int(r): k for k, r in enumerate(tr_runs)}
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            P = probs[(f, j)]
            blocks = grp.predict_runs(P, rw[np.isin(rw, hold)], len(tr_runs))
            sub, _, _ = grp.aggregate_block(blocks, ag, slot, len(tr_runs))
            Xagg_tr[assign == j] = sub[assign == j]
        del probs
        gc.collect()

        mw = lgb.LGBMClassifier(n_jobs=threads, **R.WINDOW_PARAMS)
        mw.fit(Xw[np.isin(rw, tr)], yw[np.isin(rw, tr)],
               sample_weight=CW[yw[np.isin(rw, tr)]])
        Pva = mw.predict_proba(Xw[np.isin(rw, va)])
        blocks = grp.predict_runs(Pva, rw[np.isin(rw, va)], n)
        slot_va = {int(r): k for k, r in enumerate(va)}
        Xagg_va, pk, pp = grp.aggregate_block(blocks, ag, slot_va, len(va))
        del mw
        gc.collect()

        cols = list(found.columns) + list(ag.AGG_NAMES)
        seen_cols.append(list(cols))
        Xb_tr = pd.DataFrame(np.hstack([found.iloc[tr].values, Xagg_tr]),
                             columns=cols)
        Xb_va = pd.DataFrame(np.hstack([found.iloc[va].values, Xagg_va]),
                             columns=cols)
        clf = lgb.LGBMClassifier(**params).fit(Xb_tr, yi[tr])
        oof[va] = clf.predict_proba(Xb_va)
        log("    fold %d: n_tr=%d n_val=%d  label seed=%d  %.1fs"
            % (f, len(tr), len(va), seed, time.time() - t_f))
        del clf, Xb_tr, Xb_va, Xagg_tr, Xagg_va
        gc.collect()

    for c2 in seen_cols:
        if c2 != seen_cols[0]:
            raise AssertionError("fold %d fitted a different column set/order"
                                 % seen_cols.index(c2))
    return oof.astype(np.float32), seen_cols[0]


# ---------------------------------------------------------------------------
# 4. the ensemble
# ---------------------------------------------------------------------------

def average(probas):
    """``(P1 + P2 + P3) / 3`` in float64 - equal weights, declared in advance.

    The inputs are the float32-cast per-seed matrices, the same precision the
    incumbent is stored at, so the seed-42 member is bit-identical to the
    sealed baseline.
    """
    mats = [np.asarray(P, dtype=np.float32) for P in probas]
    if len(mats) != len(SEEDS):
        raise ValueError("the ensemble is exactly %d models" % len(SEEDS))
    acc = np.zeros((N_RUNS, N_CLASSES), dtype=np.float64)
    for M in mats:
        acc += M.astype(np.float64)
    out = acc / float(len(mats))
    if not np.isfinite(out).all():
        raise ValueError("the ensemble contains non-finite values")
    return out


def corpus_for(P, d, rob, labels=None):
    """Exp10's ``Corpus`` with ``proba_A is proba_B = P``.

    ``blend(1.0)`` is then ``1.0 * P + 0.0 * P == P`` bit-for-bit, so Exp10's
    verified fast path and official path score EXACTLY this matrix - nothing
    about the scoring is reimplemented here.
    """
    P = np.asarray(P, dtype=np.float32)
    return Corpus(P, P, d["yi"], d["train"]["fault_turn"].values,
                  d["peak_pos"], rob,
                  labels_B=(np.asarray(labels, dtype=np.int64) if labels is not None
                            else P.argmax(1)))


def full_official(c, pred):
    """Official composite over ALL 10000 rows, through ``evaluation.metrics``.

    Mirrors Exp10's ``official_full``: the faulty-only alignment hazard is
    removed by passing the truly-faulty rows with their labels AND turns
    together, because ``fault_turn_hit_at_k`` zips positionally.
    """
    from metrics import (composite, fault_turn_hit_at_k, f1_per_class, macro_f1)
    yt = c.name(c.yi)
    yp = c.name(pred)
    turns = c.turns_for(pred)
    f = np.where(c.yi != CLEAN)[0]
    h2 = fault_turn_hit_at_k([yt[i] for i in f], [yp[i] for i in f],
                             [int(c.fturn[i]) for i in f],
                             [int(turns[i]) for i in f], k=2)
    rob_pos = np.where(c.rob)[0]
    rf = (macro_f1([yt[i] for i in rob_pos], [yp[i] for i in rob_pos], LABELS)
          if len(rob_pos) else 0.0)
    mf = macro_f1(yt, yp, LABELS)
    return {"macro_f1": mf, "robustness_f1": rf,
            "success_f1": SUCCESS_F1_PINNED, "fault_turn_hit2": h2,
            "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                    "success_f1": SUCCESS_F1_PINNED,
                                    "fault_turn_hit2": h2})}, turns


def confusion(yt, yp):
    ti = np.array([L2I[t] for t in yt], dtype=np.int64)
    pi = np.array([L2I[p] for p in yp], dtype=np.int64)
    return np.bincount(ti * N_CLASSES + pi,
                       minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES,
                                                                N_CLASSES)


def conf_str(m):
    lines = ["      %-17s %s" % ("true \\ pred",
                                 " ".join("%5d" % i for i in range(N_CLASSES)))]
    for i, lab in enumerate(LABELS):
        lines.append("      %-17s %s" % (lab, " ".join("%5d" % v for v in m[i])))
    lines.append("      %-17s %s" % ("", " ".join("%5s" % l[:5] for l in LABELS)))
    return "\n".join(lines)


def sha32(P):
    return ca.sha256_array(np.ascontiguousarray(np.asarray(P, np.float32)))
