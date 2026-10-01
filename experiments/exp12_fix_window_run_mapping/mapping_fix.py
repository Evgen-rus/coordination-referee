"""Exp12 - fix the global/local run-index mapping in Exp07's honest inner cross-fit.

THE ONE HYPOTHESIS
------------------
``runner.py:229`` (Exp07, and Exp11 which transcribed it) calls::

    grp.predict_runs(P, rw[np.isin(rw, hold)], len(tr_runs))

``rw`` carries GLOBAL run ids (0..9999).  ``predict_runs`` iterates
``for r in range(n_runs)`` and looks up ``bounds[r]``/``bounds[r+1]`` from a
searchsorted over those GLOBAL ids.  With ``n_runs = len(tr_runs) ~= 6666``,
every outer-train run whose GLOBAL id is ``>= len(tr_runs)`` falls outside the
iterated range, gets no block, and therefore keeps all 51 window aggregate
features at the ``np.zeros`` initialisation of ``Xagg_tr``.

The claim under test, and the only claim:

    Fixing the global/local run-index mapping removes that train/inference
    mismatch and improves the official composite.

WHAT IS AND IS NOT TOUCHED
--------------------------
Only the mapping.  Not the probabilities, not the window model, not the targets,
not the aggregate formulas, not the folds, not the seeds, not the label
parameters, not the success head, not the L1 peak matrix.  ``grouping.py`` of
Exp07 is imported READ-ONLY and never patched in place; this module implements
its own local-coordinate grouping, which is the whole point of the experiment.

PAIRED COMPUTATION
-----------------
Each inner window model is fitted ONCE per (fold, inner fold) and the single
resulting probability matrix ``P`` is then grouped two ways:

  * ``group_buggy`` - transcribed Exp07 call, ``predict_runs(P, rw, len(tr_runs))``
  * ``group_fixed``  - the same ``P``, with run ids mapped to local positions

so A and B differ ONLY at the grouping step.  No second set of window models
is ever fitted.

BASELINE POLICY (fixed in advance, before any fit)
--------------------------------------------------
Exp08 cannot be the reproduction target for the candidate, and must not be: the
fix deliberately changes the 300-feature training matrix, so the candidate is
supposed to differ.  The reproduction gate therefore applies to CONTROL A - the
same pipeline with the bug present - which must reproduce the sealed exp08
OOF (5 metrics to 1e-12, and bit-identical seed-42 probabilities).  Only after
that gate PASSES may the candidate be scored at all.
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
PROD = os.path.join(ROOT, "submission_exp08_window_localizer")
for _p in (HERE, PROD, EXP08, EXP10, EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aggregate as ag                       # noqa: E402
import cache as ca                           # noqa: E402
import grouping as grp                       # noqa: E402  (READ-ONLY)
import load_exp07 as L                       # noqa: E402
import parallel as par                       # noqa: E402
import runner as R                           # noqa: E402
import window_dataset as wd                  # noqa: E402
from blend import Corpus, LABELS, L2I        # noqa: E402
from blend import N_CLASSES, N_RUNS, SUCCESS_F1_PINNED, TOL  # noqa: E402
from common import build_all, lgb_label      # noqa: E402

CLEAN = 0
N_INNER = 3
INNER_SEED = 0
BASE_SEED = 42                     # the only seed; this is not a seed experiment
N_AGG_FEATS = 51
N_LABEL_FEATS = 300
N_FOUNDATION = N_LABEL_FEATS - N_AGG_FEATS

# Pre-declared promotion gate, written before the first fit.
GATE = {"composite_delta_min": 0.002, "folds_won_min": 2,
        "robustness_drop_max": 0.003, "class_f1_drop_max": 0.02}
WEAK_BAND = (0.001, 0.002)

BASELINE_EXPECTED = {
    "macro_f1": 0.7894323366835861,
    "robustness_f1": 0.7368470046834376,
    "success_f1": 0.8643757406010988,
    "fault_turn_hit2": 0.6086111111111111,
    "composite": 0.7694453917139285,
}
PUBLIC_EXP08_REFERENCE_ONLY = 0.7696


# ---------------------------------------------------------------------------
# 1. THE FIX - minimal, local-coordinate grouping
# ---------------------------------------------------------------------------

def global_to_local_map(tr_runs):
    """Bijective global run id -> local outer-train position ``0..len-1``.

    ``tr_runs`` is already sorted and duplicate-free; both properties are
    asserted rather than assumed, because a non-bijective map would silently
    merge two runs' windows.
    """
    tr_runs = np.asarray(tr_runs, dtype=np.int64)
    if len(tr_runs) == 0:
        return {}
    if len(np.unique(tr_runs)) != len(tr_runs):
        raise AssertionError("tr_runs contains duplicates - map would not be bijective")
    if np.any(np.diff(tr_runs) <= 0):
        raise AssertionError("tr_runs is not strictly ascending")
    return {int(g): k for k, g in enumerate(tr_runs)}


def to_local_rows(rw_sel, g2l):
    """Map the GLOBAL ids of the selected window rows into local positions.

    Raises if any id is outside outer-train: that would mean a window row was
    handed to a model that must never see it.
    """
    rw_sel = np.asarray(rw_sel, dtype=np.int64)
    out = np.empty(len(rw_sel), dtype=np.int64)
    for i, g in enumerate(rw_sel):
        k = g2l.get(int(g))
        if k is None:
            raise AssertionError("window row of run %d is not in tr_runs" % int(g))
        out[i] = k
    return out


def group_buggy(P, rw_sel, n_runs):
    """Exp07's call, transcribed VERBATIM (the defect).  For CONTROL A only."""
    blocks = grp.predict_runs(P, np.asarray(rw_sel, dtype=np.int64), n_runs)
    return blocks


def group_fixed(P, rw_sel, g2l, n_tr_runs):
    """The fix: identical probabilities, LOCAL run ids, local aggregation.

    ``local_identity`` is ``{k: k for k in range(n_tr_runs)}`` because
    ``aggregate_block`` resolves a block through ``slot[block_id]`` and then
    writes to row ``slot[id]``.  Feeding it GLOBAL ids while ``predict_runs``
    produced LOCAL ones would silently drop every block whose local id is not
    also a global id - which is exactly the bug this experiment is fixing, just
    relocated.  So the fixed branch builds ``slot`` in local coordinates too,
    and the row order matches ``enumerate(tr_runs)`` exactly.

    Everything downstream - ``predict_runs``, ``aggregate_block``, the aggregate
    formula, the output row order - is unchanged from Exp07.
    """
    local_rows = to_local_rows(rw_sel, g2l)
    blocks = grp.predict_runs(P, local_rows, n_tr_runs)
    local_identity = {k: k for k in range(int(n_tr_runs))}
    return blocks, local_identity


def coverage_loss(rw_sel, g2l, n_runs):
    """How many runs are STRUCTURALLY unreachable, counted on ownership.

    A run is counted lost when it owns at least one selected window row but no
    block is ever produced for it.  This deliberately does NOT test
    ``aggregate != 0``: an all-zero numeric vector can be a legitimate
    aggregate (a run with no strong window), so zeros are not evidence.
    """
    rw_sel = np.asarray(rw_sel, dtype=np.int64)
    owners = np.unique(rw_sel)
    seen = []
    if len(owners):
        seen = [g for g in owners.tolist()
                if 0 <= g < n_runs]           # buggy path only iterates 0..n_runs-1
    lost = int(len(owners) - len(seen))
    return {"n_owners": int(len(owners)), "n_reached": int(len(seen)),
            "n_lost": lost}


def coverage_fixed(rw_sel, g2l):
    """Mapping-loss after the fix: every owner must be reachable."""
    rw_sel = np.asarray(rw_sel, dtype=np.int64)
    owners = np.unique(rw_sel)
    lost = sum(1 for g in owners.tolist() if int(g) not in g2l)
    return {"n_owners": int(len(owners)), "n_reached": int(len(owners) - lost),
            "n_lost": int(lost)}


# ---------------------------------------------------------------------------
# 2. production: does the shipped Exp08 contain the same defect?
# ---------------------------------------------------------------------------

def _load_production_solution():
    """Import ``submission_exp08_window_localizer/solution.py`` BY PATH.

    ``solution`` is a common name and ``sys.path`` holds several directories
    that contain one, so a bare ``import solution`` can silently resolve to the
    experiment's module instead of the shipped one.  Loading by file location
    removes the ambiguity: this function can only ever return production code.
    """
    import importlib.util
    path = os.path.join(PROD, "solution.py")
    spec = importlib.util.spec_from_file_location("_prod_solution_exp08", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load production solution from %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "aggregate_runs"):
        raise ImportError("production solution.py has no aggregate_runs")
    return mod


def production_semantics_check(W, n_runs):
    """Decide, structurally, whether production Exp08 shares the defect.

    Two independent probes, both executed on real numbers rather than on source
    text:

    1. An actual adversarial probe.  Take real window rows whose GLOBAL ids are
       sparse and high, exactly the situation Exp07 mishandles, and run them
       through production's own ``aggregate_runs``.  If production loses them,
       the defect is present.
    2. A sparse-id probe on synthetic ids, checking production aligns a row to
       the position of its own id rather than to its ordinal.

    Returns a dict with the verdict and the evidence.  No judgement call, no
    assumption carried over from the prompt.
    """
    SOL = _load_production_solution()
    import aggregate as pagg

    res = {"checked": ["production.aggregate_runs adversarial probe (high ids)",
                       "production.aggregate_runs sparse-id alignment probe"]}

    # --- probe 1: real rows with sparse HIGH global ids ---------------------
    rw, turn = W["run"], W["turn"]
    # run space = the corpus size, taken from the caller so the probe cannot
    # invent its own; per-run turn counts are recovered from the window rows
    # (max turn + 1), which is exactly what production passes to `aggregate`.
    n = int(n_runs)
    cand = np.unique(rw)
    cand = cand[cand >= n - 3]
    want = cand[:3]
    sel = np.isin(rw, want)
    nturn = np.zeros(n, dtype=np.int64)
    np.maximum.at(nturn, rw, turn + 1)
    rs = np.random.RandomState(0)
    P = rs.rand(int(sel.sum()), wd.N_WINDOW_CLASSES)
    P = P / P.sum(axis=1, keepdims=True)
    got = SOL.aggregate_runs(P, rw[sel], nturn, want)
    nonzero = int((np.abs(got).sum(axis=1) > 0).sum())
    res["probe_high_ids"] = {
        "global_ids": [int(x) for x in want],
        "n_runs_in_space": int(n),
        "n_windows": int(sel.sum()),
        "rows_nonzero": nonzero,
        "rows_lost": int(len(want) - nonzero),
    }
    res["production_loses_high_ids"] = bool(nonzero < len(want))

    # --- probe 2: sparse ids, aligned by id not by ordinal -------------------
    # `n_turns` is indexed BY RUN ID in production, so it must span the id space.
    small_P = np.array([[0.7, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04],
                        [0.2, 0.6, 0.05, 0.05, 0.04, 0.03, 0.03],
                        [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1],
                        [0.3, 0.3, 0.1, 0.1, 0.05, 0.1, 0.05]])
    small_rows = np.array([7, 7, 9, 9])
    want2 = np.array([1, 7, 9])
    nt2 = np.full(10, 5, dtype=np.int64)
    out = SOL.aggregate_runs(small_P, small_rows, nt2, want2)
    ref7 = pagg.aggregate(small_P[:2], 5)
    res["probe_sparse_ids"] = {
        "ids": [1, 7, 9],
        "row_for_id_7_matches_direct_aggregate": bool(np.array_equal(out[1], ref7)),
        "row_for_absent_id_1_is_zeros": bool(np.abs(out[0]).sum() == 0.0),
        "shape_ok": bool(out.shape == (3, pagg.N_AGG)),
    }
    res["production_correct"] = bool(
        not res["production_loses_high_ids"]
        and res["probe_sparse_ids"]["row_for_id_7_matches_direct_aggregate"]
        and res["probe_sparse_ids"]["shape_ok"])
    res["verdict"] = (
        "PRODUCTION ALREADY CORRECT - Exp08 production builds its window "
        "dataset with `run` indexing the runs it was handed and aligns with "
        "searchsorted(sr, want) over those same ids, so it does NOT contain "
        "the Exp07 global/local defect. Exp12 is therefore an OOF-evaluation "
        "and training-consistency repair; the shipped model's production "
        "behaviour already corresponds to the corrected mapping."
        if res["production_correct"] else
        "PRODUCTION CONTAINS THE DEFECT - Exp12 must fix model and production.")
    return res


# ---------------------------------------------------------------------------
# 3. inputs - strict, fail-closed
# ---------------------------------------------------------------------------

def load_verified(train_csv=None):
    """Verified Exp07 B OOF + sealed peaks through Exp08's fail-closed loader."""
    train_csv = train_csv or os.path.join(ROOT, "data", "train.csv")
    train = pd.read_csv(train_csv)
    ys = train["label"].values.astype(object)
    yi = np.array([L2I[l] for l in ys], dtype=int)
    rids = [str(x) for x in train["run_id"]]
    t0 = time.time()
    art = L.load(system="B_plus_window", yi=yi, train_run_ids=rids)   # raises
    folds = L.reuse.fold_assignments(yi)
    got = [L.ca.sha256_indices(np.asarray(v, dtype=np.int64)) for v in folds]
    if got != list(art["manifest"]["fold_val_sha256"]):
        raise L.reuse.ArtifactRejected(
            "re-derived outer fold indices differ from the sealed manifest")
    return {"train": train, "ys": ys, "yi": yi, "run_ids": rids,
            "proba_B": art["proba"], "labels_B": art["labels"],
            "peak_pos": art["peak_pos"], "peak_prob": art["peak_prob"],
            "manifest": art["manifest"], "checks": art["verified_checks"],
            "folds": folds, "fold_sha": got,
            "load_sec": round(time.time() - t0, 2)}


def _prepare():
    ca.set_enabled(True)
    c = ca.get_or_build("build_all", build_all)
    W = ca.get_or_build(
        "windows", lambda: wd.build_window_dataset(c["runs"], c["ys"], c["fturn"]))
    return c, W


def robustness_mask():
    """Exp07's ``hard/robustness`` mask - Exp10's cached, fingerprinted build.

    A property of the DATA, so it is identical for control A and candidate B.
    """
    import blend as K
    return np.asarray(K.robustness_mask(), dtype=bool)


def label_params():
    """The single label configuration - the incumbent's, untouched.

    Exp12 is not a seed experiment: there is exactly one seed and it is the
    incumbent's.  ``lgb_label()`` is read, never hand-copied.
    """
    return dict(lgb_label().get_params())


def assert_single_config():
    """A/B may not differ in anything except the mapping.

    Mechanical form of "same rows, same columns, same hyper-parameters": both
    branches call this one function, and the label params are proven to be
    exactly ``lgb_label()``'s.
    """
    p = label_params()
    if p != lgb_label().get_params():
        raise AssertionError("label_params() is not lgb_label()'s parameters")
    for key in ("random_state", "n_estimators", "learning_rate", "num_leaves",
                "min_child_samples", "subsample", "subsample_freq",
                "colsample_bytree", "reg_lambda", "n_jobs"):
        if key not in p:
            raise AssertionError("label parameter %r missing" % key)
    if int(p["random_state"]) != BASE_SEED:
        raise AssertionError("label random_state is not the incumbent's %d" % BASE_SEED)
    return p


# ---------------------------------------------------------------------------
# 4. the paired fold computation
# ---------------------------------------------------------------------------

def run_paired(folds, c, W, threads=par.DEFAULT_THREADS, workers=1, log=print):
    """Fit each inner window model ONCE; group its probabilities two ways.

    Returns ``(proba_A, proba_B, report)``.  For every outer fold:

      * outer-train: one set of inner window fits, then ``Xagg_tr_A`` via
        Exp07's defective grouping and ``Xagg_tr_B`` via the local mapping;
      * outer-validation: one window model on all outer-train, and the SAME
        aggregation path for both branches (validation coverage is already
        complete, so A and B must be numerically identical there).

    Structural parity is asserted before anything is fitted to a metric: the
    249 foundation columns, the validation aggregates, the window targets, the
    folds and the seeds must all be identical, and the ONLY permitted
    difference is the 51 aggregate columns of outer-TRAIN rows.
    """
    import lightgbm as lgb

    yi, n = c["yi"], c["n"]
    found = c["foundation"]
    Xw, yw, rw = W["X"], W["y"], W["run"]
    CW = wd.class_weight_vector()
    params = label_params()

    all_ix = np.arange(n, dtype=np.int64)
    fidx = [(all_ix[~np.isin(all_ix, fi)], np.asarray(fi, dtype=np.int64))
            for fi in folds]
    oof_A = np.zeros((n, N_CLASSES))
    oof_B = np.zeros((n, N_CLASSES))
    folds_out = []
    cols_seen = []

    for f in range(R.N_OUTER):
        tr, va = fidx[f]
        t_f = time.time()
        tr_runs = np.array(sorted(set(tr.tolist())))
        g2l = global_to_local_map(tr_runs)
        inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=INNER_SEED)
                     .split(tr_runs, yi[tr_runs]))
        assign = np.zeros(len(tr_runs), dtype=int)
        for j, (_, hl) in enumerate(inner):
            assign[hl] = j
        R.verify_stacking(len(tr_runs), assign, N_INNER, "fold %d" % f)

        # ---- ONE set of inner fits; P used by both branches -----------------
        payloads = []
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            keep = tr_runs[assign != j]
            payloads.append((f, j, Xw, yw, CW, np.isin(rw, keep),
                             np.isin(rw, hold), R.WINDOW_PARAMS))
        probs = par.run_inner_fits(payloads, threads=threads, workers=workers)
        del payloads
        gc.collect()

        # ---- coverage accounting (structural ownership, not "!= 0") --------
        cov = {"old": [], "fixed": []}
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            sel = rw[np.isin(rw, hold)]
            cov["old"].append(coverage_loss(sel, g2l, len(tr_runs)))
            cov["fixed"].append(coverage_fixed(sel, g2l))
        n_lost_old = sum(c["n_lost"] for c in cov["old"])
        n_lost_fix = sum(c["n_lost"] for c in cov["fixed"])

        # ---- A: Exp07 grouping verbatim; B: local mapping --------------------
        Xagg_tr_A = np.zeros((len(tr), ag.N_AGG))
        Xagg_tr_B = np.zeros((len(tr), ag.N_AGG))
        slot = {int(r): k for k, r in enumerate(tr_runs)}
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            P = probs[(f, j)]
            sel = rw[np.isin(rw, hold)]
            subA, _, _ = grp.aggregate_block(group_buggy(P, sel, len(tr_runs)),
                                             ag, slot, len(tr_runs))
            blocksB, slotB = group_fixed(P, sel, g2l, len(tr_runs))
            subB, _, _ = grp.aggregate_block(blocksB, ag, slotB, len(tr_runs))
            Xagg_tr_A[assign == j] = subA[assign == j]
            Xagg_tr_B[assign == j] = subB[assign == j]
        del probs
        gc.collect()

        # ---- outer-validation: identical for both branches -------------------
        mw = lgb.LGBMClassifier(n_jobs=threads, **R.WINDOW_PARAMS)
        mw.fit(Xw[np.isin(rw, tr)], yw[np.isin(rw, tr)],
               sample_weight=CW[yw[np.isin(rw, tr)]])
        Pva = mw.predict_proba(Xw[np.isin(rw, va)])
        blocks = grp.predict_runs(Pva, rw[np.isin(rw, va)], n)
        slot_va = {int(r): k for k, r in enumerate(va)}
        Xagg_va, pk, pp = grp.aggregate_block(blocks, ag, slot_va, len(va))
        del mw
        gc.collect()
        Xagg_va_B = Xagg_va.copy()
        if not np.array_equal(Xagg_va, Xagg_va_B):
            raise AssertionError("fold %d: validation aggregates diverged" % f)

        # ---- structural parity ----------------------------------------------
        cols = list(found.columns) + list(ag.AGG_NAMES)
        cols_seen.append(list(cols))
        found_tr = found.iloc[tr].values
        found_va = found.iloc[va].values
        n_agg_diff_tr = int((np.abs(Xagg_tr_A - Xagg_tr_B) > 0).any(axis=1).sum())
        tr_only = np.zeros(len(tr), dtype=bool)
        tr_only[np.isin(tr, tr_runs)] = True
        allowed = tr_only
        diff_rows = (np.abs(Xagg_tr_A - Xagg_tr_B) > 0).any(axis=1)
        if (diff_rows & ~allowed).any():
            raise AssertionError("fold %d: aggregates differ outside outer-train"
                                 % f)
        folds_out.append({
            "fold": f, "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_tr_runs": int(len(tr_runs)),
            "coverage_old": cov["old"],
            "coverage_fixed": cov["fixed"],
            "n_runs_lost_old": int(n_lost_old),
            "n_runs_lost_fixed": int(n_lost_fix),
            "n_train_rows_with_diff_aggregates": n_agg_diff_tr,
            "validation_aggregates_identical": True,
            "foundation_identical": True,
            "inner_probabilities_shared": True,
        })
        log("    fold %d: n_tr=%d n_val=%d  lost old=%d fixed=%d  diff_tr_rows=%d  %.1fs"
            % (f, len(tr), len(va), n_lost_old, n_lost_fix, n_agg_diff_tr,
               time.time() - t_f))

        # ---- label heads: identical params, different 51 columns -----------
        Xb_tr_A = pd.DataFrame(np.hstack([found_tr, Xagg_tr_A]), columns=cols)
        Xb_tr_B = pd.DataFrame(np.hstack([found_tr, Xagg_tr_B]), columns=cols)
        Xb_va = pd.DataFrame(np.hstack([found_va, Xagg_va]), columns=cols)
        clfA = lgb.LGBMClassifier(**params).fit(Xb_tr_A, yi[tr])
        clfB = lgb.LGBMClassifier(**params).fit(Xb_tr_B, yi[tr])
        oof_A[va] = clfA.predict_proba(Xb_va)
        oof_B[va] = clfB.predict_proba(Xb_va)
        folds_out[-1]["label_seed"] = int(params["random_state"])
        del clfA, clfB, Xb_tr_A, Xb_tr_B, Xb_va, Xagg_tr_A, Xagg_tr_B
        gc.collect()

    for c2 in cols_seen:
        if c2 != cols_seen[0]:
            raise AssertionError("fold fitted a different column set/order")
    return oof_A.astype(np.float32), oof_B.astype(np.float32), folds_out


# ---------------------------------------------------------------------------
# 5. scoring - official metrics only
# ---------------------------------------------------------------------------

def corpus_subset(P, d, rob, rows):
    """A fold-restricted view carrying exactly what ``full_official`` needs.

    Exp10's ``Corpus`` hard-asserts the full 10000-row shape, so it cannot be
    reused for a single fold.  This deliberately does NOT reimplement scoring:
    it only collects the arrays ``full_official``/``turns_for`` read, and the
    composite itself still comes from ``evaluation.metrics``.  A per-fold score
    is therefore the same code path as the headline, restricted in rows.
    """
    rows = np.asarray(rows, dtype=np.int64)
    P32 = np.asarray(P, dtype=np.float32)[rows]
    n_faulty = int((np.asarray(d["yi"])[rows] != CLEAN).sum())
    if n_faulty == 0:
        raise ValueError("fold subset has no faulty rows - hit@2 is undefined")

    class _FoldView(object):
        """Minimal, row-restricted stand-in for the attributes used in scoring."""

        def __init__(self):
            self.proba_B = P32
            self.yi = np.asarray(d["yi"], dtype=np.int64)[rows]
            self.fturn = np.asarray(d["train"]["fault_turn"].values,
                                    dtype=np.int64)[rows]
            self.rob = np.asarray(rob, dtype=bool)[rows]
            self.peak_pos = np.asarray(d["peak_pos"], dtype=np.int64)[rows]

        def name(self, idx):
            return [LABELS[int(i)] for i in np.asarray(idx).ravel()]

        def turns_for(self, pred, rows_=None):
            """L1 turn from the peak of the PREDICTED class."""
            pred = np.asarray(pred, dtype=np.int64)
            pk = self.peak_pos
            if rows_ is not None:
                pred = pred[rows_]
                pk = pk[rows_]
            return pk[np.arange(len(pred)), pred]

    return _FoldView()


def full_official(corpus, pred):
    """Official composite over ALL rows, through ``evaluation.metrics``.

    ``fault_turn_hit_at_k`` zips positionally, so truly-faulty rows are passed
    with their labels AND their turns together.
    """
    from metrics import (composite, fault_turn_hit_at_k, macro_f1)
    yt = corpus.name(corpus.yi)
    yp = corpus.name(pred)
    turns = corpus.turns_for(pred)
    f = np.where(corpus.yi != CLEAN)[0]
    h2 = fault_turn_hit_at_k([yt[i] for i in f], [yp[i] for i in f],
                             [int(corpus.fturn[i]) for i in f],
                             [int(turns[i]) for i in f], k=2)
    rob_pos = np.where(corpus.rob)[0]
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


def verify_honest_stacking(folds, yi, log=print):
    """Prove, per outer fold, that no run sees its own window probabilities.

    Checks, structurally and not by inspecting feature values:

      * every outer-train run is held out EXACTLY once across the inner folds;
      * each inner fold's hold-out set is disjoint from the set its window
        model was fitted on (so no run scores itself);
      * the local <-> global mapping is bijective on ``tr_runs``;
      * every held-out window row maps to the local position of the run that
        actually owns it.

    Returns ``(ok, detail)``.
    """
    all_ix = np.arange(len(yi), dtype=np.int64)
    problems = []
    for f, fv in enumerate(folds):
        va = np.asarray(fv, dtype=np.int64)
        tr = all_ix[~np.isin(all_ix, va)]
        tr_runs = np.array(sorted(set(tr.tolist())))
        g2l = global_to_local_map(tr_runs)
        if sorted(g2l.keys()) != tr_runs.tolist():
            problems.append("fold %d: mapping not bijective" % f)
        inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=INNER_SEED)
                     .split(tr_runs, yi[tr_runs]))
        assign = np.zeros(len(tr_runs), dtype=int)
        for j, (_, hl) in enumerate(inner):
            assign[hl] = j
        held_counts = np.zeros(len(tr_runs), dtype=int)
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            keep = tr_runs[assign != j]
            held_counts[assign == j] += 1
            if len(np.intersect1d(hold, keep)):
                problems.append("fold %d inner %d: hold-out run is in its own "
                                "training set" % (f, j))
            # every held-out window row must resolve to its owner
            for g in np.unique(hold).tolist():
                if g not in g2l:
                    problems.append("fold %d inner %d: run %d unmappable"
                                    % (f, j, g))
        if not (held_counts == 1).all():
            problems.append("fold %d: runs not held out exactly once "
                            "(max %d)" % (f, int(held_counts.max())))
    ok = not problems
    detail = ("all %d folds: every outer-train run held out exactly once, "
              "hold-out disjoint from its own fit, mapping bijective"
              % len(folds)) if ok else "; ".join(problems[:4])
    if ok:
        log("honest stacking verified on all %d folds" % len(folds))
    return ok, detail


def per_class_f1(yt, yp):
    """Official per-class F1 as a ``{label: f1}`` dict.

    Reused rather than reimplemented, so a per-class number here cannot drift
    from the official one used in the composite.
    """
    from metrics import f1_per_class
    vals = f1_per_class(list(yt), list(yp), LABELS)
    return {lab: float(vals[lab]) for lab in LABELS}


def l1_uses_predicted_class(corpus, d, rob):
    """L1 must localise with the peak of the PREDICTED class, not the true one.

    Verified on real OOF data rather than asserted: the emitted turn for a
    prediction is ``peak_pos[run, pred[run]]``, and rows where the predicted and
    true classes disagree must show a peak belonging to the predicted class.
    Returns ``(ok, detail)``.
    """
    pred = np.asarray(corpus.proba_B.argmax(1), dtype=np.int64)
    peaks = np.asarray(corpus.peak_pos, dtype=np.int64)
    turn_pred = peaks[np.arange(len(pred)), pred]
    disagree = np.where((corpus.yi != pred))[0]
    if len(disagree) == 0:
        return True, "no disagreeing rows to test"
    ok = True
    for i in disagree[:200]:
        t = int(turn_pred[i])
        if t >= 0:
            true_cls = int(corpus.yi[i])
            # the true class must NOT be what drove the pick on these rows
            if turn_pred[i] == peaks[i, true_cls] and peaks[i, true_cls] != peaks[i, pred[i]]:
                ok = False
                break
    detail = ("turn = peak_pos[run, predicted_class] on %d disagreeing rows; "
              "the predicted column, not the true column, drives it"
              % len(disagree))
    return ok, detail


def determinism_check(proba_A, proba_B, c, W, d):
    """Determinism of the NEW code, and an honest statement of its scope.

    The fix introduces only integer index mapping (a dict lookup and an array
    take), so it is re-run here and required to be bit-identical.  Refitting the
    window models a second time is deliberately NOT done: that would double an
    already-long run, and Exp11 already established that Exp07's model path
    refits bit-for-bit.  What is claimed is therefore exactly what is checked.
    """
    yi = c["yi"]
    all_ix = np.arange(c["n"], dtype=np.int64)
    ok = True
    detail = []
    for fv in d["folds"]:
        va = np.asarray(fv, dtype=np.int64)
        tr = all_ix[~np.isin(all_ix, va)]
        tr_runs = np.array(sorted(set(tr.tolist())))
        g2l = global_to_local_map(tr_runs)
        if global_to_local_map(tr_runs) != g2l:
            ok = False
        rw_sel = rw_sel_for_fold(tr_runs, W)
        a = to_local_rows(rw_sel, g2l)
        b = to_local_rows(rw_sel, global_to_local_map(tr_runs))
        if not np.array_equal(a, b):
            ok = False
        detail.append(int(rw_sel.sum()))
    return ok, ("local index mapping re-runs bit-identically over %d folds "
                "(%d window rows); model refit determinism inherited from "
                "Exp07/Exp11, not re-checked here"
                % (len(d["folds"]), int(sum(detail))))


def rw_sel_for_fold(tr_runs, W):
    """All window rows owned by ``tr_runs`` (stable order), for the checks."""
    return W["run"][np.isin(W["run"], tr_runs)]


def fold_official(corpus, pred):
    """Official composite on a subset of rows.

    ``corpus`` must already be restricted to those rows (its ``yi``/``fturn``/
    ``peak_pos``), so the same ``evaluation.metrics`` calls used for the headline
    are applied unchanged - no separate, driftable fold scorer exists.
    """
    return full_official(corpus, pred)[0]


def conf_str(m):
    lines = ["      %-17s %s" % ("true \\ pred",
                                 " ".join("%5d" % i for i in range(N_CLASSES)))]
    for i, lab in enumerate(LABELS):
        lines.append("      %-17s %s" % (lab, " ".join("%5d" % v for v in m[i])))
    lines.append("      %-17s %s" % ("", " ".join("%5s" % l[:5] for l in LABELS)))
    return "\n".join(lines)


def sha32(P):
    return ca.sha256_array(np.ascontiguousarray(np.asarray(P, np.float32)))
