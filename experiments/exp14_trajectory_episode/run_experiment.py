"""Exp14 - explicit bounded episode representation for `runaway_loop`.

THE ONE HYPOTHESIS
------------------
    Explicit bounded repeat-episode representation improves classification of
    `runaway_loop`, because the current 312-feature Exp13 representation stores
    only order-free repeat counts and adjacency streaks, but loses episode
    identity, chronology, interleaving, agent-ring structure and progress
    inside the episode.

WHAT IS AND IS NOT TOUCHED
--------------------------
Touched: ONE appended block of 16 fold-independent columns on the label head,
built from observable messages/artifacts/shared_state only.

NOT touched, deliberately:
  * no semantic model, no TF-IDF, no embeddings
  * no threshold, class weight, hyperparameter or seed search
  * no ensemble, no calibration, no decision-rule change
  * no L1 / fault_turn change - episode onset is NEVER used as a localizer,
    because the audit measured episode-start hit@2 at 0.252 against the
    incumbent L1 conditional runaway_loop hit@2 of 0.831
  * no new normaliser - the detector reuses production `features._norm`
  * no assignment-delivery features, no goal_drift work, no success work
  * the frozen 12 wg_* block, the 300 corrected columns, the window fits,
    the folds, the seed and the success head are all reused verbatim

BASELINE POLICY (fixed before any fit)
--------------------------------------
    CONTROL A   = Exp13 production semantics: the 300 corrected columns + the
                  frozen 12 wg_* columns  (= Exp13 CANDIDATE D, verbatim)
    CANDIDATE B = the IDENTICAL 312 columns + the frozen 16 ep_* columns

A must reproduce the committed Exp13 corrected OOF bit for bit before B is
scored at all. Both arms are produced from ONE shared corrected fold matrix, so
they differ in exactly 16 appended columns and nothing else - no second window
fit, no second fold, no second seed.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP13 = os.path.join(ROOT, "experiments", "exp13_wait_dependency_graph")
EXP12 = os.path.join(ROOT, "experiments", "exp12_fix_window_run_mapping")
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
EXP11 = os.path.join(ROOT, "experiments", "exp11_label_seed_ensemble")
for _p in (EXP13, EXP12, EXP11,
           os.path.join(ROOT, "experiments", "exp10_ab_probability_blend"),
           os.path.join(ROOT, "experiments", "exp08_window_localizer"),
           EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cache as ca                                       # noqa: E402
import ensemble as ENS                                   # noqa: E402
import lightgbm as lgb                                   # noqa: E402
from blend import LABELS, L2I                            # noqa: E402
from blend import N_CLASSES, SUCCESS_F1_PINNED, TOL      # noqa: E402
from metrics import f1_per_class                         # noqa: E402

# The frozen 12 wg_* block, byte-identical to production wait_graph.py.
_spec = importlib.util.spec_from_file_location(
    "exp13_wait_features", os.path.join(EXP13, "features.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

# Exp13's own corrected fold-matrix generator. Imported, never executed: every
# np.save / json.dump in that file lives inside main(), so importing it writes
# nothing. Reusing the generator is what makes A and B provably identical
# everywhere except the 16 appended columns.
_spec2 = importlib.util.spec_from_file_location(
    "exp13_production_confirmation",
    os.path.join(EXP13, "production_confirmation.py"))
P13 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(P13)

# Production Exp13 feature semantics: the normaliser the detector must use.
_spec3 = importlib.util.spec_from_file_location(
    "exp14_production_features",
    os.path.join(ROOT, "submission_exp13_wait_dependency_graph", "features.py"))
FT = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(FT)

# This experiment's own frozen detector + block.
_spec4 = importlib.util.spec_from_file_location(
    "exp14_episode_features", os.path.join(HERE, "features.py"))
E = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(E)

CLEAN = 0
RL = L2I["runaway_loop"]
N_LABEL = 300                       # 249 cached foundation + 51 aggregates
N_WG = W.N_FEATURES                 # 12, frozen in Exp13
N_EP = E.N_FEATURES                 # 16, frozen here
N_BASE = N_LABEL + N_WG             # 312
N_ALL = N_BASE + N_EP               # 328

# ---------------------------------------------------------------------------
# Reproduction targets: Exp13 CANDIDATE D, corrected production semantics.
# Read from the committed artifact rather than hand-copied.
# ---------------------------------------------------------------------------
with open(os.path.join(EXP13, "results_production.json"), encoding="utf-8") as _fh:
    _P13 = json.load(_fh)
EXP13_METRICS = {k: float(_P13["metrics_D"][k]) for k in
                 ("macro_f1", "robustness_f1", "success_f1",
                  "fault_turn_hit2", "composite")}

# sha256 of the committed corrected OOF, computed from the artifact itself.
def sha32(P):
    return ca.sha256_array(np.ascontiguousarray(np.asarray(P, np.float32)))


EXP13_SHA_D = sha32(np.load(os.path.join(
    EXP13, "oof_D_corrected_wg.npy")))

# ---------------------------------------------------------------------------
# PREDECLARED GATE - written down before the first fit of candidate B.
# Not retuned after any number is seen.
# ---------------------------------------------------------------------------
GATE = {
    "composite_delta_min": 0.002,
    "composite_folds_won_min": 2,
    "runaway_loop_f1_delta_min": 0.015,
    "runaway_loop_folds_won_min": 2,
    "robustness_drop_max": 0.003,
    "class_f1_drop_max": 0.02,
}
WEAK_BAND = (0.001, 0.002)

OUT = []
RES = {}


def log(m=""):
    print(m)
    OUT.append(str(m))


def bar(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def hdr(t):
    log()
    log("-- %s" % t)


def prf(yi, pred, cls):
    tp = int(np.sum((yi == cls) & (pred == cls)))
    fp = int(np.sum((yi != cls) & (pred == cls)))
    fn = int(np.sum((yi == cls) & (pred != cls)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (fn + tp) if fn + tp else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f}


def per_class_f1(yi, pred):
    return {LABELS[c]: prf(yi, pred, c)["f1"] for c in range(len(LABELS))}


def official_full(P, d, rob, pred):
    """Exp11's scoring path, unchanged."""
    c = ENS.corpus_for(P, d, rob, labels=np.asarray(pred, np.int64))
    m, _turns = ENS.full_official(c, np.asarray(pred, np.int64))
    return m


def official_subset(d, rob, rows, pred_rows):
    """Official score over one fold, through the same metric code path.

    L1 turns come from the Corpus, whose turns_for() already forces -1 for a
    clean prediction. A raw peak_pos[:, 0] lookup would instead return the
    BACKGROUND peak, which is not a turn and would silently inflate hit@2.
    """
    rows = np.asarray(rows, np.int64)
    pr = np.asarray(pred_rows, np.int64)
    yt = [LABELS[int(d["yi"][i])] for i in rows]
    yp = [LABELS[int(i)] for i in pr]
    fturn = d["train"]["fault_turn"].values
    corpus = ENS.corpus_for(np.zeros((len(d["yi"]), N_CLASSES), np.float32),
                            d, rob)
    turns = corpus.turns_for(pr, rows)
    faulty = np.where(d["yi"][rows] != CLEAN)[0]
    from metrics import composite, fault_turn_hit_at_k, macro_f1
    h2 = fault_turn_hit_at_k([yt[i] for i in faulty], [yp[i] for i in faulty],
                             [int(fturn[rows][i]) for i in faulty],
                             [int(turns[i]) for i in faulty], k=2)
    rp = np.where(rob[rows])[0]
    rf = macro_f1([yt[i] for i in rp], [yp[i] for i in rp], LABELS) if len(rp) else 0.0
    mf = macro_f1(yt, yp, LABELS)
    return {"macro_f1": mf, "robustness_f1": rf,
            "success_f1": SUCCESS_F1_PINNED, "fault_turn_hit2": h2,
            "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                    "success_f1": SUCCESS_F1_PINNED,
                                    "fault_turn_hit2": h2})}


def leakage_probe(runs, n=400):
    """Target perturbation: replacing every target must not move one number."""
    import copy
    base = E.build_matrix(runs[:n], FT._norm)
    pert = []
    for i, r in enumerate(runs[:n]):
        r2 = copy.deepcopy(r)
        r2["label"] = ["runaway_loop", "clean", "goal_drift"][i % 3]
        r2["success"] = 1 - (i % 2)
        r2["fault_turn"] = (i * 7) % 11
        pert.append(r2)
    other = E.build_matrix(pert, FT._norm)
    return {"identical": bool(np.array_equal(base, other)),
            "n_rows_probed": int(n),
            "max_abs_diff": float(np.max(np.abs(base - other))) if base.size else 0.0}


def duplicate_check(Xnew, Xold, names):
    """Flag any ep_* column numerically identical to an existing column."""
    rows = []
    for j, nm in enumerate(names):
        v = Xnew[:, j]
        dup = [int(i) for i in range(Xold.shape[1])
               if np.array_equal(v, Xold[:, i])]
        rows.append({"feature": nm, "n_unique": int(len(np.unique(v))),
                     "identical_to_existing": bool(dup),
                     "identical_cols": dup})
    return rows


def corr_diagnostic(Xnew, Xold, names, old_names, top=5):
    """max |r| per new column against every existing one. Report only."""
    from scipy.stats import pearsonr, spearmanr
    out = []
    for j, nm in enumerate(names):
        v = Xnew[:, j]
        if np.std(v) == 0:
            out.append({"feature": nm, "max_abs_r": 0.0, "max_abs_rho": 0.0,
                        "top_partners": []})
            continue
        pr, rho = [], []
        for i in range(Xold.shape[1]):
            u = Xold[:, i]
            if np.std(u) == 0:
                continue
            pr.append(abs(float(pearsonr(v, u)[0])))
            rho.append(abs(float(spearmanr(v, u)[0])))
        order = np.argsort(-np.asarray(pr))
        out.append({"feature": nm,
                    "max_abs_r": float(np.max(pr)),
                    "max_abs_rho": float(np.max(rho)),
                    "top_partners": [{"feature": old_names[int(i)],
                                      "abs_r": float(pr[int(i)])}
                                     for i in order[:top]]})
    return out


def write_results(obj):
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=float)
    with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")


def main():
    t0 = time.time()
    bar("EXP14 - EXPLICIT BOUNDED EPISODE REPRESENTATION")
    log("CONTROL A   = Exp13 corrected 300 + frozen 12 wg_*  (= Exp13 Candidate D)")
    log("CANDIDATE B = the SAME 312 columns + the frozen 16 ep_* columns")
    log("frozen: detector, features, params, seeds, folds, thresholds, gate")
    log("NOT touched: L1/fault_turn, thresholds, class weights, seeds, ensemble,")
    log("             normaliser, success head, wg_* block, window fits")

    # ---------------------------------------------------------------- inputs
    hdr("1. VERIFIED INPUTS")
    d = ENS.load_verified()
    yi = d["yi"]
    folds = [np.asarray(v, np.int64) for v in d["folds"]]
    n = len(yi)
    rob = np.asarray(ENS.robustness_mask(), dtype=bool)
    log("   runs=%d folds=%d checks=%d robust=%d"
        % (n, len(folds), len(d["checks"]), int(rob.sum())))

    c, Wd = ENS._prepare()
    log("   foundation %s  windows %s" % (c["foundation"].shape, Wd["X"].shape))

    hdr("2. FROZEN BLOCKS")
    Xwg = W.build_matrix(c["runs"])
    assert Xwg.shape == (n, N_WG), Xwg.shape
    Xep = E.build_matrix(c["runs"], FT._norm)
    assert Xep.shape == (n, N_EP), Xep.shape
    log("   wg_* %s (frozen in Exp13, untouched)" % (Xwg.shape,))
    log("   ep_* %s (frozen here)" % (Xep.shape,))
    log("   episode constants: EP_MIN_LEN=%d EP_GAP_MAX=%d EP_CYCLE_MIN=%d"
        % (E.EP_MIN_LEN, E.EP_GAP_MAX, E.EP_CYCLE_MIN))
    fires = int((Xep[:, 1] > 0).sum())
    log("   runs with >=1 episode: %d / %d (%.1f%%)" % (fires, n, 100.0 * fires / n))

    # ---------------------------------------------------------------- leakage
    hdr("3. LEAKAGE / DUPLICATION (before any fit)")
    probe = leakage_probe(c["runs"], n=400)
    log("   target perturbation: identical=%s max|d|=%.1e"
        % (probe["identical"], probe["max_abs_diff"]))
    if not probe["identical"]:
        log("   STOP: the episode block moved under target perturbation.")
        RES["decision"] = "STOP_LEAKAGE"
        write_results(dict(RES, leakage_probe=probe))
        return 1

    base_names = list(c["foundation"].columns) + list(
        P13.ag.AGG_NAMES) + list(W.FEATURE_NAMES)
    assert len(base_names) == N_BASE, len(base_names)

    # a fold-independent view of the current representation for the r audit
    fold_indep = np.hstack([c["foundation"].values, Xwg])
    assert fold_indep.shape == (n, N_LABEL - 51 + N_WG), fold_indep.shape
    n_fi = fold_indep.shape[1]
    log("   fold-independent existing columns: %d "
        "(249 foundation + %d wg_*)" % (n_fi, N_WG))

    dups = duplicate_check(Xep, fold_indep, list(E.FEATURE_NAMES))
    dropped = [d for d in dups if d["identical_to_existing"]]
    log("   exact duplicates against existing columns: %d"
        % len(dropped))
    for d_ in dropped:
        log("      DROP %s (identical to col %s)" % (d_["feature"],
                                                    d_["identical_cols"]))
    corr = corr_diagnostic(Xep, fold_indep, list(E.FEATURE_NAMES),
                           base_names[:n_fi])

    log()
    log("   max |r| of each ep_* against the %d existing fold-independent cols"
        % n_fi)
    for c_ in corr:
        tp = c_["top_partners"][0]["feature"] if c_["top_partners"] else "-"
        log("      %-34s max|r| %.4f  max|rho| %.4f  nearest %s"
            % (c_["feature"], c_["max_abs_r"], c_["max_abs_rho"], tp))

    keep = [i for i, c_ in enumerate(dups)
            if not c_["identical_to_existing"]]
    dropped_names = [dups[i]["feature"] for i in range(len(dups)) if i not in keep]
    Xep_used = Xep[:, keep] if len(keep) != N_EP else Xep
    names_used = [E.FEATURE_NAMES[i] for i in keep]
    log()
    log("   feature block after the duplicate rule: %d columns (%s)"
        % (len(names_used), ",".join(dropped_names) if dropped_names else
           "none dropped"))

    # ------------------------------------------------------------------- CV
    hdr("4. PAIRED CV - one shared corrected matrix per fold")

    # TWO PHASES, ON PURPOSE.
    #
    # Phase 1 runs every corrected window fit back to back and keeps the
    # matrices. Phase 2 fits A and B. Interleaving the two (fitting A/B for
    # fold f, then asking the generator for fold f+1) was measured to break
    # the reproduction gate: exp07/parallel.py documents that the window model
    # at n_jobs=-1 is bit-sensitive because the 51 aggregates are built from
    # argmax-derived positions, so 49/10000 predictions can flip on last-bit
    # reduction-order changes. Running the window phase contiguously and alone
    # is what makes CONTROL A reproduce the committed Exp13 OOF exactly.
    log("   phase 1: building all corrected fold matrices (window fits only)")
    pending = []
    for f, tr, va, XC_tr, XC_va in P13.corrected_fold_matrices(
            folds, c, Wd, log=log):
        pending.append((f, tr, va, XC_tr, XC_va))
    log("   phase 1 complete: %d fold matrices built" % len(pending))
    gc.collect()

    params = ENS.label_params(42)
    oof_A = np.zeros((n, N_CLASSES), np.float64)
    oof_B = np.zeros((n, N_CLASSES), np.float64)
    imp_B = np.zeros(N_BASE + len(keep), np.float64)
    fold_idx = []

    log("   phase 2: fitting A and B on the stored matrices")
    for f, tr, va, XC_tr, XC_va in pending:
        fold_idx.append((f, tr, va))

        XA_tr = pd.DataFrame(np.hstack([XC_tr.values, Xwg[tr]]),
                             columns=base_names)
        XA_va = pd.DataFrame(np.hstack([XC_va.values, Xwg[va]]),
                             columns=base_names)
        assert list(XA_tr.columns) == base_names
        assert XA_tr.shape[1] == N_BASE, XA_tr.shape

        XB_tr = pd.DataFrame(np.hstack([XA_tr.values, Xep_used[tr]]),
                             columns=base_names + names_used)
        XB_va = pd.DataFrame(np.hstack([XA_va.values, Xep_used[va]]),
                             columns=base_names + names_used)

        # A and B must differ in exactly the appended columns, nothing else.
        assert np.array_equal(XA_tr.values, XB_tr.values[:, :N_BASE]), \
            "train base block diverged"
        assert np.array_equal(XA_va.values, XB_va.values[:, :N_BASE]), \
            "val base block diverged"
        assert XB_tr.shape[1] == N_BASE + len(keep)

        clfA = lgb.LGBMClassifier(**params).fit(XA_tr, yi[tr])
        oof_A[va] = clfA.predict_proba(XA_va)
        clfB = lgb.LGBMClassifier(**params).fit(XB_tr, yi[tr])
        oof_B[va] = clfB.predict_proba(XB_va)
        im = clfB.booster_.feature_importance(importance_type="gain")
        imp_B += (im / (im.sum() or 1.0)) / len(folds)
        log("    fold %d: A fitted on %d rows / %d cols; B on the same rows /"
            " %d cols" % (f, len(tr), N_BASE, N_BASE + len(keep)))
        del clfA, clfB, XA_tr, XA_va, XB_tr, XB_va, XC_tr, XC_va
        gc.collect()
    del pending
    gc.collect()

    oof_A32 = oof_A.astype(np.float32)
    oof_B32 = oof_B.astype(np.float32)
    np.save(os.path.join(HERE, "oof_A.npy"), oof_A32)
    np.save(os.path.join(HERE, "oof_B.npy"), oof_B32)
    pred_A = oof_A32.argmax(1)
    pred_B = oof_B32.argmax(1)

    # -------------------------------------------------------------- baseline
    hdr("5. BASELINE REPRODUCTION GATE (CONTROL A vs committed Exp13)")
    mA = official_full(oof_A32, d, rob, pred_A)
    sha_A = sha32(oof_A32)
    rows = []
    ok = True
    for k, exp in EXP13_METRICS.items():
        got = mA[k]
        good = abs(got - exp) <= TOL
        ok &= good
        rows.append({"metric": k, "expected": exp, "got": got,
                     "abs_diff": abs(got - exp), "pass": bool(good)})
        log("   %-18s expected %.16f got %.16f diff %.2e %s"
            % (k, exp, got, abs(got - exp), "PASS" if good else "FAIL"))
    sha_ok = bool(sha_A == EXP13_SHA_D)
    log("   sha256 A         : %s" % sha_A)
    log("   sha256 exp13 D   : %s" % EXP13_SHA_D)
    log("   sha identical    : %s" % sha_ok)
    repro_ok = bool(ok and sha_ok)
    log("   BASELINE REPRODUCTION: %s" % ("PASS" if repro_ok else "FAIL"))
    RES["baseline_reproduction"] = {
        "passed": repro_ok, "rows": rows, "sha_A": sha_A,
        "sha_exp13_d": EXP13_SHA_D, "sha_identical": sha_ok,
        "note": ("sha is NOT bit-stable across runs on this machine: the "
                 "window model is pinned to n_jobs=-1 and exp07/parallel.py "
                 "documents that argmax-derived aggregates make the result "
                 "sensitive to floating-point reduction order. Measured "
                 "sha drift across three runs of identical code; the metric "
                 "drift is reported above and is the quantity that matters.")}
    if not repro_ok:
        log()
        log("   STOP: CONTROL A does not reproduce the committed Exp13 corrected")
        log("   numbers by more than the run-to-run noise floor. CANDIDATE B is")
        log("   NOT scored and nothing is built.")
        RES["decision"] = "STOP_BASELINE"
        write_results(dict(RES, metrics_A=mA, leakage_probe=probe,
                           duplicates=dups, corr_diagnostic=corr))
        return 1

    # --------------------------------------------------------------- metrics
    hdr("6. A vs B METRICS")
    mB = official_full(oof_B32, d, rob, pred_B)
    keys = ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
            "composite")
    deltas = {k: mB[k] - mA[k] for k in keys}
    log("   %-20s %16s %16s %12s" % ("metric", "A (Exp13)", "B (+episodes)",
                                     "delta"))
    for k in keys:
        log("   %-20s %16.6f %16.6f %+12.6f" % (k, mA[k], mB[k], deltas[k]))

    f1A, f1B = per_class_f1(yi, pred_A), per_class_f1(yi, pred_B)
    log()
    log("   per-class F1:")
    for lab in LABELS:
        log("   %-20s %.4f -> %.4f  (%+.4f)"
            % (lab, f1A[lab], f1B[lab], f1B[lab] - f1A[lab]))

    hdr("7. MAIN TARGET: runaway_loop")
    rlA, rlB = prf(yi, pred_A, RL), prf(yi, pred_B, RL)
    for k in ("precision", "recall", "f1"):
        log("   %-10s %.4f -> %.4f  (%+.4f)"
            % (k, rlA[k], rlB[k], rlB[k] - rlA[k]))
    log("   tp %d->%d  fp %d->%d  fn %d->%d"
        % (rlA["tp"], rlB["tp"], rlA["fp"], rlB["fp"], rlA["fn"], rlB["fn"]))

    hdr("8. CONFUSIONS")
    for t, p, nm in ((RL, CLEAN, "runaway_loop -> clean"),
                     (RL, L2I["dropped_handoff"], "runaway_loop -> dropped_handoff"),
                     (RL, L2I["deadlock"], "runaway_loop -> deadlock"),
                     (RL, L2I["goal_drift"], "runaway_loop -> goal_drift"),
                     (CLEAN, RL, "clean -> runaway_loop"),
                     (L2I["dropped_handoff"], RL, "dropped_handoff -> runaway_loop"),
                     (L2I["deadlock"], RL, "deadlock -> runaway_loop")):
        a_ = int(np.sum((yi == t) & (pred_A == p)))
        b_ = int(np.sum((yi == t) & (pred_B == p)))
        log("   %-36s %5d -> %5d  (%+d)" % (nm, a_, b_, b_ - a_))

    hdr("9. CHANGED LABELS")
    changed = int((pred_A != pred_B).sum())
    wA, wB = pred_A != yi, pred_B != yi
    wr = int((wA & ~wB).sum())
    rw = int((~wA & wB).sum())
    ww = int((wA & wB).sum())
    log("   changed labels %d / %d (%.2f%%)"
        % (changed, n, 100.0 * changed / n))
    log("   wrong -> right %d    right -> wrong %d    wrong -> wrong %d"
        % (wr, rw, ww))
    tr_rl = np.where(yi == RL)[0]
    fixed = int(((pred_A[tr_rl] != RL) & (pred_B[tr_rl] == RL)).sum())
    broken = int(((pred_A[tr_rl] == RL) & (pred_B[tr_rl] != RL)).sum())
    still = int(((pred_A[tr_rl] != RL) & (pred_B[tr_rl] != RL)).sum())
    log("   over the %d true runaway_loop rows:" % len(tr_rl))
    log("      fixed   %d" % fixed)
    log("      broken  %d" % broken)
    log("      still wrong %d" % still)

    hdr("10. FEATURE IMPORTANCE")
    all_names = base_names + names_used
    order = np.argsort(-imp_B)
    top20 = set(int(i) for i in order[:20])
    top50 = set(int(i) for i in order[:50])
    ep_idx = set(range(N_BASE, N_BASE + len(keep)))
    share = float(imp_B[N_BASE:].sum())
    log("   episode block gain share: %.4f" % share)
    log("   episode features in top-20: %d / %d"
        % (len(top20 & ep_idx), len(names_used)))
    log("   episode features in top-50: %d / %d"
        % (len(top50 & ep_idx), len(names_used)))
    for i in order:
        if int(i) in ep_idx:
            log("      %-34s %.5f" % (all_names[int(i)], imp_B[int(i)]))

    hdr("11. FOLD STABILITY")
    fold_rows = []
    for f, tr, va in fold_idx:
        fa = official_subset(d, rob, va, pred_A[va])
        fb = official_subset(d, rob, va, pred_B[va])
        ya = yi[va]
        ra = prf(ya, pred_A[va], RL)["f1"]
        rb = prf(ya, pred_B[va], RL)["f1"]
        row = {"fold": f, "n_val": int(len(va)),
               "composite_A": fa["composite"], "composite_B": fb["composite"],
               "composite_delta": fb["composite"] - fa["composite"],
               "macro_delta": fb["macro_f1"] - fa["macro_f1"],
               "robustness_delta": fb["robustness_f1"] - fa["robustness_f1"],
               "hit2_delta": fb["fault_turn_hit2"] - fa["fault_turn_hit2"],
               "runaway_f1_A": ra, "runaway_f1_B": rb,
               "runaway_f1_delta": rb - ra}
        fold_rows.append(row)
        log("   fold %d n=%4d composite %+.6f  macro %+.6f  rob %+.6f  "
            "hit2 %+.6f  RL_F1 %.4f->%.4f (%+.4f)"
            % (f, row["n_val"], row["composite_delta"], row["macro_delta"],
               row["robustness_delta"], row["hit2_delta"], ra, rb,
               row["runaway_f1_delta"]))
    folds_won = sum(1 for r in fold_rows if r["composite_delta"] > 0)
    rl_won = sum(1 for r in fold_rows if r["runaway_f1_delta"] > 0)

    hdr("12. GATE (declared before the fit, reported unchanged)")
    worst_lab, worst_drop = None, 0.0
    for lab in LABELS:
        ddrop = f1A[lab] - f1B[lab]
        if ddrop > worst_drop:
            worst_lab, worst_drop = lab, ddrop
    rl_delta = rlB["f1"] - rlA["f1"]
    checks = {
        "composite_delta>=+0.002": (deltas["composite"],
                                     deltas["composite"] >= GATE["composite_delta_min"]),
        "composite_folds_won>=2": (float(folds_won),
                                   folds_won >= GATE["composite_folds_won_min"]),
        "runaway_f1_delta>=+0.015": (rl_delta,
                                     rl_delta >= GATE["runaway_loop_f1_delta_min"]),
        "runaway_folds_won>=2": (float(rl_won),
                                 rl_won >= GATE["runaway_loop_folds_won_min"]),
        "robustness>=-0.003": (deltas["robustness_f1"],
                               deltas["robustness_f1"] >= -GATE["robustness_drop_max"]),
        "no_other_class_worse_than_-0.02": (-worst_drop,
                                            worst_drop <= GATE["class_f1_drop_max"]),
    }
    for k, (v, o) in checks.items():
        log("   %-38s %+10.5f  %s" % (k, v, "PASS" if o else "FAIL"))
    gate_pass = all(o for _v, o in checks.values())
    log("   worst class drop: %s %+.5f" % (worst_lab, -worst_drop))
    log("   GATE: %s (%d/6)" % ("PASS" if gate_pass else "FAIL",
                                sum(1 for _v, o in checks.values() if o)))

    if gate_pass:
        decision = "PROMOTE"
        log("   VERDICT: PROMOTE")
    elif WEAK_BAND[0] <= deltas["composite"] < WEAK_BAND[1]:
        decision = "STOP_WEAK_INCONCLUSIVE"
        log("   VERDICT: weak / inconclusive -> STOP")
    else:
        decision = "STOP"
        log("   VERDICT: STOP (gate not passed; no second episode variant, no")
        log("           threshold tweaks, no production build, no Exp15)")
    RES["decision"] = decision

    RES.update({
        "metrics_A": mA, "metrics_B": mB, "deltas": deltas,
        "per_class_f1_A": f1A, "per_class_f1_B": f1B,
        "runaway_A": rlA, "runaway_B": rlB,
        "runaway_f1_delta": rl_delta,
        "changed_labels": changed, "wrong_to_right": wr,
        "right_to_wrong": rw, "wrong_to_wrong": ww,
        "runaway_fixed": fixed, "runaway_broken": broken,
        "runaway_still_wrong": still,
        "episode_gain_share": share,
        "episode_in_top20": len(top20 & ep_idx),
        "episode_in_top50": len(top50 & ep_idx),
        "episode_importance": [{"feature": all_names[int(i)],
                                "gain": float(imp_B[int(i)])}
                               for i in order if int(i) in ep_idx],
        "folds": fold_rows, "folds_won": folds_won,
        "runaway_folds_won": rl_won,
        "episode_features": list(names_used),
        "episode_n_features": len(names_used),
        "n_base_features": N_BASE,
        "leakage_probe": probe, "duplicates": dups,
        "corr_diagnostic": corr,
        "fold_independent_columns": n_fi,
        # protocol_guard.py reads promotion_gate["passed"] to cross-check the
        # results.csv verdict, so the object must carry that exact key.
        "promotion_gate": {
            "passed": bool(gate_pass),
            "criteria": {k: {"value": float(v), "pass": bool(o)}
                         for k, (v, o) in checks.items()},
            "n_passed": int(sum(1 for _v, o in checks.values() if o)),
            "n_criteria": len(checks),
        },
        "promotion_gate_passed": bool(gate_pass),
        "weak_band": list(WEAK_BAND),
        "confusions_named": {
            "%s->%s" % (LABELS[t], LABELS[p]):
                {"A": int(np.sum((yi == t) & (pred_A == p))),
                 "B": int(np.sum((yi == t) & (pred_B == p)))}
            for t, p in ((RL, CLEAN), (RL, L2I["dropped_handoff"]),
                         (RL, L2I["deadlock"]), (RL, L2I["goal_drift"]),
                         (CLEAN, RL), (L2I["dropped_handoff"], RL),
                         (L2I["deadlock"], RL))},
        "confusions_full": {
            LABELS[t]: {LABELS[p]: int(np.sum((yi == t) & (pred_B == p)))
                        for p in range(len(LABELS))
                        if int(np.sum((yi == t) & (pred_B == p)))}
            for t in range(len(LABELS))},
        "episode_counts_corpus": {
            "runs_with_episode": fires, "n_runs": n},
    })
    log()
    log("   runtime %.1fs" % (time.time() - t0))
    write_results(RES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())