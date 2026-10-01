"""Exp13 (corrected) - fold-local paired A/B on the wait-dependency graph.

WHY THIS FILE EXISTS
--------------------
A first Exp13 attempt built ONE global ``(10000, 300)`` matrix and filled its 51
window-aggregate columns fold by fold.  ``diag_fold_specific.py`` proved that
construction invalid: 6667 of 10000 runs (66.67%) receive DIFFERENT train
aggregates depending on which outer fold is being built, because each fold's
aggregates come from that fold's own honest inner window models.  The global
matrix therefore showed each run the wrong honest representation to one of the
two label fits that must see it, and CONTROL A failed to reproduce the sealed
Exp08 probabilities (max|dP| = 0.913).

Those preliminary numbers (composite +0.016190, deadlock F1 +0.1287) are
INVALID / PRELIMINARY DUE TO NON-COMPARABLE BASELINE.  They are preserved in
``results_preliminary_INVALID.json`` and are not a result.

THE CORRECTED SCHEME
--------------------
Per outer fold, and only per outer fold, the exact Exp07 training path is
rebuilt and CONTROL A is fitted on it BEFORE any candidate is scored:

    tr, va  = the sealed outer split (re-derived and compared, never trusted)
    tr_runs = sorted(set(tr))
    3 inner StratifiedKFold(3, seed=0) over tr_runs, verify_stacking
    one set of inner window fits  ->  Xagg_tr_f  via Exp07's grouping VERBATIM
    one window model on all outer-train -> Xagg_va_f
    X_A_tr_f = [found[tr], Xagg_tr_f]      X_A_va_f = [found[va], Xagg_va_f]
    clf_A.fit(X_A_tr_f, y[tr]) -> oof_A[va]

The window fits are computed ONCE per fold and the resulting 300-column
fold-local matrices are reused verbatim by the candidate, so A and B differ in
exactly the 12 appended ``wg_*`` columns and in nothing else:

    X_B_tr_f = [X_A_tr_f, Xwg[tr]]         X_B_va_f = [X_A_va_f, Xwg[va]]

`Xwg` is a pure function of a run's messages, so it is computed once and is
identical for both arms.

STOP CONDITIONS
---------------
  * CONTROL A does not reproduce the sealed Exp08 OOF bit for bit
    -> STOP_REPRODUCTION, and candidate B is never scored;
  * the reproduction gate fails on any official metric -> same STOP;
  * otherwise the pre-declared gate decides PROMOTE or STOP.  The thresholds
    are unchanged from the original Exp13 declaration.
"""

from __future__ import annotations

import gc
import hashlib
import json
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
EXP11 = os.path.join(ROOT, "experiments", "exp11_label_seed_ensemble")
for _p in (EXP11, EXP08, EXP10, EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aggregate as ag                       # noqa: E402
import cache as ca                           # noqa: E402
import grouping as grp                       # noqa: E402
import lightgbm as lgb                       # noqa: E402
import parallel as par                       # noqa: E402
import runner as R                           # noqa: E402
import window_dataset as wd                   # noqa: E402

import ensemble as ENS                        # noqa: E402  (Exp11, the working path)
from blend import LABELS, L2I                # noqa: E402
from blend import N_CLASSES, TOL             # noqa: E402
from blend import SUCCESS_F1_PINNED          # noqa: E402

# Exp13's own parser, loaded by path: Exp07 also ships a module named features.
import importlib.util                        # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "exp13_wait_features", os.path.join(HERE, "features.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

CLEAN = 0
DEADLOCK = LABELS.index("deadlock")
CLEAN_I = LABELS.index("clean")
N_INNER = ENS.N_INNER
INNER_SEED = ENS.INNER_SEED
N_AGG = ENS.N_AGG_FEATS
N_LABEL = ENS.N_LABEL_FEATS
N_WG = W.N_FEATURES

# The ORIGINAL pre-declared Exp13 gate.  Unchanged.
GATE = {"composite_delta_min": 0.002, "folds_won_min": 2,
        "deadlock_f1_delta_min": 0.015, "deadlock_folds_won_min": 2,
        "robustness_drop_max": 0.003, "class_f1_drop_max": 0.02}
WEAK_BAND = (0.001, 0.002)

# The sealed Exp08 baseline numbers, as declared by Exp11 against the same
# sealed artifact.  Read, not re-derived.
BASELINE_EXPECTED = {
    "macro_f1": 0.7894323366835861,
    "robustness_f1": 0.7368470046834376,
    "success_f1": 0.8643757406010988,
    "fault_turn_hit2": 0.6086111111111111,
    "composite": 0.7694453917139285,
}

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


def sha32(P):
    return ca.sha256_array(np.ascontiguousarray(np.asarray(P, np.float32)))


def prf(yi, pred, cls):
    tp = int(np.sum((yi == cls) & (pred == cls)))
    fp = int(np.sum((yi != cls) & (pred == cls)))
    fn = int(np.sum((yi == cls) & (pred != cls)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def per_class_f1(yi, pred):
    return {LABELS[c]: prf(yi, pred, c)["f1"] for c in range(len(LABELS))}


def confusion_counts(yi, pred):
    out = {}
    for t in range(len(LABELS)):
        row = {}
        for p in range(len(LABELS)):
            c = int(np.sum((yi == t) & (pred == p)))
            if c:
                row[LABELS[p]] = c
        if row:
            out[LABELS[t]] = row
    return out


def official_full(P, d, rob, pred):
    """Exp11's scoring, unchanged: ``evaluation.metrics`` is the only definition."""
    c = ENS.corpus_for(P, d, rob, labels=np.asarray(pred, np.int64))
    m, turns = ENS.full_official(c, np.asarray(pred, np.int64))
    return m, turns


def official_subset(P, d, rob, rows, pred_rows):
    """Per-fold official score over the SAME metric code path, restricted in rows."""
    rows = np.asarray(rows, np.int64)
    pr = np.asarray(pred_rows, np.int64)
    yt = [LABELS[int(d["yi"][i])] for i in rows]
    yp = [LABELS[int(i)] for i in pr]
    from metrics import composite, fault_turn_hit_at_k, macro_f1
    fturn = d["train"]["fault_turn"].values
    # L1 turn = peak position of the PREDICTED class, exactly as Exp08.
    turns = np.asarray(d["peak_pos"])[rows][np.arange(len(rows)), pr]
    faulty = np.where(d["yi"][rows] != CLEAN)[0]
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


def build_fold_matrices(folds, c, W, log=log):
    """Yield the fold-local 300-column train/val matrices, Exp07 verbatim.

    Window fits are executed ONCE per fold here; both arms consume the output.
    """
    yi, n = c["yi"], c["n"]
    found = c["foundation"]
    Xw, yw, rw = W["X"], W["y"], W["run"]
    CW = wd.class_weight_vector()

    # Re-derive the outer folds and refuse to continue if they differ from the
    # sealed ones, so a fold can never be swapped between verification and fit.
    rederived = [np.asarray(v, np.int64) for _, v in
                 StratifiedKFold(R.N_OUTER, shuffle=True, random_state=R.SEED)
                 .split(np.zeros(n), yi)]
    for a, b in zip(folds, rederived):
        if not np.array_equal(np.asarray(a, np.int64), b):
            raise AssertionError("outer fold is not the sealed seed-0 fold")

    all_ix = np.arange(n, dtype=np.int64)
    fidx = [(all_ix[~np.isin(all_ix, fi)], np.asarray(fi, np.int64)) for fi in folds]
    cols = list(found.columns) + list(ag.AGG_NAMES)
    assert len(cols) == N_LABEL, len(cols)

    for f in range(R.N_OUTER):
        t_f = time.time()
        tr, va = fidx[f]
        tr_runs = np.array(sorted(set(tr.tolist())))
        inner = list(StratifiedKFold(N_INNER, shuffle=True,
                                     random_state=INNER_SEED)
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
        probs = par.run_inner_fits(payloads)
        del payloads
        gc.collect()

        Xagg_tr = np.zeros((len(tr_runs), ag.N_AGG))
        slot = {int(r): k for k, r in enumerate(tr_runs)}
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            # Exp07's own call, with its index-range quirk, VERBATIM.
            blocks = grp.predict_runs(probs[(f, j)], rw[np.isin(rw, hold)],
                                      len(tr_runs))
            sub, _, _ = grp.aggregate_block(blocks, ag, slot, len(tr_runs))
            Xagg_tr[assign == j] = sub[assign == j]
        del probs
        gc.collect()

        mw = lgb.LGBMClassifier(n_jobs=par.DEFAULT_THREADS, **R.WINDOW_PARAMS)
        mw.fit(Xw[np.isin(rw, tr)], yw[np.isin(rw, tr)],
               sample_weight=CW[yw[np.isin(rw, tr)]])
        Pva = mw.predict_proba(Xw[np.isin(rw, va)])
        blocks = grp.predict_runs(Pva, rw[np.isin(rw, va)], n)
        slot_va = {int(r): k for k, r in enumerate(va)}
        Xagg_va, _, _ = grp.aggregate_block(blocks, ag, slot_va, len(va))
        del mw
        gc.collect()

        XA_tr = pd.DataFrame(np.hstack([found.iloc[tr].values, Xagg_tr]),
                             columns=cols)
        XA_va = pd.DataFrame(np.hstack([found.iloc[va].values, Xagg_va]),
                             columns=cols)
        log("    fold %d: window fits done, 300-col matrices built  %.1fs"
            % (f, time.time() - t_f))
        yield f, tr, va, XA_tr, XA_va
        del XA_tr, XA_va, Xagg_tr, Xagg_va
        gc.collect()


def main():
    t0 = time.time()
    bar("EXP13 (CORRECTED) - FOLD-LOCAL PAIRED A/B ON THE WAIT GRAPH")
    log("the earlier global-X300 result is INVALID (non-comparable baseline)")
    log("no feature, parser, seed, parameter, threshold or gate has changed")

    hdr("1. VERIFIED INPUTS")
    d = ENS.load_verified()
    yi = d["yi"]
    folds = [np.asarray(v, np.int64) for v in d["folds"]]
    n = len(yi)
    rob = np.asarray(ENS.robustness_mask(), dtype=bool)
    log("   runs=%d folds=%d checks=%d sealed_proba=%s %s"
        % (n, len(folds), len(d["checks"]), np.shape(d["proba_B"]),
           np.asarray(d["proba_B"]).dtype))
    sealed = np.asarray(d["proba_B"], np.float32)
    sealed_sha = sha32(sealed)
    log("   sealed sha256 %s..." % sealed_sha[:32])

    c, Wd = ENS._prepare()
    log("   foundation %s  windows %s" % (c["foundation"].shape, Wd["X"].shape))

    # ------------------------------------------------------------------
    hdr("2. WAIT-GRAPH BLOCK (fold-independent, computed once)")
    runs = c["runs"]
    Xwg = W.build_matrix(runs)
    assert Xwg.shape == (n, N_WG), Xwg.shape
    log("   %s from %d runs: %s" % (Xwg.shape, n, ",".join(W.FEATURE_NAMES)))
    import copy
    base = W.build_matrix(runs[:300])
    pert = []
    for i, r in enumerate(runs[:300]):
        r2 = copy.deepcopy(r)
        r2["label"] = ["deadlock", "clean", "goal_drift"][i % 3]
        r2["success"] = 1 - (i % 2)
        r2["fault_turn"] = (i * 7) % 11
        pert.append(r2)
    leak_same = bool(np.array_equal(base, W.build_matrix(pert)))
    log("   leakage probe (targets overwritten on 300 runs): identical=%s" % leak_same)

    # ------------------------------------------------------------------
    hdr("3. FOLD-LOCAL PAIRED FITS - CONTROL A first, then B, same windows")
    params = ENS.label_params(42)
    log("   label params from common.lgb_label(), random_state=%s"
        % params["random_state"])
    oof_A = np.zeros((n, N_CLASSES), np.float64)
    oof_B = np.zeros((n, N_CLASSES), np.float64)
    imp_A = np.zeros(N_LABEL, np.float64)
    imp_B = np.zeros(N_LABEL + N_WG, np.float64)
    fold_idx = []

    for f, tr, va, XA_tr, XA_va in build_fold_matrices(folds, c, Wd):
        fold_idx.append((f, tr, va))

        clfA = lgb.LGBMClassifier(**params).fit(XA_tr, yi[tr])
        oof_A[va] = clfA.predict_proba(XA_va)
        im = clfA.booster_.feature_importance(importance_type="gain")
        imp_A += (im / (im.sum() or 1.0)) / len(folds)
        log("    fold %d: A fitted on %d rows" % (f, len(tr)))

        XB_tr = pd.DataFrame(np.hstack([XA_tr.values, Xwg[tr]]),
                             columns=list(XA_tr.columns) + list(W.FEATURE_NAMES))
        XB_va = pd.DataFrame(np.hstack([XA_va.values, Xwg[va]]),
                             columns=list(XA_va.columns) + list(W.FEATURE_NAMES))
        assert np.array_equal(XA_tr.values, XB_tr.values[:, :N_LABEL])
        clfB = lgb.LGBMClassifier(**params).fit(XB_tr, yi[tr])
        oof_B[va] = clfB.predict_proba(XB_va)
        im2 = clfB.booster_.feature_importance(importance_type="gain")
        imp_B += (im2 / (im2.sum() or 1.0)) / len(folds)
        log("    fold %d: B fitted on the SAME rows + %d columns"
            % (f, N_WG))
        del clfA, clfB, XB_tr, XB_va
        gc.collect()

    oof_A32 = oof_A.astype(np.float32)
    oof_B32 = oof_B.astype(np.float32)
    np.save(os.path.join(HERE, "oof_A_foldlocal.npy"), oof_A32)
    np.save(os.path.join(HERE, "oof_B_foldlocal.npy"), oof_B32)

    # ------------------------------------------------------------------
    hdr("4. CONTROL A SEALED PARITY (gate before any candidate scoring)")
    a_sha = sha32(oof_A32)
    bit_identical = bool(a_sha == sealed_sha)
    lab_seal = sealed.argmax(1)
    lab_a = oof_A32.argmax(1)
    n_lab_diff = int((lab_a != lab_seal).sum())
    max_d = float(np.max(np.abs(oof_A32.astype(np.float64)
                               - sealed.astype(np.float64))))
    log("   A sha256      : %s..." % a_sha[:32])
    log("   sealed sha256 : %s..." % sealed_sha[:32])
    log("   bit identical : %s" % bit_identical)
    log("   argmax labels : %d / %d differ" % (n_lab_diff, n))
    log("   max|dP|       : %.3e" % max_d)
    RES["control_a_parity"] = {"bit_identical": bit_identical,
                               "sha_a": a_sha, "sha_sealed": sealed_sha,
                               "argmax_labels_differ": n_lab_diff,
                               "max_abs_prob_diff": max_d}

    mA, _ = official_full(oof_A32, d, rob, lab_a)
    gate_rows = []
    ok_metrics = True
    for k, exp in BASELINE_EXPECTED.items():
        got = mA[k]
        good = abs(got - exp) <= TOL
        ok_metrics &= good
        gate_rows.append({"metric": k, "expected": exp, "got": got,
                          "abs_diff": abs(got - exp), "pass": bool(good)})
        log("   %-18s expected %.16f got %.16f diff %.2e %s"
            % (k, exp, got, abs(got - exp), "PASS" if good else "FAIL"))
    repro_ok = bool(bit_identical and n_lab_diff == 0 and ok_metrics)
    RES["reproduction_gate"] = {"passed": repro_ok, "rows": gate_rows,
                                "bit_identical": bit_identical,
                                "labels_differ": n_lab_diff}
    log("   CONTROL A REPRODUCTION: %s" % ("PASS" if repro_ok else "FAIL"))

    if not repro_ok:
        log()
        log("   CONTROL A DOES NOT REPRODUCE THE SEALED Exp08 OOF.")
        log("   Exp13 = INVALID / STOP_REPRODUCTION.  Candidate B is NOT scored,")
        log("   no gate is evaluated, no promotion, no production, no ZIP.")
        RES["decision"] = "STOP_REPRODUCTION"
        dump(t0)
        return 1

    # ------------------------------------------------------------------
    hdr("5. PAIRED METRICS - A vs B (official evaluation.metrics only)")
    pred_A, pred_B = lab_a, oof_B32.argmax(1)
    mB, _ = official_full(oof_B32, d, rob, pred_B)
    deltas = {}
    log("   %-20s %14s %14s %12s" % ("metric", "A (sealed exp08)", "B (paired)",
                                    "delta"))
    for k in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
              "composite"):
        deltas[k] = mB[k] - mA[k]
        log("   %-20s %14.6f %14.6f %+12.6f" % (k, mA[k], mB[k], deltas[k]))

    f1A, f1B = per_class_f1(yi, pred_A), per_class_f1(yi, pred_B)
    log()
    log("   per-class F1:")
    for lab in LABELS:
        log("   %-20s %.4f -> %.4f  (%+.4f)"
            % (lab, f1A[lab], f1B[lab], f1B[lab] - f1A[lab]))

    hdr("6. DEADLOCK")
    pa, pb = prf(yi, pred_A, DEADLOCK), prf(yi, pred_B, DEADLOCK)
    log("   F1        %.4f -> %.4f  (%+.4f)" % (pa["f1"], pb["f1"],
                                                pb["f1"] - pa["f1"]))
    log("   precision %.4f -> %.4f" % (pa["precision"], pb["precision"]))
    log("   recall    %.4f -> %.4f" % (pa["recall"], pb["recall"]))
    log("   tp %d->%d  fp %d->%d  fn %d->%d"
        % (pa["tp"], pb["tp"], pa["fp"], pb["fp"], pa["fn"], pb["fn"]))
    cA, cB = confusion_counts(yi, pred_A), confusion_counts(yi, pred_B)
    dl_clean_A = cA.get("deadlock", {}).get("clean", 0)
    dl_clean_B = cB.get("deadlock", {}).get("clean", 0)
    log("   deadlock -> clean : %d -> %d  (%+d)"
        % (dl_clean_A, dl_clean_B, dl_clean_B - dl_clean_A))

    hdr("7. FOLD STABILITY")
    fold_rows = []
    for f, tr, va in fold_idx:
        fa = official_subset(oof_A32, d, rob, va, pred_A[va])
        fb = official_subset(oof_B32, d, rob, va, pred_B[va])
        ya = yi[va]
        da = prf(ya, pred_A[va], DEADLOCK)["f1"]
        db = prf(ya, pred_B[va], DEADLOCK)["f1"]
        row = {"fold": f, "n_val": int(len(va)),
               "composite_A": fa["composite"], "composite_B": fb["composite"],
               "composite_delta": fb["composite"] - fa["composite"],
               "macro_delta": fb["macro_f1"] - fa["macro_f1"],
               "robustness_delta": fb["robustness_f1"] - fa["robustness_f1"],
               "hit2_delta": fb["fault_turn_hit2"] - fa["fault_turn_hit2"],
               "deadlock_f1_A": da, "deadlock_f1_B": db,
               "deadlock_f1_delta": db - da}
        fold_rows.append(row)
        log("   fold %d n=%4d composite %+.6f  macro %+.6f  rob %+.6f  "
            "hit2 %+.6f  dl_F1 %.4f->%.4f (%+.4f)"
            % (f, row["n_val"], row["composite_delta"], row["macro_delta"],
               row["robustness_delta"], row["hit2_delta"], da, db,
               row["deadlock_f1_delta"]))
    folds_won = sum(1 for r in fold_rows if r["composite_delta"] > 0)
    dl_folds_won = sum(1 for r in fold_rows if r["deadlock_f1_delta"] > 0)

    hdr("8. DIAGNOSTICS")
    names_B = list(c["foundation"].columns) + list(ag.AGG_NAMES) \
        + list(W.FEATURE_NAMES)
    order = np.argsort(-imp_B)
    wg_top = [names_B[i] for i in order if names_B[i].startswith("wg_")][:12]
    log("   wg_* gain share: %.4f" % float(imp_B[N_LABEL:].sum()))
    for i in order:
        if names_B[i].startswith("wg_"):
            log("      %-32s %.5f" % (names_B[i], imp_B[i]))
    changed = int((pred_A != pred_B).sum())
    wb = pred_A != yi
    wf = pred_B != yi
    w2r = int((wb & ~wf).sum())
    r2w = int((~wb & wf).sum())
    log("   changed labels %d/%d (%.2f%%)" % (changed, n, 100.0 * changed / n))
    log("   wrong->right %d   right->wrong %d" % (w2r, r2w))
    dlr = yi == DEADLOCK
    log("   true deadlock: fixed %d / broken %d (of %d)"
        % (int((dlr & wb & ~wf).sum()), int((dlr & ~wb & wf).sum()),
           int(dlr.sum())))

    hdr("9. GATE (unchanged thresholds)")
    worst_lab, worst_drop = None, 0.0
    for lab in LABELS:
        d_ = f1A[lab] - f1B[lab]
        if d_ > worst_drop:
            worst_lab, worst_drop = lab, d_
    crit = {
        "composite_delta>=+0.002": (deltas["composite"],
                                     deltas["composite"] >= GATE["composite_delta_min"]),
        "folds_won>=2": (float(folds_won), folds_won >= GATE["folds_won_min"]),
        "deadlock_f1_delta>=+0.015": (pb["f1"] - pa["f1"],
                                      (pb["f1"] - pa["f1"]) >= GATE["deadlock_f1_delta_min"]),
        "deadlock_folds_won>=2": (float(dl_folds_won),
                                  dl_folds_won >= GATE["deadlock_folds_won_min"]),
        "robustness>=-0.003": (deltas["robustness_f1"],
                               deltas["robustness_f1"] >= -GATE["robustness_drop_max"]),
        "no_class_worse_than_-0.02": (-worst_drop,
                                      worst_drop <= GATE["class_f1_drop_max"]),
    }
    for k, (v, o) in crit.items():
        log("   %-32s %+10.5f  %s" % (k, v, "PASS" if o else "FAIL"))
    log("   worst class drop: %s %+.5f" % (worst_lab, -worst_drop))
    promote = all(o for _v, o in crit.values())
    log("   VERDICT: %s" % ("PROMOTE" if promote else "STOP"))

    RES.update({
        "metrics_A": mA, "metrics_B": mB, "deltas": deltas,
        "per_class_f1_A": f1A, "per_class_f1_B": f1B,
        "deadlock_A": pa, "deadlock_B": pb,
        "deadlock_to_clean_A": dl_clean_A, "deadlock_to_clean_B": dl_clean_B,
        "folds": fold_rows, "folds_won": folds_won,
        "deadlock_folds_won": dl_folds_won,
        "leakage_identical": leak_same,
        "wg_gain_share": float(imp_B[N_LABEL:].sum()),
        "wg_gain_ranking": wg_top,
        "changed_labels": changed, "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "gate": {k: {"value": float(v), "pass": bool(o)}
                 for k, (v, o) in crit.items()},
        "gate_pass": bool(promote),
        "decision": "PROMOTE" if promote else "STOP",
    })
    dump(t0)
    return 0


def dump(t0):
    RES["runtime_sec"] = round(time.time() - t0, 2)
    RES["preliminary_status"] = (
        "the first Exp13 run (global X300) is INVALID / PRELIMINARY DUE TO "
        "NON-COMPARABLE BASELINE; kept in results_preliminary_INVALID.json")
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=2, sort_keys=True, default=float)
    with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    log()
    log("   runtime %.1fs; results.json + run.log written" % RES["runtime_sec"])


if __name__ == "__main__":
    raise SystemExit(main())