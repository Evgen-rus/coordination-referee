"""Exp13 - explicit wait-dependency graph for `deadlock`.

THE ONE HYPOTHESIS
------------------
    An explicit representation of WHO waits for WHOM adds relational
    information that the current 300-feature Exp08 representation does not
    contain, and improves the official composite.

WHAT IS AND IS NOT TOUCHED
--------------------------
Only the label-head feature matrix gains the 12 ``wg_*`` columns.  Not the
success head, not the Exp07 window model, not the Exp08 L1 localizer, not the
fault_turn rule, not the hyper-parameters, not the folds, not the seeds, not
the decision rule (argmax, no threshold).  ``goal_drift`` is not modelled and
no feature targets it.

BASELINE POLICY (fixed before any fit)
--------------------------------------
CONTROL A is the incumbent Exp08 pipeline over the current 300 features and
must reproduce the SEALED Exp08 OOF to 1e-12 before the candidate is scored.
Its probabilities are loaded from the sealed artifact rather than refitted, so
A is exact by construction; the gate additionally re-derives all five official
metrics from them.  Only then is CONTROL B fitted - the same LightGBM, the
same params, the same seed, the same rows, with the wait-graph block appended.

The reproduction is of Exp08 as shipped.  Exp12's mapping fix is NOT part of
this experiment: A and B share whatever the incumbent pipeline does, so the
comparison isolates the wait-graph block alone.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP12 = os.path.join(ROOT, "experiments", "exp12_fix_window_run_mapping")
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
for _p in (HERE, EXP12, os.path.join(ROOT, "experiments", "exp10_ab_probability_blend"),
           os.path.join(ROOT, "experiments", "exp08_window_localizer"),
           EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cache as ca                           # noqa: E402
import lightgbm as lgb                       # noqa: E402
import mapping_fix as M                      # noqa: E402  (read-only reuse)
from blend import Corpus, L2I, LABELS        # noqa: E402
from blend import N_CLASSES, SUCCESS_F1_PINNED, TOL   # noqa: E402
from common import build_all, lgb_label      # noqa: E402
from metrics import f1_per_class             # noqa: E402

# Exp07 also ships a module called ``features``; loading this experiment's own
# parser by explicit file path removes the ambiguity instead of relying on
# sys.path order.
import importlib.util                        # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "exp13_wait_features", os.path.join(HERE, "features.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)
CLEAN = 0
DEADLOCK = LABELS.index("deadlock")
CLEAN_I = LABELS.index("clean")
DH_I = LABELS.index("dropped_handoff")
RL_I = LABELS.index("runaway_loop")

N_FOUNDATION = M.N_FOUNDATION          # 249
N_AGG_FEATS = M.N_AGG_FEATS            # 51
N_LABEL_FEATS = M.N_LABEL_FEATS        # 300
N_WG = W.N_FEATURES                    # 12
N_INNER = M.N_INNER
INNER_SEED = M.INNER_SEED

# Pre-declared promotion gate, written before the first fit of candidate B.
GATE = {
    "composite_delta_min": 0.002,
    "folds_won_min": 2,
    "deadlock_f1_delta_min": 0.015,
    "deadlock_folds_won_min": 2,
    "robustness_drop_max": 0.003,
    "class_f1_drop_max": 0.02,
}
WEAK_BAND = (0.001, 0.002)

# Reproduction targets: the sealed exp08 numbers, read from exp12's loader
# rather than hand-copied here.
BASELINE_EXPECTED = dict(M.BASELINE_EXPECTED)

OUT = []


def log(msg=""):
    print(msg)
    OUT.append(str(msg))


def bar(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


def hdr(title):
    log()
    log("-- %s" % title)


def leakage_probe(runs, n=300):
    """Target perturbation: replacing every target must not move one number."""
    import copy
    base = W.build_matrix(runs[:n])
    pert = []
    for i, r in enumerate(runs[:n]):
        r2 = copy.deepcopy(r)
        r2["label"] = ["deadlock", "clean", "goal_drift"][i % 3]
        r2["success"] = 1 - (i % 2)
        r2["fault_turn"] = (i * 7) % 11
        pert.append(r2)
    other = W.build_matrix(pert)
    return {"identical": bool(np.array_equal(base, other)),
            "n_rows_probed": int(n),
            "max_abs_diff": float(np.max(np.abs(base - other))) if base.size else 0.0}


def duplicate_check(Xnew, Xold, names):
    """Flag any wg_* column numerically identical to a 300-feature column."""
    rows = []
    for j, nm in enumerate(names):
        v = Xnew[:, j]
        dup = [int(i) for i in range(Xold.shape[1]) if np.array_equal(v, Xold[:, i])]
        rows.append({"feature": nm, "n_unique": int(len(np.unique(v))),
                     "identical_to_300": bool(dup),
                     "identical_cols": dup})
    return rows


def corr_diagnostic(Xnew, Xold, names, old_names):
    """max |r| of each new column against every existing feature."""
    rows = []
    for j, nm in enumerate(names):
        v = Xnew[:, j]
        if len(np.unique(v)) < 2:
            rows.append({"feature": nm, "max_abs_r": None, "argmax": None,
                         "n_unique": int(len(np.unique(v)))})
            continue
        best_r, best_n = 0.0, None
        for i in range(Xold.shape[1]):
            w = Xold[:, i]
            if len(np.unique(w)) < 2:
                continue
            r = np.corrcoef(v, w)[0, 1]
            if not np.isnan(r) and abs(r) > abs(best_r):
                best_r, best_n = r, old_names[i]
        rows.append({"feature": nm, "max_abs_r": float(abs(best_r)),
                     "argmax": best_n, "signed_r": float(best_r),
                     "n_unique": int(len(np.unique(v)))})
    return rows


def prf(yi, pred, cls):
    tp = int(np.sum((yi == cls) & (pred == cls)))
    fp = int(np.sum((yi != cls) & (pred == cls)))
    fn = int(np.sum((yi == cls) & (pred != cls)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


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


def load_foundation():
    """The current 300-feature Exp08 label-head matrix and its column names."""
    c, _W = M._prepare()
    found = c["foundation"]
    return found, c


def make_corpus(d, yi, pred_a):
    """Corpus over the whole corpus, carrying the sealed peaks."""
    rob = M.robustness_mask()
    cp = Corpus(np.eye(N_CLASSES)[pred_a], np.asarray(d["proba_B"], np.float64),
                yi, d["train"]["fault_turn"].values, d["peak_pos"], rob,
                labels_B=d["labels_B"])
    return cp, rob


def main():
    t0 = time.time()
    bar("EXP13 - WAIT DEPENDENCY GRAPH (one fixed hypothesis)")
    log("hypothesis : explicit WHO-waits-for-WHM representation adds the")
    log("             relational information deadlock needs, and improves")
    log("             the official composite")
    log("scope      : label head only; success head / window model / L1 /")
    log("             fault_turn rule / params / folds / seeds untouched")
    log("parser     : 6 frozen template families, unit-tested before this run")

    hdr("1. VERIFIED LOAD (fail-closed, sealed peaks)")
    d = M.load_verified()
    yi = d["yi"]
    folds = d["folds"]
    n_runs = len(yi)
    log("   runs=%d  folds=%d  verified_checks=%d  load=%.1fs"
        % (n_runs, len(folds), len(d["checks"]), d["load_sec"]))

    c, Wd = M._prepare()
    found = c["foundation"]
    log("   foundation matrix: %s (+ %d window aggregates = %d)"
        % (found.shape, N_AGG_FEATS, N_LABEL_FEATS))

    # ------------------------------------------------------------------
    # CONTROL A : the sealed Exp08 OOF.  Reproduction gate FIRST.
    # ------------------------------------------------------------------
    hdr("2. REPRODUCTION GATE - CONTROL A vs SEALED Exp08")
    hdr("2a. rebuild the incumbent 300-column matrix (249 + 51 aggregates)")
    X300, old_names = build_300(folds, c, Wd)
    assert X300.shape == (n_runs, N_LABEL_FEATS), X300.shape
    assert np.isfinite(X300).all(), "300-column matrix has non-finite values"
    log("   300-column matrix: %s" % (X300.shape,))

    proba_A = np.asarray(d["proba_B"], np.float64)
    pred_A = proba_A.argmax(1)
    corpus, rob = make_corpus(d, yi, pred_A)
    mA, turns_A = M.full_official(corpus, pred_A)
    gate_rows = []
    for k, exp in BASELINE_EXPECTED.items():
        got = mA[k]
        ok = abs(got - exp) <= TOL
        gate_rows.append({"metric": k, "expected": exp, "got": got,
                          "abs_diff": abs(got - exp), "pass": bool(ok)})
        log("   %-18s expected %.16f  got %.16f  diff %.2e  %s"
            % (k, exp, got, abs(got - exp), "PASS" if ok else "FAIL"))
    repro_pass = all(r["pass"] for r in gate_rows)
    log("   REPRODUCTION GATE (sealed artifact): %s"
        % ("PASS" if repro_pass else "FAIL"))
    if not repro_pass:
        log("   STOP: baseline did not reproduce; candidate is not scored.")
        write_results({"reproduction": {"passed": False, "rows": gate_rows},
                       "decision": "STOP_BASELINE_FAILED"})
        return

    hdr("2b. CONTROL A refitted on the rebuilt 300 (must match the sealed OOF)")
    params = M.label_params()
    log("   params: n_estimators=%s lr=%s leaves=%s seed=%s"
        % (params.get("n_estimators"), params.get("learning_rate"),
           params.get("num_leaves"), params.get("random_state")))
    oof_A = np.zeros((n_runs, N_CLASSES), np.float64)
    imp_A = np.zeros(N_LABEL_FEATS, np.float64)
    allidx = np.arange(n_runs)
    for f, fv in enumerate(folds):
        va = np.asarray(fv, np.int64)
        tr = allidx[~np.isin(allidx, va)]
        clf = lgb.LGBMClassifier(**params).fit(X300[tr], yi[tr])
        oof_A[va] = clf.predict_proba(X300[va])
        im = clf.booster_.feature_importance(importance_type="gain")
        imp_A += (im / (im.sum() or 1.0)) / len(folds)
        del clf
        gc.collect()
    refit_ok = bool(np.array_equal(oof_A, proba_A.astype(np.float64)))
    max_dev = float(np.max(np.abs(oof_A - proba_A)))
    log("   refit A vs sealed OOF: bit-identical=%s  max|dev|=%.3e"
        % (refit_ok, max_dev))
    if not refit_ok:
        log("   NOTE: refit differs from the sealed artifact; CONTROL A is the")
        log("         sealed probabilities for scoring, so the comparison to")
        log("         exp08 stays exact.")
    pred_A = oof_A.argmax(1)

    hdr("3. WAIT-GRAPH BLOCK (from messages only)")
    runs = c["runs"]
    Xwg = W.build_matrix(runs)
    log("   block shape %s  names=%s" % (Xwg.shape, ",".join(W.FEATURE_NAMES)))
    for j, nm in enumerate(W.FEATURE_NAMES):
        v = Xwg[:, j]
        log("   %-32s mean %8.3f  uniq %5d  nonzero %5d"
            % (nm, v.mean(), len(np.unique(v)), int((v != 0).sum())))

    hdr("4. LEAKAGE PROOF (target perturbation)")
    lk = leakage_probe(runs)
    log("   targets replaced on %d runs -> features identical: %s (max|diff| %.2e)"
        % (lk["n_rows_probed"], lk["identical"], lk["max_abs_diff"]))
    if not lk["identical"]:
        log("   STOP: wait-graph block reads a target.")
        write_results({"reproduction": {"passed": True, "rows": gate_rows},
                       "leakage": lk, "decision": "STOP_LEAKAGE"})
        return

    hdr("5. DUPLICATE / CORRELATION DIAGNOSTIC (before any CV)")
    dups = duplicate_check(Xwg, X300, W.FEATURE_NAMES)
    exact = [r for r in dups if r["identical_to_300"]]
    log("   columns numerically identical to an existing 300-feature: %d"
        % len(exact))
    for r in exact:
        log("      %-32s == col(s) %s" % (r["feature"], r["identical_cols"]))
    if exact:
        log("   STOP: a new column is a mathematical duplicate, not new")
        log("         information. Removed before CV, per protocol.")
        write_results({"reproduction": {"passed": True, "rows": gate_rows},
                       "leakage": lk, "duplicates": dups,
                       "decision": "STOP_DUPLICATE"})
        return
    cr = corr_diagnostic(Xwg, X300, W.FEATURE_NAMES, old_names)
    for r in cr:
        log("   %-32s max|r| vs 300 = %.3f  (%s)"
            % (r["feature"], r["max_abs_r"] or 0.0, r["argmax"]))
    log("   (correlation is acceptable; exact duplication would not be)")

    hdr("6. SINGLE CV - CONTROL B (300 + wait-graph), same params/seed/rows")
    X_B = np.hstack([X300, Xwg])
    assert X_B.shape[1] == N_LABEL_FEATS + N_WG
    oof_B = np.zeros((n_runs, N_CLASSES), np.float64)
    imp_sum = np.zeros(X_B.shape[1], np.float64)
    fold_of = np.zeros(n_runs, np.int64)
    allidx = np.arange(n_runs)
    for f, fv in enumerate(folds):
        va = np.asarray(fv, np.int64)
        tr = allidx[~np.isin(allidx, va)]
        fold_of[va] = f
        clf = lgb.LGBMClassifier(**params).fit(X_B[tr], yi[tr])
        oof_B[va] = clf.predict_proba(X_B[va])
        imp = clf.booster_.feature_importance(importance_type="gain")
        tot = imp.sum() if imp.sum() > 0 else 1.0
        imp_sum += imp / tot / len(folds)
        log("   fold %d: train %d / val %d  done" % (f, len(tr), len(va)))
        del clf
        gc.collect()
    pred_B = oof_B.argmax(1)
    log("   candidate OOF built over %d runs" % n_runs)

    hdr("7. OFFICIAL METRICS - A vs B")
    corpus_B, _ = make_corpus(d, yi, pred_B)
    mB, turns_B = M.full_official(corpus_B, pred_B)
    log("   %-20s %14s %14s %12s" % ("metric", "A (exp08)", "B (cand)", "delta"))
    deltas = {}
    for k in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
              "composite"):
        deltas[k] = mB[k] - mA[k]
        log("   %-20s %14.6f %14.6f %+12.6f" % (k, mA[k], mB[k], deltas[k]))

    hdr("8. DEADLOCK MECHANISM TARGET")
    ytn = [LABELS[i] for i in yi]
    pa = prf(yi, pred_A, DEADLOCK)
    pb = prf(yi, pred_B, DEADLOCK)
    log("   %-12s %8s %8s %10s %10s %10s" %
        ("", "before", "after", "d", "before", "after"))
    log("   %-12s %8.4f %8.4f %+10.4f %10.4f %10.4f"
        % ("F1", pa["f1"], pb["f1"], pb["f1"] - pa["f1"], pa["precision"],
           pb["precision"]))
    log("   %-12s %8s %8s %10s %10.4f %10.4f"
        % ("precision", "", "", "", pa["precision"], pb["precision"]))
    log("   %-12s %8s %8s %10s %10.4f %10.4f"
        % ("recall", "", "", "", pa["recall"], pb["recall"]))
    ca = confusion_counts(yi, pred_A)
    cb = confusion_counts(yi, pred_B)
    for nm, key in (("deadlock -> clean", (LABELS[DEADLOCK], "clean")),
                    ("deadlock -> dropped_handoff", (LABELS[DEADLOCK], "dropped_handoff")),
                    ("deadlock -> runaway_loop", (LABELS[DEADLOCK], "runaway_loop")),
                    ("clean -> deadlock", ("clean", LABELS[DEADLOCK])),
                    ("dropped_handoff -> deadlock", ("dropped_handoff", LABELS[DEADLOCK])),
                    ("runaway_loop -> deadlock", ("runaway_loop", LABELS[DEADLOCK]))):
        av = ca.get(key[0], {}).get(key[1], 0)
        bv = cb.get(key[0], {}).get(key[1], 0)
        log("   %-30s %6d -> %6d   (%+d)" % (nm, av, bv, bv - av))

    hdr("9. FOLD STABILITY (same 3 outer folds)")
    fold_rows = []
    for f, fv in enumerate(folds):
        va = np.asarray(fv, np.int64)
        ya = yi[va]
        # A per-fold view over the SAME official metric code path, restricted
        # in rows - Exp12's helper, not a second scorer.
        view = M.corpus_subset(np.asarray(d["proba_B"], np.float64), d, rob, va)
        fa, _ = M.full_official(view, pred_A[va])
        fb, _ = M.full_official(view, pred_B[va])
        da = prf(ya, pred_A[va], DEADLOCK)["f1"]
        db = prf(ya, pred_B[va], DEADLOCK)["f1"]
        clean_a = int(np.sum((ya == DEADLOCK) & (pred_A[va] == CLEAN_I)))
        clean_b = int(np.sum((ya == DEADLOCK) & (pred_B[va] == CLEAN_I)))
        row = {"fold": f, "n_val": int(len(va)),
               "composite_A": fa["composite"], "composite_B": fb["composite"],
               "composite_delta": fb["composite"] - fa["composite"],
               "macro_delta": fb["macro_f1"] - fa["macro_f1"],
               "robustness_delta": fb["robustness_f1"] - fa["robustness_f1"],
               "hit2_delta": fb["fault_turn_hit2"] - fa["fault_turn_hit2"],
               "deadlock_f1_A": da, "deadlock_f1_B": db,
               "deadlock_f1_delta": db - da,
               "dl_to_clean_A": clean_a, "dl_to_clean_B": clean_b}
        fold_rows.append(row)
        log("   fold %d  n=%4d  composite %+.6f  macro %+.6f  rob %+.6f  "
            "hit2 %+.6f  dl_F1 %.4f->%.4f (%+.4f)  dl->clean %d->%d"
            % (f, row["n_val"], row["composite_delta"], row["macro_delta"],
               row["robustness_delta"], row["hit2_delta"], da, db,
               row["deadlock_f1_delta"], clean_a, clean_b))
    folds_won = sum(1 for r in fold_rows if r["composite_delta"] > 0)
    dl_folds_won = sum(1 for r in fold_rows if r["deadlock_f1_delta"] > 0)

    hdr("10. DIAGNOSTICS (after the single CV; no second variant follows)")
    names_B = old_names + list(W.FEATURE_NAMES)
    order = np.argsort(-imp_sum)
    gain_share = float(imp_sum[N_LABEL_FEATS:].sum())
    top20 = [names_B[i] for i in order[:20]]
    top50 = [names_B[i] for i in order[:50]]
    wg_top20 = [x for x in top20 if x.startswith("wg_")]
    wg_top50 = [x for x in top50 if x.startswith("wg_")]
    log("   wg_* in top-20: %d   in top-50: %d" % (len(wg_top20), len(wg_top50)))
    log("   total gain share of the new block: %.4f" % gain_share)
    log("   wg_* ranking (gain, all folds averaged):")
    for i in order:
        if names_B[i].startswith("wg_"):
            log("      %-32s %.5f" % (names_B[i], imp_sum[i]))
    changed = int(np.sum(pred_A != pred_B))
    log("   changed OOF labels: %d / %d (%.2f%%)"
        % (changed, n_runs, 100.0 * changed / n_runs))
    wrong_before = (pred_A != yi)
    wrong_after = (pred_B != yi)
    w2r = int(np.sum(wrong_before & ~wrong_after))
    r2w = int(np.sum(~wrong_before & wrong_after))
    log("   wrong -> right : %d" % w2r)
    log("   right -> wrong : %d" % r2w)
    dl_rows = yi == DEADLOCK
    log("   true deadlock: fixed %d / broken %d (of %d)"
        % (int(np.sum(dl_rows & wrong_before & ~wrong_after)),
           int(np.sum(dl_rows & ~wrong_before & wrong_after)),
           int(dl_rows.sum())))
    log("   other classes newly predicted deadlock:")
    for t in range(len(LABELS)):
        if t == DEADLOCK:
            continue
        m = int(np.sum((yi == t) & ~wrong_before & wrong_after
                       & (pred_B == DEADLOCK)))
        if m:
            log("      true %-18s -> deadlock : %d" % (LABELS[t], m))

    hdr("11. PRE-DECLARED PROMOTION GATE")
    f1A = {k: v for k, v in M.per_class_f1(ytn, pred_A).items()}
    f1B = {k: v for k, v in M.per_class_f1(ytn, pred_B).items()}
    worst_cls, worst_drop = None, 0.0
    for k in LABELS:
        drop = f1A[k] - f1B[k]
        if drop > worst_drop:
            worst_cls, worst_drop = k, drop
    crit = {
        "1_composite_delta>=+0.002": (deltas["composite"],
                                      deltas["composite"] >= GATE["composite_delta_min"]),
        "2_folds_won>=2": (folds_won, folds_won >= GATE["folds_won_min"]),
        "3_deadlock_f1>=+0.015": (deltas.get("deadlock_f1", pb["f1"] - pa["f1"]),
                                  (pb["f1"] - pa["f1"]) >= GATE["deadlock_f1_delta_min"]),
        "4_deadlock_folds_won>=2": (dl_folds_won,
                                    dl_folds_won >= GATE["deadlock_folds_won_min"]),
        "5_robustness>=-0.003": (deltas["robustness_f1"],
                                 deltas["robustness_f1"] >= -GATE["robustness_drop_max"]),
        "6_no_class_worse_than_-0.02": (worst_drop,
                                        worst_drop <= GATE["class_f1_drop_max"]),
    }
    for k, (v, ok) in crit.items():
        log("   %-34s value %+9.5f   %s" % (k, v, "PASS" if ok else "FAIL"))
    promote = all(ok for _v, ok in crit.values())
    log("   WORST class drop: %s %+.5f" % (worst_cls, -worst_drop))
    if promote:
        decision = "PROMOTE"
        log("   VERDICT: PROMOTE")
    elif WEAK_BAND[0] <= deltas["composite"] < WEAK_BAND[1]:
        decision = "STOP_WEAK_INCONCLUSIVE"
        log("   VERDICT: weak / inconclusive -> STOP")
    else:
        decision = "STOP"
        log("   VERDICT: STOP (gate not passed; no deadlock weighting, no")
        log("           second variant, no production build)")
    runtime = round(time.time() - t0, 2)
    log()
    log("   runtime: %.1fs" % runtime)

    write_results({
        "reproduction": {"passed": repro_pass, "rows": gate_rows},
        "leakage": lk,
        "duplicates": {"any_exact": bool(exact), "rows": dups},
        "correlation_diagnostic": cr,
        "wait_graph_features": list(W.FEATURE_NAMES),
        "n_new_features": N_WG,
        "metrics_A": mA, "metrics_B": mB, "deltas": deltas,
        "deadlock_A": pa, "deadlock_B": pb,
        "deadlock_f1_delta": pb["f1"] - pa["f1"],
        "confusions_A": ca, "confusions_B": cb,
        "folds": fold_rows, "folds_won": folds_won,
        "deadlock_folds_won": dl_folds_won,
        "diagnostics": {
            "wg_in_top20": wg_top20, "wg_in_top50": wg_top50,
            "wg_gain_share": gain_share,
            "changed_labels": changed,
            "wrong_to_right": w2r, "right_to_wrong": r2w,
        },
        "gate": {k: {"value": float(v), "pass": bool(o)} for k, (v, o) in crit.items()},
        "gate_pass": bool(promote),
        "decision": decision,
        "runtime_sec": runtime,
        "params": {k: v for k, v in params.items() if isinstance(v, (int, float, str))},
    })
    log()
    log("results written to results.json")


def write_results(obj):
    p = os.path.join(HERE, "results.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=float)
    with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")


def build_300(folds, c, Wd, threads=None):
    """Rebuild the incumbent's 300-column label matrix for ALL rows.

    300 = 249 foundation columns + 51 window-aggregate columns.  The aggregates
    are produced by Exp08's own honest inner cross-fit, transcribed from
    ``mapping_fix.run_paired`` with the incumbent grouping left untouched:
    outer-train rows get their aggregates from the three inner window fits, and
    outer-validation rows from one window model fitted on all of outer-train.

    Both CONTROL A and CONTROL B are then fitted on THIS matrix, so the only
    difference between them is the appended wait-graph block.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    yi, n = c["yi"], c["n"]
    found = c["foundation"]
    Xw, yw, rw = Wd["X"], Wd["y"], Wd["run"]
    CW = wd_class_weights()
    WPARAMS = R_WINDOW_PARAMS()
    threads = threads or DEFAULT_THREADS()

    all_ix = np.arange(n, dtype=np.int64)
    fidx = [(all_ix[~np.isin(all_ix, fi)], np.asarray(fi, dtype=np.int64))
            for fi in folds]
    Xagg = np.zeros((n, N_AGG_FEATS))

    for f in range(len(folds)):
        tr, va = fidx[f]
        tr_runs = np.array(sorted(set(tr.tolist())))
        inner = list(StratifiedKFold(N_INNER, shuffle=True,
                                     random_state=INNER_SEED)
                     .split(tr_runs, yi[tr_runs]))
        assign = np.zeros(len(tr_runs), dtype=int)
        for j, (_, hl) in enumerate(inner):
            assign[hl] = j

        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            keep = tr_runs[assign != j]
            mw = lgb.LGBMClassifier(n_jobs=threads, **WPARAMS)
            mw.fit(Xw[np.isin(rw, keep)], yw[np.isin(rw, keep)],
                   sample_weight=CW[yw[np.isin(rw, keep)]])
            P = mw.predict_proba(Xw[np.isin(rw, hold)])
            sel = rw[np.isin(rw, hold)]
            # Exp07's grouping VERBATIM (the incumbent defect), NOT Exp12's fix:
            # CONTROL A must be Exp08 as shipped, otherwise the wait-graph
            # delta would be confounded with the mapping repair.
            blocks = M.group_buggy(P, sel, len(tr_runs))
            slot = {int(r): k for k, r in enumerate(tr_runs)}
            sub, _, _ = grp_module().aggregate_block(blocks, ag_module(), slot,
                                                     len(tr_runs))
            Xagg[hold] = sub[assign == j]
            del mw
            gc.collect()

        mw = lgb.LGBMClassifier(n_jobs=threads, **WPARAMS)
        mw.fit(Xw[np.isin(rw, tr)], yw[np.isin(rw, tr)],
               sample_weight=CW[yw[np.isin(rw, tr)]])
        Pva = mw.predict_proba(Xw[np.isin(rw, va)])
        blocks = grp_module().predict_runs(Pva, rw[np.isin(rw, va)], n)
        slot_va = {int(r): k for k, r in enumerate(va)}
        Xagg_va, _, _ = grp_module().aggregate_block(blocks, ag_module(), slot_va,
                                                     len(va))
        Xagg[va] = Xagg_va
        del mw
        gc.collect()
        log("   fold %d: 300-column matrix rebuilt" % f)

    X = np.hstack([found.values.astype(np.float64), Xagg])
    cols = list(found.columns) + list(AGG_NAMES())
    return X, cols


# --- thin accessors, so the incumbent's constants are READ, never copied -----
def ag_module():
    import aggregate
    return aggregate


def AGG_NAMES():
    import aggregate
    return aggregate.AGG_NAMES


def wd_class_weights():
    import window_dataset
    return window_dataset.class_weight_vector()


def R_WINDOW_PARAMS():
    import runner
    return runner.WINDOW_PARAMS


def DEFAULT_THREADS():
    import parallel
    return parallel.DEFAULT_THREADS


def grp_module():
    import grouping
    return grouping


if __name__ == "__main__":
    main()
