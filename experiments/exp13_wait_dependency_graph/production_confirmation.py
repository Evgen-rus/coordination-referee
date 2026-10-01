"""Exp13 validation - production-semantics confirmation (corrected mapping).

WHY
---
Exp13's research result compares the candidate against the SEALED Exp08 OOF, and
that comparison is valid research evidence: A reproduces the sealed artifact bit
for bit.  But Exp12 established that the shipped production Exp08 uses CORRECTED
aggregation semantics, while the historical OOF carries the global/local indexing
defect.  Production Exp08 is therefore not measured by the historical OOF, and
before building anything we must confirm the wait-graph gain survives when BOTH
arms use the corrected semantics.

THIS IS NOT A NEW HYPOTHESIS.  The 12 `wg_*` features, the parser, LightGBM and
its parameters, the seeds, the folds and the thresholds are all frozen.  Only the
window-aggregation semantics of the shared base change, from Exp07's defective
grouping to Exp12's corrected local mapping, for BOTH arms at once.

    CONTROL C = corrected representation, no wait graph   (= Exp12's arm B)
    CANDIDATE D = the SAME corrected matrices + the frozen 12 wg_* columns

C and D consume ONE set of inner window fits per fold and ONE pair of 300-column
fold-local matrices, so they differ in exactly the 12 appended columns.

STOP CONDITIONS
---------------
  * CONTROL C does not reproduce Exp12's committed corrected numbers -> STOP;
  * otherwise the production-confirmation gate and the original strict Exp13
    gate are both reported, and neither is retuned.
"""

from __future__ import annotations

import gc
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
EXP12 = os.path.join(ROOT, "experiments", "exp12_fix_window_run_mapping")
for _p in (EXP12, EXP11, EXP08, EXP10, EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aggregate as ag                       # noqa: E402
import lightgbm as lgb                       # noqa: E402
import mapping_fix as M                      # noqa: E402  (Exp12, read-only)
import parallel as par                       # noqa: E402
import runner as R                           # noqa: E402
import window_dataset as wd                   # noqa: E402

import ensemble as ENS                        # noqa: E402
from blend import LABELS, L2I                # noqa: E402
from blend import N_CLASSES, SUCCESS_F1_PINNED, TOL   # noqa: E402
from metrics import composite, fault_turn_hit_at_k, macro_f1   # noqa: E402

import importlib.util                        # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "exp13_wait_features", os.path.join(HERE, "features.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

CLEAN = 0
DEADLOCK = LABELS.index("deadlock")
CLEAN_I = LABELS.index("clean")
N_INNER = 3
INNER_SEED = 0
N_LABEL = 300
N_WG = W.N_FEATURES

# Exp12's committed corrected numbers (arm B = the fixed mapping).
EXP12_COMMITTED = {
    "macro_f1": 0.7873785254579829,
    "robustness_f1": 0.7368422139406114,
    "success_f1": 0.8643757406010988,
    "fault_turn_hit2": 0.6093055555555555,
    "composite": 0.7684867328598647,
}
EXP12_SHA_B = "b7e92bd4e14fa59c8c340e65c70317d49d7cca6c18a841fd248217617d1fc9d1"

# The production-confirmation gate (declared here, before any fit).
PROD_GATE = {"composite_delta_min_exclusive": 0.0, "folds_won_min": 2,
             "deadlock_f1_delta_min_exclusive": 0.0,
             "deadlock_folds_won_min": 2, "class_f1_drop_max": 0.02}
# The ORIGINAL strict Exp13 gate, reported unchanged.
STRICT_GATE = {"composite_delta_min": 0.002, "folds_won_min": 2,
               "deadlock_f1_delta_min": 0.015, "deadlock_folds_won_min": 2,
               "robustness_drop_max": 0.003, "class_f1_drop_max": 0.02}

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
    import cache as ca
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
    """Exp11's scoring path, unchanged: evaluation.metrics is the only definition."""
    c = ENS.corpus_for(P, d, rob, labels=np.asarray(pred, np.int64))
    m, _ = ENS.full_official(c, np.asarray(pred, np.int64))
    return m


def official_subset(d, rob, rows, pred_rows):
    """Per-fold official score over the same metric code path, restricted in rows."""
    rows = np.asarray(rows, np.int64)
    pr = np.asarray(pred_rows, np.int64)
    yt = [LABELS[int(d["yi"][i])] for i in rows]
    yp = [LABELS[int(i)] for i in pr]
    fturn = d["train"]["fault_turn"].values
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


def corrected_fold_matrices(folds, c, Wd, log=log):
    """Per outer fold: Exp07's honest window fits + Exp12's CORRECTED grouping.

    Identical probabilities to Exp12's run; only the global->local mapping of the
    selected window rows differs from Exp07.  Emits one pair of 300-column
    fold-local matrices, consumed by both CONTROL C and CANDIDATE D.
    """
    yi, n = c["yi"], c["n"]
    found = c["foundation"]
    Xw, yw, rw = Wd["X"], Wd["y"], Wd["run"]
    CW = wd.class_weight_vector()

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
        g2l = M.global_to_local_map(tr_runs)
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
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            sel = rw[np.isin(rw, hold)]
            # THE CORRECTED MAPPING - Exp12's group_fixed, verbatim.
            blocks, slot = M.group_fixed(probs[(f, j)], sel, g2l, len(tr_runs))
            sub, _, _ = M.grp.aggregate_block(blocks, ag, slot, len(tr_runs))
            Xagg_tr[assign == j] = sub[assign == j]
        del probs
        gc.collect()

        mw = lgb.LGBMClassifier(n_jobs=par.DEFAULT_THREADS, **R.WINDOW_PARAMS)
        mw.fit(Xw[np.isin(rw, tr)], yw[np.isin(rw, tr)],
               sample_weight=CW[yw[np.isin(rw, tr)]])
        Pva = mw.predict_proba(Xw[np.isin(rw, va)])
        # outer-validation aggregates are UNCHANGED (production semantics, n=10000)
        blocks = M.grp.predict_runs(Pva, rw[np.isin(rw, va)], n)
        slot_va = {int(r): k for k, r in enumerate(va)}
        Xagg_va, _, _ = M.grp.aggregate_block(blocks, ag, slot_va, len(va))
        del mw
        gc.collect()

        XC_tr = pd.DataFrame(np.hstack([found.iloc[tr].values, Xagg_tr]),
                             columns=cols)
        XC_va = pd.DataFrame(np.hstack([found.iloc[va].values, Xagg_va]),
                             columns=cols)
        log("    fold %d: corrected 300-col matrices built  %.1fs"
            % (f, time.time() - t_f))
        yield f, tr, va, XC_tr, XC_va
        del XC_tr, XC_va, Xagg_tr, Xagg_va
        gc.collect()


def main():
    t0 = time.time()
    bar("EXP13 VALIDATION - PRODUCTION SEMANTICS (corrected mapping)")
    log("CONTROL C = corrected representation, no wait graph  (= Exp12 arm B)")
    log("CANDIDATE D = the SAME corrected matrices + the frozen 12 wg_* columns")
    log("frozen: features, parser, params, seeds, folds, thresholds, no tuning")

    hdr("1. VERIFIED INPUTS")
    d = ENS.load_verified()
    yi = d["yi"]
    folds = [np.asarray(v, np.int64) for v in d["folds"]]
    n = len(yi)
    rob = np.asarray(ENS.robustness_mask(), dtype=bool)
    log("   runs=%d folds=%d checks=%d" % (n, len(folds), len(d["checks"])))

    c, Wd = ENS._prepare()
    log("   foundation %s  windows %s" % (c["foundation"].shape, Wd["X"].shape))

    hdr("2. WAIT-GRAPH BLOCK (frozen, fold-independent)")
    Xwg = W.build_matrix(c["runs"])
    assert Xwg.shape == (n, N_WG), Xwg.shape
    log("   %s  (%s)" % (Xwg.shape, ",".join(W.FEATURE_NAMES)))

    hdr("3. PAIRED C/D ON THE CORRECTED MATRICES (one set of window fits)")
    params = ENS.label_params(42)
    log("   label params from common.lgb_label(), random_state=%s"
        % params["random_state"])
    oof_C = np.zeros((n, N_CLASSES), np.float64)
    oof_D = np.zeros((n, N_CLASSES), np.float64)
    imp_D = np.zeros(N_LABEL + N_WG, np.float64)
    fold_idx = []

    for f, tr, va, XC_tr, XC_va in corrected_fold_matrices(folds, c, Wd):
        fold_idx.append((f, tr, va))
        clfC = lgb.LGBMClassifier(**params).fit(XC_tr, yi[tr])
        oof_C[va] = clfC.predict_proba(XC_va)
        log("    fold %d: C fitted on %d rows" % (f, len(tr)))

        XD_tr = pd.DataFrame(np.hstack([XC_tr.values, Xwg[tr]]),
                             columns=list(XC_tr.columns) + list(W.FEATURE_NAMES))
        XD_va = pd.DataFrame(np.hstack([XC_va.values, Xwg[va]]),
                             columns=list(XC_va.columns) + list(W.FEATURE_NAMES))
        assert np.array_equal(XC_tr.values, XD_tr.values[:, :N_LABEL])
        clfD = lgb.LGBMClassifier(**params).fit(XD_tr, yi[tr])
        oof_D[va] = clfD.predict_proba(XD_va)
        im = clfD.booster_.feature_importance(importance_type="gain")
        imp_D += (im / (im.sum() or 1.0)) / len(folds)
        log("    fold %d: D fitted on the SAME rows + %d columns" % (f, N_WG))
        del clfC, clfD, XD_tr, XD_va
        gc.collect()

    oof_C32 = oof_C.astype(np.float32)
    oof_D32 = oof_D.astype(np.float32)
    np.save(os.path.join(HERE, "oof_C_corrected.npy"), oof_C32)
    np.save(os.path.join(HERE, "oof_D_corrected_wg.npy"), oof_D32)

    # ------------------------------------------------------------------
    hdr("4. CONTROL C REPRODUCES EXP12's COMMITTED CORRECTED NUMBERS")
    pred_C = oof_C32.argmax(1)
    mC = official_full(oof_C32, d, rob, pred_C)
    c_sha = sha32(oof_C32)
    rows = []
    okm = True
    for k, exp in EXP12_COMMITTED.items():
        got = mC[k]
        good = abs(got - exp) <= TOL
        okm &= good
        rows.append({"metric": k, "expected": exp, "got": got,
                     "abs_diff": abs(got - exp), "pass": bool(good)})
        log("   %-18s expected %.16f got %.16f diff %.2e %s"
            % (k, exp, got, abs(got - exp), "PASS" if good else "FAIL"))
    sha_ok = bool(c_sha == EXP12_SHA_B)
    log("   C sha256      : %s..." % c_sha[:32])
    log("   exp12 sha256 B: %s..." % EXP12_SHA_B[:32])
    log("   sha identical : %s" % sha_ok)
    repro_ok = bool(okm and sha_ok)
    RES["control_c_reproduction"] = {"passed": repro_ok, "rows": rows,
                                     "sha_c": c_sha, "sha_exp12_b": EXP12_SHA_B,
                                     "sha_identical": sha_ok}
    log("   CONTROL C REPRODUCTION: %s" % ("PASS" if repro_ok else "FAIL"))
    if not repro_ok:
        log()
        log("   STOP: CONTROL C does not reproduce Exp12's committed corrected")
        log("   numbers. Candidate D is NOT scored and nothing is built.")
        RES["decision"] = "STOP_C_REPRODUCTION"
        dump(t0)
        return 1

    # ------------------------------------------------------------------
    hdr("5. C vs D METRICS")
    pred_D = oof_D32.argmax(1)
    mD = official_full(oof_D32, d, rob, pred_D)
    deltas = {k: mD[k] - mC[k] for k in
              ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
               "composite")}
    log("   %-20s %16s %16s %12s" % ("metric", "C (corrected)", "D (+wg)",
                                     "delta"))
    for k in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
              "composite"):
        log("   %-20s %16.6f %16.6f %+12.6f" % (k, mC[k], mD[k], deltas[k]))

    f1C, f1D = per_class_f1(yi, pred_C), per_class_f1(yi, pred_D)
    log()
    log("   per-class F1:")
    for lab in LABELS:
        log("   %-20s %.4f -> %.4f  (%+.4f)"
            % (lab, f1C[lab], f1D[lab], f1D[lab] - f1C[lab]))

    hdr("6. MAIN TARGET: DEADLOCK")
    pc, pd_ = prf(yi, pred_C, DEADLOCK), prf(yi, pred_D, DEADLOCK)
    log("   F1        %.4f -> %.4f  (%+.4f)" % (pc["f1"], pd_["f1"],
                                                pd_["f1"] - pc["f1"]))
    log("   precision %.4f -> %.4f  (%+.4f)" % (pc["precision"], pd_["precision"],
                                                pd_["precision"] - pc["precision"]))
    log("   recall    %.4f -> %.4f  (%+.4f)" % (pc["recall"], pd_["recall"],
                                                pd_["recall"] - pc["recall"]))
    log("   tp %d->%d  fp %d->%d  fn %d->%d"
        % (pc["tp"], pd_["tp"], pc["fp"], pd_["fp"], pc["fn"], pd_["fn"]))
    cC, cD = confusion_counts(yi, pred_C), confusion_counts(yi, pred_D)
    for nm, key in (("deadlock -> clean", ("deadlock", "clean")),
                    ("clean -> deadlock", ("clean", "deadlock")),
                    ("dropped_handoff -> deadlock", ("dropped_handoff", "deadlock")),
                    ("runaway_loop -> deadlock", ("runaway_loop", "deadlock"))):
        av = cC.get(key[0], {}).get(key[1], 0)
        bv = cD.get(key[0], {}).get(key[1], 0)
        log("   %-30s %5d -> %5d  (%+d)" % (nm, av, bv, bv - av))

    hdr("7. FOLD STABILITY")
    fold_rows = []
    for f, tr, va in fold_idx:
        fc = official_subset(d, rob, va, pred_C[va])
        fd = official_subset(d, rob, va, pred_D[va])
        ya = yi[va]
        dcl = prf(ya, pred_C[va], DEADLOCK)["f1"]
        ddl = prf(ya, pred_D[va], DEADLOCK)["f1"]
        row = {"fold": f, "n_val": int(len(va)),
               "composite_C": fc["composite"], "composite_D": fd["composite"],
               "composite_delta": fd["composite"] - fc["composite"],
               "macro_delta": fd["macro_f1"] - fc["macro_f1"],
               "robustness_delta": fd["robustness_f1"] - fc["robustness_f1"],
               "hit2_delta": fd["fault_turn_hit2"] - fc["fault_turn_hit2"],
               "deadlock_f1_C": dcl, "deadlock_f1_D": ddl,
               "deadlock_f1_delta": ddl - dcl}
        fold_rows.append(row)
        log("   fold %d n=%4d composite %+.6f  macro %+.6f  rob %+.6f  "
            "hit2 %+.6f  dl_F1 %.4f->%.4f (%+.4f)"
            % (f, row["n_val"], row["composite_delta"], row["macro_delta"],
               row["robustness_delta"], row["hit2_delta"], dcl, ddl,
               row["deadlock_f1_delta"]))
    folds_won = sum(1 for r in fold_rows if r["composite_delta"] > 0)
    dl_folds_won = sum(1 for r in fold_rows if r["deadlock_f1_delta"] > 0)

    hdr("8. DIAGNOSTICS")
    names = list(c["foundation"].columns) + list(ag.AGG_NAMES) \
        + list(W.FEATURE_NAMES)
    order = np.argsort(-imp_D)
    for i in order:
        if names[i].startswith("wg_"):
            log("      %-32s %.5f" % (names[i], imp_D[i]))
    log("   wg_* gain share: %.4f" % float(imp_D[N_LABEL:].sum()))
    changed = int((pred_C != pred_D).sum())
    wb, wf = pred_C != yi, pred_D != yi
    log("   changed labels %d/%d (%.2f%%)" % (changed, n, 100.0 * changed / n))
    log("   wrong->right %d   right->wrong %d"
        % (int((wb & ~wf).sum()), int((~wb & wf).sum())))

    hdr("9. PRODUCTION CONFIRMATION GATE")
    worst_lab, worst_drop = None, 0.0
    for lab in LABELS:
        d_ = f1C[lab] - f1D[lab]
        if d_ > worst_drop:
            worst_lab, worst_drop = lab, d_
    prod = {
        "composite_delta>0": (deltas["composite"], deltas["composite"] > 0),
        "folds_won>=2": (float(folds_won), folds_won >= PROD_GATE["folds_won_min"]),
        "deadlock_f1_delta>0": (pd_["f1"] - pc["f1"],
                                (pd_["f1"] - pc["f1"]) > 0),
        "deadlock_folds_won>=2": (float(dl_folds_won),
                                  dl_folds_won >= PROD_GATE["deadlock_folds_won_min"]),
        "no_class_drop>0.02": (-worst_drop, worst_drop <= PROD_GATE["class_f1_drop_max"]),
    }
    for k, (v, o) in prod.items():
        log("   %-28s %+10.5f  %s" % (k, v, "PASS" if o else "FAIL"))
    prod_pass = all(o for _v, o in prod.values())
    log("   worst class drop: %s %+.5f" % (worst_lab, -worst_drop))
    log("   PRODUCTION CONFIRMATION: %s" % ("PASS" if prod_pass else "FAIL"))

    hdr("10. ORIGINAL STRICT Exp13 GATE (reported unchanged, not retuned)")
    strict = {
        "composite_delta>=+0.002": (deltas["composite"],
                                     deltas["composite"] >= STRICT_GATE["composite_delta_min"]),
        "folds_won>=2": (float(folds_won), folds_won >= STRICT_GATE["folds_won_min"]),
        "deadlock_f1_delta>=+0.015": (pd_["f1"] - pc["f1"],
                                      (pd_["f1"] - pc["f1"]) >= STRICT_GATE["deadlock_f1_delta_min"]),
        "deadlock_folds_won>=2": (float(dl_folds_won),
                                  dl_folds_won >= STRICT_GATE["deadlock_folds_won_min"]),
        "robustness>=-0.003": (deltas["robustness_f1"],
                               deltas["robustness_f1"] >= -STRICT_GATE["robustness_drop_max"]),
        "no_class_worse_than_-0.02": (-worst_drop,
                                      worst_drop <= STRICT_GATE["class_f1_drop_max"]),
    }
    for k, (v, o) in strict.items():
        log("   %-32s %+10.5f  %s" % (k, v, "PASS" if o else "FAIL"))
    strict_pass = all(o for _v, o in strict.values())
    log("   ORIGINAL STRICT Exp13 GATE: %s" % ("PASS" if strict_pass else "FAIL"))

    hdr("11. HISTORICAL vs CORRECTED SEMANTICS")
    hist = {"composite_delta": 0.023176, "deadlock_f1_delta": 0.11495}
    log("   historical Exp08 (defective) -> +wg : composite %+0.6f  deadlock %+0.5f"
        % (hist["composite_delta"], hist["deadlock_f1_delta"]))
    log("   corrected  Exp12 (fixed)    -> +wg : composite %+0.6f  deadlock %+0.5f"
        % (deltas["composite"], pd_["f1"] - pc["f1"]))
    survives = bool(deltas["composite"] > 0 and (pd_["f1"] - pc["f1"]) > 0)
    log("   wait-graph finding SURVIVES the mapping fix: %s" % survives)

    RES.update({
        "metrics_C": mC, "metrics_D": mD, "deltas": deltas,
        "per_class_f1_C": f1C, "per_class_f1_D": f1D,
        "deadlock_C": pc, "deadlock_D": pd_,
        "confusions_C": cC, "confusions_D": cD,
        "folds": fold_rows, "folds_won": folds_won,
        "deadlock_folds_won": dl_folds_won,
        "wg_gain_share": float(imp_D[N_LABEL:].sum()),
        "changed_labels": changed,
        "wrong_to_right": int((wb & ~wf).sum()),
        "right_to_wrong": int((~wb & wf).sum()),
        "prod_gate": {k: {"value": float(v), "pass": bool(o)}
                      for k, (v, o) in prod.items()},
        "prod_gate_pass": bool(prod_pass),
        "strict_gate": {k: {"value": float(v), "pass": bool(o)}
                        for k, (v, o) in strict.items()},
        "strict_gate_pass": bool(strict_pass),
        "historical_vs_corrected": {"historical": hist,
                                    "corrected": {"composite_delta": deltas["composite"],
                                                  "deadlock_f1_delta":
                                                      pd_["f1"] - pc["f1"]},
                                    "finding_survives": survives},
    })
    dump(t0)
    return 0


def dump(t0):
    RES["runtime_sec"] = round(time.time() - t0, 2)
    with open(os.path.join(HERE, "results_production.json"), "w",
              encoding="utf-8") as fh:
        json.dump(RES, fh, indent=2, sort_keys=True, default=float)
    with open(os.path.join(HERE, "run_production.log"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    log()
    log("   runtime %.1fs; results_production.json + run_production.log written"
        % RES["runtime_sec"])


if __name__ == "__main__":
    raise SystemExit(main())