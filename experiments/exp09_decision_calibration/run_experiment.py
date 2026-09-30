"""Exp09 - decision-layer calibration: 6 additive class offsets, cross-fitted.

HEADLINE, and what it is not
----------------------------
This is ``OOF meta-CV``.  A 7-dimensional additive decision layer is fitted on
Exp07's existing OOF probabilities by cross-fitting over Exp07's 3 existing
outer folds, and the cross-fitted predictions are then scored ONCE.  It is NOT
a fully nested retrain of the Exp07 base model: those probabilities were
produced once by a single Exp07 run, and no offset ever saw a held-out fold
during the fold that validated it - but the base model itself was not re-fitted
inside a further inner loop.  So the public leaderboard remains the external
validation, and this number is an honest estimate of the OFFSET LAYER's
contribution only.

The one hypothesis: ``pred = argmax_c (log P_c + delta_c)``, ``delta_clean = 0``,
6 free fault offsets, deterministic coordinate search inside a search space
declared before any number was seen.  See ``calib.py`` for the frozen space and
the exact tie-break.

STOP CONDITIONS, all of which halt the run rather than produce a number:
  * the Exp08 baseline does not reproduce to 1e-12;
  * any strict-loader rejection (there is no lenient path);
  * the fast objective disagrees with the official ``evaluation.metrics`` on any
    checked candidate;
  * a held-out fold touches its own offsets;
  * the promotion gate fails (no production build, no second method tried).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(ROOT, "experiments", "exp08_window_localizer"),
           os.path.join(ROOT, "experiments", "exp07_fault_windows"),
           os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calib as K                                          # noqa: E402
from calib import Corpus, Objective, LABELS, N_CLASSES, CLEAN, FREE_CLASSES  # noqa: E402
from metrics import composite, f1_per_class                # noqa: E402

SYSTEM = K.SYSTEM
ZERO = np.zeros(N_CLASSES)

# The task brief's pre-registered Exp08 baseline.  These are reproduction
# TARGETS, not tuning knobs: nothing in the search may read them.
EXPECTED_BASELINE = {
    "macro_f1": 0.7894323366835861,
    "robustness_f1": 0.7368470046834376,
    "success_f1": 0.8643757406010988,
    "fault_turn_hit2": 0.6086111111111111,
    "composite": 0.7694453917139285,
}

# The promotion gate, pre-declared.
GATE = {
    "composite_delta_min": 0.003,
    "folds_won_min": 2,
    "robustness_drop_max": 0.003,
    "class_f1_drop_max": 0.02,
}

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL  %s" % msg)


def ok(msg):
    print("  ok    %s" % msg)


class Tee(object):
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")
        self.orig = sys.stdout

    def write(self, s):
        self.orig.write(s)
        self.fh.write(s)

    def flush(self):
        self.orig.flush()
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        finally:
            sys.stdout = self.orig


def confusion(yt, yp):
    """Rows = true, cols = predicted, both in the fixed LABELS order."""
    ti = np.array([K.L2I[t] for t in yt], dtype=np.int64)
    pi = np.array([K.L2I[p] for p in yp], dtype=np.int64)
    return np.bincount(ti * N_CLASSES + pi,
                       minlength=N_CLASSES * N_CLASSES).reshape(
                           N_CLASSES, N_CLASSES)


def conf_str(m):
    lines = []
    lines.append("      %-17s %s" % ("true \\ pred", " ".join("%5d" % i
                                                           for i in range(N_CLASSES))))
    for i, lab in enumerate(LABELS):
        lines.append("      %-17s %s" % (lab, " ".join("%5d" % v
                                                       for v in m[i])))
    lines.append("      %-17s %s" % ("", " ".join("%5s" % l[:5] for l in LABELS)))
    return "\n".join(lines)


def main():
    t_start = time.time()
    tee = Tee(os.path.join(HERE, "run.log"))
    sys.stdout = tee
    try:
        return _run(t_start)
    finally:
        tee.close()


def _run(t_start):
    print("=" * 78)
    print("Exp09 - decision-layer calibration (6 additive class offsets)")
    print("=" * 78)

    res = {"experiment": "exp09_decision_calibration", "system": SYSTEM,
           "n_runs": K.N_RUNS, "search_space": {
               "form": "pred = argmax_c (log(max(P_c, 1e-15)) + delta_c)",
               "start_units": [0] * N_CLASSES,
               "clean_offset": "pinned to 0",
               "bound": K.BOUND, "grid_unit": K.UNIT,
               "step_schedule": list(K.STEP_SCHEDULE),
               "rule": "deterministic coordinate descent, best single move by "
                       "TRAINING composite, tie-break composite > sum(delta^2) "
                       "> fixed class order",
               "objective": "official evaluation.metrics.composite()"},
           "baseline_expected": EXPECTED_BASELINE, "gate": GATE}

    # ---- 1. strict, verified inputs --------------------------------------
    print("\n[1] verified Exp07 OOF + sealed window peaks (strict, fail closed)")
    d = K.load_verified()
    ys, yi = d["ys"], d["yi"]
    print("  ok    %d provenance checks passed, load %.2fs"
          % (len(d["checks"]), d["load_sec"]))
    folds, fold_sha, fold_sha_want = K.fold_ids(yi, d["manifest"])
    if fold_sha != fold_sha_want:
        fail("re-derived outer fold indices differ from the sealed manifest")
    else:
        ok("the 3 outer folds re-derived from seed 0 match the sealed manifest "
           "exactly (index-vector hashes)")

    proba = d["proba"]
    peak_pos = d["peak_pos"]
    from_labels = np.asarray(LABELS)[d["labels"]].astype(str)
    if not (proba.argmax(1) == d["labels"]).all():
        fail("argmax(verified proba) != verified labels")
    else:
        ok("argmax(verified proba) == verified labels on all %d rows" % K.N_RUNS)

    fturn = d["train"]["fault_turn"].values
    rob = K.robustness_mask()
    corpus = Corpus(proba, yi, fturn, peak_pos, rob, labels=d["labels"])
    ok("corpus precomputed: logP %s, candidate L1 turn matrix %s, "
       "robustness rows %d"
       % (corpus.logP.shape, corpus.T.shape, int(rob.sum())))

    full_ix = np.arange(K.N_RUNS)
    obj_full = Objective(corpus, full_ix)

    # ---- 2. REPRODUCTION GATE --------------------------------------------
    print("\n" + "=" * 78)
    print("[2] REPRODUCTION GATE - Exp08 baseline, zero offsets, official metrics")
    print("=" * 78)
    zero_units = np.zeros(N_CLASSES, dtype=np.int64)
    zero_offsets = K.to_offsets(zero_units)
    m0 = obj_full.official(zero_offsets)     # no fast path involved

    base_pred = corpus.predict(zero_offsets)
    committed = pd.read_csv(os.path.join(ROOT, "experiments",
                                         "exp07_fault_windows",
                                         "oof_%s.csv" % SYSTEM))
    n_bad = int((committed["y_pred"].values.astype(str)
                 != np.asarray(LABELS)[base_pred]).sum())
    if n_bad:
        fail("zero-offset labels differ from the committed Exp07 OOF CSV at "
             "%d row(s)" % n_bad)
    else:
        ok("zero offsets reproduce the committed Exp07 OOF labels exactly "
           "on all %d rows" % K.N_RUNS)
    if not np.array_equal(base_pred, corpus.labels):
        fail("zero-offset argmax != the verified hard labels")
    else:
        ok("log(P) + 0 reproduces argmax(P) exactly on all %d rows" % K.N_RUNS)

    good = True
    for key in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
                "composite"):
        got, want = m0[key], EXPECTED_BASELINE[key]
        if abs(got - want) > K.TOL:
            good = False
            fail("BASELINE %s: got %.16f, expected %.16f (diff %.3e)"
                 % (key, got, want, abs(got - want)))
        else:
            ok("BASELINE %-16s %.16f  (exact)" % (key, got))
    if abs(m0["composite"] - composite({
            "macro_f1": m0["macro_f1"], "robustness_f1": m0["robustness_f1"],
            "success_f1": m0["success_f1"],
            "fault_turn_hit2": m0["fault_turn_hit2"]})) > K.TOL:
        fail("composite() is not self-consistent")
    print("  ok    composite() called directly from evaluation.metrics")
    res["baseline"] = dict(m0)
    res["reproduction_gate"] = {"passed": bool(good), "expected": EXPECTED_BASELINE,
                                "got": dict(m0), "tolerance": K.TOL}
    if not good:
        print("\nREPRODUCTION GATE FAILED - STOPPING.  No calibration is run on "
              "top of a baseline that does not reproduce.")
        res["stopped"] = "reproduction gate failed"
        _dump(res, t_start)
        return 1
    print("\n  REPRODUCTION GATE PASSED - Exp08 baseline exact on all 5 numbers.")

    # ---- 3. fast objective vs the official metrics -----------------------
    print("\n" + "=" * 78)
    print("[3] FAST OBJECTIVE must equal the official metrics, always")
    print("=" * 78)
    # Real, non-trivial candidates - not just the zero vector - because a fast
    # path that only agrees at delta=0 has proved nothing.
    probe_units = [np.zeros(N_CLASSES, dtype=np.int64),
                   np.array([0, 4, -4, 0, 0, 2, -2], dtype=np.int64),
                   np.array([0, -8, 8, 4, 4, -4, 8], dtype=np.int64),
                   np.array([0, 16, 8, -4, 0, -8, 4], dtype=np.int64),
                   np.array([0, 32, -32, 0, 0, 0, 0], dtype=np.int64)]
    probe_units += [np.array([0, int(round(v / K.UNIT)), 0, 0, 0, 0, 0],
                             dtype=np.int64) for v in (-0.05, -0.01, 0.01, 0.05)]
    worst = 0.0
    for u in probe_units:
        off = K.to_offsets(u)
        K.check_bounds(off)
        a = obj_full.metrics(off)           # fast path
        b = obj_full.official(off)         # evaluation.metrics, no fast path
        for k in ("macro_f1", "robustness_f1", "fault_turn_hit2", "composite"):
            worst = max(worst, abs(a[k] - b[k]))
        if max(abs(a[k] - b[k]) for k in
               ("macro_f1", "robustness_f1", "fault_turn_hit2", "composite")) > K.TOL:
            fail("fast objective disagrees with the official metric at units %r: "
                 "%r vs %r" % (u.tolist(), a, b))
    if worst <= K.TOL:
        ok("fast objective == official metrics on %d probe candidates, "
           "max abs diff %.2e" % (len(probe_units), worst))
    res["fast_objective_check"] = {"n_probes": len(probe_units),
                                   "max_abs_diff": worst, "passed": worst <= K.TOL}
    if worst > K.TOL:
        _dump(res, t_start)
        return 1

    # ---- 4. CROSS-FITTED CALIBRATION ------------------------------------
    print("\n" + "=" * 78)
    print("[4] OOF meta-CV - 6 additive offsets, cross-fitted over the 3 folds")
    print("=" * 78)
    print("  For each fold f: fit offsets on the OTHER two folds only, freeze,")
    print("  apply to fold f.  Nothing about fold f is used to pick its offsets.")
    print("  NOTE: OOF meta-CV is not a fully nested retrain of Exp07; the")
    print("  public leaderboard remains the external validation.")

    fold_of = np.zeros(K.N_RUNS, dtype=np.int64)
    for f, va in enumerate(folds):
        fold_of[va] = f
    assert (np.bincount(fold_of, minlength=3) > 0).all()

    cf_pred = np.full(K.N_RUNS, -1, dtype=np.int64)
    cf_units = [None] * 3
    fold_search_info = [None] * 3
    res["folds"] = []

    for f in range(3):
        val_ix = np.asarray(folds[f], dtype=np.int64)
        tr_ix = np.where(fold_of != f)[0]
        # ---- the discipline: the held-out rows are physically absent -----
        obj_tr = Objective(corpus, tr_ix)
        assert not np.intersect1d(val_ix, tr_ix).size, "train/val overlap"
        print("\n  --- fold %d: calibration-train n=%d, held-out n=%d ---"
              % (f, len(tr_ix), len(val_ix)))
        t_f = time.time()
        units, info = K.coordinate_search(obj_tr)
        dt = time.time() - t_f
        offsets = K.to_offsets(units)
        K.check_bounds(offsets)
        cf_units[f] = units
        fold_search_info[f] = info
        print("    offsets found on calibration-train:")
        for c in range(N_CLASSES):
            print("      %-18s %+.4f%s" % (LABELS[c], offsets[c],
                                           "   (PINNED)" if c == CLEAN else ""))
        print("    %d candidates in %.1fs; train composite %.16f -> %.16f"
              % (info["n_candidates"], dt, info["start_composite"],
                 info["final_composite"]))

        # ---- apply the FROZEN offsets to the held-out fold --------------
        obj_va = Objective(corpus, val_ix)
        # guard: the training Objective cannot see these rows at all
        assert not set(np.asarray(obj_tr.ix).tolist()) & set(val_ix.tolist()), \
            "held-out fold leaked into its own calibration"
        pred_va = obj_va.predict(offsets)
        cf_pred[val_ix] = pred_va
        zero_va = obj_va.predict(np.zeros(N_CLASSES))
        changed = int((pred_va != zero_va).sum())
        m_va = obj_va.official(offsets)
        m_va0 = obj_va.official(np.zeros(N_CLASSES))
        d_va = m_va["composite"] - m_va0["composite"]

        res["folds"].append({
            "fold": f, "n_train": int(len(tr_ix)), "n_val": int(len(val_ix)),
            "val_sha256": fold_sha[f],
            "offsets_train_units": units.tolist(),
            "offsets_train": [float(x) for x in offsets],
            "search": {"n_candidates": info["n_candidates"],
                       "start_composite": info["start_composite"],
                       "train_composite": info["final_composite"],
                       "moves_per_step": [s["n_moves"] for s in info["steps"]],
                       "search_sec": round(dt, 2)},
            "heldout": {"n_labels_changed": changed,
                        "A_zero_offsets": m_va0, "B_calibrated": m_va,
                        "delta_composite": d_va,
                        "wins": bool(d_va > 0)},
        })
        print("    held-out fold %d: labels changed %d/%d; composite "
              "%.16f -> %.16f (%+.16f) %s"
              % (f, changed, len(val_ix), m_va0["composite"], m_va["composite"],
                 d_va, "WIN" if d_va > 0 else "LOSS"))

    if (cf_pred < 0).any():
        fail("some rows received no cross-fitted prediction")
    ok("cross-fitted prediction vector assembled for all %d rows" % K.N_RUNS)

    # ---- 5. A vs B on all 10000 -----------------------------------------
    print("\n" + "=" * 78)
    print("[5] A (Exp08, no offsets) vs B (cross-fitted offsets), all %d rows"
          % K.N_RUNS)
    print("=" * 78)
    yt = [LABELS[i] for i in corpus.yi]
    ypA = [LABELS[i] for i in base_pred]
    ypB = [LABELS[i] for i in cf_pred]

    mA = obj_full.official(np.zeros(N_CLASSES))
    # through the labels, so the METRIC code path is identical for A and B
    turnsA = corpus.T[np.arange(K.N_RUNS), base_pred]
    turnsB = corpus.T[np.arange(K.N_RUNS), cf_pred]
    from metrics import fault_turn_hit_at_k, macro_f1   # noqa: E402
    h2A = fault_turn_hit_at_k(yt, ypA, [int(t) for t in fturn],
                              [int(t) for t in turnsA], k=2)
    h2B = fault_turn_hit_at_k(yt, ypB, [int(t) for t in fturn],
                              [int(t) for t in turnsB], k=2)
    mfA = macro_f1(yt, ypA, LABELS)
    mfB = macro_f1(yt, ypB, LABELS)
    rob_ix = np.where(corpus.rob)[0]
    rfA = macro_f1([yt[i] for i in rob_ix], [ypA[i] for i in rob_ix], LABELS)
    rfB = macro_f1([yt[i] for i in rob_ix], [ypB[i] for i in rob_ix], LABELS)
    A = {"macro_f1": mfA, "robustness_f1": rfA, "success_f1": K.SUCCESS_F1_PINNED,
         "fault_turn_hit2": h2A}
    B = {"macro_f1": mfB, "robustness_f1": rfB, "success_f1": K.SUCCESS_F1_PINNED,
         "fault_turn_hit2": h2B}
    A["composite"] = composite(A)
    B["composite"] = composite(B)
    accA = float((base_pred == corpus.yi).mean())
    accB = float((cf_pred == corpus.yi).mean())
    pcA = f1_per_class(yt, ypA, LABELS)
    pcB = f1_per_class(yt, ypB, LABELS)
    cmA = confusion(yt, ypA)
    cmB = confusion(yt, ypB)
    n_changed = int((cf_pred != base_pred).sum())
    n_clean_turn = int(((cf_pred == CLEAN) & (turnsB == -1)).sum())
    if n_clean_turn != int((cf_pred == CLEAN).sum()):
        fail("a cross-fitted clean prediction did not receive fault_turn -1")
    else:
        ok("every cross-fitted clean prediction has fault_turn == -1 (%d rows)"
           % n_clean_turn)

    for tag, M in (("A", A), ("B", B)):
        print("  %s  macro %.16f  robustness %.16f  success %.16f  hit@2 %.16f"
              % (tag, M["macro_f1"], M["robustness_f1"], M["success_f1"],
                 M["fault_turn_hit2"]))
        print("     composite %.16f" % M["composite"])
    print("  delta  macro %+.16f  robustness %+.16f  hit@2 %+.16f  "
          "composite %+.16f" % (B["macro_f1"] - A["macro_f1"],
                                B["robustness_f1"] - A["robustness_f1"],
                                B["fault_turn_hit2"] - A["fault_turn_hit2"],
                                B["composite"] - A["composite"]))
    print("  accuracy (diagnostic only) %.6f -> %.6f (%+.6f)"
          % (accA, accB, accB - accA))
    print("  OOF labels changed by the cross-fitted offsets: %d / %d (%.2f%%)"
          % (n_changed, K.N_RUNS, 100.0 * n_changed / K.N_RUNS))

    print("\n  per-class F1 (official f1_per_class):")
    print("    %-18s %10s %10s %10s" % ("class", "A", "B", "delta"))
    for c in LABELS:
        print("    %-18s %10.6f %10.6f %+10.6f" % (c, pcA[c], pcB[c],
                                                  pcB[c] - pcA[c]))
    print("\n  A confusion (rows = true, cols = pred):")
    print(conf_str(cmA))
    print("\n  B confusion (rows = true, cols = pred):")
    print(conf_str(cmB))
    n_turn_changed = int((turnsB != turnsA).sum())

    res["comparison"] = {
        "A_exp08_no_offsets": A, "B_cross_fitted": B,
        "delta": {k: B[k] - A[k] for k in A},
        "accuracy": {"A": accA, "B": accB, "delta": accB - accA,
                     "note": "diagnostic only, accuracy is NOT the objective"},
        "per_class_f1": {"A": pcA, "B": pcB,
                         "delta": {c: pcB[c] - pcA[c] for c in LABELS}},
        "confusion_A": cmA.tolist(), "confusion_B": cmB.tolist(),
        "n_labels_changed": n_changed,
        "n_fault_turn_changed": n_turn_changed,
        "note": "hit@2 uses the peak of the PREDICTED class, so it moves with "
                "the label; it is not held fixed",
    }

    # ---- 6. offset stability across folds -------------------------------
    print("\n" + "=" * 78)
    print("[6] OFFSET STABILITY across the three calibration-train fits")
    print("=" * 78)
    U = np.array([u for u in cf_units], dtype=np.int64)
    signs = np.sign(U[:, FREE_CLASSES])
    unstable = []
    for j, c in enumerate(FREE_CLASSES):
        col = U[:, c]
        s = set(int(x) for x in np.sign(col))
        span = int(col.max() - col.min())
        agree = len(s) == 1 and 0 not in s
        print("    %-18s units %-18s offsets %s  spread %d  %s"
              % (LABELS[c], str(col.tolist()),
                 " ".join("%+.4f" % (x * K.UNIT) for x in col), span,
                 "consistent" if agree else "SIGN/VALUE DISAGREEMENT"))
        if not agree:
            unstable.append(LABELS[c])
    same_dir = bool((signs == signs[0]).all())
    print("\n  identical direction on all six classes in all three fits: %s"
          % same_dir)
    if unstable or not same_dir:
        print("  ** INSTABILITY: %d of 6 fault offsets are not consistent in "
              "sign across folds. A single vector is a choice, not a "
              "measurement; treat the cross-fitted number as the estimate and "
              "the shipped offsets as a compromise." % len(set(unstable)))
    else:
        print("  all six offsets agree in sign across the three fits, so the "
              "vector is at least directionally stable.")
    res["offset_stability"] = {
        "units_per_fold": [[int(x) for x in u] for u in cf_units],
        "offsets_per_fold": {str(f): [float(x) for x in K.to_offsets(u)]
                             for f, u in enumerate(cf_units)},
        "inconsistent_classes": sorted(set(unstable)),
        "all_six_same_direction": same_dir,
    }

    # ---- 7. PROMOTION GATE ----------------------------------------------
    print("\n" + "=" * 78)
    print("[7] PROMOTION GATE (pre-declared)")
    print("=" * 78)
    wins = sum(1 for r in res["folds"] if r["heldout"]["wins"])
    checks = []

    checks.append(("composite delta >= %+.3f" % GATE["composite_delta_min"],
                   B["composite"] - A["composite"], GATE["composite_delta_min"],
                   (B["composite"] - A["composite"]) >= GATE["composite_delta_min"]))
    checks.append(("wins >= %d/3 folds" % GATE["folds_won_min"], wins, 3,
                   wins >= GATE["folds_won_min"]))
    d_rob = B["robustness_f1"] - A["robustness_f1"]
    checks.append(("robustness drop <= %.3f" % GATE["robustness_drop_max"],
                   -d_rob, GATE["robustness_drop_max"],
                   (-d_rob) <= GATE["robustness_drop_max"]))
    worst_cls = min(pcA, key=lambda c: pcB[c] - pcA[c])
    worst_drop = pcA[worst_cls] - pcB[worst_cls]
    checks.append(("worst class F1 drop <= %.2f (%s)" % (GATE["class_f1_drop_max"],
                                                         worst_cls),
                   worst_drop, GATE["class_f1_drop_max"],
                   worst_drop <= GATE["class_f1_drop_max"]))
    gate_passed = all(c[3] for c in checks)
    for name, got, want, good_ in checks:
        print("  [%s] %-46s %+.6f" % ("PASS" if good_ else "FAIL", name, got))
    res["promotion_gate"] = {
        "passed": bool(gate_passed),
        "criteria": [{"name": c[0], "value": c[1], "threshold": c[2],
                      "passed": bool(c[3])} for c in checks],
        "folds_won": wins,
        "worst_class_f1_drop": {"class": worst_cls, "drop": worst_drop},
    }
    print("\n  PROMOTION GATE: %s" % ("PASS" if gate_passed else "FAIL"))

    # ---- 8. FINAL offsets for production, fitted on ALL 10000 OOF -------
    print("\n" + "=" * 78)
    print("[8] FINAL production offsets - same search, same frozen space,")
    print("    fitted on ALL %d OOF rows.  DEPLOYMENT ONLY." % K.N_RUNS)
    print("    The score of this fit is NOT a validation score and is reported")
    print("    separately from the cross-fitted evaluation above.")
    print("=" * 78)
    if not gate_passed:
        print("\n  GATE FAILED - no production offsets are fitted and no second")
        print("  calibration method is tried.  The experiment stops here.")
        res["final_offsets"] = {"fitted": False,
                                "reason": "promotion gate failed"}
        res["promoted"] = False
        _dump(res, t_start)
        return 0

    t_f = time.time()
    final_units, finfo = K.coordinate_search(obj_full)
    final_offsets = K.to_offsets(final_units)
    K.check_bounds(final_offsets)
    fin_sec = time.time() - t_f
    print("\n  FINAL OFFSETS (deployment constants):")
    print("    %r" % ([round(float(x), 4) for x in final_offsets],))
    for c in range(N_CLASSES):
        print("      %-18s %+.4f%s" % (LABELS[c], final_offsets[c],
                                         "   (PINNED)" if c == CLEAN else ""))
    print("    %d candidates in %.1fs" % (finfo["n_candidates"], fin_sec))
    final_pred = obj_full.predict(final_offsets)
    m_in = obj_full.official(final_offsets)      # IN-SAMPLE, not validation
    n_fin_changed = int((final_pred != base_pred).sum())
    print("    in-sample (NOT a validation score) composite %.16f, "
          "%d/%d OOF labels would change" % (m_in["composite"], n_fin_changed,
                                              K.N_RUNS))
    res["final_offsets"] = {
        "fitted": True,
        "units": [int(x) for x in final_units],
        "offsets": [float(x) for x in final_offsets],
        "n_candidates": finfo["n_candidates"],
        "search_sec": round(fin_sec, 2),
        "in_sample_full_oof": m_in,
        "in_sample_labels_changed": n_fin_changed,
        "moves_per_step": [s["n_moves"] for s in finfo["steps"]],
        "warning": "fitted on all 10000 OOF rows - deployment only, NOT a "
                   "validation score; the headline is the cross-fitted B",
    }
    res["promoted"] = True
    print("\n  PROMOTED.  Writing the production constants for the submission.")
    with open(os.path.join(HERE, "final_offsets.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"labels": LABELS, "offsets": [float(x) for x in final_offsets],
                   "units": [int(x) for x in final_units],
                   "grid_unit": K.UNIT, "bound": K.BOUND,
                   "fitted_on": "all 10000 OOF rows, Exp07 B_plus_window",
                   "in_sample_composite": m_in["composite"],
                   "note": res["final_offsets"]["warning"]}, fh, indent=2)

    _dump(res, t_start)
    return 0


def _dump(res, t_start):
    res["runtime_sec"] = round(time.time() - t_start, 2)
    res["failed_checks"] = FAILS
    res["caveat"] = ("OOF meta-CV is not a fully nested retrain of Exp07; the "
                     "public leaderboard remains the external validation.")
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("\nwrote %s" % os.path.join(HERE, "results.json"))
    print("total research runtime %.1fs (%.2f min)"
          % (time.time() - t_start, (time.time() - t_start) / 60.0))


if __name__ == "__main__":
    sys.exit(main())
