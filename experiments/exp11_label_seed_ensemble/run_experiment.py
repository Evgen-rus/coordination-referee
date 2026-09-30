"""Exp11 - does a FIXED 3-seed probability ensemble of the label head beat Exp08?

ONE HYPOTHESIS
--------------
Exp08's composite rests on a single label head, Exp07's system B fitted by
``common.lgb_label()`` with ``random_state=42``.  LightGBM is stochastic under
``subsample=0.9``/``subsample_freq=1`` and ``colsample_bytree=0.8``, so the SAME
rows and the SAME 300 columns fitted with a different ``random_state`` is a
genuinely different model.  Exp11 asks whether averaging three of them beats the
incumbent:

    P_ensemble = (P_seed42 + P_seed137 + P_seed2026) / 3
    pred       = argmax(P_ensemble)

The seed set is FROZEN and was fixed before any number was seen.  No other
seeds, no 2 or 4 models, no weights, no best-seed selection, no offsets, no
calibration, no thresholds, no A/B blend, no new features, no new classifier,
no Optuna, no leaderboard tuning.  The headline is the plain average.

    20|NOTHING ELSE IS TOUCHED
---------------------------
Window model, its ``random_state=42``, the 51 aggregates, the sealed window
peaks, the Exp08 L1 localiser, the success head and the folds are all unchanged.
Success F1 is the pinned Exp03 B constant.  The only thing this experiment
replaces is the single fitted label model with three, and only the seed differs.

HONEST OOF
----------
The same 3 outer folds, re-derived from ``StratifiedKFold(3, shuffle=True,
random_state=0)`` and asserted equal to the sealed manifest's index hashes.  On
each fold every seed is fitted on outer-train ONLY and predicts outer-val only;
the three outer-val probability matrices are averaged per row and the 3 folds
are concatenated into one full 10000-row OOF vector, scored once.  No
parameter is chosen from the results of any fold, because there is nothing to
choose: the seed set is fixed a priori.

    30|STOP CONDITIONS - each halts the run instead of producing a number:
  * the Exp08 baseline does not reproduce to 1e-12;
  * the strict loader rejects the artifacts (there is no lenient path);
  * the rebuilt 300-feature matrix does not reproduce the sealed seed-42
    probabilities BIT for bit;
  * the fast objective disagrees with ``evaluation.metrics``;
  * the promotion gate fails (no production build, no other seed set).

THE GATE, PRE-DECLARED
----------------------
PROMOTE only if ALL of: composite delta >= +0.002; at least 2 of 3 outer folds
win; robustness drops by no more than 0.003; no class F1 drops by more than
0.02.  A composite delta inside [+0.001, +0.002) is declared WEAK/INCONCLUSIVE
up front and resolves to STOP.
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
for _p in (HERE, os.path.join(ROOT, "experiments", "exp10_ab_probability_blend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ensemble as K                                          # noqa: E402
from ensemble import (BASE_SEED, GATE, N_RUNS, SEEDS, SUCCESS_F1_PINNED,  # noqa: E402
                      TOL, WEAK_BAND, LABELS, L2I, CLEAN)

# Reproduction TARGETS, read from current best (exp08) - never tuning knobs.
EXPECTED_BASELINE = {
    "macro_f1": 0.7894323366835861,
    "robustness_f1": 0.7368470046834376,
    "success_f1": 0.8643757406010988,
    "fault_turn_hit2": 0.6086111111111111,
    "composite": 0.7694453917139285,
}
PUBLIC_EXP08 = 0.7696          # reference only; no search reads it.

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


def main():
    t0 = time.time()
    tee = Tee(os.path.join(HERE, "run.log"))
    sys.stdout = tee
    try:
        return _run(t0)
    finally:
        tee.close()


def _run(t0):
    print("=" * 78)
    print("Exp11 - FIXED 3-seed probability ensemble of the final label head")
    print("=" * 78)
    print("  seeds (frozen): %r   equal weights, plain average" % (list(SEEDS),))
    print("  gate (pre-declared): %r" % GATE)
    print("  weak/inconclusive band: [%.3f, %.3f) -> STOP" % WEAK_BAND)

    res = {"experiment": "exp11_label_seed_ensemble", "n_runs": N_RUNS,
           "system": "B_plus_window (300 features)",
           "hypothesis": "P_ensemble = (P_seed42 + P_seed137 + P_seed2026) / 3; "
                         "pred = argmax(P_ensemble)",
           "seeds": list(SEEDS), "weights": [1.0 / 3.0] * 3,
           "search_space": {
               "form": "unweighted arithmetic mean of the three seed "
                       "probabilities, then argmax",
               "free_parameters": 0,
               "seed_set": list(SEEDS),
               "forbidden": ["other seeds", "2 or 4 models", "seed weights",
                             "best-seed selection", "A/B blend", "offsets",
                             "calibration", "thresholds", "window-model "
                             "ensemble", "new features", "new classifiers",
                             "Optuna", "public leaderboard tuning"],
               "determinism": "float32 cast per seed, float64 mean; seed 42 is "
                              "bit-identical to the sealed baseline, so the "
                              "one-model case is an identity, not a tolerance"},
           "baseline_expected": EXPECTED_BASELINE,
           "public_exp08_reference_only": PUBLIC_EXP08,
           "gate": dict(GATE), "weak_band": list(WEAK_BAND)}

    # ---- 1. verified inputs ---------------------------------------------
    print("\n" + "=" * 78)
    print("[1] STRICT VERIFIED INPUTS (fail closed) + parameter discipline")
    print("=" * 78)
    d = K.load_verified()
    ok("%d provenance checks passed in %.2fs; fold index hashes match the "
       "sealed manifest" % (len(d["checks"]), d["load_sec"]))
    print("  ok    sealed P_B %s %s   peak_pos %s %s"
          % (d["proba_B"].shape, d["proba_B"].dtype, d["peak_pos"].shape,
             d["peak_pos"].dtype))
    if not (d["proba_B"].argmax(1) == d["labels_B"]).all():
        fail("argmax(verified P_B) != verified B labels")
    else:
        ok("argmax(verified P_B) == verified B labels on all %d rows" % N_RUNS)
    rob = K.robustness_mask()
    ok("robustness mask: %d rows (a property of the data, identical for every "
       "seed)" % int(rob.sum()))

    params = K.assert_params_differ_only_in_random_state()
    ok("the 3 label parameter dicts are identical except random_state: %r"
       % [params[s]["random_state"] for s in SEEDS])
    print("  ok    locked: objective=%r num_class=%r n_estimators=%r lr=%r "
          "num_leaves=%r mcs=%r subsample=%r/%r colsample=%r reg_lambda=%r"
          % (params[42]["objective"], params[42]["num_class"],
             params[42]["n_estimators"], params[42]["learning_rate"],
             params[42]["num_leaves"], params[42]["min_child_samples"],
             params[42]["subsample"], params[42]["subsample_freq"],
             params[42]["colsample_bytree"], params[42]["reg_lambda"]))

    # ---- 2. REPRODUCTION GATE -------------------------------------------
    print("\n" + "=" * 78)
    print("[2] REPRODUCTION GATE - Exp08 baseline, official metrics, before "
          "any fit")
    print("=" * 78)
    c_base = K.corpus_for(d["proba_B"], d, rob, labels=d["labels_B"])
    pred_base = c_base.predict(1.0)
    if not np.array_equal(pred_base, d["labels_B"]):
        fail("alpha=1 labels differ from the verified B labels")
    else:
        ok("the Exp10 scoring path at alpha=1 reproduces the verified B labels "
           "exactly on all %d rows" % N_RUNS)
    committed = pd.read_csv(os.path.join(ROOT, "experiments",
                                         "exp07_fault_windows",
                                         "oof_B_plus_window.csv"))
    n_bad = int((committed["y_pred"].values.astype(str)
                 != np.asarray(LABELS)[pred_base]).sum())
    if n_bad:
        fail("baseline labels differ from the committed Exp07 OOF CSV at %d "
             "row(s)" % n_bad)
    else:
        ok("baseline labels == the committed Exp07 OOF CSV on all %d rows"
           % N_RUNS)

    m0, turns0 = K.full_official(c_base, pred_base)
    good = True
    for key in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
                "composite"):
        got, want = m0[key], EXPECTED_BASELINE[key]
        if abs(got - want) > TOL:
            good = False
            fail("BASELINE %s: got %.16f, expected %.16f (diff %.3e)"
                 % (key, got, want, abs(got - want)))
        else:
            ok("BASELINE %-16s %.16f  (exact)" % (key, got))
    if m0["composite"] != sum(w * m0[k] for k, w in
                              (("macro_f1", 0.5), ("robustness_f1", 0.25),
                               ("success_f1", 0.15), ("fault_turn_hit2", 0.1))):
        fail("composite() is not self-consistent with the official weights")
    else:
        ok("composite() matches a hand expansion of the official weights")
    # the Exp08 L1 rule, re-read from the sealed peaks
    turns_ref = np.full(N_RUNS, -1, dtype=np.int64)
    for i in range(N_RUNS):
        lab = LABELS[pred_base[i]]
        if lab != "clean":
            turns_ref[i] = int(d["peak_pos"][i, K.W2I[lab]])
    if not np.array_equal(turns_ref, turns0):
        fail("the vectorised L1 turn differs from a direct re-read of the "
             "sealed window_peak_pos")
    else:
        ok("L1 fault_turn == a direct re-read of the sealed peaks; every clean "
           "prediction -> -1, every fault prediction gets a real turn")
    print("  NOTE public Exp08 leaderboard %.4f recorded as a REFERENCE ONLY; "
          "nothing here reads it." % PUBLIC_EXP08)
    res["baseline"] = dict(m0)
    res["reproduction_gate"] = {"passed": bool(good and not FAILS),
                                "expected": EXPECTED_BASELINE, "got": dict(m0),
                                "tolerance": TOL,
                                "metric_source": "evaluation.metrics",
                                "committed_csv_labels_match": n_bad == 0,
                                "l1_turn_matches_sealed_peaks": True}
    if not (good and not FAILS):
        print("\nREPRODUCTION GATE FAILED - STOPPING. No ensemble is fitted on "
              "top of a baseline that does not reproduce.")
        res["stopped"] = "reproduction gate failed"
        _dump(res, t0)
        return 1
    print("\n  REPRODUCTION GATE PASSED - exact on all 5 numbers, before any "
          "fit.")

    # ---- 3. build the three honest OOF matrices -------------------------
    print("\n" + "=" * 78)
    print("[3] HONEST OOF: 3 outer folds x 3 seeds, fitted on outer-train only")
    print("=" * 78)
    c, W = K._prepare()
    print("  foundation %s   windows %s   (Exp07's own content-addressed cache)"
          % (c["foundation"].shape, W["X"].shape))
    probas, cols_ref, per_seed_fold = {}, None, {}
    sealed_sha = K.sha32(d["proba_B"])
    for seed in SEEDS:
        t_s = time.time()
        P, cols = K.build_features(seed, d["folds"], c, W)
        if len(cols) != K.N_LABEL_FEATS:
            fail("seed %d fitted %d columns, expected %d"
                 % (seed, len(cols), K.N_LABEL_FEATS))
        if cols_ref is None:
            cols_ref = cols
        elif cols != cols_ref:
            fail("seed %d fitted a different column set or order" % seed)
        probas[seed] = P
        # Persisted so the tests can re-derive the headline from the exact
        # arrays the report was computed from, instead of refitting them.
        np.save(os.path.join(HERE, "seed_%d_oof.npy" % seed), P)
        sha = K.sha32(P)
        same = sha == sealed_sha
        if seed == BASE_SEED:
            per_seed_fold[seed] = {"bit_identical_to_sealed": bool(same)}
            if same:
                ok("seed 42 reproduces the sealed Exp07 OOF probabilities BIT "
                   "for BIT (sha256 %s) - the rebuilt 300-feature matrix is "
                   "bit-exactly the matrix current best was fitted on"
                   % sha[:16])
            else:
                fail("seed 42 does NOT reproduce the sealed Exp07 OOF "
                     "probabilities bit for bit (sha %s vs %s) - the feature "
                     "matrix or the fit differs from current best"
                     % (sha[:16], sealed_sha[:16]))
        else:
            nd = int((P.argmax(1) != d["labels_B"]).sum())
            mx = float(np.abs(P.astype(np.float64)
                              - d["proba_B"].astype(np.float64)).max())
            per_seed_fold[seed] = {"bit_identical_to_sealed": bool(same),
                                   "labels_differ_vs_seed42": nd,
                                   "max_abs_prob_diff_vs_seed42": mx}
            print("  ok    seed %d: %d/%d labels differ from seed 42, "
                  "max|dP| = %.3e - the seeds really are different models"
                  % (seed, nd, N_RUNS, mx))
        print("      seed %d done in %.1fs" % (seed, time.time() - t_s))
    res["feature_matrix"] = {
        "n_features": len(cols_ref), "n_agg": K.N_AGG_FEATS,
        "column_order": list(cols_ref),
        "column_sha256": hashlib_sha(cols_ref),
        "same_columns_all_seeds": True,
        "exp07_quirk_reproduced": "predict_runs(..., len(tr_runs)) leaves the "
                                  "51 aggregates ZERO for outer-train runs with "
                                  "a global index >= len(tr_runs) (~33% per "
                                  "fold); reproduced verbatim, see ensemble.py",
    }
    res["per_seed_bit_parity"] = per_seed_fold
    if any(not v.get("bit_identical_to_sealed", True) and s == BASE_SEED
           for s, v in per_seed_fold.items()):
        print("\nSEED-42 BIT PARITY FAILED - STOPPING.")
        res["stopped"] = "seed 42 does not reproduce the sealed probabilities"
        _dump(res, t0)
        return 1
    res["feature_matrix"]["seed42_bit_identical_to_sealed"] = True
    res["feature_matrix"]["seed42_sha256"] = K.sha32(probas[BASE_SEED])

    # ---- 4. the fixed average -------------------------------------------
    print("\n" + "=" * 78)
    print("[4] THE FIXED 3-SEED AVERAGE, assembled over all %d rows" % N_RUNS)
    print("=" * 78)
    P_ens = K.average([probas[s] for s in SEEDS])
    c_ens = K.corpus_for(P_ens, d, rob)
    pred_ens = c_ens.predict(1.0)
    n_chg = int((pred_ens != pred_base).sum())
    mx = float(np.abs(P_ens - d["proba_B"].astype(np.float64)).max())
    print("  P_ensemble = (P_42 + P_137 + P_2026) / 3, float64 mean of the "
          "float32 casts")
    ok("argmax(P_ensemble) differs from the baseline on %d/%d rows (%.2f%%); "
       "max|dP| vs baseline %.3e" % (n_chg, N_RUNS, 100.0 * n_chg / N_RUNS, mx))
    ok("the ensemble is an average of exactly 3 models with equal weights and "
       "no per-row selection")

    # ---- 5. fast path vs official ---------------------------------------
    print("\n" + "=" * 78)
    print("[5] FAST PATH must equal the official metrics, on real candidates")
    print("=" * 78)
    worst = 0.0
    probes = []
    for tag, cc in (("baseline", c_base), ("ensemble", c_ens),
                    ("seed137", K.corpus_for(probas[137], d, rob)),
                    ("seed2026", K.corpus_for(probas[2026], d, rob))):
        o = K.Objective(cc, np.arange(N_RUNS))
        fast, off = o.metrics(1.0), o.official(1.0)
        for k in ("macro_f1", "robustness_f1", "fault_turn_hit2", "composite"):
            dd = abs(fast[k] - off[k])
            worst = max(worst, dd)
            if dd > TOL:
                fail("fast path disagrees with the official metric on %s/%s: "
                     "%r vs %r" % (tag, k, fast[k], off[k]))
        probes.append(tag)
    if worst <= TOL:
        ok("fast objective == official evaluation.metrics on %d real candidates "
           "(baseline, ensemble, and both extra seeds), max abs diff %.2e"
           % (len(probes), worst))
    res["fast_objective_check"] = {"probes": probes, "max_abs_diff": worst,
                                   "passed": worst <= TOL}
    if worst > TOL:
        _dump(res, t0)
        return 1

    # ---- 6. HEADLINE ----------------------------------------------------
    print("\n" + "=" * 78)
    print("[6] HEADLINE - current best (seed 42) vs the fixed 3-seed average")
    print("=" * 78)
    m1, turns1 = K.full_official(c_ens, pred_ens)
    acc_b = float((pred_base == c_base.yi).mean())
    acc_e = float((pred_ens == c_ens.yi).mean())
    yt = c_base.name(c_base.yi)
    from metrics import f1_per_class
    pc_b = f1_per_class(yt, c_base.name(pred_base), LABELS)
    pc_e = f1_per_class(yt, c_ens.name(pred_ens), LABELS)
    cm_b = K.confusion(yt, c_base.name(pred_base))
    cm_e = K.confusion(yt, c_ens.name(pred_ens))
    n_turn_chg = int((turns1 != turns0).sum())
    for tag, M in (("baseline (seed 42)", m0), ("ensemble (42/137/2026)", m1)):
        print("  %-24s macro %.16f  robustness %.16f  success %.16f  "
              "hit@2 %.16f" % (tag, M["macro_f1"], M["robustness_f1"],
                               M["success_f1"], M["fault_turn_hit2"]))
        print("  %-24s composite %.16f" % ("", M["composite"]))
    d_comp = m1["composite"] - m0["composite"]
    print("  delta  macro %+.16f  robustness %+.16f  success %+.16f  hit@2 %+.16f"
          % (m1["macro_f1"] - m0["macro_f1"], m1["robustness_f1"] - m0["robustness_f1"],
             m1["success_f1"] - m0["success_f1"], m1["fault_turn_hit2"] - m0["fault_turn_hit2"]))
    print("  delta  composite %+.16f" % d_comp)
    print("  accuracy (diagnostic only) %.6f -> %.6f (%+.6f)"
          % (acc_b, acc_e, acc_e - acc_b))
    print("  OOF labels changed: %d / %d (%.2f%%)"
          % (n_chg, N_RUNS, 100.0 * n_chg / N_RUNS))
    print("  fault_turn values changed as a consequence: %d / %d"
          % (n_turn_chg, N_RUNS))
    clean_ens = pred_ens == CLEAN
    if not bool((turns1[clean_ens] == -1).all()):
        fail("an ensemble clean prediction did not receive fault_turn -1")
    else:
        ok("every ensemble clean prediction has fault_turn == -1 (%d rows)"
           % int(clean_ens.sum()))

    print("\n  per-class F1 (official f1_per_class):")
    print("    %-18s %10s %10s %10s" % ("class", "baseline", "ensemble", "delta"))
    for cc in LABELS:
        print("    %-18s %10.6f %10.6f %+10.6f"
              % (cc, pc_b[cc], pc_e[cc], pc_e[cc] - pc_b[cc]))
    print("\n  baseline confusion (rows = true, cols = pred):")
    print(K.conf_str(cm_b))
    print("\n  ensemble confusion (rows = true, cols = pred):")
    print(K.conf_str(cm_e))

    trans, fixes, breaks, wrong_to_wrong = {}, 0, 0, 0
    for i in range(N_RUNS):
        a, b = int(pred_base[i]), int(pred_ens[i])
        if a == b:
            continue
        k = "%s->%s" % (LABELS[a], LABELS[b])
        trans[k] = trans.get(k, 0) + 1
        ya = int(c_base.yi[i])
        if b == ya and a != ya:
            fixes += 1
        if a == ya and b != ya:
            breaks += 1
        if a != ya and b != ya:
            # a run can move from one WRONG class to a different WRONG class
            # and stay wrong; counting only the two correctness flips would
            # leave this tail unaccounted for
            wrong_to_wrong += 1
    print("\n  confusion transitions (baseline -> ensemble), changed rows only:")
    for k in sorted(trans, key=lambda k: -trans[k]):
        print("    %-40s %5d" % (k, trans[k]))
    print("  of the %d changed labels: %d became correct, %d became wrong, "
          "%d wrong->wrong  (net %+d)" % (n_chg, fixes, breaks, wrong_to_wrong,
                                         fixes - breaks))
    res["comparison"] = {
        "baseline_seed42": dict(m0), "ensemble_3seed": dict(m1),
        "delta": {k: m1[k] - m0[k] for k in m0},
        "accuracy": {"baseline": acc_b, "ensemble": acc_e,
                     "delta": acc_e - acc_b,
                     "note": "diagnostic only; accuracy is NOT the objective"},
        "per_class_f1": {"baseline": pc_b, "ensemble": pc_e,
                         "delta": {c: pc_e[c] - pc_b[c] for c in LABELS}},
        "confusion_baseline": cm_b.tolist(), "confusion_ensemble": cm_e.tolist(),
        "confusion_transitions": trans,
        "changed_label_stats": {"changed": n_chg, "became_correct": fixes,
                                "became_wrong": breaks,
                                "wrong_to_wrong": wrong_to_wrong,
                                "net": fixes - breaks},
        "n_labels_changed": n_chg, "n_fault_turn_changed": n_turn_chg,
        "note": "hit@2 reads the peak of the ENSEMBLE PREDICTED class, so it "
                "moves with the label; it is not held fixed",
    }

    # ---- 7. individual seeds, DIAGNOSTIC ONLY ---------------------------
    print("\n" + "=" * 78)
    print("[7] INDIVIDUAL SEED SCORES - DIAGNOSTIC ONLY, NOT A SELECTION")
    print("=" * 78)
    print("  These are reported to show the spread.  No best seed is chosen,")
    print("  the headline is and remains the fixed average.")
    res["individual_seeds_diagnostic"] = {"note": "reported for spread only; "
                                   "the headline is the fixed 3-seed average "
                                   "and no seed is selected",
                                   "seeds": {}}
    for seed in SEEDS:
        c_s = K.corpus_for(probas[seed], d, rob)
        p_s = c_s.predict(1.0)
        ms, _ = K.full_official(c_s, p_s)
        nds = int((p_s != pred_ens).sum())
        res["individual_seeds_diagnostic"]["seeds"][str(seed)] = {
            **ms, "labels_differ_vs_baseline": int((p_s != pred_base).sum()),
            "labels_differ_vs_ensemble": nds}
        print("  seed %-5d macro %.6f  rob %.6f  hit@2 %.6f  composite %.16f"
              "   (diff vs baseline %+.16f; %d labels differ from the ensemble)"
              % (seed, ms["macro_f1"], ms["robustness_f1"],
                 ms["fault_turn_hit2"], ms["composite"],
                 ms["composite"] - m0["composite"], nds))

    # ---- 8. fold-by-fold ------------------------------------------------
    print("\n" + "=" * 78)
    print("[8] FOLD-BY-FOLD - the same 3 outer folds, held-out rows only")
    print("=" * 78)
    fold_of = np.zeros(N_RUNS, dtype=np.int64)
    for f, va in enumerate(d["folds"]):
        fold_of[np.asarray(va, dtype=np.int64)] = f
    assert (np.bincount(fold_of, minlength=3) > 0).all()
    res["folds"] = []
    wins = 0
    for f in range(3):
        val_ix = np.asarray(d["folds"][f], dtype=np.int64)
        tr_ix = np.where(fold_of != f)[0]
        assert not np.intersect1d(val_ix, tr_ix).size, "train/val overlap"
        ob = K.Objective(c_base, val_ix)
        oe = K.Objective(c_ens, val_ix)
        mb, me = ob.official(1.0), oe.official(1.0)
        df_ = me["composite"] - mb["composite"]
        w = bool(df_ > 0)
        wins += int(w)
        ch = int((c_ens.predict(1.0)[val_ix] != pred_base[val_ix]).sum())
        res["folds"].append({"fold": f, "n_train": int(len(tr_ix)),
                             "n_val": int(len(val_ix)),
                             "val_sha256": d["fold_sha"][f],
                             "baseline": mb, "ensemble": me,
                             "delta_composite": df_, "wins": w,
                             "n_labels_changed": ch})
        print("  fold %d (n_val=%5d, %d labels changed): composite %.16f -> "
              "%.16f  (%+.16f)  %s"
              % (f, len(val_ix), ch, mb["composite"], me["composite"], df_,
                 "WIN" if w else "LOSS"))
        print("           macro %+.6f  robustness %+.6f  hit@2 %+.6f"
              % (me["macro_f1"] - mb["macro_f1"],
                 me["robustness_f1"] - mb["robustness_f1"],
                 me["fault_turn_hit2"] - mb["fault_turn_hit2"]))
    res["fold_wins"] = "%d/%d" % (wins, 3)
    print("\n  ensemble wins on %d/3 outer folds" % wins)
    print("  NOTE this is a CONSISTENCY check over 3 folds, not significance:")
    print("  the seed set is fixed a priori, so nothing was selected per fold,")
    print("  but n=3 and the folds share one dataset.")

    # ---- 9. PROMOTION GATE ---------------------------------------------
    print("\n" + "=" * 78)
    print("[9] PROMOTION GATE (pre-declared before the first fit)")
    print("=" * 78)
    checks = []
    checks.append(("composite delta >= %+.3f" % GATE["composite_delta_min"],
                   d_comp, GATE["composite_delta_min"],
                   d_comp >= GATE["composite_delta_min"]))
    checks.append(("wins >= %d/3 folds" % GATE["folds_won_min"], wins, 3,
                   wins >= GATE["folds_won_min"]))
    d_rob = m1["robustness_f1"] - m0["robustness_f1"]
    checks.append(("robustness drop <= %.3f" % GATE["robustness_drop_max"],
                   -d_rob, GATE["robustness_drop_max"],
                   (-d_rob) <= GATE["robustness_drop_max"]))
    worst_cls = min(pc_b, key=lambda c: pc_e[c] - pc_b[c])
    worst_drop = pc_b[worst_cls] - pc_e[worst_cls]
    checks.append(("worst class F1 drop <= %.2f (%s)"
                   % (GATE["class_f1_drop_max"], worst_cls),
                   worst_drop, GATE["class_f1_drop_max"],
                   worst_drop <= GATE["class_f1_drop_max"]))
    gate_passed = all(c4[3] for c4 in checks) and not FAILS
    for name, got, want, good_ in checks:
        print("  [%s] %-46s %+.6f" % ("PASS" if good_ else "FAIL", name, got))
    weak = (WEAK_BAND[0] <= d_comp < WEAK_BAND[1])
    if weak:
        print("  NOTE composite delta %+.6f lies in the pre-declared WEAK band "
              "[%.3f, %.3f) -> weak/inconclusive -> STOP regardless."
              % (d_comp, WEAK_BAND[0], WEAK_BAND[1]))
    print("\n  PROMOTION GATE: %s" % ("PASS" if gate_passed else "FAIL"))
    res["promotion_gate"] = {
        "passed": bool(gate_passed),
        "criteria": [{"name": c4[0], "value": c4[1], "threshold": c4[2],
                      "passed": bool(c4[3])} for c4 in checks],
        "folds_won": wins, "worst_class_f1_drop": {"class": worst_cls,
                                                    "drop": worst_drop},
        "weak_inconclusive_band": bool(weak),
        "unexpected_failures": list(FAILS)}
    res["promoted"] = bool(gate_passed)

    if not gate_passed:
        print("\n  GATE FAILED - no production ensemble is fitted, no submission")
        print("  is built, no other seed set is tried, and Exp12 is not started.")
        res["final_ensemble"] = {"fitted": False,
                                 "reason": "promotion gate failed",
                                 "note": "no 3-seed production model was fitted "
                                         "and no artifact was written"}
        _dump(res, t0)
        return 0

    # ---- 10. production, only on PASS ----------------------------------
    print("\n" + "=" * 78)
    print("[10] PRODUCTION BUILD (gate PASSED)")
    print("=" * 78)
    import build_production
    rc = build_production.build(t0)
    res["production"] = rc
    _dump(res, t0)
    return 0 if rc.get("ok") else 1


def hashlib_sha(seq):
    import hashlib
    h = hashlib.sha256()
    for s in seq:
        h.update(str(s).encode())
    return h.hexdigest()


def _dump(res, t0):
    res["runtime_sec"] = round(time.time() - t0, 2)
    res["failed_checks"] = list(FAILS)
    res["caveat"] = ("Honest 3-fold OOF with the seed set fixed a priori: no "
                     "parameter was chosen from the results, so this is not a "
                     "selected-parameter estimate. It is still one dataset and "
                     "three folds; the public leaderboard remains the external "
                     "validation and public Exp08 is a reference only.")
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("\nwrote %s" % os.path.join(HERE, "results.json"))
    print("total research runtime %.1fs (%.2f min)"
          % (time.time() - t0, (time.time() - t0) / 60.0))


if __name__ == "__main__":
    sys.exit(main())
