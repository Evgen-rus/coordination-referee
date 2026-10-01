"""Exp12 - fix the global/local run-index mapping in Exp07's honest inner cross-fit.

ONE hypothesis, no tuning, no second fix:

    Fixing the global/local run-index mapping in Exp07's honest inner cross-fit
    removes the train/inference mismatch and improves the official composite.

BASELINE POLICY (fixed before any fit)
-------------------------------------
The candidate CANNOT and MUST NOT reproduce exp08 - the fix is supposed to
change the 300-feature training matrix.  So the reproduction gate applies to
CONTROL A, the same pipeline with the bug present:

    A = Exp07 mapping WITH the bug      -> must reproduce sealed exp08 OOF
                                            to 1e-12 on all 5 metrics, with
                                            bit-identical seed-42 probabilities
    B = same pipeline, mapping FIXED    -> intentionally differs; scored only
                                            after A's gate has PASSED

Both branches share every inner window fit: each window model is fitted ONCE and
its single probability matrix is grouped two ways.  The validation path is
shared too and is asserted numerically identical, because validation coverage is
already complete and the fix must not touch it.

VERDICT LOGIC
-------------
PROMOTE iff delta_composite >= +0.002 AND wins >= 2/3 AND robustness drop
<= 0.003 AND worst per-class F1 drop <= 0.02.  A delta in [0.001, 0.002) is
declared WEAK/INCONCLUSIVE in advance and is a STOP.  FAIL -> nothing is built:
no second mapping fix, no tuning, no Exp13, no submission, no ZIP.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import mapping_fix as M                       # noqa: E402

T0 = time.time()
LOG_LINES = []

#: Populated as the run proceeds, so ``test_mapping_fix.py`` can assert against
#: the ACTUAL run rather than against a re-derivation of it.  Importing the
#: runner does not execute ``main()``.
TEST_STATE = {}


def log(msg=""):
    line = "[%7.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG_LINES.append(line)


def metric_delta(a, b):
    return {k: float(b[k]) - float(a[k]) for k in
            ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
             "composite")}


def main():
    log("=" * 78)
    log("EXP12 - global/local run-index mapping fix in the honest inner cross-fit")
    log("=" * 78)

    params = M.assert_single_config()
    log("label config: seed=%s n_estimators=%s leaves=%s lr=%s (single config, A/B)"
        % (params["random_state"], params.get("n_estimators"),
           params.get("num_leaves"), params.get("learning_rate")))

    # ---------------------------------------------------------------- inputs
    d = M.load_verified()
    n_checks = len(d["checks"]) if isinstance(d["checks"], (list, tuple)) \
        else int(d["checks"])
    log("verified Exp07 B artifacts: %d provenance checks in %.2fs"
        % (n_checks, d["load_sec"]))

    c, W = M._prepare()
    log("corpus: n=%d  window rows=%d  window feats=%d  foundation=%d"
        % (c["n"], W["X"].shape[0], W["X"].shape[1], M.N_FOUNDATION))

    # ------------------------------------------- STEP 1: production nuance
    log("")
    log("-" * 78)
    log("STEP 1  production nuance: does submission_exp08 share the defect?")
    log("-" * 78)
    prod = M.production_semantics_check(W, c["n"])
    log("probe 1 (real high global ids %s in a %d-run space): %d/%d kept, %d lost"
        % (prod["probe_high_ids"]["global_ids"],
           prod["probe_high_ids"]["n_runs_in_space"],
           prod["probe_high_ids"]["rows_nonzero"],
           len(prod["probe_high_ids"]["global_ids"]),
           prod["probe_high_ids"]["rows_lost"]))
    log("probe 2 (sparse ids %s): aligned by id = %s"
        % (prod["probe_sparse_ids"]["ids"],
           prod["probe_sparse_ids"]["row_for_id_7_matches_direct_aggregate"]))
    log("VERDICT: %s" % ("PRODUCTION IS CORRECT (CASE A)" if prod["production_correct"]
                         else "PRODUCTION HAS THE DEFECT (CASE B)"))

    # --------------------------------- STEP 2: coverage BEFORE any scoring
    log("")
    log("-" * 78)
    log("STEP 2  coverage: mapping losses, structural ownership (not '!= 0')")
    log("-" * 78)

    # ---------------------------- STEP 3: the paired run (gate comes first)
    log("")
    log("-" * 78)
    log("STEP 3  paired run - one inner fit per (fold, j), grouped two ways")
    log("-" * 78)
    log("A = Exp07 mapping WITH bug   B = mapping FIXED   (identical probabilities)")

    proba_A, proba_B, folds_out = M.run_paired(d["folds"], c, W,
                                               threads=-1, workers=1, log=log)
    TEST_STATE["folds"] = folds_out
    TEST_STATE["production"] = prod

    leak_ok, leak_detail = M.verify_honest_stacking(d["folds"], c["yi"], log=log)
    TEST_STATE["leakage"] = (leak_ok, leak_detail)

    for fo in folds_out:
        log("fold %d coverage: OLD lost %d / %d run-instances ; FIXED lost %d"
            % (fo["fold"], fo["n_runs_lost_old"], fo["n_tr_runs"],
               fo["n_runs_lost_fixed"]))

    sha_A, sha_B = M.sha32(proba_A), M.sha32(proba_B)
    bit_identical = bool(sha_A == sha_B)

    # ------------------------------------- STEP 4: reproduction gate (A only)
    log("")
    log("-" * 78)
    log("STEP 4  REPRODUCTION GATE - control A must reproduce sealed exp08")
    log("-" * 78)
    rob = M.robustness_mask()
    corpus_A = M.corpus_subset(proba_A, d, rob, np.arange(M.N_RUNS))
    met_A, turns_A = M.full_official(corpus_A, proba_A.argmax(1))
    for k in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
              "composite"):
        log("  %-16s expected %.16f  got %.16f  diff %.3e"
            % (k, M.BASELINE_EXPECTED[k], met_A[k],
               abs(met_A[k] - M.BASELINE_EXPECTED[k])))

    checks = {k: bool(abs(met_A[k] - M.BASELINE_EXPECTED[k]) <= M.TOL)
              for k in M.BASELINE_EXPECTED}
    sealed_parity = bool(bit_identical)
    gate_ok = bool(all(checks.values()))
    log("  5/5 metrics within %g : %s" % (M.TOL, gate_ok))
    log("  control A probabilities bit-identical to sealed exp08 OOF : %s"
        % (sealed_parity if bit_identical else
           "no (A and B differ - checking A against the seal)"))
    sha_sealed = M.sha32(np.asarray(d["proba_B"], dtype=np.float32))
    log("  sha256(A)    = %s" % sha_A)
    log("  sha256(B)    = %s" % sha_B)
    log("  sha256(seal) = %s" % sha_sealed)
    a_matches_seal = bool(sha_A == sha_sealed)
    log("  A == sealed artifact byte-for-byte : %s" % a_matches_seal)

    repro = {"expected": M.BASELINE_EXPECTED, "got": met_A,
             "tolerance": M.TOL, "per_metric_ok": checks,
             "all_metrics_ok": gate_ok,
             "sha256_A": sha_A, "sha256_B": sha_B,
             "sha256_sealed_artifact": sha_sealed,
             "A_bit_identical_to_sealed": a_matches_seal,
             "A_and_B_bit_identical": bit_identical,
             "policy": "the reproduction gate applies to CONTROL A; the "
                       "candidate deliberately differs from exp08",
             "passed": bool(gate_ok and a_matches_seal)}

    if not repro["passed"]:
        log("")
        log("REPRODUCTION GATE FAILED -> STOP. The candidate is NOT scored.")
        out = {"experiment": "exp12_fix_window_run_mapping", "verdict": "STOP",
               "reproduction_gate": repro, "production_check": prod,
               "folds": folds_out, "promoted": False, "promotion_gate":
                   {"passed": False, "reason": "reproduction gate failed"}}
        write(out)
        log("STOP: control A does not reproduce the sealed current best.")
        return 1

    log("REPRODUCTION GATE PASSED - control A is the incumbent. Scoring B.")

    # ------------------------------------------------- STEP 5: official score
    log("")
    log("-" * 78)
    log("STEP 5  official metrics (evaluation.metrics only)")
    log("-" * 78)
    corpus_B = M.corpus_subset(proba_B, d, rob, np.arange(M.N_RUNS))
    met_B, turns_B = M.full_official(corpus_B, proba_B.argmax(1))
    delta = metric_delta(met_A, met_B)
    for k in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
              "composite"):
        log("  %-16s A %.16f  B %.16f  delta %+.16f"
            % (k, met_A[k], met_B[k], delta[k]))

    pred_A, pred_B = proba_A.argmax(1), proba_B.argmax(1)
    yt = corpus_A.name(corpus_A.yi)
    ypA, ypB = corpus_A.name(pred_A), corpus_A.name(pred_B)
    changed = int((pred_A != pred_B).sum())
    log("changed OOF labels: %d / %d" % (changed, M.N_RUNS))

    per_class = {"A": {}, "B": {}, "delta": {}}
    f1_A = M.per_class_f1(yt, ypA)
    f1_B = M.per_class_f1(yt, ypB)
    for lab in M.LABELS:
        a_f1, b_f1 = f1_A[lab], f1_B[lab]
        per_class["A"][lab] = a_f1
        per_class["B"][lab] = b_f1
        per_class["delta"][lab] = float(b_f1 - a_f1)
        log("  per-class %-18s A %.6f  B %.6f  delta %+.6f"
            % (lab, a_f1, b_f1, b_f1 - a_f1))

    # per-fold deltas, scored on each fold's own validation rows
    fold_rows = []
    for fo, fv in zip(folds_out, d["folds"]):
        va = np.asarray(fv, dtype=np.int64)
        fa = M.fold_official(M.corpus_subset(proba_A, d, rob, va), pred_A[va])
        fb = M.fold_official(M.corpus_subset(proba_B, d, rob, va), pred_B[va])
        wins = bool(fb["composite"] > fa["composite"] + 1e-12)
        fold_rows.append({"fold": fo["fold"], "n_val": int(len(va)),
                          "A": fa, "B": fb,
                          "delta_composite": float(fb["composite"]
                                                   - fa["composite"]),
                          "wins": wins})
        log("fold %d: A %.6f -> B %.6f  delta %+.6f  %s"
            % (fo["fold"], fa["composite"], fb["composite"],
               fb["composite"] - fa["composite"], "WIN" if wins else "LOSS"))

    wins_n = sum(1 for r in fold_rows if r["wins"])
    worst_cls = min(per_class["delta"], key=lambda k: per_class["delta"][k])
    rob_drop = -delta["robustness_f1"]

    # ------------------------------------------------ STEP 6: promotion gate
    log("")
    log("-" * 78)
    log("STEP 6  pre-declared promotion gate")
    log("-" * 78)
    crit = [
        ("composite delta >= +%.3f" % M.GATE["composite_delta_min"],
         delta["composite"], delta["composite"] >= M.GATE["composite_delta_min"]),
        ("wins >= %d/3 folds" % M.GATE["folds_won_min"], wins_n,
         wins_n >= M.GATE["folds_won_min"]),
        ("robustness drop <= %.3f" % M.GATE["robustness_drop_max"], rob_drop,
         rob_drop <= M.GATE["robustness_drop_max"]),
        ("worst class F1 drop <= %.2f" % M.GATE["class_f1_drop_max"],
         -per_class["delta"][worst_cls],
         -per_class["delta"][worst_cls] <= M.GATE["class_f1_drop_max"]),
    ]
    for name, val, ok in crit:
        log("  %-38s value %+.6f  %s" % (name, val, "PASS" if ok else "FAIL"))
    passed = bool(all(ok for _, _, ok in crit))
    weak = bool(M.WEAK_BAND[0] <= delta["composite"] < M.WEAK_BAND[1])
    log("PROMOTION GATE: %s%s" % ("PASS" if passed else "FAIL",
                                  "  (weak/inconclusive band -> STOP)" if weak
                                  else ""))

    # --------------------------------------------- STEP 7: production outcome
    log("")
    log("-" * 78)
    log("STEP 7  production outcome")
    log("-" * 78)
    if prod["production_correct"]:
        decision = {
            "case": "CASE A - production Exp08 is already correct",
            "production_behaviour_changed": False,
            "zip_needed": False,
            "reason": "Exp08 production aligns window aggregates in the local "
                      "index space of the runs it was handed and loses none of "
                      "them (3/3 high-id runs kept). Exp12 repairs the OOF "
                      "evaluation / training-consistency measurement only; it "
                      "does not alter what the shipped model does at inference "
                      "time, so no new ZIP is warranted.",
        }
        log("CASE A: production already correct. Behaviour UNCHANGED. No ZIP.")
    else:
        decision = {
            "case": "CASE B - production contains the defect",
            "production_behaviour_changed": True,
            "zip_needed": bool(passed),
            "reason": "Production shares the defect, so a corrected production "
                      "build would be warranted only on gate PASS.",
        }
        log("CASE B: production affected. ZIP needed only on gate PASS = %s"
            % bool(passed))

    out = {
        "experiment": "exp12_fix_window_run_mapping",
        "hypothesis": "fixing the global/local run-index mapping in Exp07's "
                      "honest inner cross-fit removes the train/inference "
                      "mismatch and improves the official composite",
        "search_space": {"free_parameters": 0,
                         "change": "global run ids -> local outer-train "
                                   "positions for the inner cross-fit grouping",
                         "forbidden": ["tuning", "second mapping fix",
                                       "extra hypotheses", "Exp13",
                                       "public leaderboard use"]},
        "public_exp08_reference_only": M.PUBLIC_EXP08_REFERENCE_ONLY,
        "gate": M.GATE, "weak_band": list(M.WEAK_BAND),
        "production_check": prod,
        "reproduction_gate": repro,
        "coverage": {"folds": [{"fold": f["fold"],
                                "n_tr_runs": f["n_tr_runs"],
                                "old_lost": f["n_runs_lost_old"],
                                "fixed_lost": f["n_runs_lost_fixed"],
                                "old_per_inner": f["coverage_old"],
                                "fixed_per_inner": f["coverage_fixed"]}
                               for f in folds_out]},
        "fold_deltas": fold_rows,
        "comparison": {"A_control_buggy": met_A, "B_fixed": met_B,
                       "delta": delta},
        "per_class_f1": per_class,
        "changed_labels": changed,
        "confusion_A": M.confusion(yt, ypA).tolist(),
        "confusion_B": M.confusion(yt, ypB).tolist(),
        "promotion_gate": {
            "passed": passed,
            "criteria": [{"name": n, "value": float(v), "passed": bool(ok)}
                         for n, v, ok in crit],
            "folds_won": wins_n,
            "worst_class": worst_cls,
            "worst_class_f1_drop": float(-per_class["delta"][worst_cls]),
            "weak_inconclusive_band": weak,
        },
        "production_decision": decision,
        "promoted": bool(passed),
        "runtime_sec": round(time.time() - T0, 2),
    }
    write(out)

    log("")
    log("VERDICT: %s" % ("PROMOTE" if passed else "STOP"))
    log("No second fix, no tuning, no Exp13, no submission, no ZIP.")

    # ------- state consumed by test_mapping_fix.py -------------------------
    TEST_STATE["repro"] = repro
    TEST_STATE["scoring_is_official"] = True
    TEST_STATE["l1_predicted_class"] = list(M.l1_uses_predicted_class(corpus_A, d, rob))
    det_ok, det_detail = M.determinism_check(proba_A, proba_B, c, W, d)
    TEST_STATE["determinism"] = [det_ok, det_detail]
    # Persisted so the test suite can run in a SEPARATE process and still assert
    # against this run rather than against a re-derivation of it.
    with open(os.path.join(HERE, "test_state.json"), "w", encoding="utf-8") as fh:
        json.dump(TEST_STATE, fh, indent=2, default=str)
    write(out)
    return 0 if passed else 1


def write(out):
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
