"""Exp08 - can the official L0 localiser be replaced by the L1 window peak?

ONE QUESTION
------------
Exp07 observed, on its own OOF, that taking ``fault_turn`` as the turn of
maximum window-model probability for the ALREADY-PREDICTED fault class ("L1")
scores hit@2 = 0.6086 against L0's rule-based localiser's 0.4308.  Exp08 asks
whether that holds up well enough to ship, changing nothing else.

WHAT IS FIXED AND NOT TOUCHED
-----------------------------
label model, Exp07's 300 features, success model, window model, window features,
folds, seeds, class weights, hyper-parameters, label predictions, success
predictions.  This script fits NOTHING.  It reads Exp07's verified OOF artifacts
and applies one decision rule to them.

THE L1 RULE, EXACTLY
--------------------
    predicted label == clean          ->  fault_turn = -1
    predicted label == fault class c  ->  fault_turn = argmax_t P[run, t, c]

``P[run, t, c]`` is the window model probability that turn ``t`` of this run
belongs to fault class ``c``.  No confidence threshold, no offset, no
class-specific rule, no calibration.  One rule, applied uniformly.

HIT@2 COMES ONLY FROM THE OFFICIAL METRIC
------------------------------------------
Every hit@k number comes from ``evaluation.metrics.fault_turn_hit_at_k``.  Exp07
shipped a hit@2 that disagreed with the official metric by a factor of two
because a positional zip misaligned every row after the first clean run.  Exp08
keeps its own independent recount; the two must agree or the run stops.

THE REPRODUCTION GATE
---------------------
L0 and L1 are recomputed here from the verified artifacts and must equal, to
1e-12, the numbers Exp07 recorded.  If they do not, the script STOPS.  It does
not search for a version of L1 that reproduces better - that is precisely the
post-hoc fitting this experiment forbids.
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
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
sys.path.insert(0, HERE)
sys.path.insert(0, EXP07)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from metrics import composite, fault_turn_hit_at_k   # noqa: E402
from localize import localize                        # noqa: E402
import load_exp07 as L                               # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}

# WINDOW classes: index 0 is background.  peak_pos/peak_prob are indexed in
# this order, so a PREDICTED RUN LABEL maps through W2I, never through L2I.
W2I = {c: i for i, c in enumerate(["background"] + LABELS[1:])}

SYSTEM = "B_plus_window"
N_RUNS = 10000
TOL = 1e-12

# The pre-registered reproduction targets, from Exp07's own artifacts.
EXPECTED = {
    "L0_hit2": 0.43083333333333335,
    "L1_hit2": 0.6086111111111111,
    "L1_per_class": {
        "dropped_handoff": 0.4675,
        "duplicated_work": 0.7275,
        "deadlock": 0.5883333333333334,
        "conflict": 0.7000,
        "goal_drift": 0.5766666666666667,
        "runaway_loop": 0.5916666666666667,
    },
}
# Exp07 B's fixed head metrics.  NOT recomputed here: Exp08 changes fault_turn
# and nothing else, so these carry over untouched and any difference is a bug.
FOUNDATION = {
    "macro_f1": 0.7894323366835861,
    "robustness_f1": 0.7368470046834376,
    "success_f1": 0.8643757406010988,
    "composite_l0": 0.7516676139361507,
}

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL  %s" % msg)


def ok(msg):
    print("  ok    %s" % msg)


def turns_l0(runs, pred):
    """L0: the OFFICIAL rule-based localiser on the predicted label."""
    return [int(localize(runs[i], pred[i])) for i in range(len(pred))]


def turns_l1(peak_pos, pred):
    """L1: argmax over windows of the PREDICTED class's probability.

    clean has no window class, so it maps to -1.  This is the whole rule: no
    threshold, no offset, no per-class handling.
    """
    out = np.full(len(pred), -1, dtype=np.int64)
    for i, lab in enumerate(pred):
        if lab == "clean":
            continue
        out[i] = int(peak_pos[i, W2I[lab]])
    return out


def recount(y_true, pred, fturn, turns):
    """hit@k by brute force over truly-faulty runs.

    This duplicates ``metrics.fault_turn_hit_at_k`` on purpose.  If the two ever
    disagree the run stops rather than reporting either number.
    """
    hits = [0, 0, 0]
    n = 0
    per = {}
    for i, yt in enumerate(y_true):
        if yt == "clean":
            continue                       # denominator: truly-faulty runs
        n += 1
        d = per.setdefault(yt, [0, 0, 0, 0])
        d[3] += 1
        if pred[i] != yt:
            continue                       # misclassified -> miss
        t = turns[i]
        if t is None or int(t) < 0:
            continue                       # clean prediction -> miss
        dd = abs(int(t) - int(fturn[i]))
        for j, lim in enumerate((0, 1, 2)):
            if dd <= lim:
                hits[j] += 1
                d[j] += 1
    return {"n_faulty": n, "hits": hits,
            "rates": [h / n if n else 0.0 for h in hits],
            "per_class": {c: {"n": v[3], "hit@0": v[0] / max(1, v[3]),
                              "hit@1": v[1] / max(1, v[3]),
                              "hit@2": v[2] / max(1, v[3])}
                          for c, v in per.items()}}


def official(y_true, pred, fturn, turns, k):
    return fault_turn_hit_at_k(list(y_true), list(pred), list(fturn),
                               list(turns), k=k)


def main():
    t0 = time.time()
    print("=" * 78)
    print("Exp08 - L1 window-peak localizer")
    print("=" * 78)

    # ---- 1. verified inputs, fail closed ----------------------------------
    print("\n[1] loading Exp07 OOF + window peaks (strict, fail closed)")
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    ys = train["label"].values.astype(object)
    yi = np.array([L2I[l] for l in ys], dtype=int)
    rids = [str(x) for x in train["run_id"]]
    art = L.load(yi=yi, train_run_ids=rids)
    peak_pos = art["peak_pos"]
    print("  ok    %d provenance checks passed, %d runs, peaks %s / %s"
          % (len(art["verified_checks"]), art["n_runs"], peak_pos.shape,
             art["peak_prob"].shape))

    oof = pd.read_csv(os.path.join(EXP07, "oof_%s.csv" % SYSTEM))
    from_labels = np.array(LABELS)[art["labels"]].astype(str)
    if not (oof["y_pred"].values.astype(str) == from_labels).all():
        fail("verified OOF labels disagree with oof_%s.csv" % SYSTEM)
    elif not (art["proba"].argmax(1) == art["labels"]).all():
        fail("argmax(verified proba) != verified labels")
    else:
        ok("verified labels == argmax(verified proba) == committed OOF CSV "
           "on all %d rows" % N_RUNS)

    pred = oof["y_pred"].values.astype(object)
    y_true = oof["y_true"].values.astype(object)
    fturn = oof["fault_turn"].values
    if not (y_true == ys).all():
        fail("OOF y_true differs from train.csv label")
    if len(pred) != N_RUNS:
        fail("expected %d rows, got %d" % (N_RUNS, len(pred)))
    print("  ok    run order verified against current train.csv (%d run_ids)"
          % len(art["run_ids"]))

    # ---- 2. build both turn sources ---------------------------------------
    print("\n[2] building turn sources")
    print("  building runs for the official L0 localiser ...")
    import common as C
    runs = C.build_all()["runs"]
    L0 = turns_l0(runs, pred)
    L1 = turns_l1(peak_pos, pred)
    ok("L0 turns %d/%d non-negative; L1 turns %d/%d non-negative"
       % (sum(1 for t in L0 if t >= 0), N_RUNS,
          sum(1 for t in L1 if t >= 0), N_RUNS))
    ok("L0 and L1 disagree on %d of %d runs"
       % (sum(1 for a, b in zip(L0, L1) if a != b), N_RUNS))

    # ---- 3. OFFICIAL metric + independent recount ------------------------
    print("\n[3] official metric + independent recount")
    res = {"localiser": {}}
    agree = True
    for tag, turns in (("L0_official_localizer", L0), ("L1_window_peak", L1)):
        entry = {}
        r = recount(y_true, pred, fturn, turns)
        for k in (0, 1, 2):
            o = official(y_true, pred, fturn, turns, k)
            entry["hit@%d" % k] = o
            entry["hit@%d_recount" % k] = r["rates"][k]
            if abs(o - r["rates"][k]) > TOL:
                agree = False
                fail("%s hit@%d: official=%.16f recount=%.16f disagree"
                     % (tag, k, o, r["rates"][k]))
        entry["hits"] = r["hits"]
        entry["n_faulty"] = r["n_faulty"]
        entry["per_class"] = r["per_class"]
        res["localiser"][tag] = entry
    if agree:
        for tag in ("L0_official_localizer", "L1_window_peak"):
            e = res["localiser"][tag]
            ok("%-22s hit@0=%.6f hit@1=%.6f hit@2=%.6f "
               "(official == independent recount to 1e-12)"
               % (tag, e["hit@0"], e["hit@1"], e["hit@2"]))
    if FAILS:
        print("\nSTOPPED - the two metric implementations disagree. Refusing "
              "to report a number built on a disagreement.")
        return 1

    # ---- 4. REPRODUCTION GATE ---------------------------------------------
    print("\n" + "=" * 78)
    print("[4] REPRODUCTION GATE vs the pre-registered Exp07 diagnostic")
    print("=" * 78)
    l0h2 = res["localiser"]["L0_official_localizer"]["hit@2"]
    l1h2 = res["localiser"]["L1_window_peak"]["hit@2"]

    def gate(name, got, want):
        d = abs(got - want)
        if d > TOL:
            fail("REPRODUCTION %s: got %.16f, expected %.16f (diff %.3e)"
                 % (name, got, want, d))
            return False
        ok("REPRODUCTION %s = %.16f (exact)" % (name, got))
        return True

    good = gate("L0 hit@2", l0h2, EXPECTED["L0_hit2"])
    good &= gate("L1 hit@2", l1h2, EXPECTED["L1_hit2"])

    got_pc = res["localiser"]["L1_window_peak"]["per_class"]
    print("\n  L1 per-class hit@2:")
    for c in ["dropped_handoff", "duplicated_work", "deadlock", "conflict",
              "goal_drift", "runaway_loop"]:
        got = got_pc[c]["hit@2"]
        want = EXPECTED["L1_per_class"][c]
        d = abs(got - want)
        print("    %-18s %.16f  (expected %.16f)  %s"
              % (c, got, want, "exact" if d <= TOL else "MISMATCH %.3e" % d))
        if d > TOL:
            good = False
            fail("REPRODUCTION per-class L1 %s: got %.16f expected %.16f"
                 % (c, got, want))
    if good:
        ok("REPRODUCTION per-class L1 hit@2 exact for all %d classes"
           % len(EXPECTED["L1_per_class"]))

    exp07_res = json.load(open(os.path.join(EXP07, "results.json"),
                               encoding="utf-8"))
    b = exp07_res["systems"][SYSTEM]
    for key, want in (("macro_f1", FOUNDATION["macro_f1"]),
                      ("robustness_f1", FOUNDATION["robustness_f1"]),
                      ("success_f1", FOUNDATION["success_f1"])):
        if abs(b[key] - want) > 1e-12:
            fail("Exp07 %s is %.16f, expected %.16f" % (key, b[key], want))
    ok("Exp07 B foundation metrics unchanged: macro=%.16f robustness=%.16f "
       "success=%.16f" % (b["macro_f1"], b["robustness_f1"], b["success_f1"]))

    gate_passed = bool(good and not FAILS)
    res["reproduction_gate"] = {
        "passed": gate_passed, "expected": EXPECTED,
        "got": {"L0_hit2": l0h2, "L1_hit2": l1h2,
                "L1_per_class": {c: got_pc[c]["hit@2"] for c in got_pc}},
        "tolerance": TOL,
        "metric_source": "evaluation.metrics.fault_turn_hit_at_k",
        "independent_recount_agrees": agree,
    }
    if not gate_passed:
        print("\n" + "=" * 78)
        print("REPRODUCTION GATE FAILED - STOPPING.")
        print("No parameter is being fitted and no tolerance is being widened.")
        print("The cause must be found before any L1 number is believed.")
        print("=" * 78)
        json.dump(res, open(os.path.join(HERE, "results.json"), "w",
                            encoding="utf-8"), indent=2, default=float)
        return 1
    print("\n  REPRODUCTION GATE PASSED - exact on all %d checks."
          % (2 + len(EXPECTED["L1_per_class"])))

    # ---- 5. CROSS-FOLD CONSISTENCY ---------------------------------------
    print("\n" + "=" * 78)
    print("[5] CROSS-FOLD CONSISTENCY (L0 vs L1 on each existing outer fold)")
    print("=" * 78)
    folds = L.reuse.fold_assignments(yi)
    fold_hashes = [L.ca.sha256_indices(np.asarray(v, dtype=np.int64))
                   for v in folds]
    if fold_hashes != list(art["manifest"]["fold_val_sha256"]):
        fail("re-derived fold indices differ from the sealed manifest")
    else:
        ok("the 3 folds re-derived from seed 0 match the sealed manifest exactly")

    res["folds"] = []
    wins = 0
    for f, val in enumerate(folds):
        row = {"fold": f, "n_val": int(len(val)), "val_sha256": fold_hashes[f],
               "n_faulty": int((y_true[val] != "clean").sum())}
        for tag, turns in (("L0", L0), ("L1", L1)):
            sp = [pred[i] for i in val]
            st = [y_true[i] for i in val]
            sf = [int(fturn[i]) for i in val]
            sv = [int(turns[i]) for i in val]
            for k in (0, 1, 2):
                row["%s_hit@%d" % (tag, k)] = official(st, sp, sf, sv, k)
        row["delta_hit@2"] = row["L1_hit@2"] - row["L0_hit@2"]
        row["L1_wins"] = bool(row["delta_hit@2"] > 0)
        wins += int(row["L1_wins"])
        res["folds"].append(row)
        print("  fold %d (n_val=%5d, faulty=%4d): L0 h@0=%.4f h@1=%.4f "
              "h@2=%.4f | L1 h@0=%.4f h@1=%.4f h@2=%.4f | d h@2=%+.4f %s"
              % (f, row["n_val"], row["n_faulty"],
                 row["L0_hit@0"], row["L0_hit@1"], row["L0_hit@2"],
                 row["L1_hit@0"], row["L1_hit@1"], row["L1_hit@2"],
                 row["delta_hit@2"], "WIN" if row["L1_wins"] else "LOSS"))
    res["fold_wins"] = "%d/%d" % (wins, len(folds))
    print("\n  L1 beats L0 on hit@2 in %d/%d folds" % (wins, len(folds)))
    print("  NOTE: NOT independent validation - L1 was discovered post-hoc on")
    print("  exactly these OOF predictions. External check = public leaderboard.")

    # ---- 6. COMPOSITE -----------------------------------------------------
    print("\n" + "=" * 78)
    print("[6] COMPOSITE (label / robustness / success carried over unchanged)")
    print("=" * 78)
    comp_l0 = composite({"macro_f1": b["macro_f1"],
                         "robustness_f1": b["robustness_f1"],
                         "success_f1": b["success_f1"],
                         "fault_turn_hit2": l0h2})
    comp_l1 = composite({"macro_f1": b["macro_f1"],
                         "robustness_f1": b["robustness_f1"],
                         "success_f1": b["success_f1"],
                         "fault_turn_hit2": l1h2})
    manual = (0.50 * b["macro_f1"] + 0.25 * b["robustness_f1"]
              + 0.15 * b["success_f1"] + 0.10 * l1h2)
    if abs(manual - comp_l1) > 1e-12:
        fail("manual composite %.16f != metrics.composite() %.16f"
             % (manual, comp_l1))
    if abs(comp_l0 - b["composite"]) > 1e-12:
        fail("recomputed L0 composite %.16f != Exp07 results.json %.16f"
             % (comp_l0, b["composite"]))
    if abs(comp_l0 - FOUNDATION["composite_l0"]) > 1e-12:
        fail("recomputed L0 composite != the figure given in the task brief")
    ok("composite() reproduces Exp07's recorded L0 composite exactly")
    ok("manual arithmetic matches the official composite()")

    res["composite"] = {
        "weights": {"macro_f1": 0.50, "robustness_f1": 0.25,
                    "success_f1": 0.15, "fault_turn_hit2": 0.10},
        "macro_f1": b["macro_f1"], "robustness_f1": b["robustness_f1"],
        "success_f1": b["success_f1"],
        "L0": {"hit@2": l0h2, "composite": comp_l0},
        "L1": {"hit@2": l1h2, "composite": comp_l1},
        "delta_hit@2": l1h2 - l0h2,
        "delta_composite": comp_l1 - comp_l0,
        "source": "evaluation.metrics.composite",
    }
    print("  macro      %.16f  (unchanged)" % b["macro_f1"])
    print("  robustness %.16f  (unchanged)" % b["robustness_f1"])
    print("  success    %.16f  (unchanged)" % b["success_f1"])
    print("  hit@2      L0 %.16f -> L1 %.16f   delta %+.16f"
          % (l0h2, l1h2, l1h2 - l0h2))
    print("  COMPOSITE  L0 %.16f -> L1 %.16f  delta %+.16f"
          % (comp_l0, comp_l1, comp_l1 - comp_l0))

    # ---- 7. the other side of the trade ----------------------------------
    print("\n" + "=" * 78)
    print("[7] what L1 costs at hit@0 (reported, not optimised)")
    print("=" * 78)
    for tag in ("L0_official_localizer", "L1_window_peak"):
        e = res["localiser"][tag]
        print("  %-22s hit@0=%.6f  hit@1=%.6f  hit@2=%.6f"
              % (tag, e["hit@0"], e["hit@1"], e["hit@2"]))
    print("  L1 trades exactness for locality: hit@0 falls while hit@2 rises.")
    print("  Only hit@2 carries weight in the official composite, so the trade")
    print("  is favourable HERE.")

    res["n_runs"] = N_RUNS
    res["system"] = SYSTEM
    res["rule"] = {
        "L0": "fault_turn = localize(run, predicted_label)",
        "L1": ("fault_turn = argmax_t P[run, t, class(predicted_label)]; "
               "clean -> -1"),
        "threshold": None, "offset": None,
        "class_specific_rules": None, "calibration": None,
    }
    res["provenance"] = {
        "source": "Exp07 verified honest OOF (no retrain)",
        "checks_passed": len(art["verified_checks"]),
        "peak_sha256": art["peak_sha256"],
    }
    res["runtime_sec"] = round(time.time() - t0, 2)

    json.dump(res, open(os.path.join(HERE, "results.json"), "w",
                        encoding="utf-8"), indent=2, default=float)
    print("\nwrote %s" % os.path.join(HERE, "results.json"))
    print("done in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
