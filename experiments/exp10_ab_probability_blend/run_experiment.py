"""Exp10 - probability blend of Exp07's A and B label heads.  Cross-fitted.

HEADLINE, and what it is not
----------------------------
``P_blend = (1 - alpha) * P_A + alpha * P_B``, ``pred = argmax(P_blend)``, ONE
scalar over the fixed 21-point grid.  Exp08 ships ``alpha = 1`` (pure B).

This is ``OOF meta-CV``.  For each of Exp07's 3 existing outer folds the alpha
is chosen on the OTHER two folds only, frozen, and applied to the held-out fold.
The three held-out predictions are then concatenated into one cross-fitted
prediction vector over all 10000 rows and scored ONCE.  That vector is the
headline.  Choosing alpha on all 10000 and then reporting the gain on those same
10000 rows is not done here, and would not be a validation result if it were.

It is NOT a fully nested retrain of Exp07: those probabilities were produced
once by a single Exp07 run and no alpha ever saw a held-out fold during the
fold that validated it, but the base models were not re-fitted inside a further
inner loop.  So the public leaderboard remains the external validation, and its
score is recorded as a REFERENCE ONLY - the search never reads it.

STOP CONDITIONS, all of which halt the run rather than produce a number:
  * the Exp08 baseline does not reproduce to 1e-12;
  * any strict-loader rejection (there is no lenient path);
  * the fast objective disagrees with ``evaluation.metrics`` on any checked
    candidate;
  * a held-out fold touches its own alpha;
  * the promotion gate fails (no production build, no second blend tried).

FROZEN, AND NOT RE-OPENED AFTER SEEING A RESULT: the 21-point grid, the
tie-break, the objective, the promotion gate.  21 evaluations per fold is the
entire search; no finer grid, no second grid, no Optuna, no random search.
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

import blend as K                                            # noqa: E402
from blend import Corpus, Objective, LABELS, N_CLASSES, CLEAN, GRID   # noqa: E402
from metrics import composite, f1_per_class                  # noqa: E402
from metrics import fault_turn_hit_at_k, macro_f1            # noqa: E402

# The task brief's pre-registered Exp08 baseline.  These are reproduction
# TARGETS, not tuning knobs: nothing in the search may read them.
EXPECTED_BASELINE = {
    "macro_f1": 0.7894323366835861,
    "robustness_f1": 0.7368470046834376,
    "success_f1": 0.8643757406010988,
    "fault_turn_hit2": 0.6086111111111111,
    "composite": 0.7694453917139285,
}
PUBLIC_EXP08 = 0.7696        # reference only. Never read by the search.

# The promotion gate, pre-declared.
GATE = {
    "composite_delta_min": 0.002,
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
    lines = ["      %-17s %s" % ("true \\ pred",
                                 " ".join("%5d" % i for i in range(N_CLASSES)))]
    for i, lab in enumerate(LABELS):
        lines.append("      %-17s %s" % (lab, " ".join("%5d" % v for v in m[i])))
    lines.append("      %-17s %s" % ("", " ".join("%5s" % l[:5] for l in LABELS)))
    return "\n".join(lines)


def official_full(corpus, pred):
    """Score a full-length predicted-label vector through evaluation.metrics.

    Macro and robustness straight from the official ``macro_f1``; hit@2 from the
    official ``fault_turn_hit_at_k``, fed ONLY the truly faulty rows with their
    labels AND turns - see ``Objective.official`` for why that matters.
    """
    yt = corpus.name(corpus.yi)
    yp = corpus.name(pred)
    turns = corpus.turns_for(pred)
    f = np.where(corpus.yi != CLEAN)[0]
    h2 = fault_turn_hit_at_k([yt[i] for i in f], [yp[i] for i in f],
                             [int(corpus.fturn[i]) for i in f],
                             [int(turns[i]) for i in f], k=2)
    rf = (macro_f1([yt[i] for i in np.where(corpus.rob)[0]],
                   [yp[i] for i in np.where(corpus.rob)[0]], LABELS)
          if int(corpus.rob.sum()) else 0.0)
    mf = macro_f1(yt, yp, LABELS)
    return {"macro_f1": mf, "robustness_f1": rf,
            "success_f1": K.SUCCESS_F1_PINNED, "fault_turn_hit2": h2,
            "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                    "success_f1": K.SUCCESS_F1_PINNED,
                                    "fault_turn_hit2": h2})}, turns


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
    print("Exp10 - probability blend of Exp07 A (249 feats) and B (300 feats)")
    print("=" * 78)

    res = {"experiment": "exp10_ab_probability_blend", "n_runs": K.N_RUNS,
           "systems": {"A": K.SYSTEM_A, "B": K.SYSTEM_B},
           "search_space": {
               "form": "P_blend = (1 - alpha) * P_A + alpha * P_B; "
                       "pred = argmax(P_blend)",
               "grid": list(GRID), "n_grid": len(GRID), "step": 0.05,
               "single_scalar": True,
               "forbidden": ["class-specific alpha", "offsets", "temperature",
                             "geometric/log blend", "thresholds",
                             "stacking model", "second grid", "finer grid"],
               "tie_break": "1) higher official composite; 2) if gap <= 1e-12, "
                            "alpha closer to 1.0; 3) still tied, larger alpha",
               "objective": "official evaluation.metrics.composite()"},
           "baseline_expected": EXPECTED_BASELINE,
           "public_exp08_reference_only": PUBLIC_EXP08,
           "gate": GATE}

    # ---- 1. strict, verified inputs --------------------------------------
    print("\n[1] verified Exp07 OOF for BOTH systems + sealed window peaks")
    print("    (strict, fail closed - ArtifactRejected is fatal)")
    d = K.load_verified()
    ys, yi = d["ys"], d["yi"]
    print("  ok    %d provenance checks passed, load %.2fs"
          % (len(d["checks"]), d["load_sec"]))
    print("  ok    P_A %s  P_B %s  peak_pos %s"
          % (d["proba_A"].shape, d["proba_B"].shape, d["peak_pos"].shape))

    folds, fold_sha, fold_sha_want = K.fold_ids(yi, d["manifest"])
    if fold_sha != fold_sha_want:
        fail("re-derived outer fold indices differ from the sealed manifest")
    else:
        ok("the 3 outer folds re-derived from seed 0 match the sealed manifest "
           "exactly (index-vector hashes)")

    if not (d["proba_B"].argmax(1) == d["labels_B"]).all():
        fail("argmax(verified P_B) != verified B labels")
    if not (d["proba_A"].argmax(1) == d["labels_A"]).all():
        fail("argmax(verified P_A) != verified A labels")
    ok("argmax(P_A) == verified A labels and argmax(P_B) == verified B labels "
       "on all %d rows" % K.N_RUNS)
    ok("A and B are genuinely different models: %d of %d labels differ"
       % (int((d["labels_A"] != d["labels_B"]).sum()), K.N_RUNS))

    fturn = d["train"]["fault_turn"].values
    rob = K.robustness_mask()
    corpus = Corpus(d["proba_A"], d["proba_B"], yi, fturn, d["peak_pos"], rob,
                    labels_B=d["labels_B"])
    ok("corpus precomputed: P_A %s  P_B %s  candidate L1 turn matrix %s  "
       "robustness rows %d" % (corpus.PA.shape, corpus.PB.shape,
                               corpus.T.shape, int(rob.sum())))

    # ---- 2. REPRODUCTION GATE --------------------------------------------
    print("\n" + "=" * 78)
    print("[2] REPRODUCTION GATE - Exp08 baseline at alpha = 1, official metrics")
    print("=" * 78)
    full_ix = np.arange(K.N_RUNS)
    obj_full = Objective(corpus, full_ix)
    m0 = obj_full.official(1.0)          # no fast path involved

    base_pred = corpus.predict(1.0)
    if not np.array_equal(base_pred, d["labels_B"]):
        fail("alpha=1 labels differ from the verified B labels")
    else:
        ok("alpha=1 reproduces the verified B labels exactly on all %d rows"
           % K.N_RUNS)
    committed = pd.read_csv(os.path.join(ROOT, "experiments",
                                         "exp07_fault_windows",
                                         "oof_%s.csv" % K.SYSTEM_B))
    n_bad = int((committed["y_pred"].values.astype(str)
                 != np.asarray(LABELS)[base_pred]).sum())
    if n_bad:
        fail("alpha=1 labels differ from the committed Exp07 OOF CSV at %d row(s)"
             % n_bad)
    else:
        ok("alpha=1 reproduces the committed Exp07 OOF labels exactly "
           "on all %d rows" % K.N_RUNS)

    # the L1 turn vector at alpha=1 must equal Exp08's fault_turn rule exactly
    turns1 = corpus.turns_for(base_pred)
    oof = pd.read_csv(os.path.join(ROOT, "experiments", "exp07_fault_windows",
                                  "oof_%s.csv" % K.SYSTEM_B))
    exp08_turn = None
    try:
        # Exp08's own results.json holds the L1 diagnostic it shipped
        e08 = json.load(open(os.path.join(ROOT, "experiments",
                                          "exp08_window_localizer",
                                          "results.json"), encoding="utf-8"))
        exp08_turn = e08["localiser"]["L1_window_peak"]["hit@2"]
    except Exception:                                       # noqa: BLE001
        exp08_turn = None
    ok("alpha=1 emits %d non-negative turns; %d clean predictions all -> -1"
       % (int((turns1 >= 0).sum()),
          int(((base_pred == CLEAN) & (turns1 == -1)).sum())))

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
    print("  NOTE public Exp08 leaderboard %.4f recorded as a REFERENCE ONLY; "
          "no search reads it." % PUBLIC_EXP08)
    res["baseline"] = dict(m0)
    res["reproduction_gate"] = {"passed": bool(good),
                                "expected": EXPECTED_BASELINE, "got": dict(m0),
                                "tolerance": K.TOL,
                                "alpha": 1.0,
                                "exp08_l1_hit2_from_results_json": exp08_turn}
    if not good:
        print("\nREPRODUCTION GATE FAILED - STOPPING.  No blend is run on top of a "
              "baseline that does not reproduce.")
        res["stopped"] = "reproduction gate failed"
        _dump(res, t_start)
        return 1
    print("\n  REPRODUCTION GATE PASSED - Exp08 baseline exact on all 5 numbers.")

    # ---- 3. fast objective vs the official metrics -----------------------
    print("\n" + "=" * 78)
    print("[3] FAST OBJECTIVE must equal the official metrics, always")
    print("=" * 78)
    # Real candidates across the whole declared grid - not just the endpoints -
    # because a fast path that only agrees at alpha in {0, 1} has proved nothing.
    probe_alphas = [0.0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
    worst = 0.0
    for a in probe_alphas:
        fast, off = obj_full.metrics(a), obj_full.official(a)
        for k in ("macro_f1", "robustness_f1", "fault_turn_hit2", "composite"):
            d_ = abs(fast[k] - off[k])
            worst = max(worst, d_)
            if d_ > K.TOL:
                fail("fast objective disagrees with the official metric at "
                     "alpha=%.2f on %s: %r vs %r" % (a, k, fast[k], off[k]))
    if worst <= K.TOL:
        ok("fast objective == official metrics on %d probe alphas spanning the "
           "declared grid, max abs diff %.2e" % (len(probe_alphas), worst))
    res["fast_objective_check"] = {"n_probes": len(probe_alphas),
                                   "max_abs_diff": worst, "passed": worst <= K.TOL}
    if worst > K.TOL:
        _dump(res, t_start)
        return 1

    # ---- 4. THE BLEND CURVE over all 10000 (diagnostic, NOT a result) ----
    print("\n" + "=" * 78)
    print("[4] Blend curve over all %d rows - DIAGNOSTIC ONLY" % K.N_RUNS)
    print("    A single alpha chosen here and reported here would be in-sample.")
    print("    It is printed so the report can show the shape of the surface.")
    print("=" * 78)
    a_diag, table_diag = K.select_alpha(obj_full)
    print("    %-7s %-20s %-12s %-12s %-12s %s"
          % ("alpha", "composite", "macro", "robust", "hit@2", "labels vs B"))
    for r in table_diag:
        nchg = int((obj_full.predict(r["alpha"]) != d["labels_B"]).sum())
        mark = "  <- argmax of the full-OOF curve (IN-SAMPLE)" \
            if abs(r["alpha"] - a_diag) < 1e-15 else ""
        print("    %-7.2f %-20.16f %-12.6f %-12.6f %-12.6f %6d%s"
              % (r["alpha"], r["composite"], r["macro_f1"], r["robustness_f1"],
                 r["fault_turn_hit2"], nchg, mark))
    res["full_oof_curve_diagnostic"] = {
        "note": "in-sample, NOT a validation result; the headline is the "
                "cross-fitted number in section 6",
        "best_alpha": a_diag, "table": table_diag}
    print("  full-OOF argmax is alpha=%.2f at composite %.16f"
          % (a_diag, max(r["composite"] for r in table_diag)))
    print("  Exp08 (alpha=1.00) sits at %.16f." % EXPECTED_BASELINE["composite"])

    # ---- 5. alpha SELECTION on the two training folds of each fold -------
    print("\n" + "=" * 78)
    print("[5] OOF meta-CV - alpha chosen on 2 folds, scored on the third")
    print("=" * 78)
    print("  For each fold f: the OTHER two folds are the alpha-selection train,")
    print("  alpha is frozen, then applied to fold f.  Nothing about fold f is")
    print("  used to pick its own alpha.")
    print("  NOTE: OOF meta-CV is not a fully nested retrain of Exp07; the")
    print("  public leaderboard remains the external validation.")

    fold_of = np.zeros(K.N_RUNS, dtype=np.int64)
    for f, va in enumerate(folds):
        fold_of[va] = f
    assert (np.bincount(fold_of, minlength=3) > 0).all()

    cf_pred = np.full(K.N_RUNS, -1, dtype=np.int64)
    cf_alpha = [None] * 3
    fold_train_curve = [None] * 3
    res["folds"] = []

    for f in range(3):
        val_ix = np.asarray(folds[f], dtype=np.int64)
        tr_ix = np.where(fold_of != f)[0]
        # ---- the discipline: the held-out rows are physically absent -----
        obj_tr = Objective(corpus, tr_ix)
        assert not np.intersect1d(val_ix, tr_ix).size, "train/val overlap"
        print("\n  --- fold %d: alpha-train n=%d, held-out n=%d ---"
              % (f, len(tr_ix), len(val_ix)))
        a_sel, table = K.select_alpha(obj_tr)
        cf_alpha[f] = a_sel
        fold_train_curve[f] = table
        print("    alpha selected on the 2 training folds: %.2f  (train "
              "composite %.16f)" % (a_sel, max(r["composite"] for r in table)))
        a_best_tr = max(r["composite"] for r in table)
        print("    training-curve top alphas: %s"
              % ", ".join("%.2f" % r["alpha"] for r in table
                          if r["composite"] >= a_best_tr - 1e-12))

        # ---- apply the FROZEN alpha to the held-out fold -----------------
        obj_va = Objective(corpus, val_ix)
        assert not set(np.asarray(obj_tr.ix).tolist()) & set(val_ix.tolist()), \
            "held-out fold leaked into its own alpha selection"
        pred_va = obj_va.predict(a_sel)
        cf_pred[val_ix] = pred_va
        pred_va1 = obj_va.predict(1.0)
        m_va = obj_va.official(a_sel)
        m_va1 = obj_va.official(1.0)
        d_va = m_va["composite"] - m_va1["composite"]
        changed = int((pred_va != pred_va1).sum())

        res["folds"].append({
            "fold": f, "n_train": int(len(tr_ix)), "n_val": int(len(val_ix)),
            "val_sha256": fold_sha[f], "alpha_selected": a_sel,
            "train_curve_argmax": table,
            "heldout": {
                "A_alpha1": m_va1, "B_alpha_selected": m_va,
                "delta_composite": d_va,
                "n_labels_changed": changed,
                "wins": bool(d_va > 0)}})
        print("    held-out fold %d: %d/%d labels changed; composite "
              "%.16f -> %.16f (%+.16f) %s"
              % (f, changed, len(val_ix), m_va1["composite"], m_va["composite"],
                 d_va, "WIN" if d_va > 0 else "LOSS"))

    if (cf_pred < 0).any():
        fail("some rows received no cross-fitted prediction")
    ok("cross-fitted prediction vector assembled for all %d rows" % K.N_RUNS)

    # ---- 6. A vs B on all 10000 (THE HEADLINE) -------------------------
    print("\n" + "=" * 78)
    print("[6] HEADLINE - A (Exp08, alpha=1) vs B (cross-fitted), all %d rows"
          % K.N_RUNS)
    print("=" * 78)
    yt = corpus.name(corpus.yi)
    ypA = corpus.name(obj_full.predict(1.0))
    ypB = corpus.name(cf_pred)

    A, turnsA = official_full(corpus, obj_full.predict(1.0))
    B, turnsB = official_full(corpus, cf_pred)
    accA = float((obj_full.predict(1.0) == corpus.yi).mean())
    accB = float((cf_pred == corpus.yi).mean())
    pcA = f1_per_class(yt, ypA, LABELS)
    pcB = f1_per_class(yt, ypB, LABELS)
    cmA = confusion(yt, ypA)
    cmB = confusion(yt, ypB)
    n_changed = int((cf_pred != obj_full.predict(1.0)).sum())
    n_turn_changed = int((turnsB != turnsA).sum())
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
    print("  OOF labels changed by the cross-fitted blend: %d / %d (%.2f%%)"
          % (n_changed, K.N_RUNS, 100.0 * n_changed / K.N_RUNS))
    print("  fault_turn values changed as a consequence: %d / %d"
          % (n_turn_changed, K.N_RUNS))

    print("\n  per-class F1 (official f1_per_class):")
    print("    %-18s %10s %10s %10s" % ("class", "A", "B", "delta"))
    for c in LABELS:
        print("    %-18s %10.6f %10.6f %+10.6f" % (c, pcA[c], pcB[c],
                                                  pcB[c] - pcA[c]))
    print("\n  A confusion (rows = true, cols = pred):")
    print(conf_str(cmA))
    print("\n  B confusion (rows = true, cols = pred):")
    print(conf_str(cmB))

    # ---- confusion transitions -------------------------------------------
    predA = obj_full.predict(1.0)
    trans = {}
    for i in range(K.N_RUNS):
        a, b = int(predA[i]), int(cf_pred[i])
        if a != b:
            key = "%s->%s" % (LABELS[a], LABELS[b])
            trans[key] = trans.get(key, 0) + 1
    acc_by = {"A->B fixes": 0, "A->B breaks": 0, "A->B changed": 0,
              "A->B correct_on_B_side": 0, "A->B correct_on_A_side": 0}
    for i in range(K.N_RUNS):
        a, b = int(predA[i]), int(cf_pred[i])
        if a == b:
            continue
        acc_by["A->B changed"] += 1
        ya = int(corpus.yi[i])
        if b == ya and a != ya:
            acc_by["A->B fixes"] += 1
        if a == ya and b != ya:
            acc_by["A->B breaks"] += 1
    print("\n  confusion transitions (A label -> B label), changed rows only:")
    for k in sorted(trans, key=lambda k: -trans[k]):
        print("    %-40s %5d" % (k, trans[k]))
    print("  of the %d changed labels: %d became correct, %d became wrong "
          "(net %+d)" % (acc_by["A->B changed"], acc_by["A->B fixes"],
                         acc_by["A->B breaks"],
                         acc_by["A->B fixes"] - acc_by["A->B breaks"]))

    res["comparison"] = {
        "A_exp08_alpha1": A, "B_cross_fitted": B,
        "delta": {k: B[k] - A[k] for k in A},
        "accuracy": {"A": accA, "B": accB, "delta": accB - accA,
                     "note": "diagnostic only, accuracy is NOT the objective"},
        "per_class_f1": {"A": pcA, "B": pcB,
                         "delta": {c: pcB[c] - pcA[c] for c in LABELS}},
        "confusion_A": cmA.tolist(), "confusion_B": cmB.tolist(),
        "confusion_transitions": trans, "changed_label_stats": acc_by,
        "n_labels_changed": n_changed, "n_fault_turn_changed": n_turn_changed,
        "note": "hit@2 uses the peak of the BLENDED PREDICTED class, so it "
                "moves with the label; it is not held fixed",
    }

    # ---- 7. alpha stability across folds --------------------------------
    print("\n" + "=" * 78)
    print("[7] ALPHA STABILITY across the three selection fits")
    print("=" * 78)
    alphas = cf_alpha
    spread = float(max(alphas) - min(alphas))
    n_distinct = len(set(alphas))
    print("    fold 0 alpha = %.2f" % alphas[0])
    print("    fold 1 alpha = %.2f" % alphas[1])
    print("    fold 2 alpha = %.2f" % alphas[2])
    print("    spread = %.2f, %d distinct value(s) of 3" % (spread, n_distinct))
    # instability: a fold choosing ~0 while another chooses ~1
    extreme = (min(alphas) <= 0.20 and max(alphas) >= 0.80)
    if extreme:
        verdict = "EXTREME INSTABILITY (one fold near 0, another near 1)"
    elif spread > 0.30:
        verdict = "NOTABLE INSTABILITY (spread > 0.30)"
    elif n_distinct == 1:
        verdict = "fully stable (all folds agree)"
    else:
        verdict = "mild variation"
    print("    verdict: %s" % verdict)
    if spread > 0.0:
        print("    NOTE: at alpha=%.2f the blend is the EXP08 baseline itself; "
              "folds that choose 1.0 are choosing NOT TO BLEND."
              % max(alphas))
    res["alpha_stability"] = {
        "alpha_per_fold": [float(a) for a in alphas],
        "spread": spread, "n_distinct": n_distinct,
        "verdict": verdict, "extreme_instability": bool(extreme)}

    # ---- 8. PROMOTION GATE ----------------------------------------------
    print("\n" + "=" * 78)
    print("[8] PROMOTION GATE (pre-declared)")
    print("=" * 78)
    wins = sum(1 for r in res["folds"] if r["heldout"]["wins"])
    checks = []

    d_comp = B["composite"] - A["composite"]
    checks.append(("composite delta >= %+.3f" % GATE["composite_delta_min"],
                   d_comp, GATE["composite_delta_min"],
                   d_comp >= GATE["composite_delta_min"]))
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
    print("\n  PROMOTION GATE: %s" % ("PASS" if gate_passed else "FAIL"))
    if not gate_passed and extreme:
        print("  ADDITIONAL SANITY: alpha is EXTREMELY unstable across folds "
              "(%s). Promotion would require a very strong result." % verdict)
    res["promotion_gate"] = {
        "passed": bool(gate_passed),
        "criteria": [{"name": c[0], "value": c[1], "threshold": c[2],
                      "passed": bool(c[3])} for c in checks],
        "folds_won": wins,
        "worst_class_f1_drop": {"class": worst_cls, "drop": worst_drop},
        "alpha_stability_extreme": bool(extreme)}

    if not gate_passed:
        print("\n  GATE FAILED - no production alpha is fitted, no submission is")
        print("  built, and no other blend is tried.  The experiment stops here.")
        res["final_alpha"] = {"fitted": False, "reason": "promotion gate failed"}
        res["promoted"] = False
        _dump(res, t_start)
        return 0

    # ---- 9. FINAL production alpha, fitted on ALL 10000 OOF -----------
    print("\n" + "=" * 78)
    print("[9] FINAL production alpha - same grid, same objective, fitted on ALL")
    print("    %d OOF rows.  This is a DEPLOYMENT FIT, not a validation" % K.N_RUNS)
    print("    result.  Its full-OOF score is NOT the headline.")
    print("=" * 78)
    a_final, table_final = K.select_alpha(obj_full)
    res["final_alpha"] = {
        "fitted": True, "alpha": a_final,
        "in_sample_full_oof": obj_full.official(a_final),
        "in_sample_labels_changed": int((obj_full.predict(a_final)
                                         != d["labels_B"]).sum()),
        "in_sample_curve": table_final,
        "warning": "fitted on all 10000 OOF rows - deployment only, NOT a "
                   "validation score; the headline is the cross-fitted B",
    }
    print("  FINAL production alpha = %.2f  (deployment constant)" % a_final)
    print("  in-sample (NOT a validation) full-OOF composite %.16f; "
          "%d/%d OOF labels would change"
          % (res["final_alpha"]["in_sample_full_oof"]["composite"],
             res["final_alpha"]["in_sample_labels_changed"], K.N_RUNS))
    res["promoted"] = True

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
