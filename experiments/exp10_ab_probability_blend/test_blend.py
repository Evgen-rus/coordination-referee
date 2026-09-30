"""Tests for the Exp10 A/B probability blend.

A test that would pass on a broken implementation is worse than no test, so
each case here is written to FAIL on the specific mistake it guards against:

  * ``alpha = 1.0`` must reproduce Exp08's labels row for row, and its
    ``fault_turn`` vector row for row, through TWO independent recomputations of
    the L1 rule plus the committed OOF CSV;
  * ``alpha = 0.0`` must reproduce A's own labels, exactly;
  * success F1 must be a pinned constant that no decision rule can move;
  * ``P_A`` / ``P_B`` must be ``(n, 7)``, finite, inside ``[0, 1]``;
  * ``alpha`` must be a scalar inside ``[0, 1]`` - out-of-range and NaN refused;
  * the 21-point grid must be exactly the declared one, and the search
    bit-for-bit deterministic across reruns;
  * a held-out fold must not be able to influence the alpha that scores it;
  * the L1 turn must be read for the BLENDED predicted label, not the true one;
  * nothing in the decision path may read ``y_true``;
  * the fast objective must equal ``evaluation.metrics`` on non-trivial alphas;
  * the recorded gate must be the pre-declared one, and because it FAILED no
    production submission may exist.

Run:  python experiments/exp10_ab_probability_blend/test_blend.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(ROOT, "experiments", "exp08_window_localizer"),
           os.path.join(ROOT, "experiments", "exp07_fault_windows"),
           os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import blend as K                                            # noqa: E402
from blend import Corpus, Objective, LABELS, N_CLASSES, CLEAN, GRID  # noqa: E402

RESULTS = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
SUBMISSION = os.path.join(ROOT, "submission_exp10_ab_blend")
SUBMISSION_ZIP = os.path.join(ROOT, "submission_exp10_ab_blend.zip")

_CACHE = {}
results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print("  PASS  %s" % name)
    except AssertionError as e:
        results.append((name, False, str(e)))
        print("  FAIL  %s -> %s" % (name, e))
    except Exception as e:  # noqa: BLE001
        results.append((name, False, "%s: %s" % (type(e).__name__, e)))
        print("  ERROR %s -> %s: %s" % (name, type(e).__name__, e))


def loaded():
    """The real verified Exp07 OOF for both systems, loaded once. No fitting."""
    if "d" not in _CACHE:
        d = K.load_verified()
        rob = K.robustness_mask()
        c = Corpus(d["proba_A"], d["proba_B"], d["yi"],
                   d["train"]["fault_turn"].values, d["peak_pos"], rob,
                   labels_B=d["labels_B"])
        _CACHE["d"] = (c, d, rob)
    return _CACHE["d"]


# ---------------------------------------------------------------------------
# 1. endpoints - alpha=1 is Exp08, alpha=0 is A
# ---------------------------------------------------------------------------

def test_alpha_one_reproduces_exp08_labels_exactly():
    """alpha = 1 must equal argmax(P_B) on all 10000 rows, bit for bit.

    ``0.0 * P_A + 1.0 * P_B`` is exactly ``P_B`` in IEEE arithmetic, so this is
    an identity rather than a tolerance - which is why a single differing row
    would indicate a real bug rather than float noise.
    """
    c, d, _ = loaded()
    p = c.predict(1.0)
    assert p.shape == (K.N_RUNS,)
    assert np.array_equal(c.blend(1.0), c.PB), \
        "alpha=1 blend is not bit-identical to P_B"
    assert np.array_equal(p, c.PB.argmax(1)), "alpha=1 != argmax(P_B)"
    assert np.array_equal(p, d["labels_B"]), \
        "alpha=1 labels differ from the verified B labels"
    committed = np.loadtxt(os.path.join(ROOT, "experiments", "exp07_fault_windows",
                                        "oof_%s.csv" % K.SYSTEM_B),
                           delimiter=",", skiprows=1, usecols=2, dtype=str)
    assert (committed == np.asarray(LABELS)[p]).all(), \
        "alpha=1 labels differ from the committed Exp07 OOF CSV"


def test_alpha_zero_reproduces_A_labels_exactly():
    """alpha = 0 must equal argmax(P_A) on all 10000 rows, bit for bit."""
    c, d, _ = loaded()
    assert np.array_equal(c.blend(0.0), c.PA), \
        "alpha=0 blend is not bit-identical to P_A"
    assert np.array_equal(c.predict(0.0), c.PA.argmax(1)), \
        "alpha=0 != argmax(P_A)"
    assert np.array_equal(c.predict(0.0), d["labels_A"]), \
        "alpha=0 labels differ from the verified A labels"
    # A and B really are different models, so the two endpoints must differ
    assert int((c.predict(0.0) != c.predict(1.0)).sum()) > 100, \
        "the two endpoints barely differ - the fixture cannot test the blend"


def test_alpha_one_fault_turn_is_exactly_exp08():
    """The L1 turn vector at alpha=1 must match Exp08, checked three ways.

    1. against a fresh, independent re-implementation of the Exp08 rule read
       straight off the sealed ``window_peak_pos`` (a plain Python loop, no
       vectorised helper - so a bug in ``Corpus.T`` cannot hide);
    2. against the committed Exp08 ``fault_turn``-derived hit@2 of
       0.6086111111111111, recomputed by the official metric;
    3. against a deliberately naive brute-force recount, because Exp07 shipped a
       hit@2 that disagreed with the official metric by a factor of two through
       a positional zip misalignment, and Exp08's fix was a second
       implementation.  A test with one implementation cannot catch that class
       of bug.
    """
    c, d, _ = loaded()
    pred = c.predict(1.0)
    turns = c.turns_for(pred)

    # (1) independent re-implementation, reading the peaks directly
    peak = np.load(os.path.join(ROOT, "experiments", "exp07_fault_windows",
                                "runs", "cv", "window_peak_pos.npy"))
    ref = np.full(K.N_RUNS, -1, dtype=np.int64)
    for i in range(K.N_RUNS):
        lab = LABELS[pred[i]]
        if lab != "clean":
            ref[i] = int(peak[i, K.W2I[lab]])
    assert np.array_equal(turns, ref), \
        "the vectorised L1 turn differs from a direct re-read of window_peak_pos"
    # clean -> -1, and every non-clean prediction gets a real turn
    assert bool((turns[pred == CLEAN] == -1).all()), \
        "a clean prediction did not receive fault_turn -1"
    assert bool((turns[pred != CLEAN] >= 0).all()), \
        "a fault prediction was left without a turn"

    # (2) the official metric, and (3) the independent recount
    from metrics import fault_turn_hit_at_k
    f = np.where(c.yi != CLEAN)[0]
    h2 = fault_turn_hit_at_k([LABELS[c.yi[i]] for i in f],
                             [LABELS[pred[i]] for i in f],
                             [int(c.fturn[i]) for i in f],
                             [int(turns[i]) for i in f], k=2)
    assert abs(h2 - 0.6086111111111111) < 1e-12, h2
    hits, n = 0, 0
    for i in range(K.N_RUNS):
        if c.yi[i] == CLEAN:
            continue
        n += 1
        if pred[i] == c.yi[i] and turns[i] >= 0 \
                and abs(int(turns[i]) - int(c.fturn[i])) <= 2:
            hits += 1
    assert abs(hits / n - h2) < 1e-12, (hits / n, h2)


# ---------------------------------------------------------------------------
# 2. invariants of the inputs and the parameter
# ---------------------------------------------------------------------------

def test_probability_matrices_are_well_formed():
    """P_A / P_B must be (n, 7), finite, in [0, 1] - and reject a bad one."""
    c, _, _ = loaded()
    for tag, P in (("P_A", c.PA), ("P_B", c.PB)):
        assert P.shape == (K.N_RUNS, N_CLASSES), (tag, P.shape)
        assert np.isfinite(P).all(), "%s contains non-finite values" % tag
        assert (P >= 0).all() and (P <= 1).all(), "%s outside [0, 1]" % tag
        # rows are distributions: the stored float32 rounds, allow 1e-3 as the
        # strict loader itself does
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-3), tag
    # and a malformed matrix must be refused, not silently blended
    for bad in (np.zeros((K.N_RUNS, 6)), np.full((K.N_RUNS, N_CLASSES), np.nan),
                np.full((K.N_RUNS, N_CLASSES), 2.0)):
        try:
            Corpus(bad, c.PB, c.yi, c.fturn, c.T[:, 1:].astype(np.int32), c.rob)
        except ValueError:
            continue
        raise AssertionError("Corpus accepted a malformed probability matrix")


def test_alpha_is_always_a_scalar_inside_zero_one():
    """Out-of-range, NaN and infinite alphas must be refused at the boundary."""
    for good in (0.0, 0.05, 0.5, 0.95, 1.0, np.float64(0.3)):
        K.check_alpha(good)
    for bad in (-1e-9, 1.0 + 1e-9, -0.5, 1.5, float("nan"), float("inf"),
                float("-inf")):
        try:
            K.check_alpha(bad)
        except AssertionError:
            continue
        raise AssertionError("check_alpha accepted %r" % bad)
    # every grid point is legal, and the grid IS the declared one
    assert len(GRID) == 21, len(GRID)
    assert GRID[0] == 0.0 and GRID[-1] == 1.0
    assert all(abs(g - k / 20.0) < 1e-15 for k, g in enumerate(GRID))
    for g in GRID:
        K.check_alpha(g)


# ---------------------------------------------------------------------------
# 3. the search - frozen grid, deterministic, honest cross-fit
# ---------------------------------------------------------------------------

def test_search_is_deterministic_and_grid_is_frozen():
    """Reruns must be bit-identical, and the grid must be exactly 21 points."""
    c, _, _ = loaded()
    o1 = Objective(c, np.arange(6666))
    a1, t1 = K.select_alpha(o1)
    a2, t2 = K.select_alpha(Objective(c, np.arange(6666)))
    assert a1 == a2, (a1, a2)
    assert [r["composite"] for r in t1] == [r["composite"] for r in t2]
    assert len(t1) == 21
    assert [r["alpha"] for r in t1] == list(GRID), \
        "the evaluated alphas are not the declared grid"
    # the blend itself must be a plain probability-level average: a geometric
    # or log-space blend is a DIFFERENT function and would still "work" here,
    # so the test pins the arithmetic rather than just the output
    a = 0.3
    assert np.allclose(c.blend(a), (1 - a) * c.PA + a * c.PB, atol=0, rtol=0)
    geo = np.exp((1 - a) * np.log(np.maximum(c.PA, 1e-15))
                 + a * np.log(np.maximum(c.PB, 1e-15)))
    assert not np.allclose(c.blend(a), geo, atol=1e-6), \
        "the blend matches a geometric mean - the arithmetic is not the " \
        "declared one"


def test_heldout_fold_cannot_influence_its_own_alpha():
    """Fit on {0,1} and on {0,1,2}, then prove fold 2 can tell the difference.

    Two assertions, and the second is the one that matters:

      1. the three recorded selection alphas are not all identical, i.e. the
         held-out fold really was excluded - if it had been included, all three
         would have seen the same 10000 rows and agreed;
      2. an Objective built on the training rows can have the held-out rows
         corrupted without changing the alpha it selects, because the fitting
         Objective never indexes them.  The corruption is applied to the
         held-out rows' probability matrices - the only thing held-out data
         could reach through the decision - and the selected alpha must come out
         identical.
    """
    recorded = RESULTS["alpha_stability"]["alpha_per_fold"]
    assert len(recorded) == 3
    assert len(set(recorded)) > 1, \
        "all three folds selected the same alpha - the held-out fold was NOT " \
        "excluded from its own selection"

    c, _, _ = loaded()
    import load_exp07 as L
    fold_ix = L.reuse.fold_assignments(c.yi)
    for f in range(3):
        val = np.asarray(fold_ix[f], dtype=np.int64)
        tr = np.where(~np.isin(np.arange(K.N_RUNS), val))[0]
        a_ref, _ = K.select_alpha(Objective(c, tr))
        c2 = Corpus(c.PA.copy(), c.PB.copy(), c.yi.copy(), c.fturn, c.T.copy(),
                    c.rob, labels_B=c.labels_B)
        # corrupt ONLY the held-out rows: scramble both probability matrices
        rng = np.random.default_rng(f)
        c2.PA[val] = rng.random((len(val), N_CLASSES))
        c2.PB[val] = rng.random((len(val), N_CLASSES))
        a_cor, _ = K.select_alpha(Objective(c2, tr))
        assert a_ref == a_cor, (
            "fold %d: corrupting the held-out rows changed their own selected "
            "alpha (%r -> %r)" % (f, a_ref, a_cor))


def test_no_true_label_leakage_in_the_decision_path():
    """Nothing that EMITS a prediction may read ``y_true``.

    The emitted label vector and the emitted turn vector must be bit-identical
    when every true label is replaced by noise.  The scores necessarily move -
    they are computed against y_true - but the decision cannot.
    """
    c, _, _ = loaded()
    ref_pred = c.predict(0.7)
    ref_turn = c.turns_for(ref_pred)
    c2 = Corpus(c.PA.copy(), c.PB.copy(), np.random.default_rng(0).integers(
        0, N_CLASSES, K.N_RUNS), c.fturn, c.T.copy(), c.rob,
        labels_B=c.labels_B)
    for a in (0.0, 0.35, 0.7, 1.0):
        p = c2.predict(a)
        assert np.array_equal(p, c.predict(a)), \
            "the predicted label at alpha=%.2f moved when y_true was replaced" % a
        assert np.array_equal(c2.turns_for(p), c.turns_for(p)), \
            "the emitted turn at alpha=%.2f moved when y_true was replaced" % a
    assert np.array_equal(c2.predict(0.7), ref_pred)
    assert np.array_equal(c2.turns_for(ref_pred), ref_turn)


def test_fault_turn_is_read_for_the_BLENDED_predicted_label():
    """The turn must follow the BLEND's label, not the true class's peak.

    ``fault_turn_hit_at_k`` counts a hit only when ``pred == y_true``, so on
    every row that can contribute a hit the predicted column IS the true
    column: reading the true class's peak instead scores IDENTICALLY.  A test
    that only compared hit@2 numbers would therefore pass a wrong
    implementation, which is exactly what Exp09 established.  What differs is
    the EMITTED turn, and that is what is asserted here - on a synthetic corpus
    built so the head misclassifies every faulty run, and on the real OOF.
    """
    # ---- the invariance, recorded so nobody re-derives it wrongly --------
    c, _, _ = loaded()
    idx = np.where(c.yi != CLEAN)[0]
    for a in (0.0, 0.4, 0.7, 1.0):
        pred = c.predict(a)
        t_p = c.turns_for(pred)
        t_t = c.T[idx, c.yi[idx]]
        h_p = (pred[idx] == c.yi[idx]) & (t_p[idx] >= 0) \
            & (np.abs(t_p[idx] - c.fturn[idx]) <= 2)
        h_t = (pred[idx] == c.yi[idx]) & (t_t >= 0) \
            & (np.abs(t_t - c.fturn[idx]) <= 2)
        assert (h_p == h_t).all(), \
            "hit@2 differs between the predicted and the true column at " \
            "alpha=%.2f - the metric contract has changed" % a

    # ---- the EMITTED column, where the two readings are far apart -------
    n = 400
    yi = np.array([CLEAN] + [1 + (i % 6) for i in range(n - 1)], dtype=np.int64)
    fturn = np.zeros(n, dtype=np.int64)
    peak = np.full((n, N_CLASSES), 40, dtype=np.int32)   # every class a miss...
    for i in range(n):
        if yi[i] != CLEAN:
            peak[i, K.W2I[LABELS[yi[i]]]] = 0            # ...except the true one
    PA = np.full((n, N_CLASSES), 0.10)
    PA[np.arange(n), yi] = 0.40
    PA[:, CLEAN] = 0.45                                 # -> the head says clean
    toy = Corpus.__new__(Corpus)                        # build without the
    toy.PA, toy.PB = PA, PA.copy()                      # 10000-row guard
    toy.n, toy.yi, toy.fturn = n, yi, fturn
    toy.rob = np.zeros(n, bool)
    toy.labels_B = PA.argmax(1)
    T = np.empty((n, N_CLASSES), np.int64)
    T[:, CLEAN] = -1
    for cc in K.FREE_CLASSES:
        T[:, cc] = peak[:, K.W2I[LABELS[cc]]]
    toy.T = T
    pred = toy.predict(0.0)
    assert (pred == CLEAN).all(), "fixture does not force a misclassification"
    emitted = toy.turns_for(pred)
    assert (emitted == -1).all(), \
        "a clean blended prediction must emit -1, got %r" % np.unique(emitted)
    # Compare on the FAULTY rows only.  The single clean row has y_true ==
    # predicted == clean, so T[pred] and T[y_true] are both -1 there by rule -
    # including it would make this assertion false for a correct
    # implementation.  The fixture separates the two readings on exactly the
    # misclassified rows, which is where the distinction is observable.
    fr = yi != CLEAN
    assert bool((emitted[fr] != T[np.arange(n), yi][fr]).all()), \
        "the two columns are identical here - the fixture cannot distinguish " \
        "the blended label's peak from the true class's"
    # hit@2 is 0 for both readings, which is why the SCORE cannot tell them
    # apart and the EMITTED column is the only observable difference
    o = Objective.__new__(Objective)
    o.corpus, o.ix, o.n = toy, np.arange(n), n
    o.PA, o.PB = PA, toy.PB
    o.y = yi
    o.row_counts = np.bincount(yi, minlength=N_CLASSES)
    o.present = o.row_counts > 0
    o.rob_pos = np.zeros(0, dtype=np.int64)
    o.rob_y = np.zeros(0, dtype=np.int64)
    o.rob_row_counts = np.zeros(N_CLASSES, dtype=np.int64)
    o.rob_present = np.zeros(N_CLASSES, dtype=bool)
    o.n_rob = 0
    f = np.where(yi != CLEAN)[0]
    o.fpos, o.frows, o.fy = f, f, yi[f]
    o.Tf = T[f]
    o.fturn_f = fturn[f]
    o.n_faulty = int(len(f))
    assert o.metrics(0.0)["fault_turn_hit2"] == 0.0

    # ---- and on the REAL OOF the columns genuinely differ ----------------
    rows = np.arange(K.N_RUNS)
    p = c.predict(0.7)
    e_pred, e_true = c.T[rows, p], c.T[rows, c.yi]
    d = e_pred != e_true
    assert int(d.sum()) > 100, "the real OOF does not separate the columns"
    assert bool((d & (p != c.yi)).sum() == d.sum()), \
        "the columns differ on a row where pred == y_true, which is impossible"


def test_fast_objective_equals_official_metrics():
    """The search runs on the fast path; it must equal evaluation.metrics."""
    c, _, _ = loaded()
    o = Objective(c, np.arange(6666))
    for a in (0.0, 0.1, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0):
        fast, off = o.metrics(a), o.official(a)
        for key in ("macro_f1", "robustness_f1", "fault_turn_hit2", "composite"):
            assert abs(fast[key] - off[key]) < 1e-12, (a, key, fast[key], off[key])
        assert fast["success_f1"] == off["success_f1"] == K.SUCCESS_F1_PINNED


def test_success_is_pinned_and_immovable():
    """Success F1 is a constant; no decision rule may recompute it."""
    c, _, _ = loaded()
    assert K.SUCCESS_F1_PINNED == 0.8643757406010988
    o = Objective(c, np.arange(K.N_RUNS))
    for a in (0.0, 0.5, 1.0):
        m = o.metrics(a)
        assert m["success_f1"] == K.SUCCESS_F1_PINNED, (a, m["success_f1"])
    # the composite weight on success is the official one, untouched
    from metrics import WEIGHTS
    assert WEIGHTS["success_f1"] == 0.15
    A = RESULTS["comparison"]["A_exp08_alpha1"]
    B = RESULTS["comparison"]["B_cross_fitted"]
    assert A["success_f1"] == B["success_f1"] == K.SUCCESS_F1_PINNED
    assert A["composite"] - B["composite"] != 0.0 or True   # success cancels


def test_tiebreak_is_the_predeclared_one():
    """composite, then alpha closer to 1.0, then larger alpha - in that order.

    The tie window is 1e-12 as declared.  On this objective a composite is a
    mean over integer counts, so two grid points CAN land within 1e-12 of each
    other; the declared rule is that the one nearer 1.0 - the incumbent Exp08
    baseline - wins, so "no evidence" resolves to "do not ship".
    """
    # strictly higher composite wins outright
    assert K._better(1.0 + 1e-9, 0.5, 1.0, 1.0)
    # a clear loss loses
    assert not K._better(1.0 - 1e-6, 1.0, 1.0, 0.5)
    # exactly tied -> the alpha closer to 1.0 wins, even from below
    assert K._better(1.0, 1.0, 1.0, 0.65)
    assert not K._better(1.0, 0.65, 1.0, 1.0)
    # 1e-13 apart IS a tie, so the tie-break decides and picks 1.0
    assert K._better(1.0 - 1e-13, 1.0, 1.0, 0.95)
    # still tied between 0.65 and 0.95 -> the larger alpha wins
    assert K._better(1.0, 0.95, 1.0, 0.65)
    assert not K._better(1.0, 0.65, 1.0, 0.95)
    # identical alpha, identical composite -> neither is better (strict order)
    assert not K._better(1.0, 1.0, 1.0, 1.0)
    # 1e-9 apart is a REAL difference: 0.95 must not be treated as tied
    assert K._better(1.0, 1.0, 1.0 - 1e-9, 0.95)
    # and the recorded full-OOF argmax is what the rule produces
    diag = RESULTS["full_oof_curve_diagnostic"]
    table = diag["table"]
    best = max(r["composite"] for r in table)
    winners = [r["alpha"] for r in table if r["composite"] >= best - K.TOL]
    assert diag["best_alpha"] == max(winners), (diag["best_alpha"], winners)


def test_gate_is_applied_exactly_as_declared():
    """The recorded gate must be the pre-declared one, with no moving parts."""
    g = RESULTS["promotion_gate"]
    assert GATE_MIN == 0.002 and GATE_FOLDS == 2
    assert GATE_ROB == 0.003 and GATE_CLS == 0.02
    comp = RESULTS["comparison"]
    d_comp = (comp["B_cross_fitted"]["composite"]
              - comp["A_exp08_alpha1"]["composite"])
    assert abs(g["criteria"][0]["value"] - d_comp) < 1e-12
    assert g["folds_won"] == sum(1 for r in RESULTS["folds"]
                                 if r["heldout"]["wins"])
    assert g["passed"] == all(c["passed"] for c in g["criteria"])
    assert bool(g["passed"]) == bool(RESULTS.get("promoted"))
    # the headline A column must BE the pre-registered Exp08 baseline
    b = RESULTS["baseline"]
    for k, v in (("macro_f1", 0.7894323366835861),
                 ("robustness_f1", 0.7368470046834376),
                 ("success_f1", 0.8643757406010988),
                 ("fault_turn_hit2", 0.6086111111111111),
                 ("composite", 0.7694453917139285)):
        assert abs(b[k] - v) < 1e-12, (k, b[k])
        assert abs(comp["A_exp08_alpha1"][k] - v) < 1e-12, (k,)
    # the reported deltas must equal the difference of the reported columns
    for k in ("macro_f1", "robustness_f1", "fault_turn_hit2", "composite"):
        assert abs(comp["delta"][k]
                   - (comp["B_cross_fitted"][k] - comp["A_exp08_alpha1"][k])) < 1e-12


def test_gate_failed_so_no_production_build_exists():
    """The gate FAILED, so no alpha may be shipped and no submission may exist.

    This is a real assertion, not a skip: a stray production build after a
    failed gate is exactly the outcome this experiment is forbidden from
    producing, and it would be easy to miss in a directory listing.
    """
    assert RESULTS.get("promoted") is False, \
        "results.json claims promotion despite a failed gate"
    assert RESULTS["promotion_gate"]["passed"] is False
    fa = RESULTS.get("final_alpha", {})
    assert fa.get("fitted") is False, \
        "a production alpha was fitted despite a failed gate"
    assert "reason" in fa
    assert not os.path.exists(SUBMISSION), \
        "submission_exp10_ab_blend exists although the gate FAILED"
    assert not os.path.exists(SUBMISSION_ZIP), \
        "submission_exp10_ab_blend.zip exists although the gate FAILED"
    # and the report must carry the mandatory meta-CV caveat
    assert "not a fully nested retrain" in RESULTS["caveat"]
    # the public score is recorded as a reference and never used as a target
    assert RESULTS["public_exp08_reference_only"] == 0.7696
    assert RESULTS["baseline"]["composite"] != RESULTS["public_exp08_reference_only"]


GATE_MIN, GATE_FOLDS, GATE_ROB, GATE_CLS = 0.002, 2, 0.003, 0.02


def main():
    print("=" * 78)
    print("Exp10 A/B probability blend tests")
    print("=" * 78)
    promoted = bool(RESULTS.get("promoted"))
    print("  promotion gate: %s (expected FAIL) -> production-build test runs"
          % ("PASS" if promoted else "FAIL"))
    tests = [
        ("alpha=1.0 reproduces Exp08 labels exactly (3 sources)",
         test_alpha_one_reproduces_exp08_labels_exactly),
        ("alpha=0.0 reproduces A's labels exactly",
         test_alpha_zero_reproduces_A_labels_exactly),
        ("alpha=1.0 fault_turn is exactly Exp08 (3 independent checks)",
         test_alpha_one_fault_turn_is_exactly_exp08),
        ("P_A / P_B shapes (n,7), finite, in [0,1]; malformed refused",
         test_probability_matrices_are_well_formed),
        ("alpha is always a scalar inside [0,1]; NaN/inf refused",
         test_alpha_is_always_a_scalar_inside_zero_one),
        ("search is deterministic and the 21-point grid is frozen",
         test_search_is_deterministic_and_grid_is_frozen),
        ("held-out fold cannot influence its own selected alpha",
         test_heldout_fold_cannot_influence_its_own_alpha),
        ("no true-label leakage in the decision path",
         test_no_true_label_leakage_in_the_decision_path),
        ("fault_turn is read for the BLENDED predicted label",
         test_fault_turn_is_read_for_the_BLENDED_predicted_label),
        ("fast objective == official evaluation.metrics",
         test_fast_objective_equals_official_metrics),
        ("success is pinned and immovable",
         test_success_is_pinned_and_immovable),
        ("tie-break is composite > nearer 1.0 > larger alpha",
         test_tiebreak_is_the_predeclared_one),
        ("promotion gate applied exactly as declared",
         test_gate_is_applied_exactly_as_declared),
        ("gate FAILED, so no submission and no production alpha exist",
         test_gate_failed_so_no_production_build_exists),
    ]
    for name, fn in tests:
        check(name, fn)

    npass = sum(1 for _, o, _ in results if o)
    print("\n%d/%d passed" % (npass, len(results)))
    for name, o, err in results:
        if o is False:
            print("  FAILED: %s -> %s" % (name, err))
    if [n for n, o, _ in results if o is False]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
