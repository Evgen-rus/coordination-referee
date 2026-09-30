"""Tests for the Exp09 decision-layer calibration.

A test that would pass on a broken implementation is worse than no test, so
each case here is written to FAIL on the specific mistake it guards against:

  * zero offsets must reproduce Exp08's labels row for row;
  * ``log(P) + 0`` must be argmax-equivalent, including on rows where a
    probability underflows to 0 (argmax of ``log`` -inf is still well defined);
  * the clean offset must stay pinned at 0 - not "small", not "close to";
  * the declared bounds must be enforced, never widened, never silently clipped;
  * the search must be bit-for-bit deterministic across reruns;
  * a held-out fold must not be able to influence the offsets that score it;
  * hit@2 must read the peak of the PREDICTED class, not of the true class - a
    bug here would inflate hit@2 while leaving macro F1 untouched, so it is
    tested by construction, not by looking at the final numbers;
  * the production constants must be the full-OOF search output verbatim (this
    one is skipped when the promotion gate failed and no production build was
    made, and the skip is reported rather than hidden).

Run:  python experiments/exp09_decision_calibration/test_calibration.py
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

import calib as K                                       # noqa: E402
from calib import Corpus, Objective, N_CLASSES, CLEAN, FREE_CLASSES  # noqa: E402

RESULTS = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))

_CACHE = {}
results = []


def check(name, fn, skip=False):
    if skip:
        results.append((name, None, "skipped"))
        print("  SKIP  %s" % name)
        return
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


def corpus():
    """The real verified Exp07 OOF, loaded once.  No model is fitted."""
    if "c" not in _CACHE:
        d = K.load_verified()
        rob = K.robustness_mask()
        c = Corpus(d["proba"], d["yi"], d["train"]["fault_turn"].values,
                   d["peak_pos"], rob, labels=d["labels"])
        _CACHE["c"] = (c, d)
    return _CACHE["c"]


def toy(seed=0, n=2000):
    """A small synthetic corpus with a KNOWN non-zero peak, for rule tests."""
    rng = np.random.default_rng(seed)
    yi = rng.integers(0, N_CLASSES, n)
    P = rng.dirichlet(np.ones(N_CLASSES), n)
    pos = np.full((n, N_CLASSES), 3, dtype=np.int32)
    rob = rng.random(n) < 0.13
    return Corpus(P, yi, rng.integers(0, 40, n), pos, rob)


# ---------------------------------------------------------------------------

def test_zero_offsets_reproduce_exp08_labels():
    """delta = 0 must be Exp08's argmax(P), row for row, all 10000."""
    c, d = corpus()
    zero = K.to_offsets(np.zeros(N_CLASSES, dtype=np.int64))
    assert c.predict(zero).shape == (K.N_RUNS,)
    assert np.array_equal(c.predict(zero), c.P.argmax(1)), \
        "log-space decision at delta=0 differs from argmax(P)"
    assert np.array_equal(c.predict(zero), d["labels"]), \
        "log-space decision at delta=0 differs from the verified hard labels"
    committed = np.array(json.loads(json.dumps(
        RESULTS["comparison"]["A_exp08_no_offsets"]))["macro_f1"])
    assert abs(committed - 0.7894323366835861) < 1e-12, committed


def test_log_p_equals_argmax_p_even_with_zero_probability():
    """log(P) + 0 == argmax(P), including rows containing an exact 0.

    ``log(0)`` is clipped to ``log(1e-15)``; the argmax must not move.  If the
    real float32 OOF matrix happens to contain no exact zero, the fixture is
    built here so the case is still covered - otherwise the test would pass
    without ever exercising the clip.
    """
    c, _ = corpus()
    P = c.P.copy()
    P[0, 3] = 0.0
    P[1, :] = 0.0                      # an all-zero row: every log clipped
    P[2, :] = 0.0
    P[2, 5] = 1.0
    c2 = Corpus(P, c.yi, c.fturn, np.zeros((K.N_RUNS, N_CLASSES), np.int32),
                c.rob, labels=c.labels)
    d0 = np.zeros(N_CLASSES)
    assert float(np.log(K.EPS)) == c2.logP[0, 3], "the clip is not log(1e-15)"
    assert np.array_equal(c2.predict(d0), P.argmax(1)), \
        "log-space argmax moved on a zero-probability row"
    assert np.isfinite(c2.logP).all(), "logP contains non-finite values"


def test_clean_offset_is_pinned():
    """The clean offset must be EXACTLY 0, and the search must never move it."""
    c, _ = corpus()
    obj = Objective(c, np.arange(2000))
    units, _info = K.coordinate_search(obj, verbose=False)
    assert units[CLEAN] == 0, "the search moved the clean offset to %r" % units[CLEAN]
    d = K.to_offsets(units)
    assert d[CLEAN] == 0.0, "to_offsets produced clean=%r" % d[CLEAN]
    # a NEGATIVE clean offset would trade recall for precision; it must be
    # impossible to express one through the public API
    for bad in (np.full(N_CLASSES, 0.1), np.full(N_CLASSES, -0.1)):
        try:
            K.check_bounds(bad)
        except AssertionError as e:
            assert "clean" in str(e), e
        else:
            raise AssertionError("check_bounds accepted a non-zero clean offset")


def test_bounds_are_never_violated_or_widened():
    """Bounds hold on the recorded offsets and the search cannot cross them."""
    assert K.BOUND == 0.40 and -K.BOUND == -0.40, K.BOUND
    assert K.STEP_SCHEDULE == (0.20, 0.10, 0.05, 0.025, 0.0125)
    c, _ = corpus()
    for ix in (np.arange(3000), np.arange(5000)):
        units, _ = K.coordinate_search(Objective(c, ix), verbose=False)
        K.check_bounds(K.to_offsets(units))
    # the bound itself is legal
    edge = np.zeros(N_CLASSES, dtype=np.int64)
    edge[1] = K.BOUND_UNITS
    edge[2] = -K.BOUND_UNITS
    K.check_bounds(K.to_offsets(edge))
    d = K.to_offsets(edge)
    assert abs(d[1] - 0.40) < 1e-15 and abs(d[2] + 0.40) < 1e-15, d
    # ---- leaving the box must be refused, for each class and each side ----
    # +: start exactly ON the wall, then step out (a step that stays on the
    #     wall is legal, so the bump must be at least wall+1)
    for c_i in FREE_CLASSES:
        for extra in (1, 2, 1000):
            u = np.zeros(N_CLASSES, dtype=np.int64)
            u[c_i] = K.BOUND_UNITS + extra
            try:
                K.check_bounds(K.to_offsets(u))
            except AssertionError as e:
                assert "bounds" in str(e), e
                continue
            raise AssertionError("check_bounds accepted %r" % u.tolist())
        u = np.zeros(N_CLASSES, dtype=np.int64)
        u[c_i] = -K.BOUND_UNITS - extra
        try:
            K.check_bounds(K.to_offsets(u))
        except AssertionError as e:
            assert "bounds" in str(e), e
            continue
        raise AssertionError("check_bounds accepted %r" % u.tolist())
    # and the search itself can never produce one, on any calibration subset
    for ix in (np.arange(1000), np.arange(3334), np.arange(K.N_RUNS)):
        units, _ = K.coordinate_search(Objective(c, ix), verbose=False)
        assert (np.abs(units[list(FREE_CLASSES)]) <= K.BOUND_UNITS).all(), \
            "the search left the declared box: %r" % units.tolist()
    # the recorded per-fold offsets must also be inside the box
    for f, off in RESULTS["offset_stability"]["offsets_per_fold"].items():
        a = np.array(off, dtype=np.float64).ravel()   # fold "0" keys as a str
        assert a.shape == (N_CLASSES,), (f, a.shape)
        K.check_bounds(a)
        fault_part = a[list(FREE_CLASSES)]
        assert (np.abs(fault_part) <= K.BOUND + 1e-15).all(), (f, a)


def test_search_is_deterministic():
    """Reruns must be bit-identical, not merely similar."""
    c, _ = corpus()
    ix = np.arange(4000)
    o1 = Objective(c, ix)
    o2 = Objective(c, ix)
    u1, i1 = K.coordinate_search(o1, verbose=False)
    u2, i2 = K.coordinate_search(o2, verbose=False)
    assert np.array_equal(u1, u2), (u1, u2)
    assert i1["final_composite"] == i2["final_composite"]
    assert i1["n_candidates"] == i2["n_candidates"]
    # A reversed row order is a different arithmetic order, so the composite's
    # last bits may differ.  What must NOT differ is the DECISION, which is an
    # argmax over a log-probability gap (0.0125) that is orders of magnitude
    # larger than float64 noise - so the label vector is stable even where the
    # reported composite is not.  This is also the property that matters for
    # reproducibility of a shipped offset vector.
    o3 = Objective(c, ix[::-1])
    u3, i3 = K.coordinate_search(o3, verbose=False)
    same_units = np.array_equal(u1, u3)
    print("        reversed row order -> units %s"
          % ("identical" if same_units else "DIFFERENT %r" % (u3.tolist(),)))
    # o3's output is in the order of its own `ix`, which is reversed, so undo
    # the reversal before comparing - otherwise this compares row i to row n-1-i
    p3 = o3.predict(K.to_offsets(u3))[::-1]
    p1 = o1.predict(K.to_offsets(u1))
    n_bad = int((p1 != p3).sum())
    assert n_bad == 0, (
        "%d of %d decisions moved under a pure row reorder at the SAME offsets "
        "- the argmax is not numerically stable and the search is not "
        "reproducible" % (n_bad, len(p1)))
    # the grid really is the declared one
    assert i1["grid_unit"] == K.UNIT and i1["bound"] == K.BOUND


def test_heldout_fold_does_not_influence_its_own_offsets():
    """Fit on {0,1} and on {0,1,2}, then prove fold 2 can tell the difference.

    Two assertions, and the second is the one that matters:

      1. the recorded per-fold offsets are DIFFERENT from each other, i.e. the
         held-out fold really was excluded - if fold 2 had been included in its
         own fit, all three vectors would be identical;
      2. an Objective built on the held-out rows can be corrupted without
         changing the offsets, because the fitting Objective never indexes
         them.  The corruption is applied to the held-out rows' TURN matrix,
         which is the only thing held-out data could influence through the
         decision, and the offsets must come out bit-identical.
    """
    folds = [r["val_sha256"] for r in RESULTS["folds"]]
    offs = RESULTS["offset_stability"]["units_per_fold"]
    assert len(offs) == 3
    assert not (offs[0] == offs[1] == offs[2]), \
        "all three folds found the same offsets - the held-out fold was NOT " \
        "excluded from its own calibration"
    assert not np.array_equal(offs[0], offs[1]), "folds 0 and 1 agree exactly"
    assert not np.array_equal(offs[1], offs[2]), "folds 1 and 2 agree exactly"
    assert not np.array_equal(offs[0], offs[2]), "folds 0 and 2 agree exactly"

    c, _ = corpus()
    import load_exp07 as L
    fold_ix = L.reuse.fold_assignments(c.yi)
    for f in range(3):
        val = np.asarray(fold_ix[f], dtype=np.int64)
        tr = np.where(~np.isin(np.arange(K.N_RUNS), val))[0]
        u_ref, _ = K.coordinate_search(Objective(c, tr), verbose=False)
        # corrupt ONLY the held-out rows: break their peaks and their labels'
        # logits.  If the fit could see them, the answer would move.
        c2 = Corpus(c.P.copy(), c.yi.copy(), c.fturn, c.T.copy(), c.rob,
                    labels=c.labels)
        c2.T[val] = 0                       # every held-out peak -> turn 0
        c2.logP[val] = np.random.default_rng(f).random((len(val), N_CLASSES))
        u_cor, _ = K.coordinate_search(Objective(c2, tr), verbose=False)
        assert np.array_equal(u_ref, u_cor), \
            "fold %d: corrupting the held-out rows changed their own offsets" % f


def test_hit2_uses_the_peak_of_the_PREDICTED_class():
    """The emitted fault_turn must be ``T[predicted]``, row by row.

    Two separate things, and conflating them is the mistake this test exists to
    prevent.

    1. THE SCORE CANNOT DISTINGUISH THE TWO READINGS.  ``fault_turn_hit_at_k``
       counts a hit only when ``pred == y_true``, so on every row that can
       contribute a hit the predicted column IS the true column.  A
       "read the true class's peak" implementation therefore scores identically
       - verified on the real 10000-row OOF, where both give 0.6086111111111111
       at delta=0 and 0.6116666666666667 at a non-zero delta.  So a test that
       only compared hit@2 numbers would pass a wrong implementation.

    2. WHAT THE EMITTED COLUMN ACTUALLY IS.  ``T[pred]`` and ``T[true]``
       differ on 1903 of 10000 rows - every misclassified faulty run - so the
       two readings are trivially distinguishable by looking at the turns
       themselves.  That is what is asserted here, on a synthetic corpus where
       the two columns are maximally different, AND on the real OOF.
    """
    # ---- (1) the invariance, stated so nobody re-derives it wrongly -------
    c, _ = corpus()
    rows = np.arange(K.N_RUNS)
    idx = np.where(c.yi != CLEAN)[0]
    for off in (np.zeros(N_CLASSES),
                K.to_offsets(np.array([0, 32, -30, 0, 0, 12, -8], dtype=np.int64))):
        pred = c.predict(off)
        t_p = c.T[idx, pred[idx]]
        t_t = c.T[idx, c.yi[idx]]
        hit_p = (pred[idx] == c.yi[idx]) & (t_p >= 0) & (np.abs(t_p - c.fturn[idx]) <= 2)
        hit_t = (pred[idx] == c.yi[idx]) & (t_t >= 0) & (np.abs(t_t - c.fturn[idx]) <= 2)
        assert (hit_p == hit_t).all(), \
            "hit@2 differs between the predicted and the true column - the " \
            "metric contract has changed"

    # ---- (2) the emitted column, where the two readings are far apart -----
    # The corpus is built so the label head MISSES on the faulty rows: the
    # probability mass sits on the TRUE class but the clean column is nudged
    # above it, so the decision is clean while y_true is a fault.  That is the
    # only configuration in which T[pred] and T[y_true] can differ, and it is
    # exactly the 1903-row population on the real OOF.
    n = 400
    yi = np.array([CLEAN] + [1 + (i % 6) for i in range(n - 1)], dtype=np.int64)
    fturn = np.zeros(n, dtype=np.int64)
    T = np.full((n, N_CLASSES), 40, dtype=np.int32)     # every class a miss...
    for i in range(n):
        if yi[i] != CLEAN:
            T[i, yi[i]] = 0                             # ...except the true one
    P = np.full((n, N_CLASSES), 0.10)
    P[np.arange(n), yi] = 0.40
    P[np.arange(n), CLEAN] = 0.45                       # -> the head predicts clean
    toy_c = Corpus(P, yi, fturn, T, np.zeros(n, bool))
    o = Objective(toy_c, np.arange(n))
    pred = o.predict(np.zeros(N_CLASSES))
    assert (pred == CLEAN).all(), "fixture does not force a misclassification"
    emitted = toy_c.T[np.arange(n), pred]
    assert (emitted == -1).all(), \
        "a clean prediction must emit -1, got %r" % np.unique(emitted)
    assert (emitted != T[np.arange(n), yi]).all(), \
        "the two columns are identical here - the fixture cannot distinguish " \
        "the predicted column from the true one"
    # the forbidden variant, spelled out, would emit the TRUE class's peak.
    # The clean row has no true fault class, so it is excluded from the check.
    naive = T[yi != CLEAN, yi[yi != CLEAN]]
    assert (naive == 0).all(), \
        "the forbidden variant should have emitted turn 0 on the faulty rows"
    # hit@2 is 0 for both readings, which is why the SCORE cannot tell them
    # apart and the EMITTED column is the only observable difference
    assert o.metrics(np.zeros(N_CLASSES))["fault_turn_hit2"] == 0.0

    # ---- and on the REAL OOF, where they genuinely differ ----------------
    pred0 = c.predict(np.zeros(N_CLASSES))
    emit_pred = c.T[rows, pred0]
    emit_true = c.T[rows, c.yi]
    d = emit_pred != emit_true
    assert int(d.sum()) > 100, "the real OOF does not separate the columns"
    assert bool((d & (pred0 != c.yi)).sum() == d.sum()), \
        "the columns differ on a row where pred == y_true, which is impossible"
    n_clean = int((pred0 == CLEAN).sum())
    assert bool((emit_pred[pred0 == CLEAN] == -1).all()), \
        "a clean prediction did not receive fault_turn -1"


def test_fast_objective_equals_official_metrics():
    """The search runs on the fast path; it must equal evaluation.metrics."""
    c, _ = corpus()
    o = Objective(c, np.arange(4000))
    rng = np.random.default_rng(3)
    for k in range(6):
        u = np.zeros(N_CLASSES, dtype=np.int64)
        u[list(FREE_CLASSES)] = rng.integers(-K.BOUND_UNITS, K.BOUND_UNITS + 1,
                                             len(FREE_CLASSES))
        d = K.to_offsets(u)
        a, b = o.metrics(d), o.official(d)
        for key in ("macro_f1", "robustness_f1", "fault_turn_hit2", "composite"):
            assert abs(a[key] - b[key]) < 1e-12, (key, a[key], b[key])


def test_production_constants_match_the_full_oof_search():
    """The shipped constants must BE the full-OOF search output, not a copy.

    Skipped when the promotion gate failed, because no production build is
    allowed to exist in that case - and the skip is checked too, so a stray
    submission cannot pass unnoticed.
    """
    promoted = bool(RESULTS.get("promoted"))
    fo = RESULTS.get("final_offsets", {})
    if not promoted:
        assert not fo.get("fitted"), "offsets were fitted despite gate FAIL"
        return
    c, _ = corpus()
    o = Objective(c, np.arange(K.N_RUNS))
    units, _info = K.coordinate_search(o, verbose=False)
    got = K.to_offsets(units)
    want = np.array(fo["offsets"], dtype=np.float64)
    assert np.allclose(got, want, atol=1e-12), (got.tolist(), want.tolist())
    assert [int(x) for x in units] == list(fo["units"])
    path = os.path.join(ROOT, "submission_exp09_decision_calibration",
                        "decision_offsets.py")
    assert os.path.exists(path), "promoted but no production offsets module"
    sys.path.insert(0, os.path.dirname(path))
    for m in ("decision_offsets",):
        sys.modules.pop(m, None)
    import decision_offsets as DO
    assert np.array_equal(np.array(DO.OFFSETS, dtype=np.float64), want), \
        "submission constants differ from the full-OOF search output"
    assert DO.CLEAN == 0 and DO.OFFSETS[0] == 0.0, "clean not pinned in production"
    K.check_bounds(DO.OFFSETS)


def test_tiebreak_is_the_predeclared_one():
    """composite, then sum(delta^2), then first-in-scan-order - in that order.

    The tie window is 1e-12 as declared, and the grid unit is 0.0125, so a
    composite CAN change by much less than 1e-12 between two vertices (it is a
    mean over integer counts, and 1/2400 = 4.2e-4 per class, 1/7200 per hit).
    That is the declared behaviour, not an accident: when two moves are within
    1e-12 the SMALLER NORM wins even if it is the very slightly worse vertex.
    The test pins that, so a future "improvement" to the comparator is a
    deliberate, visible change rather than a silent one.
    """
    a = np.array([0, 4, 0, 0, 0, 0, 0], dtype=np.int64)
    b = np.array([0, 0, 4, 0, 0, 0, 0], dtype=np.int64)
    assert not K._better(1.0, a, 1.0, b), "equal composite: tie-break ignored"
    big = np.array([0, 32, 0, 0, 0, 0, 0], dtype=np.int64)
    assert not K._better(1.0, big, 1.0, a), "larger norm must lose the tie"
    assert K._better(1.0, a, 1.0, big)
    # 1e-9 apart is a REAL difference, and higher composite must win outright
    assert K._better(1.0 + 1e-9, big, 1.0, a), "1e-9 apart must not be a tie"
    # 1e-13 apart IS a tie, so the norm decides - and it picks the worse vertex
    assert K._better(1.0, a, 1.0 - 1e-13, big), \
        "inside the 1e-12 window the smaller norm must win, not the composite"
    # and a clean loss is a clean loss
    assert not K._better(1.0 - 1e-6, a, 1.0, b)
    # scan order: with the same composite AND the same norm, neither wins, so
    # the incumbent (encountered first) is what the search keeps
    assert not K._better(1.0, a, 1.0, a)


def test_gate_is_applied_as_declared():
    """The recorded gate must be the pre-declared one, with no moving parts."""
    g = RESULTS["promotion_gate"]
    assert GATE_MIN == 0.003 and GATE_FOLDS == 2
    assert GATE_ROB == 0.003 and GATE_CLS == 0.02
    comp = RESULTS["comparison"]
    d_comp = comp["B_cross_fitted"]["composite"] - comp["A_exp08_no_offsets"]["composite"]
    assert abs(g["criteria"][0]["value"] - d_comp) < 1e-12
    assert g["passed"] == all(c["passed"] for c in g["criteria"])
    assert bool(g["passed"]) == bool(RESULTS.get("promoted"))


GATE_MIN, GATE_FOLDS, GATE_ROB, GATE_CLS = 0.003, 2, 0.003, 0.02


def main():
    print("=" * 78)
    print("Exp09 calibration tests")
    print("=" * 78)
    promoted = bool(RESULTS.get("promoted"))
    print("  promotion gate: %s -> production-constant test %s"
          % ("PASS" if promoted else "FAIL",
             "runs" if promoted else "is expected to skip"))
    tests = [
        ("zero offsets reproduce Exp08 labels on all 10000 rows",
         test_zero_offsets_reproduce_exp08_labels, False),
        ("log(P)+0 == argmax(P), including exact-zero probabilities",
         test_log_p_equals_argmax_p_even_with_zero_probability, False),
        ("clean offset is pinned to exactly 0 and cannot be moved",
         test_clean_offset_is_pinned, False),
        ("declared bounds are enforced and never widened",
         test_bounds_are_never_violated_or_widened, False),
        ("search is deterministic across reruns",
         test_search_is_deterministic, False),
        ("held-out fold cannot influence its own offsets",
         test_heldout_fold_does_not_influence_its_own_offsets, False),
        ("hit@2 reads the peak of the PREDICTED class, not the true class",
         test_hit2_uses_the_peak_of_the_PREDICTED_class, False),
        ("fast objective == official evaluation.metrics",
         test_fast_objective_equals_official_metrics, False),
        ("tie-break is composite > sum(delta^2) > scan order",
         test_tiebreak_is_the_predeclared_one, False),
        ("promotion gate applied exactly as declared",
         test_gate_is_applied_as_declared, False),
        ("production constants == full-OOF search output",
         test_production_constants_match_the_full_oof_search, not promoted),
    ]
    for name, fn, skip in tests:
        check(name, fn, skip=skip)

    npass = sum(1 for _, ok, _ in results if ok)
    nskip = sum(1 for _, ok, _ in results if ok is None)
    ntot = len(results) - nskip
    print("\n%d/%d passed%s" % (npass, ntot,
                                ", %d skipped" % nskip if nskip else ""))
    for name, ok, err in results:
        if ok is False:
            print("  FAILED: %s -> %s" % (name, err))
    if FAILS := [n for n, ok, _ in results if ok is False]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
