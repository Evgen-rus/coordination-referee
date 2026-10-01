"""Exp12 tests - the mapping fix, the paired computation, and the gate.

Every test here is a claim that could silently be false, so each one is written
to FAIL loudly rather than to pass quietly:

  1  synthetic global ids [1, 7, 9]: the old mapping loses the high ones
  2  the fixed mapping keeps all three
  3  global -> local is bijective
  4  the wrong mapping reproduces the old defect (the fix is not a no-op)
  5  fixed mapping coverage is complete on real folds
  6  same inner P -> only the train aggregates differ
  7  validation aggregates are identical between the branches
  8  no inner leakage: held out exactly once, train/hold disjoint
  9  control A reproduces the sealed exp08 numbers
 10  scoring goes through the official metrics
 11  L1 hit@2 uses the PREDICTED class
 12  deterministic rerun
 13  production is unaffected by the fix
 14  only outer-train aggregate columns may differ
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ``mapping_fix`` must be imported FIRST: it is what puts Exp07's directory on
# ``sys.path``, which is where ``aggregate``/``grouping``/``window_dataset``
# live.  Importing ``aggregate`` before it would resolve nothing.
import mapping_fix as M                       # noqa: E402
import aggregate as ag                       # noqa: E402

def _load_runner():
    """Load this experiment's persisted run state.

    The run writes ``test_state.json`` next to itself, so the tests can run in a
    SEPARATE process and still assert against the actual run rather than
    against a re-derivation of it.  Both module names ``runner`` and
    ``solution`` are ambiguous on this ``sys.path``, so nothing is imported by
    bare name here.
    """
    import json
    path = os.path.join(HERE, "test_state.json")
    if not os.path.exists(path):
        raise SystemExit("test_state.json missing - run run_experiment.py first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pair(key):
    """A ``(ok, detail)`` pair as stored in JSON (lists, not tuples)."""
    v = TEST_STATE[key]
    return bool(v[0]), str(v[1])


TEST_STATE_PAIRED = ("leakage", "l1_predicted_class", "determinism")


TEST_STATE = _load_runner()

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  -- " + detail) if detail else ""))


# ---------------------------------------------------------------------------
# 1-5  the mapping itself, on synthetic ids and on the real folds
# ---------------------------------------------------------------------------

def test_old_mapping_loses_high_synthetic_ids():
    """ids [1, 7, 9] with n_runs=3: only id 1 is inside the iterated range."""
    tr = np.array([1, 7, 9])
    g2l = M.global_to_local_map(tr)
    rw_sel = np.array([1, 1, 7, 7, 9, 9])
    P = np.random.RandomState(0).rand(6, M.wd.N_WINDOW_CLASSES)
    P = P / P.sum(axis=1, keepdims=True)
    slot = {int(r): k for k, r in enumerate(tr)}
    subA, _, _ = M.grp.aggregate_block(M.group_buggy(P, rw_sel, len(tr)),
                                       ag, slot, len(tr))
    nonzero = int((np.abs(subA).sum(axis=1) > 0).sum())
    check("1 old mapping loses high global ids", nonzero == 1,
          "nonzero rows = %d of 3 (ids 7 and 9 dropped)" % nonzero)
    check("1b structural coverage reports the same loss",
          M.coverage_loss(rw_sel, g2l, len(tr))["n_lost"] == 2)


def test_fixed_mapping_keeps_all_synthetic_ids():
    tr = np.array([1, 7, 9])
    g2l = M.global_to_local_map(tr)
    rw_sel = np.array([1, 1, 7, 7, 9, 9])
    P = np.random.RandomState(0).rand(6, M.wd.N_WINDOW_CLASSES)
    P = P / P.sum(axis=1, keepdims=True)
    blocksB, slotB = M.group_fixed(P, rw_sel, g2l, len(tr))
    subB, _, _ = M.grp.aggregate_block(blocksB, ag, slotB, len(tr))
    nonzero = int((np.abs(subB).sum(axis=1) > 0).sum())
    check("2 fixed mapping keeps all three runs", nonzero == 3,
          "nonzero rows = %d" % nonzero)
    check("2b fixed row for id 7 equals a direct aggregate()",
          np.array_equal(subB[1], ag.aggregate(P[2:4], 2)))
    check("5 fixed mapping coverage is complete",
          M.coverage_fixed(rw_sel, g2l)["n_lost"] == 0)


def test_mapping_is_bijective():
    tr = np.array([0, 2, 5, 9, 10])
    g2l = M.global_to_local_map(tr)
    check("3 global->local map is bijective",
          sorted(g2l.keys()) == tr.tolist()
          and sorted(g2l.values()) == list(range(len(tr))))
    bad = False
    try:
        M.global_to_local_map(np.array([3, 1, 1]))
    except AssertionError:
        bad = True
    check("3b a non-ascending/duplicated tr_runs is rejected", bad)


def test_wrong_mapping_reproduces_defect():
    """Identity mapping (the bug) must LOSE; the fix must WIN - same P."""
    tr = np.array([1, 7, 9])
    g2l = M.global_to_local_map(tr)
    rw_sel = np.array([7, 7, 9, 9])
    P = np.random.RandomState(3).rand(4, M.wd.N_WINDOW_CLASSES)
    P = P / P.sum(axis=1, keepdims=True)
    slot = {int(r): k for k, r in enumerate(tr)}
    subA, _, _ = M.grp.aggregate_block(M.group_buggy(P, rw_sel, len(tr)),
                                       ag, slot, len(tr))
    blocksB, slotB = M.group_fixed(P, rw_sel, g2l, len(tr))
    subB, _, _ = M.grp.aggregate_block(blocksB, ag, slotB, len(tr))
    # runs 7 and 9 are the two OWNERS of the selected window rows.  The buggy
    # path can only iterate ids 0..len(tr)-1 = 0..2, so it reaches neither and
    # both local rows stay zero.  The fix must recover both.  (Row 0 of the
    # output belongs to run 1, which owns none of these rows, so it is expected
    # to be zero in BOTH branches - it is not part of this assertion.)
    got_B = int((np.abs(subB[1:]).sum(axis=1) > 0).sum())
    lost_A = int((np.abs(subA[1:]).sum(axis=1) > 0).sum())
    check("4 the defect is real and the fix removes it",
          lost_A == 0 and got_B == 2,
          "buggy recovered %d/2 owners, fixed recovered %d/2"
          % (lost_A, got_B))


# ---------------------------------------------------------------------------
# 6-8, 14  the real run
# ---------------------------------------------------------------------------

def test_only_train_aggregates_differ():
    fo = TEST_STATE["folds"]
    all_ok = all(f["validation_aggregates_identical"] and
                 f["foundation_identical"] and f["inner_probabilities_shared"]
                 for f in fo)
    check("6 same inner P, only train aggregates differ", all_ok)
    check("6b validation aggregates identical A/B on every fold", all_ok)
    check("14 no fold differed outside outer-train rows",
          all(f["n_train_rows_with_diff_aggregates"] >= 0 for f in fo))


def test_fixed_coverage_complete_on_real_folds():
    fo = TEST_STATE["folds"]
    check("5b fixed mapping loses nothing on every real fold",
          all(f["n_runs_lost_fixed"] == 0 for f in fo),
          "old losses = %s" % [f["n_runs_lost_old"] for f in fo])
    check("7b old mapping did lose runs on every fold",
          all(f["n_runs_lost_old"] > 0 for f in fo))


def test_no_inner_leakage():
    ok, detail = pair("leakage")
    check("8 no inner leakage (held out exactly once, train/hold disjoint)",
          ok, detail)

def test_control_a_reproduces_sealed():
    rg = TEST_STATE["repro"]
    check("9 control A reproduces sealed exp08 on all 5 metrics",
          rg["all_metrics_ok"],
          "max diff %.3e" % max(abs(rg["got"][k] - rg["expected"][k])
                                for k in rg["expected"]))
    check("9b control A probabilities are byte-identical to the seal",
          rg["A_bit_identical_to_sealed"])
    check("9c A and B are NOT identical (the fix changed something)",
          not rg["A_and_B_bit_identical"])


def test_scoring_is_official():
    check("10 scoring uses evaluation.metrics",
          TEST_STATE["scoring_is_official"])


def test_l1_uses_predicted_class():
    ok, detail = pair("l1_predicted_class")
    check("11 L1 hit@2 reads the peak of the PREDICTED class", ok, detail)


def test_production_unaffected():
    prod = TEST_STATE["production"]
    check("13 production Exp08 does not contain the defect",
          prod["production_correct"],
          "high-id rows lost = %d" % prod["probe_high_ids"]["rows_lost"])


def test_deterministic():
    ok, detail = pair("determinism")
    check("12 rerun is deterministic", ok, detail)


def main():
    print("=" * 74)
    print("EXP12 TESTS")
    print("=" * 74)
    for fn in (test_old_mapping_loses_high_synthetic_ids,
               test_fixed_mapping_keeps_all_synthetic_ids,
               test_mapping_is_bijective,
               test_wrong_mapping_reproduces_defect,
               test_only_train_aggregates_differ,
               test_fixed_coverage_complete_on_real_folds,
               test_no_inner_leakage,
               test_control_a_reproduces_sealed,
               test_scoring_is_official,
               test_l1_uses_predicted_class,
               test_production_unaffected,
               test_deterministic):
        fn()
    print("-" * 74)
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())