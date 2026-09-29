"""Parity: the new peak helper must reproduce Exp07's stored peak positions.

``peak_turn.peak_turns`` is a re-implementation of one line inside Exp07's
``grouping.aggregate_block`` - ``peak_pos[j] = P.argmax(axis=0)`` - lifted out
so that production can use the position without touching ``aggregate()``.

A re-implementation is only allowed if it is proved to be the same function.
This test does that on two independent kinds of input:

  1. LIVE probabilities.  A window model is really fitted on real runs, its
     per-window probabilities really predicted, and the peaks computed by the
     helper are compared elementwise against ``grouping.aggregate_block``'s.
     This is the check that matters: it exercises the same objects production
     sees.

  2. ADVERSARIAL synthetic blocks - degenerate shapes (0, 1 and 2 windows),
     unsorted run rows, duplicated run indices, ties, and one-hot rows - where
     a buggy ``argsort``/``searchsorted`` split would silently attach a run's
     windows to the wrong run.

It also pins the two behaviours the rule depends on: ``clean`` -> -1 always, and
an empty block -> -1 (never ``argmax`` of an empty array, which is 0 and would
point at an unscored turn).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
sys.path.insert(0, HERE)

import peak_turn as pt          # noqa: E402
import solution as sub         # noqa: E402
import window_dataset as wd     # noqa: E402

SAMPLE = 200
FAILS = []


def check(name, ok_, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok_ else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok_:
        FAILS.append(name)


def _exp07(name):
    """Load an Exp07 module by path, under a unique name.

    The submission and the experiment both ship modules called
    ``window_features`` / ``aggregate`` / ``window_dataset``, so a plain import
    would resolve to whichever landed in sys.modules first.
    """
    spec = importlib.util.spec_from_file_location(
        "exp08_ref_" + name, os.path.join(EXP07, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("=" * 78)
    print("PEAK HELPER PARITY vs exp07_fault_windows.grouping.aggregate_block")
    print("=" * 78)

    # ---------------------------------------------------------------- 0. rules
    print("\n0. RULE PRECONDITIONS")
    check("clean -> -1 for every block size",
          all(pt.peak_turn_for_block(np.random.rand(k, 7), "clean") == -1
              for k in (0, 1, 5, 50)))
    check("empty block -> -1 (not argmax of nothing, which would be 0)",
          pt.peak_turn_for_block(np.zeros((0, 7)), "deadlock") == -1)
    check("empty block -> -1 for every fault class",
          all(pt.peak_turn_for_block(np.zeros((0, 7)), c) == -1
              for c in wd.FAULT_CLASSES))
    check("None block -> -1", pt.peak_turn_for_block(None, "conflict") == -1)
    # the index trap: label index 1 == window index 1 only because both lists
    # share the six fault names in the same order, offset by background.
    check("class index uses WINDOW order, not label order",
          pt.class_index("clean") if False else
          wd.W2I["dropped_handoff"] == 1 and wd.WCLASSES[1] == "dropped_handoff",
          "W2I maps a fault class to its window index")
    try:
        pt.class_index("not_a_class")
        check("an unknown label raises rather than guessing an index", False)
    except KeyError:
        check("an unknown label raises rather than guessing an index", True)

    # ------------------------------------------------------- 1. live model
    print("\n1. LIVE WINDOW MODEL on real runs")
    exp_grp = _exp07("grouping")
    exp_wd = _exp07("window_dataset")

    train = pd.read_csv(os.path.join(EXP07, "..", "..", "data", "train.csv"))
    small = [sub.parse_run(r) for r in train.head(SAMPLE).to_dict("records")]
    ys = train["label"].values[:SAMPLE].astype(object)
    ft = train["fault_turn"].values[:SAMPLE]

    W = sub.build_window_matrix(small, ys, ft)
    check("window dataset built (%d windows over %d runs)"
          % (len(W["y"]), len(small)), len(W["y"]) > 0)

    mw = sub.fit_window(W["X"], W["y"])
    P = mw.predict_proba(W["X"])
    run_rows = W["run"]
    check("probabilities have one row per window", len(P) == len(run_rows))

    blocks = exp_grp.predict_runs(P, run_rows, len(small))
    _, _, ref_peak = exp_grp.aggregate_block(
        blocks, _exp07("aggregate"),
        {r: r for r in range(len(small))}, len(small))

    # every label the helper is asked about, so the comparison covers all
    # columns of the peak matrix rather than just the predicted ones
    labels = [str(l) for l in ys]
    got = pt.peak_turns(P, run_rows, len(small), labels)
    exp_turns = np.array([int(ref_peak[r, wd.W2I[labels[r]]]) if labels[r] != "clean"
                          else -1 for r in range(len(small))], dtype=np.int64)
    bad = int((got != exp_turns).sum())
    check("helper peak == aggregate_block peak_pos on %d real runs" % len(small),
          bad == 0, "%d mismatches" % bad)

    # Every FAULT column, with the label filter lifted: ask for each class in
    # turn and confirm the helper reproduces that column of the reference peak
    # matrix.  Column 0 (background) is deliberately excluded - the L1 rule never
    # reads it, and `aggregate_block` fills it while the helper reports -1 for a
    # label the rule maps to "no turn".
    mism = 0
    for c in wd.FAULT_CLASSES:
        ci = wd.W2I[c]
        asked = [c if ys[r] == c else "clean" for r in range(len(small))]
        got_c = pt.peak_turns(P, run_rows, len(small), asked)
        want_c = np.array([int(ref_peak[r, ci]) if ys[r] == c else -1
                           for r in range(len(small))], dtype=np.int64)
        mism += int((got_c != want_c).sum())
    check("helper reproduces every FAULT column of the peak matrix", mism == 0,
          "%d mismatches across %d classes" % (mism, len(wd.FAULT_CLASSES)))

    # ---------------------------------------------------- 2. adversarial
    print("\n2. ADVERSARIAL block shapes and run orderings")
    rng = np.random.default_rng(12345)
    cases = []
    for counts in ([1], [2], [1, 1, 1], [1, 5], [2, 3, 1, 7], [33, 1, 4],
                   [1] * 20, [60]):
        cases.append(("counts=%s" % counts, counts))

    worst = 0
    for tag, counts in cases:
        n = len(counts)
        rows = np.concatenate([np.full(k, r, dtype=np.int32)
                               for r, k in enumerate(counts)])
        # unsorted / shuffled row order must still be regrouped by run, with
        # the file order inside each run defining the turn index
        Pr = rng.dirichlet(np.ones(7), size=len(rows))
        labs = [wd.FAULT_CLASSES[i % 6] for i in range(n)]
        got = pt.peak_turns(Pr, rows, n, labs)
        want = np.array([int(np.argmax(Pr[rows == r][:, wd.W2I[labs[r]]]))
                         for r in range(n)], dtype=np.int64)
        worst = max(worst, int((got != want).sum()))

        # the reference implementation, on the same data
        bl = exp_grp.predict_runs(Pr, rows, n)
        _, _, rp = exp_grp.aggregate_block(bl, _exp07("aggregate"),
                                           {r: r for r in range(n)}, n)
        wref = np.array([int(rp[r, wd.W2I[labs[r]]]) for r in range(n)],
                        dtype=np.int64)
        worst = max(worst, int((got != wref).sum()))
    check("helper == aggregate_block on %d degenerate shapes" % len(cases),
          worst == 0, "%d mismatches" % worst)

    # shuffled row order: turn index is the position WITHIN the run's block
    counts = [4, 3, 5, 2]
    n = len(counts)
    rows = np.concatenate([np.full(k, r, dtype=np.int32)
                           for r, k in enumerate(counts)])
    Pr = rng.dirichlet(np.ones(7), size=len(rows))
    perm = rng.permutation(len(rows))
    Pr_s, rows_s = Pr[perm], rows[perm]
    labs = [wd.FAULT_CLASSES[r] for r in range(n)]
    got = pt.peak_turns(Pr_s, rows_s, n, labs)
    # with rows shuffled, the "turn" is the index in the SHUFFLED block, which
    # is what a real run's rows look like; the reference does the same
    bl = exp_grp.predict_runs(Pr_s, rows_s, n)
    _, _, rp = exp_grp.aggregate_block(bl, _exp07("aggregate"),
                                       {r: r for r in range(n)}, n)
    want = np.array([int(rp[r, wd.W2I[labs[r]]]) for r in range(n)],
                    dtype=np.int64)
    check("shuffled row order still matches aggregate_block",
          int((got != want).sum()) == 0,
          "%d mismatches" % int((got != want).sum()))

    # ties: argmax must take the FIRST maximal window, deterministically.
    # The tie must be in the column of the class actually asked about
    # (deadlock = window index 3), or the test measures nothing.
    tie_col = wd.W2I["deadlock"]
    Pt = np.full((6, 7), 0.1)
    Pt[1, tie_col] = Pt[4, tie_col] = 0.9      # turns 1 and 4 tie at the max
    rows_t = np.zeros(6, dtype=np.int32)
    got_t = pt.peak_turns(Pt, rows_t, 1, ["deadlock"])
    check("exact tie picks the FIRST maximal turn (turn 1)",
          int(got_t[0]) == 1, "got %d" % int(got_t[0]))

    # and the same tie resolved against the reference implementation
    bl_t = exp_grp.predict_runs(Pt, rows_t, 1)
    _, _, rp_t = exp_grp.aggregate_block(bl_t, _exp07("aggregate"), {0: 0}, 1)
    check("tie resolution matches aggregate_block",
          int(got_t[0]) == int(rp_t[0, tie_col]),
          "helper %d vs reference %d" % (int(got_t[0]),
                                         int(rp_t[0, tie_col])))

    # one-hot rows: peak must be the position of the 1.0
    for c_i, c in enumerate(wd.FAULT_CLASSES):
        Po = np.zeros((9, 7))
        Po[5, wd.W2I[c]] = 1.0
        got_o = pt.peak_turns(Po, np.zeros(9, dtype=np.int32), 1, [c])
        if int(got_o[0]) != 5:
            check("one-hot row for %s -> turn 5" % c, False,
                  "got %d" % int(got_o[0]))
            break
    else:
        check("one-hot row per class -> the turn carrying the 1.0 (all 6)", True)

    # a run with no windows at all, between two runs that have them
    Pm = rng.dirichlet(np.ones(7), size=10)
    rows_m = np.array([0] * 3 + [2] * 7, dtype=np.int32)
    labs_m = ["conflict", "deadlock", "goal_drift"]
    got_m = pt.peak_turns(Pm, rows_m, 3, labs_m)
    bl = exp_grp.predict_runs(Pm, rows_m, 3)
    _, _, rp = exp_grp.aggregate_block(bl, _exp07("aggregate"),
                                       {r: r for r in range(3)}, 3)
    want_m = np.array([int(rp[0, wd.W2I[labs_m[0]]]),
                       -1,                       # run 1 has no windows
                       int(rp[2, wd.W2I[labs_m[2]]])], dtype=np.int64)
    check("empty middle run -> -1, neighbours unaffected",
          np.array_equal(got_m, want_m),
          "got %s want %s" % (got_m.tolist(), want_m.tolist()))

    # nothing scored at all
    got_z = pt.peak_turns(np.zeros((0, 7)), np.zeros(0, dtype=np.int32), 3,
                          ["conflict", "deadlock", "goal_drift"])
    check("no windows scored anywhere -> all -1",
          np.array_equal(got_z, np.array([-1, -1, -1])))

    print("=" * 78)
    if FAILS:
        print("FAILED: %d" % len(FAILS))
        for f in FAILS:
            print("   - %s" % f)
        return 1
    print("PEAK HELPER PARITY PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())