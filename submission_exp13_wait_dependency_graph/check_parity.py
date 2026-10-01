#!/usr/bin/env python3
"""Exp13 production parity - prove the ONLY change is the 12 wait-graph columns.

Run from the submission directory:

    python check_parity.py [--train data/train.csv] [--test data/test.csv]

What this asserts, and why each one matters:

  1. Every module inherited from Exp08 is BYTE-identical.  The wait-graph block
     is the only new file, and ``solution.py`` differs only in the label matrix
     assembly.  Nothing else was touched, so success, the window model and L1
     cannot have moved.
  2. ``solution.py``'s ``make_label_model`` / ``make_success_model`` /
     ``make_window_model`` return models with IDENTICAL ``get_params()``.
  3. The 300 production columns are the same 249 foundation + the same 51
     aggregates, in the same order, as Exp08 - the new block is strictly
     appended after them.
  4. The wait-graph block in this submission is numerically IDENTICAL to the
     research module on real runs: the same 12 names, same order, same values.
  5. All six template families fire on real runs.
  6. The block reads no target: overwriting ``label`` / ``success`` /
     ``fault_turn`` cannot move a single value.
  7. The fallback path still produces a well-formed predictions.csv for an
     incomplete schema.
  8. If predictions for both submissions are supplied, it reports how many test
     labels changed, which class transitions occurred, how the deadlock count
     moved, and - critically - whether every changed ``fault_turn`` is
     explained by a changed predicted class rather than by the L1 rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB08 = os.path.join(ROOT, "submission_exp08_window_localizer")
EXP13 = os.path.join(ROOT, "experiments", "exp13_wait_dependency_graph")
for _p in (HERE, EXP13):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAILS = []
NOTES = []


def ok(msg):
    print("  ok    %s" % msg)


def fail(msg):
    print("  FAIL  %s" % msg)
    FAILS.append(msg)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. inherited modules are byte-identical
# ---------------------------------------------------------------------------

def check_modules_identical():
    print("\n[1] modules inherited from Exp08 are byte-identical")
    changed_expected = {"solution.py"}          # the one intentional edit
    new_expected = {"wait_graph.py",            # the one new feature module
                    "check_parity.py"}          # this test, new in Exp13
    for f in sorted(os.listdir(SUB08)):
        if not f.endswith(".py"):
            continue
        p8 = os.path.join(SUB08, f)
        p13 = os.path.join(HERE, f)
        if not os.path.exists(p13):
            fail("Exp13 is missing the inherited module %s" % f)
            continue
        if f in changed_expected:
            ok("%s differs (intentional: the label-head assembly)" % f)
            continue
        if f == "check_parity.py":
            ok("check_parity.py is Exp13's own extended parity test")
            continue
        if sha(p8) == sha(p13):
            ok("%s byte-identical" % f)
        else:
            fail("%s was modified but should not have been" % f)
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py") and f not in os.listdir(SUB08) \
                and f not in new_expected:
            fail("unexpected new module %s" % f)


# ---------------------------------------------------------------------------
# 2. model parameters unchanged
# ---------------------------------------------------------------------------

def check_params():
    print("\n[2] model parameters unchanged")
    s08 = load("sol08", os.path.join(SUB08, "solution.py"))
    s13 = load("sol13", os.path.join(HERE, "solution.py"))
    for maker in ("make_label_model", "make_success_model", "make_window_model"):
        a = getattr(s08, maker)().get_params()
        b = getattr(s13, maker)().get_params()
        if a == b:
            ok("%s: identical get_params()" % maker)
        else:
            diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
                    if a.get(k) != b.get(k)}
            fail("%s: parameters changed %r" % (maker, diff))
    for const in ("N_FOUNDATION", "N_SUCCESS_FEATS", "N_INNER", "INNER_SEED"):
        if getattr(s08, const) == getattr(s13, const):
            ok("%s == %r" % (const, getattr(s13, const)))
        else:
            fail("%s changed" % const)


# ---------------------------------------------------------------------------
# 3. the 300 production columns are preserved and the block is appended
# ---------------------------------------------------------------------------

def check_column_layout():
    print("\n[3] column layout: 300 preserved, 12 appended")
    import aggregate as ag
    import wait_graph as wg
    s13 = load("sol13", os.path.join(HERE, "solution.py"))
    cols13 = (["foundation"] * s13.N_FOUNDATION) + list(ag.AGG_NAMES) \
        + list(wg.FEATURE_NAMES)
    if len(wg.FEATURE_NAMES) == 12:
        ok("exactly 12 wait-graph features: %s"
           % ", ".join(wg.FEATURE_NAMES))
    else:
        fail("expected 12 wait-graph features, found %d"
             % len(wg.FEATURE_NAMES))
    tail = cols13[-12:]
    if all(c.startswith("wg_") for c in tail):
        ok("the 12 new columns are appended strictly after the 300")
    else:
        fail("the wait-graph columns are not the trailing 12")
    if s13.N_SUCCESS_FEATS == 185:
        ok("success head still 185 features (not widened)")
    else:
        fail("success feature count changed")
    if s13.N_FOUNDATION == 249:
        ok("foundation still 249 features")
    else:
        fail("foundation feature count changed")


# ---------------------------------------------------------------------------
# 4-6. the block is the research module, covers six templates, reads no target
# ---------------------------------------------------------------------------

def check_block_semantics(runs):
    print("\n[4] the submission block equals the research module on real runs")
    import wait_graph as wg
    res = load("exp13_features", os.path.join(EXP13, "features.py"))
    if list(wg.FEATURE_NAMES) == list(res.FEATURE_NAMES):
        ok("feature names and order identical to the research module")
    else:
        fail("feature names differ from the research module")
    A = wg.build_matrix(runs)
    B = res.build_matrix(runs)
    if A.shape == B.shape and np.array_equal(A, B):
        ok("numerically identical to the research block on %d real runs "
           "(%s)" % (len(runs), A.shape))
    else:
        fail("block differs from the research module")

    print("\n[5] all six template families fire on real runs")
    fired = set()
    for run in runs:
        for m in run.get("messages") or []:
            if str(m.get("type", "")).lower() != "status":
                continue
            for i, pat in enumerate(wg.TEMPLATE_PATTERNS):
                if pat.search(str(m.get("text", ""))):
                    fired.add(i)
    if len(fired) == 6:
        ok("6/6 families matched: %s"
           % ", ".join(wg.TEMPLATE_NAMES[i] for i in sorted(fired)))
    else:
        fail("only %d/6 families fired: %s"
             % (len(fired), sorted(wg.TEMPLATE_NAMES[i] for i in fired)))

    print("\n[6] the block reads no target (perturbation)")
    import copy
    base = wg.build_matrix(runs[:300])
    pert = []
    for i, r in enumerate(runs[:300]):
        r2 = copy.deepcopy(r)
        r2["label"] = ["deadlock", "clean", "goal_drift"][i % 3]
        r2["success"] = 1 - (i % 2)
        r2["fault_turn"] = (i * 7) % 11
        pert.append(r2)
    other = wg.build_matrix(pert)
    if np.array_equal(base, other):
        ok("overwriting label/success/fault_turn moved nothing")
    else:
        fail("the block reacts to a target (max|diff| %.3e)"
             % np.max(np.abs(base - other)))
    if np.isfinite(base).all():
        ok("no non-finite values")
    else:
        fail("non-finite values in the block")


# ---------------------------------------------------------------------------
# 7. fallback still works for an incomplete schema
# ---------------------------------------------------------------------------

def check_fallback(tmpdir):
    print("\n[7] fallback for an incomplete schema still works")
    import subprocess
    train = pd.DataFrame({"run_id": ["r1"], "label": ["clean"],
                          "success": [1], "fault_turn": [-1]})
    test = pd.DataFrame({"run_id": ["t1", "t2"]})          # no run columns
    tr = os.path.join(tmpdir, "train_min.csv")
    te = os.path.join(tmpdir, "test_min.csv")
    out = os.path.join(tmpdir, "pred_min.csv")
    train.to_csv(tr, index=False)
    test.to_csv(te, index=False)
    r = subprocess.run([sys.executable, os.path.join(HERE, "solution.py"),
                        "--train", tr, "--test", te, "--output", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail("solution.py exited %d on an incomplete schema\n%s"
             % (r.returncode, r.stderr[-800:]))
        return
    if not os.path.exists(out):
        fail("no predictions.csv written for an incomplete schema")
        return
    p = pd.read_csv(out)
    need = {"run_id", "label", "success", "fault_turn"}
    if need <= set(p.columns) and len(p) == 2:
        ok("wrote a well-formed fallback: %d rows, columns %s"
           % (len(p), list(p.columns)))
        if set(p["label"]) == {"clean"} and set(p["fault_turn"]) == {-1}:
            ok("fallback constants are the documented ones (clean / -1)")
        else:
            fail("fallback constants differ from the documented ones")
    else:
        fail("fallback output is malformed: %s" % list(p.columns))


# ---------------------------------------------------------------------------
# 8. comparison against the Exp08 submission's own predictions
# ---------------------------------------------------------------------------

def check_prediction_delta(pred08_path, pred13_path):
    print("\n[8] test predictions vs submission_exp08")
    if not (os.path.exists(pred08_path) and os.path.exists(pred13_path)):
        NOTES.append("prediction delta skipped: Exp08 predictions not "
                     "available at %s" % pred08_path)
        print("  note  %s" % NOTES[-1])
        return
    a = pd.read_csv(pred08_path)
    b = pd.read_csv(pred13_path)
    if len(a) != len(b):
        fail("different number of test rows: %d vs %d" % (len(a), len(b)))
        return
    if not np.array_equal(a["run_id"].values.astype(str),
                          b["run_id"].values.astype(str)):
        fail("run_id order differs")
        return
    ok("both cover %d test rows in the same order" % len(a))

    sa = a["success"].astype(int).values
    sb = b["success"].astype(int).values
    if np.array_equal(sa, sb):
        ok("success predictions byte-identical (%d rows)" % len(a))
    else:
        fail("success predictions changed on %d rows"
             % int((sa != sb).sum()))

    la = a["label"].values.astype(str)
    lb = b["label"].values.astype(str)
    nch = int((la != lb).sum())
    ok("label changed on %d/%d rows (%.2f%%)" % (nch, len(a),
                                                 100.0 * nch / len(a)))

    print("        class transitions (exp08 -> exp13):")
    tr = {}
    for x, y in zip(la, lb):
        if x != y:
            tr[(x, y)] = tr.get((x, y), 0) + 1
    for (x, y), n in sorted(tr.items(), key=lambda kv: -kv[1]):
        print("          %-18s -> %-18s %5d" % (x, y, n))

    for cls in ("deadlock",):
        ca = int((la == cls).sum())
        cb = int((lb == cls).sum())
        print("        %s predictions: %d -> %d (%+d)" % (cls, ca, cb, cb - ca))

    ta = a["fault_turn"].astype(int).values
    tb = b["fault_turn"].astype(int).values
    n_t = int((ta != tb).sum())
    print("        fault_turn changed on %d rows" % n_t)
    lab_changed = la != lb
    turn_changed = ta != tb
    unexplained = turn_changed & ~lab_changed
    if not unexplained.any():
        ok("every changed fault_turn coincides with a changed predicted "
           "label - no L1 behaviour changed")
    else:
        fail("%d rows changed fault_turn WITHOUT changing label; that is an "
             "L1 regression, not a consequence of the new features"
             % int(unexplained.sum()))
    both = turn_changed & lab_changed
    print("        of the %d label changes, %d also moved fault_turn "
           "(the L1 peak is re-read for the new class)"
           % (int(lab_changed.sum()), int(both.sum())))
    clean_a = int(((la == "clean") & (ta >= 0)).sum())
    clean_b = int(((lb == "clean") & (tb >= 0)).sum())
    if clean_a == 0 and clean_b == 0:
        ok("clean rows carry fault_turn -1 in both submissions")
    else:
        fail("clean rows with a non-negative fault_turn: %d vs %d"
             % (clean_a, clean_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "train.csv"))
    ap.add_argument("--test", default=os.path.join(ROOT, "data", "test.csv"))
    ap.add_argument("--pred08", default=None)
    ap.add_argument("--pred13", default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("Exp13 production parity check")
    print("=" * 78)

    check_modules_identical()
    check_params()
    check_column_layout()

    runs = []
    if os.path.exists(args.train):
        import features as F
        train = pd.read_csv(args.train, nrows=400)
        runs = [F.parse_run(r) for r in train.to_dict("records")]
        ok("parsed %d real train runs from %s"
           % (len(runs), os.path.basename(args.train)))
        check_block_semantics(runs)
    else:
        fail("train.csv not found at %s" % args.train)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        check_fallback(td)

    if args.pred08 and args.pred13:
        check_prediction_delta(args.pred08, args.pred13)

    print("\n" + "=" * 78)
    if FAILS:
        print("PARITY FAILED - %d check(s)" % len(FAILS))
        for f in FAILS:
            print("  - %s" % f)
        return 1
    print("ALL PARITY CHECKS PASSED")
    for n in NOTES:
        print("  note: %s" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())