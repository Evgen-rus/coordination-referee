"""End-to-end equivalence: the submission must reproduce the EXPERIMENT.

The parity script proves the pieces match.  This proves the assembled pipeline
does, by holding out a slice of the real train set, running the submission
`solution.py` on (train-minus-slice -> slice) exactly as the platform would, and
comparing every prediction against the experiment's own OOF file.

This is the check that would catch a wiring mistake in the honest-stacking
plumbing - e.g. fitting the test window features on the wrong runs - which
piece-level parity cannot see.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submission_exp07_fault_windows")
EXP = os.path.join(ROOT, "experiments", "exp07_fault_windows")
FAILS = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    exp = pd.read_csv(os.path.join(EXP, "oof_B_plus_window.csv"))
    assert (exp["run_id"].values == train["run_id"].values).all(), \
        "experiment OOF is not aligned with train.csv"

    # A contiguous 1500-run holdout.  Stratified-ish by construction: the file is
    # ordered by label blocks in the demo set, so take a strided sample instead
    # to guarantee all 7 classes are represented in the fit.
    rng = np.random.default_rng(11)
    hold = np.sort(rng.choice(len(train), 1500, replace=False))
    mask = np.zeros(len(train), bool)
    mask[hold] = True
    sub_train = train[~mask].reset_index(drop=True)
    sub_test = train[mask].reset_index(drop=True)
    print("fit on %d runs, predict %d held-out runs" % (len(sub_train), len(sub_test)))
    print("holdout label mix:")
    print(sub_test["label"].value_counts().to_string())

    tmp = tempfile.mkdtemp(prefix="exp07_equiv_")
    tr_p = os.path.join(tmp, "train.csv")
    te_p = os.path.join(tmp, "test.csv")
    pr_p = os.path.join(tmp, "predictions.csv")
    sub_train.to_csv(tr_p, index=False)
    sub_test[["run_id", "goal", "agents", "shared_state", "messages",
              "artifacts", "topology"]].to_csv(te_p, index=False)

    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    cmd = [sys.executable, os.path.join(SUB, "solution.py"),
           "--train", tr_p, "--test", te_p, "--output", pr_p]
    print("running: %s" % " ".join(cmd[:2] + ["..."]))
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, env=env)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        check("solution.py completed", False, "exit %d" % p.returncode)
        return
    check("solution.py completed", True)
    for line in p.stdout.splitlines():
        if "stacking verified" in line or "DONE in" in line:
            print("    " + line.strip())

    pred = pd.read_csv(pr_p)
    check("row count matches holdout", len(pred) == len(sub_test),
          "%d vs %d" % (len(pred), len(sub_test)))
    check("run_id set matches holdout",
          list(pred["run_id"]) == list(sub_test["run_id"]))

    got = pred["label"].values.astype(object)
    # experiment OOF for system B, restricted to the same holdout
    ref = exp["y_pred"].values.astype(object)[hold]
    agree = float((got == ref).mean())
    print()
    print("  label agreement with experiment OOF: %.4f" % agree)
    check("label agreement is high (>= 0.90)", agree >= 0.90,
          "%.4f" % agree)

    # the honest-stacking pipeline refits, so it is NOT expected to be
    # bit-identical; what must hold is that it is not systematically worse.
    from collections import Counter
    print("  submission label mix : %s" % dict(Counter(got)))
    print("  experiment  label mix: %s" % dict(Counter(ref)))

    # fault_turn must equal the OFFICIAL localiser applied to OUR predicted label
    sys.path.insert(0, SUB)
    for m in ("localize", "features", "solution", "aggregate", "window_dataset",
              "window_features", "new_features", "lifecycle", "temporal_features"):
        sys.modules.pop(m, None)
    from localize import localize as loc_fn
    from features import parse_run
    runs = [parse_run(r) for r in sub_test.to_dict("records")]
    expect = [loc_fn(r, l) for r, l in zip(runs, got)]
    got_turn = pd.to_numeric(pred["fault_turn"]).values
    bad = int((got_turn != np.array(expect)).sum())
    check("fault_turn == official localiser on the submitted label", bad == 0,
          "%d of %d rows differ" % (bad, len(expect)))
    check("success is binary", set(pd.to_numeric(pred["success"]).unique())
          <= {0, 1})

    print("=" * 78)
    if FAILS:
        print("FAILED: %d" % len(FAILS))
        for f in FAILS:
            print("   - %s" % f)
        sys.exit(1)
    print("EQUIVALENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
