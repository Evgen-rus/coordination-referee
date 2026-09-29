"""Demonstrate downstream use of the verified Exp07 OOF artifacts.

Read-only.  Runs no part of the Exp07 training pipeline: no feature build, no
window dataset, no LightGBM.  It only calls ``reuse.load(...)``, which is the
point being demonstrated - a downstream experiment can get the OOF matrix,
y_true, run_id and fold assignment in about a second.

Run:  python experiments/exp07_fault_windows/demo_downstream_use.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import modes as MD  # noqa: E402
import reuse  # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]


def main(system="B_plus_window"):
    t0 = time.perf_counter()

    # the only inputs a downstream consumer needs: the current train file
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                        usecols=["run_id", "label"])
    l2i = {l: i for i, l in enumerate(LABELS)}
    yi = np.array([l2i[l] for l in train["label"].values], dtype=int)
    run_ids = [str(x) for x in train["run_id"]]

    # verified load - raises unless every provenance check passes
    t1 = time.perf_counter()
    r = reuse.load(system, mode=MD.CV, yi=yi, train_run_ids=run_ids)
    load_s = time.perf_counter() - t1

    # ---- what the downstream experiment gets ------------------------------
    P_oof = r["proba"]                                   # (10000, 7) float32
    y_true = train["label"].values.astype(object)         # true labels
    run_id = np.array(r["run_ids"])                       # run order

    # fold assignment, re-derived from the manifest-verified seed
    from sklearn.model_selection import StratifiedKFold
    fold = np.zeros(len(yi), dtype=int)
    for k, (_, va) in enumerate(
            StratifiedKFold(reuse.N_OUTER, shuffle=True,
                            random_state=reuse.SEED)
            .split(np.zeros(len(yi)), yi)):
        fold[va] = k

    y_pred = np.array([LABELS[i] for i in P_oof.argmax(1)], dtype=object)

    from sklearn.metrics import f1_score
    macro = f1_score(y_true, y_pred, average="macro")
    # every row is the validation row of exactly one fold: counting folds per
    # row means building a (n_rows, n_folds) one-hot and summing over folds
    n_folds = int(fold.max()) + 1
    rows_per_fold = np.bincount(fold, minlength=n_folds)
    coverage = np.zeros(len(fold), dtype=int)
    for k in range(n_folds):
        coverage += (fold == k).astype(int)

    print("=" * 62)
    print("verified load            : ACCEPTED (%s, mode=%s)"
          % (r["system"], r["mode"]))
    print("shape                    : %s %s" % (P_oof.shape, P_oof.dtype))
    print("exact run count          : %d" % r["n_runs"])
    print("load time                : %.3f s" % load_s)
    print("-" * 62)
    print("OOF probabilities        : %s %s" % (P_oof.shape, P_oof.dtype))
    print("y_true                   : %s %s (%d classes)"
          % (y_true.shape, y_true.dtype, len(set(y_true))))
    print("run_id                   : %s  %s ... %s"
          % (run_id.shape, run_id[0], run_id[-1]))
    print("fold assignment          : %s rows per fold  (each row in "
          "exactly 1 fold: %s)"
          % (rows_per_fold.tolist(), bool((coverage == 1).all())))
    print("y_pred = argmax(proba)   : %s ... %s"
          % (list(y_pred[:3]), list(y_pred[-3:])))
    print("recomputed macro F1      : %.10f" % macro)
    print("-" * 62)
    print("verification checks (%d) :" % len(r["verified_checks"]))
    for c in r["verified_checks"]:
        print("   - %s" % c)
    print("-" * 62)
    print("total wall time          : %.2f s (no training, no feature build)"
          % (time.perf_counter() - t0))
    print("=" * 62)
    return {"proba": P_oof, "y_true": y_true, "run_id": run_id, "fold": fold,
            "macro_f1": macro, "load_seconds": load_s}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "B_plus_window")
