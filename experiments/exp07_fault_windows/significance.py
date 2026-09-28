"""Paired significance for Exp07.

WHAT IS AND IS NOT A PER-RUN ESTIMAND (this bit matters, and I got it wrong the
first time):

  * Macro F1 over a SLICE is a per-run estimand in the sense that the bootstrap
    resamples RUNS and then recomputes macro F1 on the resampled set.  That is
    valid.  The estimator must be the real ``macro_f1`` from
    ``evaluation/metrics.py`` - which averages only over classes PRESENT in
    ``y_true``.  A hand-rolled version that averages over all 7 labels, or that
    forgets to mask to ``y_true == c`` before counting false positives, is a
    different function entirely; one of my drafts returned 1.3058 for a quantity
    whose true value is 0.7827, and reported a correspondingly inflated delta.

  * The COMPOSITE is not a per-run estimand at all: it is a fixed weighted sum
    of three whole-set statistics.  Bootstrapping runs and re-deriving it
    double-counts the robustness slice and is biased.  With 3 paired OOF runs
    available, the honest treatment is a fold-level paired bootstrap and an
    explicit statement that n=3 is a consistency check, not a p-value.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from metrics import macro_f1                                   # noqa: E402
import common as C                                             # noqa: E402

A_KEY, B_KEY = "A_foundation", "B_plus_window"
N_RUN_BOOT = 20000
N_FOLD_BOOT = 1000
RNG = np.random.default_rng(20260928)


def main():
    res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
    oofA = pd.read_csv(os.path.join(HERE, "oof_%s.csv" % A_KEY))
    oofB = pd.read_csv(os.path.join(HERE, "oof_%s.csv" % B_KEY))
    ys = oofA["y_true"].values.astype(object)
    pa = oofA["y_pred"].values.astype(object)
    pb = oofB["y_pred"].values.astype(object)
    n = len(ys)

    c = C.build_all()
    A, B = res["systems"][A_KEY], res["systems"][B_KEY]

    # sanity gate: the estimator must reproduce the headline numbers exactly
    ma, mb = macro_f1(list(ys), list(pa)), macro_f1(list(ys), list(pb))
    assert abs(ma - A["macro_f1"]) < 1e-9 and abs(mb - B["macro_f1"]) < 1e-9, \
        "estimator disagrees with results.json: %.6f/%.6f vs %.6f/%.6f" % (
            ma, mb, A["macro_f1"], B["macro_f1"])

    def m_on(idx):
        return (macro_f1([ys[i] for i in idx], [pb[i] for i in idx]),
                macro_f1([ys[i] for i in idx], [pa[i] for i in idx]))

    out = {
        "estimator_gate": {"macro_A": ma, "macro_B": mb, "matches_results_json": True},
        "point": {
            "d_macro": B["macro_f1"] - A["macro_f1"],
            "d_robustness": B["robustness_f1"] - A["robustness_f1"],
            "d_hit2": B["fault_turn_hit2"] - A["fault_turn_hit2"],
            "d_composite": B["composite"] - A["composite"],
        },
    }

    # ---- Macro F1: run-level paired bootstrap, estimator recomputed each time --
    d = np.empty(N_RUN_BOOT)
    for b in range(N_RUN_BOOT):
        ii = RNG.integers(0, n, n)
        fb, fa = m_on(ii)
        d[b] = fb - fa
    out["macro_run_bootstrap"] = {
        "delta": float(d.mean()), "n_boot": N_RUN_BOOT,
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
        "note": "resamples runs; macro F1 recomputed on each resample",
    }

    # ---- slices: same estimator, same procedure -------------------------
    sl = {}
    for sname, m in c["SL"].items():
        idx = np.where(m)[0]
        if len(idx) < 30:
            continue
        ds = np.empty(N_RUN_BOOT)
        for b in range(N_RUN_BOOT):
            jj = idx[RNG.integers(0, len(idx), len(idx))]
            fb = macro_f1([ys[i] for i in jj], [pb[i] for i in jj])
            fa = macro_f1([ys[i] for i in jj], [pa[i] for i in jj])
            ds[b] = fb - fa
        sl[sname] = {
            "n": int(len(idx)),
            "A": float(macro_f1([ys[i] for i in idx], [pa[i] for i in idx])),
            "B": float(macro_f1([ys[i] for i in idx], [pb[i] for i in idx])),
            "delta": float(ds.mean()),
            "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
        }
    out["slices"] = sl

    # ---- composite: fold-level paired bootstrap (not a per-run estimand) ----
    fmA, fmB = np.array(A["fold_macro_f1"]), np.array(B["fold_macro_f1"])
    dc = np.empty(N_FOLD_BOOT)
    for b in range(N_FOLD_BOOT):
        ii = RNG.integers(0, 3, 3)
        dc[b] = float((fmB[ii] - fmA[ii]).mean())
    out["composite_fold_bootstrap"] = {
        "note": "composite is not a per-run estimand; n=3 folds is a "
                "consistency check, NOT a significance level",
        "n_folds": 3, "n_boot": N_FOLD_BOOT,
        "d_macro_by_fold": (fmB - fmA).tolist(),
        "wins": int((fmB > fmA).sum()),
        "d_macro_mean": float((fmB - fmA).mean()),
        "ci95": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
    }

    with open(os.path.join(HERE, "significance.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
