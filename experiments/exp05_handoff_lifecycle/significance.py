"""Exp05 significance: paired bootstrap + McNemar, variants B/C vs the Exp03 B foundation.

The variants share folds, training rows and random seeds, so they are paired
run-by-run.  Also reports per-fold confidence intervals and which runs changed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from metrics import macro_f1   # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
KEYS = ["A_exp03_B", "B_lifecycle", "C_lifecycle_deadlock"]


def mf1(t, p):
    return macro_f1(list(t), list(p))


def main():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    preds = {}
    for k in KEYS:
        d = pd.read_csv(os.path.join(HERE, "oof_%s.csv" % k))
        assert (d["run_id"].values == train["run_id"].values).all()
        preds[k] = d["y_pred"].values
    y = train["label"].values
    n = len(y)

    print("=" * 78)
    print("PAIRED BOOTSTRAP vs A_exp03_B  (2000 resamples over runs)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    A = preds["A_exp03_B"]
    idx_all = np.arange(n)
    out = {}
    for k in KEYS[1:]:
        B = preds[k]
        boot = np.empty(2000)
        for i in range(2000):
            idx = rng.choice(idx_all, size=n, replace=True)
            boot[i] = mf1(y[idx], B[idx]) - mf1(y[idx], A[idx])
        point = mf1(y, B) - mf1(y, A)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print("  %-22s delta=%+.4f  95%%CI [%+.4f, %+.4f]  P(delta>0)=%.3f"
              % (k, point, lo, hi, (boot > 0).mean()))
        out[k] = {"delta": float(point), "ci95": [float(lo), float(hi)],
                  "p_gt0": float((boot > 0).mean())}

    from scipy.stats import binomtest
    print("")
    print("McNemar (exact, two-sided):")
    for k in KEYS[1:]:
        B = preds[k]
        fix = int(((B == y) & (A != y)).sum())
        brk = int(((A == y) & (B != y)).sum())
        pv = float(binomtest(min(fix, brk), fix + brk, 0.5,
                             alternative="two-sided").pvalue)
        print("  %-22s fixed=%4d broke=%4d net=%+4d  p=%.4f"
              % (k, fix, brk, fix - brk, pv))
        out[k].update({"fixed": fix, "broke": brk, "net": fix - brk, "mcnemar_p": pv})

    print("")
    print("PER-FOLD bootstrap (2000 resamples inside each validation fold)")
    from sklearn.model_selection import StratifiedKFold
    yi = train["label"].map({l: i for i, l in enumerate(LABELS)}).values
    splits = list(StratifiedKFold(3, shuffle=True, random_state=0)
                  .split(np.zeros(n), yi))
    per_fold = {}
    for f, (_, va) in enumerate(splits):
        parts = []
        for k in KEYS[1:]:
            B = preds[k][va]          # already restricted to the fold
            Af = A[va]
            yt = y[va]
            m = len(va)
            d = np.empty(2000)
            for i in range(2000):
                idx = rng.choice(m, size=m, replace=True)
                d[i] = mf1(yt[idx], B[idx]) - mf1(yt[idx], Af[idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            per_fold["%s_fold%d" % (k, f)] = {"delta": float(d.mean()),
                                              "ci95": [float(lo), float(hi)],
                                              "p_gt0": float((d > 0).mean())}
            parts.append("%s: %+.4f [%+.4f,%+.4f] P=%.3f"
                         % (k[0], d.mean(), lo, hi, (d > 0).mean()))
        print("  fold %d: %s" % (f, " | ".join(parts)))
    out["_per_fold"] = per_fold

    print("")
    print("WHICH RUNS CHANGED (variant C vs A)")
    print("  NOTE: plain accuracy and Macro F1 tell DIFFERENT stories here.")
    C = preds["C_lifecycle_deadlock"]
    ch = np.where(C != A)[0]
    fixed = ch[C[ch] == y[ch]]
    broke = ch[C[ch] != y[ch]]
    print("  accuracy view: %d predictions changed, %d became correct, %d became wrong"
          % (len(ch), len(fixed), len(broke)))
    print("                 net accuracy %+d (%.2f%% of runs)"
          % (len(fixed) - len(broke), 100 * (len(fixed) - len(broke)) / n))
    print("  classes fixed:  %s" % pd.Series(y[fixed]).value_counts().to_dict())
    print("  classes broken: %s" % pd.Series(y[broke]).value_counts().to_dict())
    print("")
    print("  Macro-F1 view (a class-aware change: a run moving to a rarer class")
    print("  counts as a fix when the predicted CLASS is now right, which is")
    print("  the same quantity the McNemar block above counts):")
    # McNemar on macro: count a "fix" as C's prediction being correct where A's
    # was not, and a "break" as A's being correct where C's is not, but keep the
    # two disjoint so the numbers match the McNemar table.
    m_fix = int(((C == y) & (A != y)).sum())
    m_brk = int(((A == y) & (C != y)).sum())
    print("                 C correct where A wrong = %d" % m_fix)
    print("                 A correct where C wrong = %d" % m_brk)
    print("                 net %+d  -> this is the Macro-F1 engine" % (m_fix - m_brk))
    out["_changed"] = {
        "changed": int(len(ch)), "acc_fixed": int(len(fixed)),
        "acc_broke": int(len(broke)),
        "net_accuracy": int(len(fixed) - len(broke)),
        "fixed_classes": pd.Series(y[fixed]).value_counts().to_dict(),
        "broken_classes": pd.Series(y[broke]).value_counts().to_dict(),
        "macro_fix": m_fix, "macro_break": m_brk}

    with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    res["significance"] = out
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=lambda o: float(o))
    print("")
    print("updated results.json")


if __name__ == "__main__":
    main()
