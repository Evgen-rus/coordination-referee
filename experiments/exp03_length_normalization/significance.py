"""Paired significance analysis for Exp03 (A vs B) on the saved OOF predictions.

The three variants share folds, training rows and random seeds, so A and B are
paired run-by-run.  A paired bootstrap over runs gives a confidence interval on
the Macro F1 delta that does not assume independence across variants.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from metrics import f1_per_class, macro_f1  # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
A = pd.read_csv(os.path.join(HERE, "oof_A_baseline122.csv"))
B = pd.read_csv(os.path.join(HERE, "oof_B_baseline_plus_norm.csv"))
C = pd.read_csv(os.path.join(HERE, "oof_C_norm_only.csv"))
assert (A.run_id.values == B.run_id.values).all() and (A.y_true.values == C.y_true.values).all()
y = A.y_true.values
n = len(y)
print("runs=%d   classes=%d" % (n, len(set(y))))


def mf1(t, p):
    return macro_f1(list(t), list(p))


rng = np.random.default_rng(0)
NB = 2000
boot_a = np.empty(NB)
boot_b = np.empty(NB)
boot_d = np.empty(NB)
idx_all = np.arange(n)
for i in range(NB):
    idx = rng.choice(idx_all, size=n, replace=True)
    a = mf1(y[idx], A.y_pred.values[idx])
    b = mf1(y[idx], B.y_pred.values[idx])
    boot_a[i], boot_b[i], boot_d[i] = a, b, b - a

lo, hi = np.percentile(boot_d, [2.5, 97.5])
print("")
print("PAIRED BOOTSTRAP (2000 resamples over runs)")
print("  A macro_f1 point=%.4f  95%%CI [%.4f, %.4f]" % (mf1(y, A.y_pred), *np.percentile(boot_a, [2.5, 97.5])))
print("  B macro_f1 point=%.4f  95%%CI [%.4f, %.4f]" % (mf1(y, B.y_pred), *np.percentile(boot_b, [2.5, 97.5])))
print("  delta (B-A)    =%+.4f  95%%CI [%+.4f, %+.4f]" % (mf1(y, B.y_pred) - mf1(y, A.y_pred), lo, hi))
print("  P(delta > 0)   = %.3f" % (boot_d > 0).mean())

ca, cb, cc = A.y_pred.values, B.y_pred.values, C.y_pred.values
print("")
print("PAIRED DISAGREEMENT (McNemar view)")
fix = ((cb == y) & (ca != y)).sum()
brk = ((ca == y) & (cb != y)).sum()
print("  B fixed runs that A got wrong : %d" % fix)
print("  B broke runs that A got right  : %d" % brk)
from scipy.stats import binomtest  # noqa: E402

print("  net %+d" % (fix - brk))
res = binomtest(min(fix, brk), fix + brk, 0.5, alternative="two-sided")
print("  exact McNemar two-sided p      = %.4f" % res.pvalue)
print("  (n discordant = %d, so a net %+d swing over %d runs is the whole effect size)"
      % (fix + brk, fix - brk, n))

print("")
print("PER-CLASS paired change (A -> B), and per-class support")
fa, fb, fc = (f1_per_class(list(y), list(p), LABELS) for p in (ca, cb, cc))
sup = pd.Series(y).value_counts()
print("%-18s %6s %8s %8s %8s %9s" % ("class", "n", "A", "B", "C", "dB"))
for k in LABELS:
    print("%-18s %6d %8.4f %8.4f %8.4f %+9.4f" % (k, sup.get(k, 0), fa[k], fb[k], fc[k], fb[k] - fa[k]))

print("")
print("WORST REGRESSIONS of B vs A (A right -> B wrong), by true class")
m = (ca == y) & (cb != y)
print(pd.Series(y[m]).value_counts().to_string())

print("")
print("Clean-confusion check (false-clean is the expensive error)")
for nm, p in (("A", ca), ("B", cb), ("C", cc)):
    pred_clean = p == "clean"
    fp = int((pred_clean & (y != "clean")).sum())
    fn = int(((~pred_clean) & (y == "clean")).sum())
    print("  %s: predicted clean but actually faulty = %4d | clean runs missed as clean = %4d"
          % (nm, fp, fp))
    print("     %s: clean runs NOT predicted clean       = %4d"
          % (nm, fn))
