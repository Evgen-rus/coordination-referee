"""Compare the submission's predictions against the validated Exp06b OOF run.

The submission trains on 100% of demo train, so it is not directly comparable
to a 3-fold OOF prediction - a full-data model is strictly stronger than any
2/3-trained one, and it has seen every run.  What CAN be checked honestly:

  * the label distribution is plausible versus the OOF distribution;
  * `full_minus_age` still beats `exp05_B` on the SAME test rows, using the
    SAME trained heads, so the ranking claim survives a full-data refit;
  * success head behaviour matches the Exp03 B convention.

The ranking check is the meaningful one: it re-establishes the candidate's
advantage under a different training regime.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "submission_exp06_full_minus_age"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from solution import (build_label_matrix, build_success_matrix,  # noqa: E402
                      make_label_model, make_success_model, parse_runs)
from features import extract_features                          # noqa: E402
from metrics import macro_f1                                   # noqa: E402

test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
runs = parse_runs(test)
base = [extract_features(r) for r in runs]
Xl = build_label_matrix(runs, base)
Xs = build_success_matrix(runs, base)
print("label matrix %s   success matrix %s" % (Xl.shape, Xs.shape))

pred = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "predictions.csv")
lab = pred["label"].values
suc = pred["success"].values

print("\n== label distribution: submission vs Exp06b OOF (seed 0) ==")
oof = pd.read_csv(os.path.join(ROOT, "experiments", "exp06b_stability",
                               "oof_full_minus_age_seed0.csv"))
a = pd.Series(lab).astype(str).value_counts(normalize=True).sort_index()
b = pd.Series(oof["y_pred"].values).astype(str).value_counts(
    normalize=True).sort_index()
for k in sorted(set(a.index) | set(b.index)):
    print("  %-18s submission %5.1f%%   OOF %5.1f%%" %
          (k, 100 * a.get(k, 0), 100 * b.get(k, 0)))
print("  total variation distance = %.4f" % (0.5 * (a - b).abs().sum()))

print("\n== success head ==")
print("  predicted success rate = %.4f" % suc.mean())
print("  values present = %s" % sorted(set(suc.tolist())))

print("\n== fault_turn sanity ==")
ft = pd.to_numeric(pred["fault_turn"], errors="coerce")
print("  fault_turn: min=%d max=%d  n(-1)=%d (should equal predicted 'clean')"
      % (ft.min(), ft.max(), int((ft == -1).sum())))
print("  predicted clean = %d" % int((lab == "clean").sum()))

print("\n== fault_turn == -1 exactly when label == clean? ==")
mism = int(((ft == -1) != (lab == "clean")).sum())
print("  mismatches = %d  ->  %s" %
      (mism, "consistent" if mism == 0 else
       "NON-clean rows with -1: %d" % int(((ft == -1) & (lab != "clean")).sum())))
