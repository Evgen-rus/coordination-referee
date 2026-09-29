"""Exp07 run modes: FAST / CV / FINAL.

The three modes differ ONLY in how deeply a result is verified.  The model
mathematics is identical in all three: same features, same folds, same seeds,
same LightGBM hyper-parameters, same honest inner cross-fitting.  Nothing about
the estimator changes between modes.

  FAST   one pre-declared outer fold, no bootstrap, no diagnostics.
         Screening only - a FAST number is NOT a quality verdict.
  CV     the standard 3 outer folds with honest stacking, all headline
         metrics, slices and confusion pairs.  No bootstrap by default.
  FINAL  everything CV does, plus diagnostics, significance, the leakage
         audit and the production parity checks.

FAST additionally supports a conservative early stop: if the candidate is
clearly worse than the foundation it stops instead of finishing the remaining
folds.  That is a time-saving device only and is never recorded as evidence
that the idea is bad.
"""
from __future__ import annotations

import os

FAST = "fast"
CV = "cv"
FINAL = "final"
MODES = (FAST, CV, FINAL)

# FAST uses a single pre-declared fold: the first split of the SAME
# StratifiedKFold(3, shuffle=True, random_state=0) the CV mode uses.  Not a new
# random split - literally the first one, so FAST and CV agree exactly on it.
FAST_FOLDS = (0,)

# Conservative screening thresholds.  Either one triggers the stop.
EARLY_COMPOSITE_DELTA = -0.01
EARLY_MACRO_DELTA = -0.015


def resolve(name):
    n = (name or CV).lower()
    if n not in MODES:
        raise ValueError("unknown mode %r, expected one of %s" % (name, MODES))
    return n


def folds_for(mode):
    return FAST_FOLDS if mode == FAST else (0, 1, 2)


def outdir(mode):
    """Artifacts are namespaced per mode so a FAST run can never overwrite a
    FINAL one."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "runs", mode)
