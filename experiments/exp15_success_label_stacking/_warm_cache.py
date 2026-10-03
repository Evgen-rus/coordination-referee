"""Exp15 helper: rebuild the shared content-addressed Exp07 inputs once.

Not part of the experiment.  ``exp07/cache.py`` fingerprints its inputs and both
on-disk entries are stale, so the first ``ENS._prepare()`` in this project
rebuilds ``build_all`` and the window dataset.  This script does that alone and
exits, so the long experiment run does not pay for it and - more importantly -
does not pay for it while doing anything else.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
EXP08 = os.path.join(ROOT, "experiments", "exp08_window_localizer")
EXP10 = os.path.join(ROOT, "experiments", "exp10_ab_probability_blend")
EXP11 = os.path.join(ROOT, "experiments", "exp11_label_seed_ensemble")
for _p in (EXP11, EXP08, EXP10, EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ensemble as ENS  # noqa: E402


def main():
    t0 = time.time()
    c, W = ENS._prepare()
    print("build_all  : foundation=%s success=%s n=%d"
          % (c["foundation"].shape, c["success"].shape, c["n"]), flush=True)
    print("windows    : X=%s y=%s runs=%d"
          % (W["X"].shape, W["y"].shape, len(np.unique(W["run"]))), flush=True)
    print("pinned success_f1 from Exp03 B = %.16f" % c["success_f1"], flush=True)
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    import numpy as np  # noqa: E402
    main()