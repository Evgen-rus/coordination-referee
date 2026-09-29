"""Regression guard: n_jobs=-1 must reach LightGBM untouched."""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import parallel as par  # noqa: E402


def main():
    fails = []
    if par.DEFAULT_THREADS != -1:
        fails.append("DEFAULT_THREADS=%r, must be -1" % par.DEFAULT_THREADS)

    import numpy as np
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    cw = np.ones(2)

    seen = {}
    real = lgb.LGBMClassifier

    class Spy(real):
        def __init__(self, **kw):
            seen["n_jobs"] = kw.get("n_jobs")
            super().__init__(**kw)

    lgb.LGBMClassifier = Spy
    try:
        par.run_inner_fits([(0, 0, X, y, cw, np.arange(1000),
                              np.arange(1000, 2000),
                              dict(objective="binary", n_estimators=10,
                                   verbose=-1, random_state=42))],
                           threads=par.DEFAULT_THREADS, workers=1)
    finally:
        lgb.LGBMClassifier = real

    if seen.get("n_jobs") != -1:
        fails.append("LGBMClassifier got n_jobs=%r, expected -1"
                     % seen.get("n_jobs"))
    for f in fails:
        print("FAIL  %s" % f)
    if not fails:
        print("PASS  n_jobs=-1 forwarded to LightGBM unchanged")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
