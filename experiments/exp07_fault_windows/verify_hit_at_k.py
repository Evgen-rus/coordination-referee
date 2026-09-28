"""Gate: the diagnostics' hit@k must equal the OFFICIAL metric exactly.

The first version of diagnostics.py reported hit@2 = 0.1843 for the baseline
localiser while results.json reported 0.4258 for the same system and the same
turn source.  That was an index-alignment bug inside diagnostics.py, not a
difference of definition.  This module recomputes the official number with
``metrics.fault_turn_hit_at_k`` and fails if diagnostics.py ever disagrees again.
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
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from metrics import fault_turn_hit_at_k                           # noqa: E402
from localize import localize                                     # noqa: E402
import common as C                                                # noqa: E402

TOL = 1e-9


def main():
    res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
    diag = json.load(open(os.path.join(HERE, "diagnostics.json"), encoding="utf-8"))
    oofA = pd.read_csv(os.path.join(HERE, "oof_A_foundation.csv"))
    oofB = pd.read_csv(os.path.join(HERE, "oof_B_plus_window.csv"))
    runs = C.build_all()["runs"]

    ok = True
    for tag, oof, key in (("A_foundation", oofA, "A_foundation"),
                          ("B_plus_window", oofB, "B_plus_window")):
        ys = oof["y_true"].values.astype(object)
        ft = oof["fault_turn"].values
        pred = oof["y_pred"].values.astype(object)

        # official: rule-based localiser on the predicted label
        turns = [localize(runs[i], pred[i]) for i in range(len(ys))]
        official = fault_turn_hit_at_k(list(ys), list(pred), list(ft), turns)
        from_json = res["systems"][key]["fault_turn_hit2"]
        from_diag = diag["localisation"]["L0_baseline"][tag]["hit@2_rate"]
        good = (abs(official - from_json) < TOL and abs(official - from_diag) < TOL)
        ok &= good
        print("  [%s] %-15s official=%.6f  results.json=%.6f  diagnostics L0=%.6f"
              % ("PASS" if good else "FAIL", tag, official, from_json, from_diag))

    # and the headline claim: fault_turn is the OFFICIAL localiser in both systems
    for key in ("A_foundation", "B_plus_window"):
        official = res["systems"][key]["fault_turn_hit2"]
        l1 = diag["localisation"]["L1_window"][key]["hit@2_rate"]
        print("       %-15s headline uses L0=%.4f (NOT L1=%.4f)"
              % (key, official, l1))

    print("\n%s" % ("HIT@K GATE PASSED" if ok else "HIT@K GATE FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
