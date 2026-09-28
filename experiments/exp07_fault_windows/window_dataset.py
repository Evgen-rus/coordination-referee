"""Window-level training dataset for Exp07.

Target construction (the ONLY place a target is touched):

  * faulty run, candidate turn t, true ``fault_turn`` f:
        t in [f-2, f+2]  ->  the run's fault class
        otherwise          ->  ``background``
  * clean run: every window is ``background``.

Radius is FIXED at 2, not selected on validation.  A clean run has
``fault_turn == -1`` so it takes the background branch by construction: no clean
run is ever given a synthetic fault target.

``label``/``success``/``fault_turn`` appear here and ONLY here, as target
construction.  They are never features - ``window_features.py`` cannot even see
them, since it takes a parsed run, which carries run content only.

Imbalance uses a FIXED, pre-declared weight vector - no resampling, no search.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from features import parse_run                          # noqa: E402
import window_features as wf                            # noqa: E402

FAULT_CLASSES = ["dropped_handoff", "duplicated_work", "deadlock", "conflict",
                 "goal_drift", "runaway_loop"]
WCLASSES = ["background"] + FAULT_CLASSES          # index 0 is background
W2I = {c: i for i, c in enumerate(WCLASSES)}

POS_RADIUS = wf.RADIUS                            # 2, fixed in advance
N_WINDOW_CLASSES = len(WCLASSES)

# Pre-declared, never tuned, identical in every fold.
BG_WEIGHT = 0.25
FAULT_WEIGHT = (1.0 - BG_WEIGHT) / len(FAULT_CLASSES)


def window_targets(n_turns: int, label: str, fault_turn: int) -> np.ndarray:
    y = np.zeros(n_turns, dtype=np.int8)           # background
    if label != "clean" and fault_turn is not None and int(fault_turn) >= 0:
        lo = max(0, int(fault_turn) - POS_RADIUS)
        hi = min(n_turns, int(fault_turn) + POS_RADIUS + 1)
        y[lo:hi] = W2I[label]
    return y


def class_weight_vector() -> np.ndarray:
    w = np.ones(N_WINDOW_CLASSES, dtype=np.float64)
    w[W2I["background"]] = BG_WEIGHT
    for c in FAULT_CLASSES:
        w[W2I[c]] = FAULT_WEIGHT
    return w


def build_window_dataset(runs: List[Dict[str, Any]], ys: np.ndarray,
                         ft: np.ndarray) -> Dict[str, np.ndarray]:
    """X (N,F), y (N,), run (N,), turn (N,).  ``run`` indexes into ``runs`` and
    is what makes honest cross-fitting possible."""
    Xs, y_out, rid, turns = [], [], [], []
    for i, (run, lab, f) in enumerate(zip(runs, ys, ft)):
        F = wf.run_window_features(run)
        if F.shape[0] == 0:
            continue
        Xs.append(F)
        y_out.append(window_targets(F.shape[0], str(lab), f))
        rid.append(np.full(F.shape[0], i, dtype=np.int32))
        turns.append(np.arange(F.shape[0], dtype=np.int32))
    return {"X": np.vstack(Xs), "y": np.concatenate(y_out).astype(np.int32),
            "run": np.concatenate(rid), "turn": np.concatenate(turns)}


def load_runs(path: str):
    df = pd.read_csv(path)
    runs = [parse_run(r) for r in df.to_dict("records")]
    return df, runs, df["label"].values.astype(object), df["fault_turn"].values


def summarise(d: Dict[str, np.ndarray], tag: str) -> str:
    y = d["y"]
    out = ["%s: %d windows over %d runs, %d features"
           % (tag, len(y), len(np.unique(d["run"])), d["X"].shape[1])]
    for c in WCLASSES:
        k = int((y == W2I[c]).sum())
        out.append("    %-18s %8d  %5.2f%%" % (c, k, 100.0 * k / max(1.0, len(y))))
    return "\n".join(out)
