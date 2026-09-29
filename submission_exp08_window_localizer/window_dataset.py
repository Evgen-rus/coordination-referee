"""Window-level TARGET construction for the Exp07 submission.

This is the ONLY module in the submission that is allowed to touch a target
column, and it does so for one purpose: building the window model's training
labels.

    faulty run, candidate turn t, true ``fault_turn`` f:
        t in [f-2, f+2]  ->  the run's fault class
        otherwise          ->  ``background``
    clean run: every window is ``background``.

The +-2 radius is FIXED, not selected on validation.  A clean run carries
``fault_turn == -1``, so it takes the background branch by construction: no
clean run is ever given a synthetic fault target.

``window_features.py`` cannot see any of this - it takes a parsed run, which
carries run content only.

Imbalance uses a FIXED, pre-declared weight vector - no resampling, no search.

Compared with the experiment module, this file drops only ``load_runs`` and
``summarise`` (CSV reading and reporting), which the runner does not use.  The
target rule, the radius and the weight vector are byte-identical.
"""

from __future__ import annotations

import numpy as np

FAULT_CLASSES = ["dropped_handoff", "duplicated_work", "deadlock", "conflict",
                 "goal_drift", "runaway_loop"]
WCLASSES = ["background"] + FAULT_CLASSES          # index 0 is background
W2I = {c: i for i, c in enumerate(WCLASSES)}

POS_RADIUS = 2            # fixed in advance, matches window_features.RADIUS
N_WINDOW_CLASSES = len(WCLASSES)

# Pre-declared, never tuned, identical in every fold.
BG_WEIGHT = 0.25
FAULT_WEIGHT = (1.0 - BG_WEIGHT) / len(FAULT_CLASSES)


def window_targets(n_turns: int, label: str, fault_turn: int) -> np.ndarray:
    """Per-turn window class indices for one run (0 == background)."""
    y = np.zeros(n_turns, dtype=np.int8)
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
