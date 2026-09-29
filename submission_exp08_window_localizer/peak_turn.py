"""L1 fault-turn localisation: the peak window turn for the predicted class.

Exp07 already fits the window model in production - it has to, the 51 run-level
aggregates are the whole new signal for the label head.  What it did NOT do was
*use* the peak POSITION: production aggregated window probabilities into
features and then threw the positions away, falling back to the rule-based
``localize`` for ``fault_turn``.

This module recovers that discarded information with the smallest possible
change.  It reads the same per-run window probability blocks that
``aggregate_runs`` already consumes, takes the one column belonging to the
already-predicted fault class, and returns its argmax - the turn the window
model is most confident belongs to the fault it has already been told about.

THE RULE, IN FULL
-----------------
    predicted label == clean      ->  -1
    empty window block            ->  -1
    otherwise                     ->  argmax_t P[run, t, class(predicted label)]

No confidence threshold.  No offset.  No per-class behaviour.  No calibration.
One argmax.  ``class`` indices come from ``window_dataset.W2I``, because the
window model's index 0 is ``background`` while the label head's index 0 is
``clean`` - using the label index here would silently shift every class by one.

The local turn index returned is exactly the index ``window_features`` produced,
which is the same index the experiment's ``fault_turn`` column uses, so it is
directly comparable with ``localize``'s output.
"""

from __future__ import annotations

import numpy as np

import window_dataset as wd


def class_index(label: str) -> int:
    """Index of ``label`` among the WINDOW classes (0 = background).

    Raises KeyError for a label the window model was never trained on, rather
    than returning a plausible-looking wrong index.
    """
    return wd.W2I[label]


def peak_turn_for_block(P_block: np.ndarray, label: str) -> int:
    """Peak turn of ``P_block`` for ``label``, or -1 when undefined.

    ``P_block`` is that run's ``(n_windows, 7)`` window probability matrix.
    """
    if label == "clean":
        return -1                                  # clean: no fault turn
    ci = class_index(label)
    if P_block is None or len(P_block) == 0:
        return -1                                  # empty block: nothing to pick
    return int(np.argmax(P_block[:, ci]))


def peak_turns(P: np.ndarray, run_rows: np.ndarray, n_runs: int,
               labels) -> np.ndarray:
    """Peak turn for every run, from the flat window probability matrix.

    Mirrors ``grouping.predict_runs``: ``run_rows`` says which run each window
    row belongs to, and run ``r`` owns its rows in file order, so the k-th of
    its rows is turn ``k``.  The split is done with one stable argsort plus
    searchsorted - scanning all rows once per run would be O(n_runs x n_windows)
    and is exactly the kind of thing that turns a 5-minute submission into a
    multi-hour one.

    An empty block yields -1 rather than argmax of an empty array, which would
    be a silent 0 - a turn pointing at the first turn of a run whose windows the
    model never scored.
    """
    n_runs = int(n_runs)
    out = np.full(n_runs, -1, dtype=np.int64)
    labels = list(labels)
    if P is None or len(run_rows) == 0:
        # nothing was scored: every non-clean run still needs a turn, and the
        # honest answer to "no windows" is -1, never turn 0
        return out

    order = np.argsort(run_rows, kind="stable")
    sorted_runs = run_rows[order]
    bounds = np.searchsorted(sorted_runs, np.arange(n_runs + 1), side="left")
    for r in range(n_runs):
        a, b = int(bounds[r]), int(bounds[r + 1])
        if b <= a:
            continue                                  # empty block -> -1
        out[r] = peak_turn_for_block(P[order[a:b]], labels[r])
    return out
