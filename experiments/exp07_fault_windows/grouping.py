"""Grouping helper: map flat window rows back to per-run probability blocks."""

from __future__ import annotations

import numpy as np

N_WINDOW_CLASSES = 7


def predict_runs(P: np.ndarray, run_rows: np.ndarray, n_runs: int):
    """Split a flat (n_rows, 7) probability matrix into per-run blocks.

    Returns {run: (P_block, n_rows_in_block)}, including empty blocks so callers
    always get exactly ``n_runs`` entries.
    """
    out = {}
    if len(run_rows) == 0:
        return {r: (np.zeros((0, N_WINDOW_CLASSES)), 0) for r in range(n_runs)}
    order = np.argsort(run_rows, kind="stable")
    sr = run_rows[order]
    bounds = np.searchsorted(sr, np.arange(n_runs + 1))
    for r in range(n_runs):
        a, b = int(bounds[r]), int(bounds[r + 1])
        if b > a:
            out[r] = (P[order[a:b]], b - a)
        else:
            out[r] = (np.zeros((0, N_WINDOW_CLASSES)), 0)
    return out


def aggregate_block(blocks: dict, agg_mod, slot: dict, n_out: int):
    """Fill an (n_out, N_AGG) matrix from per-run probability blocks."""
    X = np.zeros((n_out, agg_mod.N_AGG), dtype=np.float64)
    peak = np.zeros((n_out, agg_mod.N_WINDOW_CLASSES if hasattr(agg_mod, "N_WINDOW_CLASSES")
                     else N_WINDOW_CLASSES), dtype=np.float64)
    peak_pos = np.zeros((n_out, N_WINDOW_CLASSES), dtype=np.int32)
    for r, (P, k) in blocks.items():
        if r not in slot:
            continue
        j = slot[int(r)]
        if k == 0:
            continue
        X[j] = agg_mod.aggregate(P, k)
        peak[j] = P.max(axis=0)
        peak_pos[j] = P.argmax(axis=0)
    return X, peak, peak_pos
