"""Run-level aggregation of window-model probabilities for Exp07.

Turns a per-window probability matrix (n_windows x 7) into a compact run-level
feature block that a normal run-level LightGBM can consume.

Design constraints honoured here:
  * the PREDICTED CLASS never becomes a feature - only probabilities and
    positions derived from them, so nothing leaks a hard decision;
  * the threshold used for "high-probability window" is FIXED at 0.5 and
    declared in advance, not fitted;
  * feature count stays in the 30-60 band requested.

Per fault class c we emit: max prob, mean of top-3, position of the max,
position and count of the first/above-threshold region, mean prob, and the
top1-vs-top2 window gap.  Plus a handful of run-level summaries.
"""

from __future__ import annotations

import numpy as np

from window_dataset import FAULT_CLASSES, WCLASSES

HIGH = 0.5          # fixed in advance, never tuned
NC = len(WCLASSES)  # 7 = background + 6 fault classes

# 6 fault classes x 7 statistics
PER_CLASS = ["amax", "atop3", "pos_max", "pos_first_hi", "n_hi", "area", "gap12"]
# run-level summaries
GLOBAL = ["strongest_prob", "strongest_gap", "n_strong_classes",
          "earliest_strong_pos", "probs_entropy", "bg_max", "bg_mean",
          "bg_peak_pos", "argmax_prob_spread"]

AGG_NAMES = ([("p_%s_%s" % (c, s)) for c in FAULT_CLASSES for s in PER_CLASS]
             + ["g_" + s for s in GLOBAL])
N_AGG = len(AGG_NAMES)
assert 30 <= N_AGG <= 60, "aggregation block out of the 30-60 design range: %d" % N_AGG


def _entropy(p: np.ndarray) -> float:
    q = p / max(1e-12, p.sum())
    return float(-(q * np.log(q + 1e-12)).sum())


def aggregate(P: np.ndarray, n_turns: int) -> np.ndarray:
    """P: (n_windows, 7) window-class probabilities for ONE run.

    Returns a fixed-length vector.  ``n_turns`` is used only to convert indices
    into relative positions.
    """
    out = np.zeros(N_AGG, dtype=np.float64)
    if P is None or len(P) == 0:
        return out
    n = len(P)
    denom = max(1.0, float(n_turns if n_turns else n))

    means = P.mean(axis=0)                       # per-class mean probability
    amaxes = P.max(axis=0)                       # per-class max probability
    k = 0
    for ci in range(1, NC):                      # 1..6 = the fault classes
        p = P[:, ci]
        srt = np.sort(p)[::-1]
        top3 = float(srt[:3].mean())
        i_max = int(np.argmax(p))
        hi = np.where(p >= HIGH)[0]
        # top1 - top2 over WINDOWS for this single class
        gap12 = float(srt[0] - srt[1]) if n > 1 else 0.0
        out[k] = amaxes[ci];                      k += 1   # amax
        out[k] = top3;                            k += 1   # atop3
        out[k] = i_max / denom;                   k += 1   # pos_max
        if len(hi):
            out[k] = int(hi[0]) / denom
        else:
            out[k] = 1.0
        k += 1   # pos_first_hi
        out[k] = float(len(hi)) / denom;          k += 1   # n_hi
        out[k] = float(means[ci]);                k += 1   # area
        out[k] = gap12;                           k += 1   # gap12

    # ---- run-level summaries ------------------------------------------
    order = np.argsort(-amaxes[1:NC])             # fault classes, best first
    best, second = int(order[0]), int(order[1])
    n_strong = int((amaxes[1:NC] >= HIGH).sum())
    strong_pos = [out[4 + 3 * ci] for ci in range(1, NC) if amaxes[ci] >= HIGH]
    bg = P[:, 0]

    vals = [
        amaxes[1 + best],                                   # strongest_prob
        float(amaxes[1 + best] - amaxes[1 + second]),       # strongest_gap
        float(n_strong),                                    # n_strong_classes
        (float(min(strong_pos)) if strong_pos else 1.0),    # earliest_strong_pos
        _entropy(means),                                    # probs_entropy
        float(bg.max()),                                    # bg_max
        float(bg.mean()),                                   # bg_mean
        (int(np.argmax(bg)) / denom),                       # bg_peak_pos
        float(np.sort(amaxes[1:NC])[-1] - np.sort(amaxes[1:NC])[0]),
    ]
    for j, v in enumerate(vals):
        out[k] = v
        k += 1
    return out
