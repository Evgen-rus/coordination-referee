"""Shared helpers for Exp06 significance testing."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CLASSES = ["clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict",
           "goal_drift", "runaway_loop"]
KEYS = ["0_B_exp05_B", "1_B_age", "2_B_backlog", "3_B_receiver", "4_B_reassign",
        "5_B_tail", "6_B_ALL", "7_full_minus_age", "8_full_minus_backlog",
        "9_full_minus_receiver", "10_full_minus_reassign", "11_full_minus_tail"]


def load():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    y = train["label"].values
    P, NM = {}, {}
    for k in KEYS:
        d = pd.read_csv(os.path.join(HERE, "oof_%s.csv" % k))
        assert (d["run_id"].values == train["run_id"].values).all()
        P[k] = d["y_pred"].values
        NM[k] = d["n_messages"].values
    yi = train["label"].map({l: i for i, l in enumerate(CLASSES)}).values
    return train, y, yi, P, NM


def long_recall(pred, mask):
    """Recall of dropped_handoff on a boolean mask over the full run set."""
    idx = np.where(mask)[0]
    tp = int((pred[idx] == "dropped_handoff").sum())
    return tp / len(idx) if len(idx) else 0.0


def recall_at(pred, idx):
    """Recall of dropped_handoff on an explicit array of row indices.

    ``long_recall`` takes a boolean MASK; passing an index array instead makes
    ``np.where`` return positions of non-zero entries, which silently yields the
    wrong denominator.  Bootstrap resamples are index arrays, so they need this.
    """
    tp = int((pred[idx] == "dropped_handoff").sum())
    return tp / len(idx) if len(idx) else 0.0


def long_f1(pred, mask):
    r = long_recall(pred, mask)
    return 2 * r / (1 + r) if r else 0.0
