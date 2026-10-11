"""Official Exp20 metric path with the Exp15 success predictions frozen."""

from __future__ import annotations

import numpy as np

from evaluation.metrics import (binary_f1, composite, fault_turn_hit_at_k,
                                f1_per_class, macro_f1)
from experiments.exp10_ab_probability_blend.blend import (
    FREE_CLASSES, LABELS, W2I,
)


def l1_turns(predicted_labels, peak_pos):
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    peaks = np.asarray(peak_pos, dtype=np.int64)
    if peaks.shape != (len(predicted_labels), len(LABELS)):
        raise ValueError("Exp13 peak matrix is not aligned to the predicted rows")
    turns = np.full(len(predicted_labels), -1, dtype=np.int64)
    for label_id in FREE_CLASSES:
        selected = predicted_labels == label_id
        turns[selected] = peaks[selected, W2I[LABELS[label_id]]]
    return turns


def official_scores(y_label, pred_label, y_success, pred_success, true_turn,
                    pred_turn, robust_mask):
    y_label = np.asarray(y_label, dtype=object)
    pred_label = np.asarray(pred_label, dtype=object)
    y_success = np.asarray(y_success, dtype=np.int64)
    pred_success = np.asarray(pred_success, dtype=np.int64)
    true_turn = np.asarray(true_turn, dtype=np.int64)
    pred_turn = np.asarray(pred_turn, dtype=np.int64)
    robust_mask = np.asarray(robust_mask, dtype=bool)
    n = len(y_label)
    if any(len(v) != n for v in (pred_label, y_success, pred_success, true_turn,
                                  pred_turn, robust_mask)):
        raise ValueError("metric inputs are not row aligned")
    macro = macro_f1(y_label.tolist(), pred_label.tolist(), LABELS)
    robust = macro_f1(y_label[robust_mask].tolist(),
                      pred_label[robust_mask].tolist(), LABELS)
    success = binary_f1(y_success.tolist(), pred_success.tolist())
    hit2 = fault_turn_hit_at_k(y_label.tolist(), pred_label.tolist(),
                               true_turn.tolist(), pred_turn.tolist(), k=2)
    scores = {"macro_f1": macro, "robustness_f1": robust,
              "success_f1": success, "fault_turn_hit2": hit2}
    scores["composite"] = composite(scores)
    scores["per_class_f1"] = f1_per_class(y_label.tolist(), pred_label.tolist(), LABELS)
    return scores
