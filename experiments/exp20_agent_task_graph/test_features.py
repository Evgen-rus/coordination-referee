from __future__ import annotations

import copy
import hashlib
import json
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.exp20_agent_task_graph.features import (
    FEATURE_NAMES, _ordered_events, build_matrix, extract_features,
)
from experiments.exp20_agent_task_graph.scoring import l1_turns, official_scores


def run(messages=None, artifacts=None, shared_state=None):
    return {
        "agents": [{"id": "a1", "role": "orchestrator"},
                   {"id": "a2", "role": "analyst"},
                   {"id": "a3", "role": "reviewer"}],
        "messages": messages or [], "artifacts": artifacts or [],
        "shared_state": shared_state or [], "topology": {"type": "star"},
        "label": "clean", "success": 1, "fault_turn": -1,
    }


class TemporalGraphTests(unittest.TestCase):
    def test_assignment_to_result_link_uses_exact_intent_and_original_t(self):
        x = run(messages=[{"t": 0, "from": "a1", "to": "a2", "type": "handoff",
                           "intent": " quotes ", "refs": []}],
                artifacts=[{"id": "art1", "by": "a2", "t": 7, "subtask": "QUOTES",
                            "type": "table", "status": "final"}])
        f = extract_features(x)
        self.assertEqual(f["g_known_assignments"], 1)
        self.assertEqual(f["g_assignments_without_later_result"], 0)
        self.assertEqual(f["g_delivery_delay_mean_turns"], 7)
        x["messages"][0]["intent"] = "different_task"
        changed = extract_features(x)
        self.assertEqual(changed["g_assignments_without_later_result"], 1)

    def test_response_without_result_and_other_task_activity(self):
        x = run(messages=[
            {"t": 0, "from": "a1", "to": "a2", "type": "handoff",
             "intent": "quotes", "refs": []},
            {"t": 1, "from": "a2", "to": "a1", "type": "inform",
             "intent": "quotes", "refs": [], "text": "Acknowledged"},
            {"t": 2, "from": "a2", "to": "a1", "type": "inform",
             "intent": "legal_review", "refs": [], "text": "Starting"},
        ])
        f = extract_features(x)
        self.assertEqual(f["g_executor_response_without_result"], 1)
        self.assertEqual(f["g_other_task_activity_before_result"], 1)

    def test_reassignment_to_another_agent_and_substitute_delivery(self):
        x = run(messages=[
            {"t": 0, "from": "a1", "to": "a2", "type": "handoff",
             "intent": "quotes", "refs": []},
            {"t": 2, "from": "a1", "to": "a3", "type": "handoff",
             "intent": "quotes", "refs": []},
        ], artifacts=[{"id": "art1", "by": "a3", "t": 8, "subtask": "quotes"}])
        f = extract_features(x)
        self.assertEqual(f["g_repeat_assignment_before_result"], 1)
        self.assertEqual(f["g_assignments_without_later_result"], 0)
        self.assertEqual(f["g_other_executor_deliveries"], 1)
        self.assertEqual(f["g_peak_open_assignments"], 2)

    def test_missing_and_conflicting_identifiers_do_not_guess_links(self):
        x = run(messages=[
            {"t": 0, "from": "a1", "to": "a2", "type": "handoff",
             "refs": []},
            {"t": 1, "from": "a1", "to": "a2", "type": "handoff",
             "intent": "quotes", "refs": []},
            {"t": 2, "from": "a3", "to": "a1", "type": "handoff",
             "refs": ["duplicate"]},
        ], artifacts=[
            {"id": "duplicate", "by": "a2", "t": 2, "subtask": ""},
            {"id": "duplicate", "by": "a3", "t": 2, "subtask": ""},
            {"id": "no_subtask", "by": "a3", "t": 3, "subtask": ""},
        ])
        f = extract_features(x)
        self.assertEqual(f["g_known_assignments"], 1)
        self.assertEqual(f["g_assignments_without_later_result"], 1)

    def test_wait_to_delivery_and_state_writer_relations(self):
        x = run(messages=[
            {"t": 0, "from": "a1", "to": "a2", "type": "status",
             "intent": "quotes", "refs": [],
             "text": "Waiting for a2 to deliver quotes."},
        ], artifacts=[{"id": "art1", "by": "a2", "t": 6, "subtask": "quotes"}],
        shared_state=[
            {"t": 1, "agent": "a1", "key": "price", "op": "set", "value": 10},
            {"t": 2, "agent": "a2", "key": "price", "op": "override", "value": 12},
            {"t": 3, "agent": "a1", "key": "price", "op": "override", "value": 12},
        ])
        f = extract_features(x)
        self.assertEqual(f["g_wait_task_links"], 1)
        self.assertEqual(f["g_wait_resolved_by_named_agent_rate"], 1)
        self.assertEqual(f["g_state_multiwriter_keys"], 1)
        self.assertEqual(f["g_state_cross_writer_events"], 2)
        self.assertEqual(f["g_state_cross_writer_overrides"], 2)
        self.assertEqual(f["g_state_cross_writer_value_changes"], 1)

    def test_event_order_is_message_then_state_then_artifact_at_equal_t(self):
        x = run(messages=[
            {"t": 4, "from": "a1", "to": "a2", "type": "handoff",
             "intent": "x", "refs": []},
            {"t": 4, "from": "a2", "to": "a1", "type": "inform",
             "intent": "x", "refs": []},
        ], shared_state=[{"t": 4, "agent": "a1", "key": "k", "op": "set", "value": 1}],
            artifacts=[{"id": "a", "by": "a2", "t": 4, "subtask": "x"}])
        self.assertEqual([e["kind"] for e in _ordered_events(x)],
                         ["message", "message", "state", "artifact"])
        self.assertEqual([e["source_index"] for e in _ordered_events(x)[:2]], [0, 1])
        self.assertEqual(extract_features(x)["g_assignments_without_later_result"], 0)

    def test_target_fields_and_agent_id_renaming_do_not_change_features(self):
        x = run(messages=[
            {"t": 0, "from": "a1", "to": "a2", "type": "handoff",
             "intent": "quotes", "refs": []},
            {"t": 1, "from": "a2", "to": "a1", "type": "inform",
             "intent": "quotes", "refs": []},
            {"t": 2, "from": "a1", "to": "a2", "type": "status",
             "intent": "quotes", "refs": [],
             "text": "Waiting for a2 to deliver quotes."},
        ], artifacts=[{"id": "art1", "by": "a2", "t": 4, "subtask": "quotes"}],
        shared_state=[{"t": 2, "agent": "a1", "key": "k", "op": "set", "value": 1},
                      {"t": 3, "agent": "a2", "key": "k", "op": "override", "value": 2}])
        base = extract_features(x)
        perturbed = copy.deepcopy(x)
        perturbed.update(label="runaway_loop", success=0, fault_turn=17)
        self.assertEqual(base, extract_features(perturbed))

        from experiments.exp20_agent_task_graph.run_experiment import rename_agent_ids
        renamed = rename_agent_ids(x)
        self.assertIn("a100001", renamed["messages"][2]["text"])
        self.assertEqual(base, extract_features(renamed))

    def test_all_rows_and_long_event_stream_are_retained(self):
        x = run(messages=[{"t": i, "from": "a1", "to": "a2", "type": "inform",
                           "intent": "task_%d" % i, "refs": []} for i in range(205)])
        self.assertEqual(len(_ordered_events(x)), 205)
        matrix = build_matrix([x, run(), copy.deepcopy(x)])
        self.assertEqual(matrix.shape, (3, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(matrix).all())

    def test_deterministic_finite_output(self):
        x = run(messages=[{"t": 0, "from": "a1", "to": "a2", "type": "handoff",
                           "intent": "x", "refs": []}])
        self.assertEqual(extract_features(x), extract_features(copy.deepcopy(x)))
        self.assertTrue(np.isfinite(build_matrix([x])).all())

    def test_fixed_folds_have_no_overlap_and_cover_every_row_once(self):
        y = np.repeat(np.arange(7), 9)
        owner = np.zeros(len(y), dtype=np.int8)
        for train_idx, val_idx in StratifiedKFold(
                3, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
            self.assertEqual(len(np.intersect1d(train_idx, val_idx)), 0)
            owner[val_idx] += 1
        self.assertTrue(np.all(owner == 1))


class FrozenBaselineTests(unittest.TestCase):
    def test_saved_exp15_baseline_metrics_and_turns_are_exact(self):
        from experiments.exp10_ab_probability_blend.blend import LABELS, robustness_mask

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        train = pd.read_csv(os.path.join(root, "data", "train.csv"))
        with open(os.path.join(root, "experiments", "exp15_success_label_stacking",
                               "results.json"), encoding="utf-8") as fh:
            exp15 = json.load(fh)
        ids = [str(value) for value in train["run_id"]]
        rid_sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(rid_sha,
                         exp15["run_id_alignment"]["run_id_sha256"])
        self.assertTrue(len(set(ids)) == len(ids) == 10000)

        saved_d = np.load(os.path.join(root, "experiments", "exp13_wait_dependency_graph",
                                       "oof_D_corrected_wg.npy"), allow_pickle=False)
        self.assertEqual(saved_d.shape, (len(train), len(LABELS)))
        labels = np.asarray(LABELS)[saved_d.argmax(1)]
        peaks = np.load(os.path.join(root, "experiments", "exp07_fault_windows",
                                     "runs", "cv", "window_peak_pos.npy"),
                        allow_pickle=False)
        turns = l1_turns(np.asarray([LABELS.index(x) for x in labels]), peaks)
        self.assertTrue(np.all(turns[labels == "clean"] == -1))
        success_pred = np.load(os.path.join(root, "experiments",
                                            "exp15_success_label_stacking",
                                            "success_pred_B.npy"), allow_pickle=False)
        metrics = official_scores(train["label"].values, labels,
                                  train["success"].values, success_pred,
                                  train["fault_turn"].values, turns,
                                  robustness_mask())
        for name, expected in exp15["metrics_B"].items():
            self.assertLessEqual(abs(metrics[name] - expected), 1e-12, name)


if __name__ == "__main__":
    unittest.main()
