import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_experiment as exp19
import stepfinder_model

EXP16 = exp19.load_exp16_module()


def record():
    return {
        "goal": "Resolve the customer claim",
        "agents": json.dumps([
            {"id": "a1", "role": "orchestrator"},
            {"id": "a2", "role": "analyst"},
        ]),
        "messages": json.dumps([
            {"t": 0, "from": "a1", "to": "a2", "type": "handoff",
             "intent": "check_claim", "tool": "", "refs": [],
             "text": "Please review the claim."},
            {"t": 1, "from": "a2", "to": "a1", "type": "inform",
             "intent": "", "tool": "db_query", "refs": ["art_1"],
             "text": "The record is missing."},
            {"t": 2, "from": "a2", "to": "a1", "type": "final",
             "intent": "close_claim", "tool": "", "refs": [],
             "text": "Review complete."},
        ]),
        "artifacts": json.dumps([
            {"t": 2, "id": "random-artifact-id", "by": "a2",
             "subtask": "check_claim", "type": "quote_table",
             "status": "final", "hash": "random-hash"},
        ]),
        "shared_state": json.dumps([
            {"t": 1, "agent": "a1", "key": "review", "op": "set",
             "value": "started"},
            {"t": 2, "agent": "a2", "key": "review", "op": "override",
             "value": "done"},
        ]),
    }


def test_exact_turn_to_event_alignment_and_deterministic_tie_order():
    source = record()
    prepared = EXP16.prepare_run(source)
    first = exp19.build_event_map(source, prepared, EXP16)
    second = exp19.build_event_map(source, EXP16.prepare_run(source), EXP16)
    assert np.array_equal(first["turn_to_event"], [1, 2, 4])
    assert np.array_equal(first["event_to_turn"], [-1, 0, 1, -1, 2, -1, -1])
    assert first["message_positions"].tolist() == [1, 2, 4]
    assert np.array_equal(first["turn_to_event"], second["turn_to_event"])
    assert prepared["event_texts"] == EXP16.prepare_run(source)["event_texts"]


def test_targets_and_future_events_do_not_change_earlier_inputs():
    source = record()
    original = EXP16.prepare_run(source)
    changed_targets = dict(source, label="goal_drift", success=0, fault_turn=2)
    changed = EXP16.prepare_run(changed_targets)
    assert original["event_texts"] == changed["event_texts"]
    assert np.array_equal(original["event_meta"], changed["event_meta"])

    later = dict(source)
    artifacts = json.loads(later["artifacts"])
    artifacts.append({"t": 99, "by": "a2", "subtask": "future",
                      "type": "report", "status": "partial"})
    later["artifacts"] = json.dumps(artifacts)
    states = json.loads(later["shared_state"])
    states.append({"t": 99, "agent": "a2", "key": "future", "op": "set",
                   "value": "later"})
    later["shared_state"] = json.dumps(states)
    expanded = EXP16.prepare_run(later)
    assert expanded["event_texts"][:len(original["event_texts"])] == original["event_texts"]
    assert np.array_equal(expanded["event_meta"][:len(original["event_meta"])],
                          original["event_meta"])


def test_all_message_turns_retained_for_long_run():
    source = record()
    source["messages"] = json.dumps([
        {"t": i, "from": "a1", "to": "a2", "type": "inform",
         "intent": "", "text": "turn %d" % i} for i in range(205)
    ])
    prepared = EXP16.prepare_run(source)
    sequence = exp19.build_event_map(source, prepared, EXP16)
    assert sequence["length"] == 209
    assert len(sequence["message_positions"]) == 205
    assert sequence["turn_to_event"].tolist() == [
        int(sequence["message_positions"][i]) for i in range(205)]


def test_outer_folds_do_not_overlap_and_partition_all_rows():
    from sklearn.model_selection import StratifiedKFold

    labels = np.tile(np.arange(3), 12)
    folds = [valid for _, valid in StratifiedKFold(
        3, shuffle=True, random_state=0).split(np.zeros(len(labels)), labels)]
    frozen = exp19.frozen_folds(labels, folds)
    for valid in frozen:
        train = exp19.split_rows(len(labels), valid)
        assert not np.intersect1d(train, valid).size
    assert sorted(np.concatenate(frozen).tolist()) == list(range(len(labels)))


def _model_inputs():
    import torch

    torch.manual_seed(7)
    content = torch.randn(2, 7, 384)
    metadata = torch.zeros(2, 7, 12)
    metadata[:, :, 4] = torch.arange(7, dtype=torch.float32) / 8
    metadata[:, 0, 0] = 1
    metadata[:, 1:, 1] = 1
    role_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 0]] * 2)
    class_ids = torch.tensor([0, 1])
    valid = torch.ones(2, 7, dtype=torch.bool)
    message = torch.tensor([[False, True, False, False, True, False, False]] * 2)
    return content, metadata, role_ids, class_ids, valid, message


def test_class_conditioning_and_message_only_outputs():
    import torch

    torch.manual_seed(42)
    model = stepfinder_model.StepFinderLocalizer(3).eval()
    content, metadata, roles, classes, valid, messages = _model_inputs()
    with torch.inference_mode():
        logits, _ = model(content[:1], metadata[:1], roles[:1], classes[:1],
                          valid[:1], messages[:1])
        changed_class, _ = model(content[:1], metadata[:1], roles[:1],
                                 torch.tensor([2]), valid[:1], messages[:1])
        again, _ = model(content[:1], metadata[:1], roles[:1], classes[:1],
                         valid[:1], messages[:1])
    assert torch.equal(logits, again)
    assert not torch.allclose(logits, changed_class)
    assert torch.all(logits[0, ~messages[0]] <= -1e8)
    assert messages[0, int(logits[0].argmax())]


def test_clean_prediction_returns_minus_one_without_localizer_call():
    class UnusedModel:
        def eval(self):
            return self

        def __call__(self, *args):
            raise AssertionError("clean rows must not enter the localizer")

    result = exp19.predict_turns(
        UnusedModel(), np.asarray([0]), np.asarray([0]), None,
        [{"message_positions": np.asarray([1])}], {})
    assert result.tolist() == [-1]


def test_official_hit_at_k_metric_contract():
    spec = importlib.util.spec_from_file_location(
        "official_metrics", ROOT / "evaluation" / "metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    assert metrics.fault_turn_hit_at_k(
        ["deadlock", "conflict"], ["deadlock", "conflict"],
        [10, 20], [12, 24], k=2) == 0.5


def test_exp15_baseline_parity_before_any_fit():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(HERE / "run_experiment.py"), "--check-only"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exp15 baseline verified: composite=0.7938624419 hit@2=0.6315277778" in result.stdout
    assert "no model fit" in result.stdout
