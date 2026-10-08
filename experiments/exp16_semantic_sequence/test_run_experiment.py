import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment as exp16


def _record():
    return {
        "goal": "Resolve the customer claim",
        "agents": json.dumps([
            {"id": "a1", "role": "orchestrator"},
            {"id": "a2", "role": "analyst"},
        ]),
        "messages": json.dumps([
            {"t": 0, "from": "a1", "to": "a2", "type": "handoff",
             "intent": "check_claim", "tool": "", "refs": [],
             "text": "Please ask a2 to review the claim."},
            {"t": 1, "from": "a2", "to": "a1", "type": "inform",
             "intent": "", "tool": "db_query", "refs": ["art_1"],
             "text": "The record is missing."},
            {"t": 2, "from": "a2", "to": "a1", "type": "final",
             "intent": "close_claim", "tool": "", "refs": [],
             "text": "Review complete."},
        ]),
        "artifacts": json.dumps([
            {"t": 2, "id": "art_1", "by": "a2", "subtask": "check_claim",
             "type": "quote_table", "status": "final", "hash": "abc123"},
        ]),
        "shared_state": json.dumps([
            {"t": 1, "agent": "a1", "key": "review", "op": "set",
             "value": "started"},
            {"t": 2, "agent": "a2", "key": "review", "op": "override",
             "value": "done"},
        ]),
    }


def _decode(seq):
    return [json.loads(text) if text.startswith("{") else text
            for text in seq["event_texts"]]


def test_raw_intent_artifact_and_state_semantics_are_encoded():
    record = _record()
    seq = exp16.prepare_run(record)
    texts = seq["event_texts"]
    message = next(text for text in texts if "event=message; type=handoff" in text)
    assert "intent=check_claim" in message
    assert "sender role=orchestrator" in message
    assert "receiver role=analyst" in message
    assert "tool=[none]" in message
    empty_intent = next(text for text in texts if "type=inform" in text)
    assert "intent=[none]" in empty_intent
    artifact = next(text for text in texts if text.startswith("event=artifact"))
    assert "producer role=analyst" in artifact
    assert "subtask=check_claim" in artifact
    assert "type=quote_table" in artifact and "status=final" in artifact
    state = next(text for text in texts if text.startswith("event=state"))
    assert "writer role=orchestrator" in state
    assert "key=review" in state and "op=set" in state and "value=started" in state
    assert seq["event_meta"].shape == (len(texts), exp16.META_DIM)

    changed = dict(record)
    messages = json.loads(changed["messages"])
    messages[0]["intent"] = "verify_invoice"
    changed["messages"] = json.dumps(messages)
    changed_message = next(text for text in exp16.prepare_run(changed)["event_texts"]
                           if "event=message; type=handoff" in text)
    assert changed_message != message
    assert "intent=verify_invoice" in changed_message


def test_goal_first_unified_order_and_agent_ids_are_not_semantic_tokens():
    texts = exp16.prepare_run(_record())["event_texts"]
    assert texts[0].startswith("event=goal;")
    kinds = [text.split(";", 1)[0] for text in texts]
    assert kinds == ["event=goal", "event=message", "event=message",
                     "event=state", "event=message", "event=state",
                     "event=artifact"]
    assert all("a1" not in text and "a2" not in text for text in texts)
    assert all("art_1" not in text and "abc123" not in text for text in texts)
    assert texts == exp16.prepare_run(_record())["event_texts"]


def test_future_artifact_and_state_do_not_change_earlier_metadata():
    record = _record()
    original = exp16.prepare_run(record)
    expanded = dict(record)
    artifacts = json.loads(expanded["artifacts"])
    artifacts.append({"t": 99, "by": "a2", "subtask": "future",
                      "type": "report", "status": "partial"})
    expanded["artifacts"] = json.dumps(artifacts)
    states = json.loads(expanded["shared_state"])
    states.append({"t": 99, "agent": "a2", "key": "future", "op": "set",
                   "value": "later"})
    expanded["shared_state"] = json.dumps(states)
    later = exp16.prepare_run(expanded)
    prefix = len(original["event_texts"])
    assert original["event_texts"] == later["event_texts"][:prefix]
    assert np.array_equal(original["event_meta"], later["event_meta"][:prefix])


def test_all_200_plus_events_are_kept_and_targets_do_not_change_inputs():
    record = _record()
    changed = dict(record, label="runaway_loop", success=0, fault_turn=0)
    first, second = exp16.prepare_run(record), exp16.prepare_run(changed)
    assert first["event_texts"] == second["event_texts"]
    assert np.array_equal(first["event_meta"], second["event_meta"])

    long_run = dict(record)
    long_run["messages"] = json.dumps([
        {"t": i, "from": "a1", "to": "a2", "type": "inform",
         "intent": "", "text": "turn %d" % i} for i in range(205)
    ])
    seq = exp16.prepare_run(long_run)
    assert len(seq["event_texts"]) == 209  # goal + 205 messages + 3 records
    assert sum(text.startswith("event=message") for text in seq["event_texts"]) == 205
    assert seq["event_meta"].shape == (209, exp16.META_DIM)
    assert seq["event_meta"][-1, 4] > seq["event_meta"][-2, 4]


def test_exp15_baseline_reproduction_is_exact():
    result = subprocess.run(
        [sys.executable, str(HERE / "run_experiment.py"), "--check-only"],
        cwd=HERE.parents[1], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exp15 baseline verified: composite=0.7938624419" in result.stdout
