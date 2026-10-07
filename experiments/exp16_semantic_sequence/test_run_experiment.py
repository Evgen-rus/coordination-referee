import json
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
             "text": "Please review the claim."},
            {"t": 1, "from": "a2", "to": "a1", "type": "inform",
             "intent": "", "tool": "db_query", "refs": ["art_1"],
             "text": "The record is missing."},
            {"t": 2, "from": "a2", "to": "a1", "type": "final",
             "intent": "close_claim", "tool": "", "refs": [],
             "text": "Review complete."},
        ]),
        "artifacts": json.dumps([
            {"t": 2, "id": "art_1", "status": "final"},
        ]),
        "shared_state": json.dumps([
            {"t": 1, "key": "review", "op": "set", "value": "started"},
            {"t": 2, "key": "review", "op": "set", "value": "done"},
        ]),
    }


def test_turn_order_and_required_metadata_are_preserved():
    seq = exp16.prepare_run(_record())
    assert len(seq["turn_texts"]) == 3
    assert "message type: handoff" in seq["turn_texts"][0]
    assert "sender role: orchestrator" in seq["turn_texts"][0]
    assert "receiver role: analyst" in seq["turn_texts"][0]
    assert "intent present: yes" in seq["turn_texts"][0]
    assert "intent present: no" in seq["turn_texts"][1]
    assert seq["turn_meta"].shape == (3, exp16.META_DIM)
    assert np.array_equal(seq["turn_meta"][:, 0], [0.0, 0.5, 1.0])


def test_progress_is_causal_and_not_looked_ahead():
    seq = exp16.prepare_run(_record())
    meta = seq["turn_meta"]
    # [position, intent, refs, log(refs), log(artifacts-so-far), artifacts-now,
    #  log(state-updates-so-far), state-updates-now]
    assert meta[0, 4] == 0.0 and meta[0, 6] == 0.0
    assert meta[1, 4] == 0.0 and meta[1, 6] > 0.0
    assert meta[2, 4] > 0.0 and meta[2, 5] > 0.0
    assert meta[2, 6] > meta[1, 6] and meta[2, 7] > 0.0


def test_targets_are_ignored_and_no_turn_is_truncated():
    record = _record()
    changed = dict(record, label="runaway_loop", success=0, fault_turn=0)
    first, second = exp16.prepare_run(record), exp16.prepare_run(changed)
    assert first["goal_text"] == second["goal_text"]
    assert first["turn_texts"] == second["turn_texts"]
    assert np.array_equal(first["turn_meta"], second["turn_meta"])
    long_run = dict(record)
    long_run["messages"] = json.dumps([
        {"t": i, "from": "a1", "to": "a2", "type": "inform",
         "intent": "", "text": "turn %d" % i} for i in range(200)
    ])
    seq = exp16.prepare_run(long_run)
    assert len(seq["turn_texts"]) == 200
    assert seq["turn_meta"][-1, 0] == 1.0
