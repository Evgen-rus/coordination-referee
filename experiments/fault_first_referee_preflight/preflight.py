"""Input invariants and measured CPU feasibility for the fixed Exp21 model."""

from __future__ import annotations

import copy
import ctypes
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from event_inputs import (EVENT_KIND_ID, _EXP16, _observable, build_dataset,
                          build_row)
from model import FaultFirstReferee
from run_experiment import (BATCH_SIZE, EPOCHS, LABEL_TO_ID, _batch_loss,
                            _predict)
from sklearn.model_selection import StratifiedShuffleSplit


def _peak_rss_mb():
    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    psapi = ctypes.WinDLL("Psapi.dll")
    kernel32 = ctypes.WinDLL("kernel32.dll")
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return None
    return counters.PeakWorkingSetSize / 1024**2


def _rename_agent_ids(record):
    out = copy.deepcopy(record)
    agents = _EXP16._json_field(out.get("agents"), [])
    mapping = {str(agent["id"]): f"a{9000 + i}"
               for i, agent in enumerate(agents) if isinstance(agent, dict)
               and agent.get("id") is not None}
    out["agents"] = json.dumps([
        dict(agent, id=mapping.get(str(agent.get("id")), agent.get("id")))
        for agent in agents], ensure_ascii=False)
    for field in ("messages", "artifacts", "shared_state"):
        events = _EXP16._json_field(out.get(field), [])
        changed = []
        for event in events:
            event = dict(event)
            for key in ("from", "to", "by", "agent"):
                if key in event:
                    event[key] = mapping.get(str(event[key]), event[key])
            if "text" in event:
                for old, new in mapping.items():
                    event["text"] = re.sub(
                        rf"(?<!\w){re.escape(old)}(?!\w)", new,
                        str(event["text"]), flags=re.IGNORECASE)
            changed.append(event)
        out[field] = json.dumps(changed, ensure_ascii=False)
    topology = _EXP16._json_field(out.get("topology"), {})
    if isinstance(topology, dict):
        topology["edges"] = [[mapping.get(str(a), a), mapping.get(str(b), b)]
                             for a, b in topology.get("edges", [])]
        out["topology"] = json.dumps(topology)
    return out


def _assert_equal_rows(left, right):
    for key in ("token_ids", "token_offsets", "ref_ids", "ref_offsets",
                "categories", "metadata", "message_positions", "message_turns"):
        a, b = left[key], right[key]
        if key in ("message_positions", "message_turns"):
            assert a == b, key
        else:
            assert np.array_equal(np.asarray(a), np.asarray(b)), key


def run_invariant_checks(record):
    changed_target = dict(record, label="clean", success=1,
                          fault_turn=-1)
    _assert_equal_rows(build_row(record), build_row(changed_target))

    renamed = _rename_agent_ids(record)
    _assert_equal_rows(build_row(record), build_row(renamed))
    prepared = _EXP16.prepare_run(_observable(record))
    for agent in _EXP16._json_field(record.get("agents"), []):
        raw_id = str(agent.get("id", ""))
        if raw_id:
            assert not re.search(rf"(?<!\w){re.escape(raw_id)}(?!\w)",
                                 " ".join(prepared["event_texts"]), re.I)

    base = copy.deepcopy(record)
    future = copy.deepcopy(record)
    messages = _EXP16._json_field(base["messages"], [])
    base["messages"] = json.dumps(messages[:1])
    base["artifacts"] = "[]"
    base["shared_state"] = "[]"
    future["messages"] = json.dumps(messages[:1])
    artifacts = _EXP16._json_field(record.get("artifacts"), [])
    states = _EXP16._json_field(record.get("shared_state"), [])
    if artifacts:
        late = dict(artifacts[-1], t=1000000)
        future["artifacts"] = json.dumps([late])
    if states:
        late_state = dict(states[-1], t=1000000)
        future["shared_state"] = json.dumps([late_state])
    a, b = build_row(base), build_row(future)
    assert np.array_equal(a["metadata"][0], b["metadata"][0])
    assert np.array_equal(a["metadata"][1], b["metadata"][1])
    assert np.array_equal(a["categories"][0], b["categories"][0])
    assert np.array_equal(a["categories"][1], b["categories"][1])

    tied = {
        "goal": "test goal", "agents": json.dumps([{"id": "a1", "role": "lead"}]),
        "messages": json.dumps([{"t": 0, "from": "a1", "to": "a1",
                                  "type": "inform", "intent": "x",
                                  "text": "message", "refs": []}]),
        "shared_state": json.dumps([{"t": 0, "agent": "a1", "key": "k",
                                      "op": "set", "value": "v"}]),
        "artifacts": json.dumps([{"t": 0, "id": "art", "by": "a1",
                                   "subtask": "x", "type": "x",
                                   "status": "final"}]),
        "topology": json.dumps({"type": "star", "edges": []}),
    }
    tie_row = build_row(tied)
    assert [x[0] for x in tie_row["categories"]] == [
        EVENT_KIND_ID["goal"], EVENT_KIND_ID["message"],
        EVENT_KIND_ID["state"], EVENT_KIND_ID["artifact"]]
    assert tie_row["metadata"][2][5] == 0.0

    long_run = dict(tied)
    long_run["messages"] = json.dumps([
        {"t": i, "from": "a1", "to": "a1", "type": "inform",
         "intent": "task", "text": f"message number {i}", "refs": []}
        for i in range(205)])
    long_run["artifacts"] = "[]"
    long_run["shared_state"] = "[]"
    long_row = build_row(long_run)
    assert len(long_row["categories"]) == 206
    assert len(long_row["message_positions"]) == 205
    assert long_row["message_turns"] == tuple(range(205))
    return {"target_perturbation": "PASS", "agent_rename": "PASS",
            "raw_agent_ids_in_semantic_text": "PASS",
            "earlier_metadata_invariant_to_future_events": "PASS",
            "same_t_order_message_state_artifact": "PASS",
            "205_messages_retained": "PASS"}


def _sample_rows(frame, count=544):
    labels = frame.label.astype(str).to_numpy()
    split = StratifiedShuffleSplit(n_splits=1, train_size=count,
                                   random_state=42)
    chosen, _ = next(split.split(np.zeros(len(frame)), labels))
    event_counts = []
    for row in frame.iloc[chosen].to_dict("records"):
        event_counts.append(
            1 + len(_EXP16._json_field(row["messages"], [])) +
            len(_EXP16._json_field(row["artifacts"], [])) +
            len(_EXP16._json_field(row["shared_state"], [])))
    longest = np.argsort(event_counts)[-32:]
    chosen = np.asarray(chosen)
    chosen[-32:] = chosen[longest]
    return np.unique(chosen)


def run_preflight():
    frame = pd.read_csv(ROOT / "data" / "train.csv")
    first = frame.iloc[0].to_dict()
    invariant_results = run_invariant_checks(first)
    rows = _sample_rows(frame)
    sample = frame.iloc[rows].reset_index(drop=True)
    input_start = time.perf_counter()
    records = sample[["goal", "agents", "messages", "artifacts",
                      "shared_state", "topology"]].to_dict("records")
    data = build_dataset(records)
    input_sec = time.perf_counter() - input_start
    assert int(data.event_offsets[-1]) == sum(
        1 + len(_EXP16._json_field(r["messages"], [])) +
        len(_EXP16._json_field(r["artifacts"], [])) +
        len(_EXP16._json_field(r["shared_state"], [])) for r in records)

    labels = np.asarray([LABEL_TO_ID[x] for x in sample.label.astype(str)],
                        dtype=np.int64)
    turns = sample.fault_turn.to_numpy(dtype=np.int64)
    device = torch.device("cpu")
    torch.manual_seed(42)
    model = FaultFirstReferee().to(device)
    model.eval()
    same_batch = data.batch(np.asarray([0, 1], dtype=np.int64))
    with torch.no_grad():
        first_out = model(same_batch)
        second_out = model(same_batch)
    assert all(torch.equal(a, b) for a, b in zip(first_out, second_out))

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(sample))
    fit_times = []
    event_batches = []
    for step in range(8):
        batch_rows = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        batch = data.batch(batch_rows)
        event_batches.append(int(sum(batch["lengths"])))
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = _batch_loss(model, batch, batch_rows, labels, turns, device)
        loss.backward()
        optimizer.step()
        fit_times.append(time.perf_counter() - started)
    del optimizer

    predict_start = time.perf_counter()
    probabilities, predicted_turns = _predict(
        model, data, np.arange(len(sample)), device)
    inference_sec = time.perf_counter() - predict_start
    assert probabilities.shape == (len(sample), len(LABEL_TO_ID))
    assert predicted_turns.shape == (len(sample),)
    assert np.all(np.isfinite(probabilities))
    assert np.all((predicted_turns >= -1))

    peak_rss_mb = _peak_rss_mb()
    sample_input_bytes = sum(getattr(data, name).nbytes for name in
                             ("token_ids", "token_offsets", "event_offsets",
                              "ref_ids", "ref_offsets", "categories", "metadata"))
    per_run_bytes = sample_input_bytes / len(sample)
    full_runs = 14000
    estimated_input_bytes = per_run_bytes * full_runs
    avg_batch_sec = float(np.mean(fit_times))
    train_updates = math.ceil(10000 / BATCH_SIZE) * EPOCHS
    cv_updates = 3 * math.ceil(math.ceil(10000 * 2 / 3) / BATCH_SIZE) * EPOCHS
    train_est = train_updates * avg_batch_sec
    cv_est = cv_updates * avg_batch_sec
    infer_batch_sec = inference_sec / math.ceil(len(sample) / BATCH_SIZE)
    infer_est = math.ceil(4000 / BATCH_SIZE) * infer_batch_sec
    input_est = input_sec / len(sample) * full_runs
    candidate_est = train_est + infer_est + input_est
    full_est = 593.4 + candidate_est
    conservative_est = 593.4 + candidate_est * 1.25
    result = {
        "rows_benchmarked": len(sample),
        "events_benchmarked": int(data.event_offsets[-1]),
        "tokens_benchmarked": int(len(data.token_ids)),
        "input_build_seconds": round(input_sec, 3),
        "encoder_plus_gru_train_batches": len(fit_times),
        "train_batch_seconds": [round(x, 4) for x in fit_times],
        "events_per_train_batch": event_batches,
        "train_batch_seconds_mean": round(avg_batch_sec, 4),
        "inference_seconds": round(inference_sec, 3),
        "inference_seconds_per_batch": round(infer_batch_sec, 4),
        "estimated_candidate_train_10000_seconds": round(train_est, 1),
        "estimated_candidate_infer_4000_seconds": round(infer_est, 1),
        "estimated_input_build_14000_seconds": round(input_est, 1),
        "estimated_exp15_plus_candidate_seconds": round(full_est, 1),
        "conservative_estimate_with_25pct_margin_seconds": round(
            conservative_est, 1),
        "estimated_three_fold_candidate_cv_seconds": round(cv_est, 1),
        "estimated_input_arrays_14000_bytes": int(estimated_input_bytes),
        "observed_process_rss_mb": (None if peak_rss_mb is None else
                                     round(peak_rss_mb, 1)),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": __import__("os").cpu_count(),
        "invariant_checks": invariant_results,
        "deterministic_cpu_inference": "PASS",
        "single_model_train_plus_infer_estimate_under_1800":
            conservative_est <= 1800,
        "note": "CPU estimate is from this local 12-logical-CPU environment; "
                "no DSWorks instance or GPU timing was measured.",
    }
    out = HERE / "diagnostics"
    out.mkdir(exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(result, indent=2),
                                         encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run_preflight()
