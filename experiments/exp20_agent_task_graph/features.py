"""Fixed, run-local temporal links for Exp20; IDs are join keys, not features."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from experiments.exp13_wait_dependency_graph.features import wait_events


FEATURE_NAMES = (
    "g_known_assignments",
    "g_assignments_without_later_result",
    "g_unresolved_assignment_rate",
    "g_executor_response_without_result",
    "g_response_without_result_rate",
    "g_other_task_activity_before_result",
    "g_other_task_activity_rate",
    "g_delivery_delay_mean_turns",
    "g_delivery_delay_p90_turns",
    "g_delivery_delay_max_rel",
    "g_peak_open_assignments",
    "g_open_assignment_area_rel",
    "g_max_open_age_rel",
    "g_repeat_assignment_before_result",
    "g_repeat_before_result_rate",
    "g_other_executor_deliveries",
    "g_other_executor_delivery_rate",
    "g_wait_task_links",
    "g_wait_resolved_by_named_agent_rate",
    "g_wait_to_delivery_delay_mean_rel",
    "g_state_multiwriter_keys",
    "g_state_cross_writer_events",
    "g_state_cross_writer_overrides",
    "g_state_cross_writer_value_changes",
)
N_FEATURES = len(FEATURE_NAMES)

_EVENT_RANK = {"message": 0, "state": 1, "artifact": 2}
_RESPONSE_TYPES = {"inform", "status", "state_update", "final"}
_ACTIVITY_TYPES = _RESPONSE_TYPES | {"handoff"}


def _key(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().casefold()
    return value or None


def _time(value: Any) -> Optional[int]:
    try:
        t = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and value != t:
        return None
    return t


def _refs(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(x) for x in value if x is not None and str(x)]


def _ordered_events(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    for kind, field in (("message", "messages"), ("state", "shared_state"),
                        ("artifact", "artifacts")):
        for source_index, obj in enumerate(run.get(field) or []):
            if not isinstance(obj, dict):
                continue
            t = _time(obj.get("t"))
            if t is None:
                continue
            events.append({"t": t, "rank": _EVENT_RANK[kind],
                           "source_index": source_index, "kind": kind,
                           "obj": obj})
    events.sort(key=lambda e: (e["t"], e["rank"], e["source_index"]))
    for seq, event in enumerate(events):
        event["seq"] = seq
    return events


def _canonical_value(obj: Dict[str, Any]) -> Optional[str]:
    if "value" not in obj:
        return None
    return json.dumps(obj["value"], sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def extract_features(run: Dict[str, Any]) -> Dict[str, float]:
    events = _ordered_events(run)
    out = dict.fromkeys(FEATURE_NAMES, 0.0)
    if not events:
        return out

    artifacts = run.get("artifacts") or []
    artifact_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for art in artifacts:
        if isinstance(art, dict) and art.get("id") is not None:
            artifact_by_id[str(art["id"])].append(art)

    # A ref may supply an omitted artifact subtask only when its explicit
    # intent links that unique artifact to one unambiguous task.
    ref_tasks: Dict[str, set] = defaultdict(set)
    for msg in run.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("type") != "handoff":
            continue
        task = _key(msg.get("intent"))
        if task:
            for ref in _refs(msg.get("refs")):
                if len(artifact_by_id.get(ref, ())) == 1:
                    ref_tasks[ref].add(task)

    art_task: Dict[int, Optional[str]] = {}
    for art in artifacts:
        if not isinstance(art, dict):
            continue
        task = _key(art.get("subtask"))
        if task is None and art.get("id") is not None:
            tasks = ref_tasks.get(str(art["id"]), set())
            task = next(iter(tasks)) if len(tasks) == 1 else None
        art_task[id(art)] = task

    assignment_by_seq = {}
    assignments = []
    deliveries_by_seq = defaultdict(list)
    delivery_index = []
    for event in events:
        obj = event["obj"]
        if event["kind"] == "message":
            if obj.get("type") == "handoff":
                task = _key(obj.get("intent"))
                refs = _refs(obj.get("refs"))
                sender = str(obj.get("from") or "")
                receiver = str(obj.get("to") or "")
                if not refs and task and sender and receiver:
                    item = {"task": task, "assignee": receiver,
                            "t": event["t"], "seq": event["seq"],
                            "result": None, "response": False,
                            "other_activity": False}
                    assignments.append(item)
                    assignment_by_seq[event["seq"]] = item
                elif refs:
                    resolved = []
                    for ref in refs:
                        rows = artifact_by_id.get(ref, ())
                        if len(rows) != 1:
                            resolved = []
                            break
                        resolved.append(rows[0])
                    ref_task_keys = {art_task.get(id(art)) for art in resolved}
                    ref_task_keys.discard(None)
                    if task is None and len(resolved) == len(refs) and len(ref_task_keys) == 1:
                        task = next(iter(ref_task_keys))
                    if task and (not ref_task_keys or ref_task_keys == {task}):
                        known_actors = [str(art.get("by")) for art in resolved
                                        if art.get("by") is not None]
                        actor = (known_actors[0]
                                 if len(resolved) == len(refs)
                                 and len(known_actors) == len(refs)
                                 and len(set(known_actors)) == 1
                                 else sender or None)
                        delivery_index.append((task, event["seq"], event["t"], actor))
        elif event["kind"] == "artifact":
            task = art_task.get(id(obj))
            if task:
                actor = str(obj.get("by")) if obj.get("by") is not None else None
                delivery_index.append((task, event["seq"], event["t"], actor))

    # Equivalent artifact and ref-bearing handoff evidence at the same source
    # time and actor is one result edge.
    seen_deliveries = set()
    for task, seq, t, actor in sorted(delivery_index, key=lambda x: (x[1], x[0])):
        identity = (task, t, actor)
        if identity in seen_deliveries:
            continue
        seen_deliveries.add(identity)
        deliveries_by_seq[seq].append((task, t, actor))

    pending: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    peak_open = repeat_count = 0
    area = max_age = 0.0
    open_count = 0
    prev_t = events[0]["t"]
    for event in events:
        t = event["t"]
        area += open_count * max(0, t - prev_t)
        prev_t = t
        assignment = assignment_by_seq.get(event["seq"])
        if assignment:
            task = assignment["task"]
            if pending[task]:
                repeat_count += 1
            pending[task].append(assignment)
            open_count += 1
            peak_open = max(peak_open, open_count)
        for task, result_t, actor in deliveries_by_seq.get(event["seq"], ()):
            open_items = pending.pop(task, [])
            open_count -= len(open_items)
            for item in open_items:
                item["result"] = (event["seq"], result_t, actor)
                max_age = max(max_age, float(max(0, result_t - item["t"])))
    end_t = events[-1]["t"]
    for rows in pending.values():
        for item in rows:
            max_age = max(max_age, float(max(0, end_t - item["t"])))

    span = float(max(1, end_t - events[0]["t"]))
    resolved = [a for a in assignments if a["result"] is not None]
    unresolved = [a for a in assignments if a["result"] is None]
    delays = [float(a["result"][1] - a["t"]) for a in resolved]
    unresolved_response = []
    unresolved_other = []
    other_actor = 0
    for assignment in assignments:
        stop = assignment["result"][0] if assignment["result"] else len(events)
        agent_events = [e for e in events
                        if e["kind"] == "message"
                        and str(e["obj"].get("from") or "") == assignment["assignee"]
                        and e["seq"] > assignment["seq"] and e["seq"] < stop]
        for event in agent_events:
            msg = event["obj"]
            msg_task = _key(msg.get("intent"))
            msg_type = msg.get("type")
            if msg_type in _RESPONSE_TYPES and msg_task == assignment["task"]:
                assignment["response"] = True
            if msg_type in _ACTIVITY_TYPES and msg_task and msg_task != assignment["task"]:
                assignment["other_activity"] = True
            if assignment["response"] and assignment["other_activity"]:
                break
        if assignment["response"] and assignment["result"] is None:
            unresolved_response.append(assignment)
        if assignment["other_activity"] and assignment["result"] is None:
            unresolved_other.append(assignment)
        result_actor = assignment["result"][2] if assignment["result"] else None
        if result_actor and result_actor != assignment["assignee"]:
            other_actor += 1

    waits = [w for w in wait_events(run) if w[2] and w[1]]
    resolved_waits = []
    # A delivery can resolve a named wait only after it occurred in the fixed
    # event sequence and only for the exact Subtask key.
    for sender, awaited, task, wait_t in waits:
        task_key = _key(task)
        wait_event = next((e for e in events if e["kind"] == "message"
                           and e["t"] == wait_t
                           and str(e["obj"].get("from") or "") == sender), None)
        if wait_event is None or not task_key:
            continue
        later = [(seq, t, actor) for seq, items in deliveries_by_seq.items()
                 if seq > wait_event["seq"] for key, t, actor in items
                 if key == task_key and actor == awaited]
        if later:
            seq, delivery_t, _ = min(later, key=lambda x: (x[1], x[0]))
            resolved_waits.append((delivery_t - wait_t, seq))

    state_by_key: Dict[str, set] = defaultdict(set)
    last_state = {}
    cross_writer = cross_override = cross_value_change = 0
    for event in events:
        if event["kind"] != "state":
            continue
        obj = event["obj"]
        key = _key(obj.get("key"))
        writer = str(obj.get("agent") or "")
        if not key or not writer:
            continue
        state_by_key[key].add(writer)
        previous = last_state.get(key)
        value = _canonical_value(obj)
        if previous is not None and previous[0] != writer:
            cross_writer += 1
            if obj.get("op") == "override":
                cross_override += 1
            if value is not None and previous[1] is not None and value != previous[1]:
                cross_value_change += 1
        last_state[key] = (writer, value)

    wait_delay_rel = [delay / span for delay, _ in resolved_waits]
    out.update({
        "g_known_assignments": float(len(assignments)),
        "g_assignments_without_later_result": float(len(unresolved)),
        "g_unresolved_assignment_rate": len(unresolved) / len(assignments) if assignments else 0.0,
        "g_executor_response_without_result": float(len(unresolved_response)),
        "g_response_without_result_rate": len(unresolved_response) / sum(
            1 for a in assignments if a["response"]) if any(a["response"] for a in assignments) else 0.0,
        "g_other_task_activity_before_result": float(len(unresolved_other)),
        "g_other_task_activity_rate": len(unresolved_other) / len(assignments) if assignments else 0.0,
        "g_delivery_delay_mean_turns": float(np.mean(delays)) if delays else 0.0,
        "g_delivery_delay_p90_turns": float(np.sort(delays)[int(np.ceil(0.9 * len(delays))) - 1]) if delays else 0.0,
        "g_delivery_delay_max_rel": max(delays) / span if delays else 0.0,
        "g_peak_open_assignments": float(peak_open),
        "g_open_assignment_area_rel": area / (len(assignments) * span) if assignments else 0.0,
        "g_max_open_age_rel": max_age / span,
        "g_repeat_assignment_before_result": float(repeat_count),
        "g_repeat_before_result_rate": repeat_count / len(assignments) if assignments else 0.0,
        "g_other_executor_deliveries": float(other_actor),
        "g_other_executor_delivery_rate": other_actor / len(resolved) if resolved else 0.0,
        "g_wait_task_links": float(len(waits)),
        "g_wait_resolved_by_named_agent_rate": len(resolved_waits) / len(waits) if waits else 0.0,
        "g_wait_to_delivery_delay_mean_rel": float(np.mean(wait_delay_rel)) if wait_delay_rel else 0.0,
        "g_state_multiwriter_keys": float(sum(len(writers) > 1 for writers in state_by_key.values())),
        "g_state_cross_writer_events": float(cross_writer),
        "g_state_cross_writer_overrides": float(cross_override),
        "g_state_cross_writer_value_changes": float(cross_value_change),
    })
    return out


def build_matrix(runs: Iterable[Dict[str, Any]]) -> np.ndarray:
    rows = [[features[name] for name in FEATURE_NAMES]
            for features in map(extract_features, runs)]
    matrix = np.asarray(rows, dtype=np.float64).reshape((-1, N_FEATURES))
    if not np.isfinite(matrix).all():
        raise ValueError("Exp20 graph feature block contains non-finite values")
    return matrix
