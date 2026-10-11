"""Full-run typed event inputs for Exp21; targets never enter this module."""

from __future__ import annotations

import array
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP16_PATH = ROOT / "experiments" / "exp16_semantic_sequence" / "run_experiment.py"
WAIT_PATH = ROOT / "experiments" / "exp13_wait_dependency_graph" / "features.py"
EVENT_ORDER = {"message": 0, "state": 1, "artifact": 2}
EVENT_KIND_ID = {"goal": 1, "message": 2, "artifact": 3, "state": 4}
TOPOLOGY_IDS = {
    "star": 1, "pipeline": 2, "hierarchical": 3, "mesh": 4,
    "blackboard": 5, "chain": 6,
}
TOKEN_BUCKETS = 1 << 15
MAX_AGENTS = 32
MAX_TASKS = 256
MAX_ARTIFACTS = 128

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixed input helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXP16 = _load_module("exp21_exp16_inputs", EXP16_PATH)
_WAIT = _load_module("exp21_exp13_wait_parser", WAIT_PATH)


def _hash_token(token: str, cache: dict[str, int]) -> int:
    value = cache.get(token)
    if value is None:
        raw = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(raw, "little") % TOKEN_BUCKETS + 1
        cache[token] = value
    return value


def _observable(record):
    """Whitelist observable input columns; drop all targets and row IDs."""
    return {key: record.get(key) for key in
            ("goal", "agents", "messages", "artifacts", "shared_state",
             "topology")}


def _slot(mapping: dict[str, int], raw, maximum: int, name: str) -> int:
    if raw is None:
        return 0
    key = str(raw).strip()
    if not key:
        return 0
    if key not in mapping:
        if len(mapping) >= maximum:
            raise ValueError(f"run exceeds supported {name} slots ({maximum})")
        mapping[key] = len(mapping) + 1
    return mapping[key]


@dataclass(frozen=True)
class EventDataset:
    """Flat, row-aligned input arrays; labels and fault turns live elsewhere."""

    token_ids: np.ndarray
    token_offsets: np.ndarray
    event_offsets: np.ndarray
    ref_ids: np.ndarray
    ref_offsets: np.ndarray
    categories: np.ndarray
    metadata: np.ndarray
    message_positions: tuple[tuple[int, ...], ...]
    message_turns: tuple[tuple[int, ...], ...]

    def __len__(self):
        return len(self.event_offsets) - 1

    def batch(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        event_lengths = self.event_offsets[rows + 1] - self.event_offsets[rows]
        event_indices = np.concatenate([
            np.arange(self.event_offsets[row], self.event_offsets[row + 1],
                      dtype=np.int64) for row in rows
        ])
        token_chunks = []
        token_batch_offsets = []
        token_count = 0
        ref_chunks = []
        ref_batch_offsets = []
        ref_count = 0
        for event_index in event_indices:
            lo, hi = self.token_offsets[event_index:event_index + 2]
            chunk = self.token_ids[lo:hi]
            if len(chunk) == 0:
                raise AssertionError("every event must have at least one token")
            token_batch_offsets.append(token_count)
            token_chunks.append(chunk)
            token_count += len(chunk)
            rlo, rhi = self.ref_offsets[event_index:event_index + 2]
            rchunk = self.ref_ids[rlo:rhi]
            ref_batch_offsets.append(ref_count)
            ref_chunks.append(rchunk if len(rchunk) else np.zeros(1, dtype=np.uint16))
            ref_count += len(ref_chunks[-1])
        return {
            "token_ids": np.concatenate(token_chunks).astype(np.int64, copy=False),
            "token_offsets": np.asarray(token_batch_offsets, dtype=np.int64),
            "ref_ids": np.concatenate(ref_chunks).astype(np.int64, copy=False),
            "ref_offsets": np.asarray(ref_batch_offsets, dtype=np.int64),
            "categories": self.categories[event_indices].astype(np.int64),
            "metadata": self.metadata[event_indices],
            "lengths": event_lengths.astype(np.int64),
            "message_positions": tuple(self.message_positions[row] for row in rows),
            "message_turns": tuple(self.message_turns[row] for row in rows),
        }


def build_row(record, token_cache=None):
    """Return exactly goal + all timestamp-ordered observable events."""
    token_cache = token_cache if token_cache is not None else {}
    obs = _observable(record)
    exp16 = _EXP16.prepare_run(obs)
    raw_agents = _EXP16._json_field(obs.get("agents"), [])
    messages = _EXP16._json_field(obs.get("messages"), [])
    artifacts = _EXP16._json_field(obs.get("artifacts"), [])
    states = _EXP16._json_field(obs.get("shared_state"), [])
    if not all(isinstance(value, list) for value in
               (raw_agents, messages, artifacts, states)):
        raise ValueError("agents, messages, artifacts, and shared_state must be lists")
    roles = {str(agent.get("id", "")): str(agent.get("role", "unknown"))
             for agent in raw_agents if isinstance(agent, dict)}
    agent_slots = {}
    for agent in raw_agents:
        if isinstance(agent, dict):
            _slot(agent_slots, agent.get("id"), MAX_AGENTS, "agent")
    task_slots = {}
    artifact_slots = {}
    wait_by_turn = {turn: (sender, awaited, subtask)
                    for sender, awaited, subtask, turn in
                    _WAIT.wait_events({"messages": messages})}
    topology = _EXP16._json_field(obs.get("topology"), {})
    topology_name = (topology.get("type", "unknown")
                     if isinstance(topology, dict) else "unknown")
    topology_id = TOPOLOGY_IDS.get(str(topology_name).casefold(), 0)
    role_context = " ".join("role=" + " ".join(role.split()).casefold()
                           for role in roles.values())
    exp16["event_texts"][0] += (
        " run_context topology=" + str(topology_name).casefold() +
        " team_roles=" + role_context)

    events = []
    for kind, values in (("message", messages), ("state", states),
                         ("artifact", artifacts)):
        for source_index, event in enumerate(values):
            if not isinstance(event, dict):
                raise ValueError(f"{kind} event {source_index} is not an object")
            turn = _EXP16._event_turn(event)
            events.append((turn is None, turn if turn is not None else 0,
                           EVENT_ORDER[kind], source_index, kind, event))
    events.sort(key=lambda item: item[:4])
    if len(exp16["event_texts"]) != len(events) + 1:
        raise AssertionError("semantic texts do not retain every event")

    token_values = []
    token_offsets = [0]
    ref_values = []
    ref_offsets = [0]
    categories = [[EVENT_KIND_ID["goal"], 0, 0, 0, 0, 0, 0, topology_id]]
    metadata = [exp16["event_meta"][0]]
    message_positions = []
    message_turns = []
    artifact_count = state_count = 0

    def append_tokens(text):
        tokens = _WORD_RE.findall(str(text).casefold()) or ["[empty]"]
        token_values.extend(_hash_token(token, token_cache) for token in tokens)
        token_offsets.append(len(token_values))

    append_tokens(exp16["event_texts"][0])
    ref_values.append(0)
    ref_offsets.append(1)
    for position, (_, _, _, source_index, kind, event) in enumerate(events, 1):
        text = exp16["event_texts"][position]
        sender = receiver = actor = task = artifact = wait_target = 0
        refs = []
        if kind == "message":
            raw_sender, raw_receiver = event.get("from"), event.get("to")
            sender = _slot(agent_slots, raw_sender, MAX_AGENTS, "agent")
            receiver = _slot(agent_slots, raw_receiver, MAX_AGENTS, "agent")
            actor = sender
            task = _slot(task_slots, event.get("intent"), MAX_TASKS, "task")
            raw_refs = event.get("refs") or []
            if not isinstance(raw_refs, list):
                raw_refs = [raw_refs]
            refs = [_slot(artifact_slots, ref, MAX_ARTIFACTS, "artifact")
                    for ref in raw_refs]
            turn = int(event["t"])
            if turn != source_index:
                raise ValueError("message t no longer matches its source turn")
            if turn in wait_by_turn:
                wait_sender, awaited, wait_task = wait_by_turn[turn]
                wait_target = _slot(agent_slots, awaited, MAX_AGENTS, "agent")
                if str(wait_sender) != str(raw_sender):
                    raise AssertionError("wait parser sender differs from message")
                if wait_task:
                    task = task or _slot(task_slots, wait_task, MAX_TASKS, "task")
            message_positions.append(position)
            message_turns.append(turn)
        elif kind == "artifact":
            raw_actor = event.get("by", event.get("agent"))
            actor = _slot(agent_slots, raw_actor, MAX_AGENTS, "agent")
            task = _slot(task_slots, event.get("subtask"), MAX_TASKS, "task")
            artifact = _slot(artifact_slots, event.get("id"),
                             MAX_ARTIFACTS, "artifact")
            artifact_count += 1
        else:
            raw_actor = event.get("agent", event.get("by"))
            actor = _slot(agent_slots, raw_actor, MAX_AGENTS, "agent")
            state_count += 1
        if refs:
            ref_values.extend(refs)
        else:
            ref_values.append(0)
        ref_offsets.append(len(ref_values))
        categories.append([EVENT_KIND_ID[kind], sender, receiver, actor, task,
                           artifact, wait_target, topology_id])
        meta = exp16["event_meta"][position].copy()
        if kind == "artifact":
            meta[10] = np.log1p(artifact_count)
        elif kind == "state":
            meta[11] = np.log1p(state_count)
        metadata.append(meta)
        append_tokens(text)

    return {
        "token_ids": token_values,
        "token_offsets": token_offsets,
        "ref_ids": ref_values,
        "ref_offsets": ref_offsets,
        "categories": categories,
        "metadata": metadata,
        "message_positions": tuple(message_positions),
        "message_turns": tuple(message_turns),
    }


def build_dataset(records):
    token_cache = {}
    all_tokens = array.array("I")
    token_offsets = [0]
    all_refs = array.array("H")
    ref_offsets = [0]
    categories = []
    metadata = []
    event_offsets = [0]
    message_positions = []
    message_turns = []
    for record in records:
        row = build_row(record, token_cache)
        token_base = len(all_tokens)
        ref_base = len(all_refs)
        all_tokens.extend(row["token_ids"])
        token_offsets.extend(token_base + offset
                             for offset in row["token_offsets"][1:])
        all_refs.extend(row["ref_ids"])
        ref_offsets.extend(ref_base + offset
                           for offset in row["ref_offsets"][1:])
        categories.extend(row["categories"])
        metadata.extend(row["metadata"])
        event_offsets.append(len(categories))
        message_positions.append(row["message_positions"])
        message_turns.append(row["message_turns"])
    if len(token_offsets) != len(categories) + 1 or \
            len(ref_offsets) != len(categories) + 1:
        raise AssertionError("ragged token/reference offsets lost an event")
    return EventDataset(
        token_ids=np.frombuffer(all_tokens, dtype=np.uint32).copy(),
        token_offsets=np.asarray(token_offsets, dtype=np.int64),
        event_offsets=np.asarray(event_offsets, dtype=np.int64),
        ref_ids=np.frombuffer(all_refs, dtype=np.uint16).copy(),
        ref_offsets=np.asarray(ref_offsets, dtype=np.int64),
        categories=np.asarray(categories, dtype=np.uint16),
        metadata=np.asarray(metadata, dtype=np.float32),
        message_positions=tuple(message_positions),
        message_turns=tuple(message_turns),
    )
