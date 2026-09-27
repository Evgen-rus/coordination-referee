"""Experiment 03: length-normalised + positional + density features.

Added ON TOP of the official 122 baseline features (never replacing them in the
main variant B), so the effect of the new signal is measured in isolation.

Design rules
------------
* Every added feature has a semantically meaningful denominator drawn from:
  n_messages, n_agents, n_assign, n_intents, n_artifacts, n_state_writes,
  g_nodes, n_senders, max_turn.  Ratios are NOT generated mechanically for
  every column.
* Features that the baseline already stores as a share/rate are skipped
  (e.g. ``share_status``, ``arts_per_msg``, ``state_per_msg``,
  ``unanswered_ratio``, ``unanswered_ratio``, ``dup_text_ratio``,
  ``intent_max_share``, ``state_override_ratio``, ``g_density``,
  ``g_pingpong_share``, ``coverage``).
* Absolute-scale features the baseline has but never normalises ARE covered:
  reply delays, repeat counters, artifact counters, shared-state counters,
  graph counters.
* Positional features are relative positions in [0, 1] of protocol events.
  They never read ``label`` or ``fault_turn``.
* Division by zero yields 0.0.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "baseline"))

from features import _norm, _tokens   # noqa: E402  (official baseline, read-only)

# ---------------------------------------------------------------------------
# which baseline columns are pure absolute-scale "volume" features.
# Variant C drops these and keeps the new normalised ones.
# ---------------------------------------------------------------------------
VOLUME_BASELINE = [
    "n_messages", "n_agents", "n_artifacts", "n_state_writes", "msgs_per_agent",
    "arts_per_msg", "state_per_msg", "n_tools_declared",
    "type_handoff", "type_inform", "type_status", "type_state_update",
    "type_final", "type_system",
    "n_senders", "sender_entropy", "sender_max_share",
    "dup_text_max", "norm_dup_max", "max_consecutive_repeat",
    "max_sender_repeat_streak", "n_intents", "intent_entropy", "intent_max_count",
    "kw_wait", "kw_conflict", "kw_repeat", "kw_drift", "kw_recovery", "kw_stop",
    "kw_done",
    "goal_ov_mean", "goal_ov_head", "goal_ov_tail", "goal_ov_delta",
    "goal_ov_zero_ratio", "goal_ov_max", "new_intents_in_tail",
    "n_assign", "n_result", "assign_result_gap", "unanswered_assign",
    "undelivered_assign", "reply_delay_mean", "reply_delay_max",
    "reassigned_intents", "max_assign_per_intent", "intents_with_multiple_owners",
    "n_unique_hashes", "dup_hash_pairs", "dup_hash_max", "max_arts_per_subtask",
    "subtasks_with_2plus_arts", "art_partial", "art_copy", "art_types",
    "state_keys", "state_overrides", "state_max_writes_per_key",
    "state_max_distinct_per_key", "state_flips", "state_alternating_flips",
    "state_keys_3plus",
    "topo_edges", "silent_agent_ratio",
    "g_nodes", "g_edges", "g_self_loops", "g_two_cycles", "g_triangles",
    "g_max_out_deg", "g_mean_out_deg", "g_max_in_deg", "g_largest_scc",
    "g_n_pairs", "g_max_pair", "g_pingpong", "has_final_msg", "has_system_msg",
]

# ---------------------------------------------------------------------------
NEW_FEATURE_NAMES = [
    # A. repetition / loop, normalised (runaway_loop leans on absolute counts)
    "nz_norm_dup_max_per_msg", "nz_dup_text_max_per_msg",
    "nz_max_consecutive_repeat_per_msg", "nz_max_sender_streak_per_msg",
    "nz_max_sender_streak_per_sender", "nz_intents_per_agent",
    "nz_dup_hash_pairs_per_msg",
    # B. handoff accounting against the number of assignments
    "nz_assign_per_msg", "nz_result_per_msg", "nz_result_per_assign",
    "nz_assign_gap_per_assign", "nz_reply_delay_mean_per_msg",
    "nz_reply_delay_max_per_msg", "nz_unanswered_per_msg",
    "nz_undelivered_per_msg", "nz_reassign_per_msg",
    # C. intent duplication, normalised by the number of intents
    "nz_max_assign_per_intent_per_intent", "nz_multi_owner_per_intent",
    "nz_intent_max_count_per_msg", "nz_intent_entropy_per_intent",
    "nz_new_intents_tail_per_intent",
    # D. artifacts, normalised by the number of artifacts
    "nz_unique_hashes_per_art", "nz_max_arts_per_subtask_per_art",
    "nz_art_partial_per_art", "nz_art_copy_per_art", "nz_art_types_per_art",
    "nz_subtasks_2plus_per_art", "nz_artifacts_per_assign",
    # E. shared state, normalised (state_alternating_flips is a top-1 feature)
    "nz_state_flips_per_write", "nz_state_altflips_per_write",
    "nz_state_max_writes_per_key_per_write", "nz_state_max_distinct_per_key_per_write",
    "nz_state_keys_per_agent", "nz_state_keys_3plus_per_write",
    "nz_state_flips_per_msg", "nz_state_altflips_per_msg",
    "nz_state_overrides_per_key", "nz_state_flips_per_key",
    # F. graph, normalised by messages and by nodes
    "nz_g_two_cycles_per_msg", "nz_g_pingpong_per_msg", "nz_g_self_loops_per_msg",
    "nz_g_edges_per_node", "nz_g_max_pair_per_msg", "nz_g_max_out_deg_per_node",
    "nz_g_max_in_deg_per_node", "nz_g_triangles_per_node",
    "nz_g_pingpong_per_pair", "nz_g_largest_scc_per_node",
    # G. agent / tool density
    "nz_tools_per_agent", "nz_senders_per_agent", "nz_n_pairs_per_node",
    "nz_msgs_per_sender",
    # H. positional features: relative position of protocol events in [0,1]
    "pos_first_assign", "pos_first_delivery", "pos_last_assign",
    "pos_last_delivery", "pos_streak_start", "pos_first_override",
    "pos_first_state_write", "pos_goal_zero_run", "pos_first_status_run",
    "pos_first_conflict_flip", "pos_first_ack",
]
N_NEW = len(NEW_FEATURE_NAMES)


def _d(a, b) -> float:
    """Safe division; 0 denominator -> 0.0."""
    try:
        return float(a) / float(b) if b else 0.0
    except (TypeError, ValueError):
        return 0.0


def _r(x) -> float:
    """Clamp to a finite float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return v


def extra_features(run: Dict[str, Any], base: Dict[str, float]) -> Dict[str, float]:
    """Return the new normalised / positional features for one run.

    ``base`` is the official 122-feature dict for the same run; a few
    aggregate counters are reused from it to avoid recomputation.
    """
    msgs: List[Dict] = run.get("messages") or []
    n = len(msgs)
    agents = run.get("agents") or []
    arts = run.get("artifacts") or []
    state = run.get("shared_state") or []
    f: Dict[str, float] = {}

    n_ag = max(1, len(agents))
    n_art = len(arts)
    n_st = len(state)
    n_assign = base.get("n_assign", 0.0)
    n_int = max(1, base.get("n_intents", 0.0))
    n_snd = max(1, base.get("n_senders", 1.0))
    g_nodes = max(1.0, base.get("g_nodes", 1.0))
    g_pairs = max(1.0, base.get("g_n_pairs", 1.0))
    max_turn = max(1.0, float(n - 1))
    active = max(1, base.get("n_senders", 1.0))

    # ---- A. repetition ----
    f["nz_norm_dup_max_per_msg"] = _d(base.get("norm_dup_max", 0.0), n)
    f["nz_dup_text_max_per_msg"] = _d(base.get("dup_text_max", 0.0), n)
    f["nz_max_consecutive_repeat_per_msg"] = _d(base.get("max_consecutive_repeat", 0.0), n)
    f["nz_max_sender_streak_per_msg"] = _d(base.get("max_sender_repeat_streak", 0.0), n)
    f["nz_max_sender_streak_per_sender"] = _d(base.get("max_sender_repeat_streak", 0.0), n_snd)
    f["nz_intents_per_agent"] = _d(base.get("n_intents", 0.0), n_ag)
    f["nz_dup_hash_pairs_per_msg"] = _d(base.get("dup_hash_pairs", 0.0), n)

    # ---- B. handoff accounting ----
    f["nz_assign_per_msg"] = _d(n_assign, n)
    f["nz_result_per_msg"] = _d(base.get("n_result", 0.0), n)
    f["nz_result_per_assign"] = _d(base.get("n_result", 0.0), n_assign)
    f["nz_assign_gap_per_assign"] = _d(base.get("assign_result_gap", 0.0), n_assign)
    f["nz_reply_delay_mean_per_msg"] = _d(base.get("reply_delay_mean", 0.0), n)
    f["nz_reply_delay_max_per_msg"] = _d(base.get("reply_delay_max", 0.0), n)
    f["nz_unanswered_per_msg"] = _d(base.get("unanswered_assign", 0.0), n)
    f["nz_undelivered_per_msg"] = _d(base.get("undelivered_assign", 0.0), n)
    f["nz_reassign_per_msg"] = _d(base.get("reassigned_intents", 0.0), n)

    # ---- C. intents ----
    f["nz_max_assign_per_intent_per_intent"] = _d(base.get("max_assign_per_intent", 0.0), n_int)
    f["nz_multi_owner_per_intent"] = _d(base.get("intents_with_multiple_owners", 0.0), n_int)
    f["nz_intent_max_count_per_msg"] = _d(base.get("intent_max_count", 0.0), n)
    f["nz_intent_entropy_per_intent"] = _d(base.get("intent_entropy", 0.0), n_int)
    f["nz_new_intents_tail_per_intent"] = _d(base.get("new_intents_in_tail", 0.0), n_int)

    # ---- D. artifacts ----
    f["nz_unique_hashes_per_art"] = _d(base.get("n_unique_hashes", 0.0), n_art)
    f["nz_max_arts_per_subtask_per_art"] = _d(base.get("max_arts_per_subtask", 0.0), n_art)
    f["nz_art_partial_per_art"] = _d(base.get("art_partial", 0.0), n_art)
    f["nz_art_copy_per_art"] = _d(base.get("art_copy", 0.0), n_art)
    f["nz_art_types_per_art"] = _d(base.get("art_types", 0.0), n_art)
    f["nz_subtasks_2plus_per_art"] = _d(base.get("subtasks_with_2plus_arts", 0.0), n_art)
    f["nz_artifacts_per_assign"] = _d(n_art, n_assign)

    # ---- E. shared state ----
    n_keys = max(1.0, base.get("state_keys", 1.0))
    f["nz_state_flips_per_write"] = _d(base.get("state_flips", 0.0), n_st)
    f["nz_state_altflips_per_write"] = _d(base.get("state_alternating_flips", 0.0), n_st)
    f["nz_state_max_writes_per_key_per_write"] = _d(
        base.get("state_max_writes_per_key", 0.0), n_st)
    f["nz_state_max_distinct_per_key_per_write"] = _d(
        base.get("state_max_distinct_per_key", 0.0), n_st)
    f["nz_state_keys_per_agent"] = _d(base.get("state_keys", 0.0), n_ag)
    f["nz_state_keys_3plus_per_write"] = _d(base.get("state_keys_3plus", 0.0), n_st)
    f["nz_state_flips_per_msg"] = _d(base.get("state_flips", 0.0), n)
    f["nz_state_altflips_per_msg"] = _d(base.get("state_alternating_flips", 0.0), n)
    f["nz_state_overrides_per_key"] = _d(base.get("state_overrides", 0.0), n_keys)
    f["nz_state_flips_per_key"] = _d(base.get("state_flips", 0.0), n_keys)

    # ---- F. graph ----
    f["nz_g_two_cycles_per_msg"] = _d(base.get("g_two_cycles", 0.0), n)
    f["nz_g_pingpong_per_msg"] = _d(base.get("g_pingpong", 0.0), n)
    f["nz_g_self_loops_per_msg"] = _d(base.get("g_self_loops", 0.0), n)
    f["nz_g_edges_per_node"] = _d(base.get("g_edges", 0.0), g_nodes)
    f["nz_g_max_pair_per_msg"] = _d(base.get("g_max_pair", 0.0), n)
    f["nz_g_max_out_deg_per_node"] = _d(base.get("g_max_out_deg", 0.0), g_nodes)
    f["nz_g_max_in_deg_per_node"] = _d(base.get("g_max_in_deg", 0.0), g_nodes)
    f["nz_g_triangles_per_node"] = _d(base.get("g_triangles", 0.0), g_nodes)
    f["nz_g_pingpong_per_pair"] = _d(base.get("g_pingpong", 0.0), g_pairs)
    f["nz_g_largest_scc_per_node"] = _d(base.get("g_largest_scc", 0.0), g_nodes)

    # ---- G. agent density ----
    f["nz_tools_per_agent"] = _d(base.get("n_tools_declared", 0.0), n_ag)
    f["nz_senders_per_agent"] = _d(base.get("n_senders", 0.0), n_ag)
    f["nz_n_pairs_per_node"] = _d(base.get("g_n_pairs", 0.0), g_nodes)
    f["nz_msgs_per_sender"] = _d(n, active)

    # ---- H. positional features (relative position in [0,1]) ----
    pos = _positional(msgs, state, n, max_turn, run.get("goal", ""))
    f.update(pos)

    out = {k: _r(f.get(k, 0.0)) for k in NEW_FEATURE_NAMES}
    return out


def _positional(msgs, state, n, max_turn, goal) -> Dict[str, float]:
    """Relative positions of protocol events. Never reads label/fault_turn."""
    p = {k: 0.0 for k in NEW_FEATURE_NAMES if k.startswith("pos_")}
    if n == 0:
        return p
    types = [m.get("type", "") for m in msgs]
    intents = [m.get("intent", "") or "" for m in msgs]
    refs = [m.get("refs") or [] for m in msgs]
    senders = [m.get("from", "") for m in msgs]

    def rel(i):
        return _d(float(i), max_turn)

    first_assign = first_deliv = last_assign = last_deliv = None
    for i, t in enumerate(types):
        if t == "handoff":
            if not refs[i]:
                if first_assign is None:
                    first_assign = i
                last_assign = i
            else:
                if first_deliv is None:
                    first_deliv = i
                last_deliv = i
    p["pos_first_assign"] = rel(first_assign) if first_assign is not None else 0.0
    p["pos_first_delivery"] = rel(first_deliv) if first_deliv is not None else 0.0
    p["pos_last_assign"] = rel(last_assign) if last_assign is not None else 0.0
    p["pos_last_delivery"] = rel(last_deliv) if last_deliv is not None else 0.0

    # start of the longest consecutive normalised-text repeat streak
    norms = [_norm(m.get("text", "") or "") for m in msgs]
    best, best_start, cur, cur_start = 0, -1, 0, -1
    for i, x in enumerate(norms):
        if i > 0 and x == norms[i - 1]:
            if cur == 0:
                cur_start = i - 1
            cur += 1
            if cur > best:
                best, best_start = cur, cur_start
        else:
            cur = 0
    p["pos_streak_start"] = rel(best_start) if best_start >= 0 else 0.0

    # first override / first state write
    if state:
        st_sorted = sorted(state, key=lambda s: int(s.get("t", 0)))
        p["pos_first_state_write"] = rel(int(st_sorted[0].get("t", 0)))
        ov = next((int(s.get("t", 0)) for s in st_sorted
                   if s.get("op") == "override"), None)
        p["pos_first_override"] = rel(ov) if ov is not None else 0.0

    # first sustained run of goal-overlap == 0 (2 consecutive)
    gt = _tokens(goal)
    if gt:
        ov = [1 if not (gt & _tokens(m.get("text", "") or "")) else 0 for m in msgs]
        run_start = -1
        for i in range(len(ov) - 1):
            if ov[i] == 1 and ov[i + 1] == 1:
                run_start = i
                break
        if run_start >= 0:
            p["pos_goal_zero_run"] = rel(run_start)

    # first run of 2+ consecutive status messages (mutual waiting onset)
    cur, start = 0, -1
    for i, t in enumerate(types):
        if t == "status":
            if cur == 0:
                start = i
            cur += 1
            if cur >= 2 and start >= 0 and p["pos_first_status_run"] == 0.0 and start != 0:
                p["pos_first_status_run"] = rel(start)
                start = -1
        else:
            cur = 0

    # first shared-state value flip on a key
    per_key = defaultdict(list)
    for s in sorted(state, key=lambda x: int(x.get("t", 0))):
        per_key[s.get("key")].append(s)
    flip_t = None
    for k, evs in per_key.items():
        for j in range(1, len(evs)):
            if evs[j].get("value") != evs[j - 1].get("value"):
                flip_t = int(evs[j].get("t", 0))
                break
        if flip_t is not None:
            break
    p["pos_first_conflict_flip"] = rel(flip_t) if flip_t is not None else 0.0

    # first plain acknowledgement (inform with the same intent, no refs)
    first_ack = None
    for i in range(1, n):
        if types[i] == "inform" and intents[i] == intents[i - 1] and not refs[i]:
            first_ack = i
            break
    p["pos_first_ack"] = rel(first_ack) if first_ack is not None else 0.0
    return p


def build_matrix(runs: List[Dict[str, Any]], base_rows: List[Dict[str, float]]):
    """Return a DataFrame-ready list of the new features for every run."""
    import pandas as pd
    rows = []
    for run, base in zip(runs, base_rows):
        rows.append(extra_features(run, base))
    return pd.DataFrame(rows, columns=NEW_FEATURE_NAMES).fillna(0.0)
