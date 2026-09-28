"""Window-level structural features for Exp07 (fault-centred window model).

Hypothesis under test: a run-level model sees the whole log aggregated and may
confuse the ROOT CAUSE with secondary anomalies that appear later.  A model that
scores individual turn-windows, supervised by the true ``fault_turn``, should
localise the faulty region better.

This module builds, for every run, one row per candidate turn ``t`` with a
compact STRUCTURAL description of the local neighbourhood.  It is deliberately
NOT the run-level feature set re-used mechanically: the feature list below is
local, scale-free and centred on the candidate turn.

HARD CONSTRAINT
--------------
``label``, ``success`` and ``fault_turn`` are NEVER read here.  The function
signature takes only the parsed run, so target leakage is impossible by
construction rather than by convention.  ``verify_no_leakage.py`` additionally
AST-scans this file to prove it.

Efficiency note
---------------
Every window count is obtained from a prefix sum over per-message indicator
arrays, so cost is O(n_messages) per run rather than O(n * window).  Only the
"distinct value" features loop, over at most 5+11 elements.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any, Dict, List

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
from features import _norm, _tokens   # noqa: E402

RADIUS = 2          # positive zone is fixed +-2 around fault_turn; NOT tuned
CTX = 5             # wider context radius for the "local change" features

TYPES = ["inform", "handoff", "status", "state_update", "final", "system"]
T_IDX = {t: i for i, t in enumerate(TYPES)}

FEATURE_NAMES = [
    # --- run context (constant within a run; lets the model condition) ------
    "w_n_msgs", "w_n_agents", "w_n_intents", "w_n_artifacts", "w_n_assign",
    "w_unresolved_ratio",
    # --- position of the candidate turn -------------------------------------
    "w_pos", "w_rel_turn", "w_is_first", "w_is_last",
    "w_dist_start", "w_dist_end", "w_dist_start_rel", "w_dist_end_rel",
    # --- window composition -------------------------------------------------
    "w_win_n", "w_win_n_rel",
    "w_win_handoff", "w_win_inform", "w_win_status", "w_win_state_update",
    "w_win_final", "w_win_system",
    "w_win_assign", "w_win_result", "w_win_refs",
    "w_win_tools", "w_win_tools_rel",
    "w_win_distinct_tools", "w_win_distinct_senders",
    "w_win_distinct_receivers", "w_win_distinct_intents",
    "w_win_new_intents", "w_win_repeat_intents",
    "w_win_state_set", "w_win_state_override", "w_win_state_keys",
    "w_win_artifacts", "w_win_artifacts_final",
    # --- waiting / deadlock signals ----------------------------------------
    "w_win_status_rel", "w_status_streak", "w_status_streak_rel",
    "w_win_assign_no_reply",
    # --- duplicated work signals -------------------------------------------
    "w_win_dup_intent", "w_run_dup_intent_rate", "w_win_dup_tool",
    "w_run_dup_artifact_hash",
    # --- conflict signals ---------------------------------------------------
    "w_win_override", "w_run_override_rel", "w_win_same_key_set",
    "w_state_contention",
    # --- goal drift signals -------------------------------------------------
    "w_win_goal_rel", "w_no_goal_streak", "w_no_goal_streak_rel",
    # --- runaway loop signals ----------------------------------------------
    "w_win_dup_text", "w_win_dup_text_rel", "w_run_dup_text_rate",
    "w_loop_streak",
    # --- lifecycle / backlog around the window ------------------------------
    "w_open_before", "w_open_after", "w_open_after_rel", "w_backlog_frac",
    "w_recv_active_after_rel", "w_recv_silence_oldest_rel",
    "w_recv_own_msgs_after_mean_rel",
    # --- local change vs neighbouring windows -------------------------------
    "w_delta_assign_prev", "w_delta_assign_next",
    "w_delta_status_prev", "w_delta_status_next",
    "w_delta_handoff_prev", "w_delta_handoff_next",
    "w_delta_tools_prev", "w_delta_tools_next",
    "w_delta_worst",
    # --- wider +-5 context --------------------------------------------------
    "w_ctx_assign", "w_ctx_status", "w_ctx_inform", "w_ctx_state_update",
    "w_ctx_tools", "w_ctx_override",
    # --- assignment timing --------------------------------------------------
    "w_since_assign", "w_since_assign_rel", "w_to_next_assign",
    "w_assign_pos_rel",
]
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "duplicate window feature name"
N_FEATURES = len(FEATURE_NAMES)

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _cs(a: np.ndarray) -> np.ndarray:
    """Prefix sum with a leading zero, so cs[j] - cs[i] sums a[i:j]."""
    out = np.zeros(len(a) + 1, dtype=np.float64)
    np.cumsum(a, out=out[1:])
    return out


def _runsum(cs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return cs[hi] - cs[lo]


def _assigns_and_results(msgs):
    a, r = [], []
    for m in msgs:
        if m.get("type") == "handoff":
            (r if m.get("refs") else a).append(m)
    return a, r


def _delivered_set(msgs, arts):
    d = {}
    for m in msgs:
        if m.get("type") == "handoff" and m.get("refs"):
            d.setdefault(m.get("from"), set()).add(m.get("intent"))
    for a in arts:
        d.setdefault(a.get("by"), set()).add(a.get("subtask"))
    return d


def run_window_features(run: Dict[str, Any]) -> np.ndarray:
    """One feature row per candidate turn of a single run.

    Returns an ``(n_messages, N_FEATURES)`` float array.  ``fault_turn`` is not
    an input and is not read, so this is identical at train and inference time.
    """
    msgs: List[Dict[str, Any]] = run["messages"]
    arts: List[Dict[str, Any]] = run["artifacts"]
    state: List[Dict[str, Any]] = run["shared_state"]
    n = len(msgs)
    if n == 0:
        return np.zeros((0, N_FEATURES))

    goal_tok = _tokens(run.get("goal", ""))
    types = np.array([T_IDX.get(m.get("type"), -1) for m in msgs])
    is_handoff = (types == T_IDX["handoff"]).astype(float)
    has_refs = np.array([1.0 if m.get("refs") else 0.0 for m in msgs])
    is_assign = (is_handoff * (1.0 - has_refs))
    is_result = (is_handoff * has_refs)
    is_status = (types == T_IDX["status"]).astype(float)
    is_tool = np.array([1.0 if m.get("tool") else 0.0 for m in msgs])
    is_set = np.array([1.0 if m.get("type") == "state_update" else 0.0 for m in msgs])
    is_fin = np.array([1.0 if m.get("type") == "final" else 0.0 for m in msgs])

    # state events indexed onto the nearest preceding message turn
    st_set = np.zeros(n); st_ovr = np.zeros(n)
    for s in state:
        t = int(s.get("t", 0))
        if 0 <= t < n:
            if s.get("op") == "override":
                st_ovr[t] += 1.0
            else:
                st_set[t] += 1.0
    st_any = st_set + st_ovr

    art_new = np.zeros(n); art_fin = np.zeros(n)
    for a in arts:
        t = int(a.get("t", 0))
        if 0 <= t < n:
            art_new[t] += 1.0
            if a.get("status") == "final":
                art_fin[t] += 1.0

    # duplicate-work indicators
    intent_seq = [m.get("intent") or "" for m in msgs]
    ic = Counter(intent_seq)
    assign_intent = [(m.get("intent") or "") if (m.get("type") == "handoff"
                                                  and not m.get("refs")) else None
                     for m in msgs]
    ai = Counter(x for x in assign_intent if x)
    dup_intent = np.array([1.0 if (assign_intent[i] and
                                   ai[assign_intent[i]] > 1) else 0.0
                           for i in range(n)])
    tool_seq = [m.get("tool") or "" for m in msgs]
    tc = Counter(x for x in tool_seq if x)
    dup_tool = np.array([1.0 if (tool_seq[i] and tc[tool_seq[i]] > 1) else 0.0
                         for i in range(n)])
    norm_seq = [_norm(m.get("text", "")) for m in msgs]
    nc = Counter(x for x in norm_seq if x)
    dup_text = np.array([1.0 if (norm_seq[i] and nc[norm_seq[i]] >= 3) else 0.0
                         for i in range(n)])

    # goal alignment
    no_goal = np.array([0.0 if (_tokens(m.get("text", "")) & goal_tok) else 1.0
                        for m in msgs])

    # per-message turn indices for artifacts/state for relative placement
    sent = [m.get("from") for m in msgs]
    recv = [m.get("to") for m in msgs]
    intents = [m.get("intent") or "" for m in msgs]

    # unresolved-assignment trajectory (structural, no target involved)
    assigns, results = _assigns_and_results(msgs)
    delivered = _delivered_set(msgs, arts)
    open_at = np.zeros(n)          # unresolved assignments strictly before t
    closed = 0
    ai_t = sorted(int(m.get("t", 0)) for m in assigns)
    ci_t = 0
    open_before = np.zeros(n)
    for t in range(n):
        while ci_t < len(ai_t) and ai_t[ci_t] < t:
            ci_t += 1
        open_before[t] = ci_t - closed
    # resolve pass: an assignment is closed once its intent reaches `to`
    resolved_at = []
    for m in assigns:
        ti = int(m.get("t", 0))
        ok = any(x.get("from") == m.get("to") and x.get("intent") == m.get("intent")
                 and int(x.get("t", 0)) > ti for x in msgs)
        if not ok:
            ok = m.get("intent") in delivered.get(m.get("to"), set())
        if ok:
            resolved_at.append(ti)
    rs = np.zeros(n)
    for ti in resolved_at:
        # an assignment closed at turn ti only stops being "open" from ti+1 on;
        # guard the boundary because an assignment may be logged on the last turn
        if ti + 1 < n:
            rs[ti + 1] += 1.0
    closed_before = _cs(rs)
    n_assign_before = _cs(is_assign)
    open_before = np.maximum(0.0, n_assign_before[:-1] - closed_before[:-1])

    # distinct-value window helpers
    def win_distinct(lo, hi, seq):
        return len({seq[i] for i in range(lo, hi) if seq[i]}) if hi > lo else 0

    # --- prefix sums for every window-aggregatable indicator ---------------
    C = {
        "handoff": _cs(is_handoff), "status": _cs(is_status), "inform":
            _cs((types == T_IDX["inform"]).astype(float)),
        "state_update": _cs(is_set), "tools": _cs(is_tool), "override": _cs(st_ovr),
        "assign": _cs(is_assign), "result": _cs(is_result), "refs": _cs(has_refs),
        "state_set": _cs(st_set), "state_any": _cs(st_any),
        "art": _cs(art_new), "art_fin": _cs(art_fin),
        "dup_intent": _cs(dup_intent), "dup_tool": _cs(dup_tool),
        "dup_text": _cs(dup_text), "no_goal": _cs(no_goal),
        "final": _cs(is_fin), "system": _cs((types == T_IDX["system"]).astype(float)),
        "lifetime": _cs(np.ones(n)),
    }

    n_assign_total = float(is_assign.sum())
    n_int_total = len({x for x in intent_seq if x})
    agents_all = set(sent) | set(recv)
    hashes = Counter(a.get("hash") for a in arts if a.get("hash"))
    n_dup_hash = sum(v - 1 for v in hashes.values() if v > 1)
    run_dup_intent_rate = (float(dup_intent.sum()) / max(1.0, n_assign_total))
    run_dup_text_rate = (float(dup_text.sum()) / max(1.0, n))
    run_dup_artifact = (n_dup_hash / max(1.0, len(arts)))
    run_override_rel = (float(st_ovr.sum()) / max(1.0, float(st_any.sum())))

    # per-message state contention: writes to a key another agent already wrote
    key_last_agent = {}
    contention = np.zeros(n)
    for s in sorted(state, key=lambda x: int(x.get("t", 0))):
        t = int(s.get("t", 0))
        if not (0 <= t < n):
            continue
        k, ag = s.get("key"), s.get("agent")
        prev = key_last_agent.get(k)
        if prev is not None and prev != ag:
            contention[t] += 1.0
        key_last_agent[k] = ag
    C["contention"] = _cs(contention)

    # streaks
    status_streak = np.zeros(n); cur = 0.0
    for i in range(n):
        cur = cur + 1.0 if is_status[i] else 0.0
        status_streak[i] = cur
    nogoal_streak = np.zeros(n); cur = 0.0
    for i in range(n):
        cur = cur + 1.0 if no_goal[i] else 0.0
        nogoal_streak[i] = cur
    loop_streak = np.zeros(n); cur = 0.0
    for i in range(n):
        cur = cur + 1.0 if dup_text[i] else 0.0
        loop_streak[i] = cur

    # assignment timing
    last_assign = np.zeros(n); nxt_assign = np.zeros(n)
    ai_arr = np.asarray(ai_t, dtype=int)
    prev_a = 0
    for i in range(n):
        last_assign[i] = prev_a
        j = int(np.searchsorted(ai_arr, i, side="right"))
        prev_a = ai_arr[j - 1] + 1 if j else 0
    if len(ai_arr):
        idx = np.searchsorted(ai_arr, np.arange(n), side="left")
        nxt_assign = np.where(idx < len(ai_arr), ai_arr[np.clip(idx, 0, len(ai_arr) - 1)], n)
    else:
        nxt_assign = np.full(n, n, dtype=float)

    # receiver activity after each assignment inside the window
    out = np.zeros((n, N_FEATURES))
    for i in range(n):
        lo, hi = max(0, i - RADIUS), min(n, i + RADIUS + 1)
        clo, chi = max(0, i - CTX), min(n, i + CTX + 1)
        w = hi - lo
        row = out[i]

        def S(k):
            return _runsum(C[k], lo, hi)

        def CS(k):
            return _runsum(C[k], clo, chi)

        row[_IDX["w_n_msgs"]] = n
        row[_IDX["w_n_agents"]] = len(agents_all)
        row[_IDX["w_n_intents"]] = n_int_total
        row[_IDX["w_n_artifacts"]] = len(arts)
        row[_IDX["w_n_assign"]] = n_assign_total
        row[_IDX["w_unresolved_ratio"]] = open_before[i] / max(1.0, n_assign_total)

        row[_IDX["w_pos"]] = i / max(1.0, n - 1)
        row[_IDX["w_rel_turn"]] = i / max(1.0, n)
        row[_IDX["w_is_first"]] = 1.0 if i == 0 else 0.0
        row[_IDX["w_is_last"]] = 1.0 if i == n - 1 else 0.0
        row[_IDX["w_dist_start"]] = i
        row[_IDX["w_dist_end"]] = n - 1 - i
        row[_IDX["w_dist_start_rel"]] = i / max(1.0, n)
        row[_IDX["w_dist_end_rel"]] = (n - 1 - i) / max(1.0, n)

        row[_IDX["w_win_n"]] = w
        row[_IDX["w_win_n_rel"]] = w / max(1.0, n)
        row[_IDX["w_win_handoff"]] = S("handoff")
        row[_IDX["w_win_inform"]] = S("inform")
        row[_IDX["w_win_status"]] = S("status")
        row[_IDX["w_win_state_update"]] = S("state_update")
        row[_IDX["w_win_final"]] = S("final")
        row[_IDX["w_win_system"]] = S("system")
        row[_IDX["w_win_assign"]] = S("assign")
        row[_IDX["w_win_result"]] = S("result")
        row[_IDX["w_win_refs"]] = S("refs")
        row[_IDX["w_win_tools"]] = S("tools")
        row[_IDX["w_win_tools_rel"]] = S("tools") / max(1.0, w)
        row[_IDX["w_win_distinct_tools"]] = win_distinct(lo, hi, tool_seq)
        row[_IDX["w_win_distinct_senders"]] = win_distinct(lo, hi, sent)
        row[_IDX["w_win_distinct_receivers"]] = win_distinct(lo, hi, recv)
        row[_IDX["w_win_distinct_intents"]] = win_distinct(lo, hi, intents)
        seen_before = {intents[j] for j in range(0, lo) if intents[j]}
        row[_IDX["w_win_new_intents"]] = sum(
            1 for j in range(lo, hi)
            if intents[j] and intents[j] not in seen_before)
        row[_IDX["w_win_repeat_intents"]] = sum(
            1 for j in range(lo, hi)
            if intents[j] and intents[j] in seen_before)
        row[_IDX["w_win_state_set"]] = S("state_set")
        row[_IDX["w_win_state_override"]] = S("override")
        row[_IDX["w_win_state_keys"]] = S("state_any")
        row[_IDX["w_win_artifacts"]] = S("art")
        row[_IDX["w_win_artifacts_final"]] = S("art_fin")

        row[_IDX["w_win_status_rel"]] = S("status") / max(1.0, w)
        row[_IDX["w_status_streak"]] = status_streak[i]
        row[_IDX["w_status_streak_rel"]] = status_streak[i] / max(1.0, n)
        replied = {(m.get("from"), m.get("intent")) for m in results}
        row[_IDX["w_win_assign_no_reply"]] = sum(
            1 for j in range(lo, hi)
            if assign_intent[j] and (recv[j], assign_intent[j]) not in replied)

        row[_IDX["w_win_dup_intent"]] = S("dup_intent")
        row[_IDX["w_run_dup_intent_rate"]] = run_dup_intent_rate
        row[_IDX["w_win_dup_tool"]] = S("dup_tool")
        row[_IDX["w_run_dup_artifact_hash"]] = run_dup_artifact

        row[_IDX["w_win_override"]] = S("override")
        row[_IDX["w_run_override_rel"]] = run_override_rel
        row[_IDX["w_win_same_key_set"]] = S("state_set")
        row[_IDX["w_state_contention"]] = S("contention")

        row[_IDX["w_win_goal_rel"]] = 1.0 - S("no_goal") / max(1.0, w)
        row[_IDX["w_no_goal_streak"]] = nogoal_streak[i]
        row[_IDX["w_no_goal_streak_rel"]] = nogoal_streak[i] / max(1.0, n)

        row[_IDX["w_win_dup_text"]] = S("dup_text")
        row[_IDX["w_win_dup_text_rel"]] = S("dup_text") / max(1.0, w)
        row[_IDX["w_run_dup_text_rate"]] = run_dup_text_rate
        row[_IDX["w_loop_streak"]] = loop_streak[i]

        row[_IDX["w_open_before"]] = open_before[i]
        aft = min(n - 1, i + RADIUS)
        row[_IDX["w_open_after"]] = open_before[aft] - open_before[i]
        row[_IDX["w_open_after_rel"]] = ((open_before[aft] - open_before[i])
                                         / max(1.0, w))
        row[_IDX["w_backlog_frac"]] = open_before[i] / max(1.0, n)
        # receiver activity after an assignment in the window, to end of run
        act, sil, own = [], [], []
        for j in range(lo, hi):
            if assign_intent[j]:
                a = recv[j]
                after = [k for k in range(i + RADIUS + 1, n) if sent[k] == a]
                act.append(len(after) / max(1.0, n - i - RADIUS - 1))
                own.append(sum(1 for k in after if intents[k] == assign_intent[j])
                           / max(1.0, len(after)) if after else 0.0)
                gap = [k for k in range(i + RADIUS + 1, n) if sent[k] == a]
                sil.append((n - 1 - gap[-1]) / max(1.0, n) if gap else 1.0)
        row[_IDX["w_recv_active_after_rel"]] = float(np.mean(act)) if act else 0.0
        row[_IDX["w_recv_silence_oldest_rel"]] = float(np.max(sil)) if sil else 1.0
        row[_IDX["w_recv_own_msgs_after_mean_rel"]] = float(np.mean(own)) if own else 0.0

        def d(k):
            return S(k) - _runsum(C[k], max(0, lo - RADIUS - 1), lo)

        def dn(k):
            return _runsum(C[k], hi, min(n, hi + RADIUS)) - S(k)

        da_p, da_n = d("assign"), dn("assign")
        ds_p, ds_n = d("status"), dn("status")
        dh_p, dh_n = d("handoff"), dn("handoff")
        dt_p, dt_n = d("tools"), dn("tools")
        row[_IDX["w_delta_assign_prev"]] = da_p
        row[_IDX["w_delta_assign_next"]] = da_n
        row[_IDX["w_delta_status_prev"]] = ds_p
        row[_IDX["w_delta_status_next"]] = ds_n
        row[_IDX["w_delta_handoff_prev"]] = dh_p
        row[_IDX["w_delta_handoff_next"]] = dh_n
        row[_IDX["w_delta_tools_prev"]] = dt_p
        row[_IDX["w_delta_tools_next"]] = dt_n
        row[_IDX["w_delta_worst"]] = max(abs(x) for x in
                                         (da_p, da_n, ds_p, ds_n, dh_p, dh_n, dt_p, dt_n))

        row[_IDX["w_ctx_assign"]] = CS("assign")
        row[_IDX["w_ctx_status"]] = CS("status")
        row[_IDX["w_ctx_inform"]] = CS("inform")
        row[_IDX["w_ctx_state_update"]] = CS("state_update")
        row[_IDX["w_ctx_tools"]] = CS("tools")
        row[_IDX["w_ctx_override"]] = CS("override")

        row[_IDX["w_since_assign"]] = i - last_assign[i]
        row[_IDX["w_since_assign_rel"]] = (i - last_assign[i]) / max(1.0, n)
        row[_IDX["w_to_next_assign"]] = nxt_assign[i] - i
        row[_IDX["w_assign_pos_rel"]] = (last_assign[i] / max(1.0, n)
                                         if n_assign_total else 0.0)
    return out
