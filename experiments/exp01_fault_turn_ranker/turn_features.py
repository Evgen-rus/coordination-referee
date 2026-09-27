"""Per-turn causal features for the learned fault_turn ranker (experiment 01).

Design constraints (see ../RESULTS.md):
  * one row per message turn, features computable WITHOUT the run's label and
    WITHOUT the run's fault_turn -> no target leakage;
  * every feature is either causal (uses only turns t' <= t plus run-level
    aggregates that are known once the log is read) or a small symmetric
    window (+/- r turns) around t.  Nothing looks past the window;
  * the baseline's 122 run-level features are deliberately NOT reused as
    turn features, because most of them are aggregates over the whole run
    and would let the ranker "see" whether a fault was ever resolved.

Imports (read-only) ``parse_run``/``_norm``/``_tokens`` from the official
baseline so that normalisation matches the starter kit exactly.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "baseline"))

from features import _norm, _tokens   # noqa: E402  (official baseline module)

PROTOCOL_TYPES = ["handoff", "inform", "status", "state_update", "final", "system"]
KEYWORD_GROUPS = ["wait", "conflict", "repeat", "drift", "recovery", "stop", "done"]
KEYWORDS = {
    "wait": ["wait", "blocked", "hold", "depend", "not arrived", "cannot proceed", "on hold"],
    "conflict": ["disagree", "contradict", "overrid", "incorrect", "reject", "conflict", "wrong"],
    "repeat": ["again", "re-check", "recheck", "iterat", "repeat", "another pass",
               "looping", "one more", "still not"],
    "drift": ["side note", "expanding scope", "pivot", "new direction", "switching focus",
              "deprioritis", "deprioritiz", "also look into", "matters more"],
    "recovery": ["recover", "re-assign", "reassign", "retry", "fallback", "escalat",
                 "mitigat", "repairing", "takes over"],
    "stop": ["terminat", "stops the run", "halting", "stopping the run", "budget of turns",
             "no progress", "turn budget"],
    "done": ["final", "complete", "deliver", "ready", "wrapping up", "consolidat",
             "aggregated"],
}
RADII = (2, 5)


def _safe_div(a, b) -> float:
    return float(a) / b if b else 0.0


def _entropy(counts) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * np.log(c / tot + 1e-12) for c in counts if c > 0)


def _sorted_map(d: Dict[Any, List[int]]) -> Dict[Any, np.ndarray]:
    return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in d.items()}


def feature_names() -> List[str]:
    names: List[str] = []
    names += ["abs_pos", "rel_pos", "turns_to_end", "run_len", "run_n_agents",
              "run_n_artifacts", "run_n_state", "run_n_intents", "run_n_types",
              "run_max_streak", "run_frac_handoff", "run_frac_status"]
    names += ["type_%s" % t for t in PROTOCOL_TYPES]
    names += ["is_assign", "is_delivery", "is_own_repeat_type"]
    names += ["has_intent", "intent_len", "refs_count", "text_len", "text_trunc_12_30",
              "tool_present", "self_loop"]
    names += ["kw_%s" % g for g in KEYWORD_GROUPS]
    names += ["goal_ov", "goal_ov_zero", "goal_ov_delta_prev", "goal_ov_minus_runmean",
              "goal_ov_runmean", "goal_ov_drop_3", "goal_ov_drop_5"]
    names += ["sender_role_orch", "sender_role_analyst", "recv_role_orch",
              "recv_self", "sender_out_deg", "recv_out_deg",
              "sender_share_so_far", "sender_distinct_recv_so_far",
              "recv_share_so_far", "sender_so_far_frac", "recv_so_far_frac"]
    names += ["norm_rep_count_so_far", "norm_rep_run_so_far", "same_norm_as_prev",
              "same_sender_as_prev", "streak_len", "streak_frac",
              "intent_msg_count_so_far", "intent_distinct_agents_so_far",
              "intent_agents_frac", "intent_first_seen",
              "n_assigns_intent_so_far", "n_deliveries_intent_so_far",
              "intent_delivered_so_far", "is_second_assign_intent",
              "assign_unanswered_so_far", "unanswered_frac_so_far"]
    names += ["status_streak", "status_streak_frac", "status_reciprocal_before",
              "pair_msg_count_so_far", "pair_recip_count_so_far",
              "status_between_pair_so_far"]
    names += ["state_writes_so_far", "state_distinct_keys_so_far", "state_overrides_so_far",
              "state_flips_so_far", "state_altflips_so_far", "state_key_writes_so_far",
              "state_key_distinct_vals", "state_is_override", "state_value_changed",
              "state_alt_flip", "state_key_is_run_status", "state_key_writes_frac"]
    names += ["arts_so_far", "arts_by_sender_so_far", "arts_for_intent_so_far",
              "intent_covered_by_arts", "edges_so_far", "graph_density_so_far",
              "edge_recip_before", "n_distinct_agents_so_far"]
    for r in RADII:
        names += ["w%d_handoff" % r, "w%d_status" % r, "w%d_state" % r, "w%d_final" % r,
                  "w%d_intents" % r, "w%d_goal_ov_mean" % r, "w%d_goal_ov_max" % r,
                  "w%d_max_norm_rep" % r, "w%d_forward" % r, "w%d_backward" % r]
    return names


N_FEATURES = len(feature_names())


def build_turn_matrix(run: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X [n_turns, N_FEATURES], y_pos [n_turns]) for one run.

    ``y_pos`` is filled with -1 here; the caller supplies the positive turn
    (it is the only place the true fault_turn is ever touched, and only for
    rows inside a training fold).
    """
    msgs: List[Dict] = run.get("messages") or []
    n = len(msgs)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int64)

    agents = run.get("agents") or []
    arts = run.get("artifacts") or []
    state = run.get("shared_state") or []
    roles = {a.get("id"): a.get("role", "") for a in agents}
    goal_tok = _tokens(run.get("goal", ""))
    n_ag = max(1, len(agents))

    # ---------- static per-message fields ----------
    types = [m.get("type", "") for m in msgs]
    texts = [m.get("text", "") or "" for m in msgs]
    norms = [_norm(t) for t in texts]
    lows = [t.lower() for t in texts]
    intents = [m.get("intent", "") or "" for m in msgs]
    refs = [m.get("refs") or [] for m in msgs]
    senders = [m.get("from", "") for m in msgs]
    receivers = [m.get("to", "") for m in msgs]
    kw = np.zeros((n, len(KEYWORD_GROUPS)), dtype=np.float32)
    for gi, (_, words) in enumerate(KEYWORDS.items()):
        for i, lt in enumerate(lows):
            c = sum(lt.count(w) for w in words)
            kw[i, gi] = c
    goal_ov = np.array([_safe_div(len(goal_tok & _tokens(t)), len(goal_tok))
                        for t in texts], dtype=np.float32)

    # ---------- prefix counts ----------
    norm_run = np.zeros(n, dtype=np.int64)
    rep_run = np.zeros(n, dtype=np.int64)
    seen_norm = defaultdict(int)
    streak = np.zeros(n, dtype=np.int64)
    for i in range(n):
        rep_run[i] = seen_norm[norms[i]]
        seen_norm[norms[i]] += 1
        norm_run[i] = seen_norm[norms[i]]
        streak[i] = (streak[i - 1] + 1) if i > 0 and norms[i] == norms[i - 1] else 1

    intent_msgs = _sorted_map(defaultdict(list, {
        k: [i for i in range(n) if intents[i] == k] for k in set(intents)}))
    intent_agents = _sorted_map({})
    ia_tmp = defaultdict(lambda: defaultdict(list))
    for i in range(n):
        if intents[i]:
            ia_tmp[intents[i]][senders[i]].append(i)
    intent_agents = {k: {ag: np.asarray(sorted(v), dtype=np.int64)
                         for ag, v in d.items()} for k, d in ia_tmp.items()}

    assigns = defaultdict(list)
    delivers = defaultdict(list)
    for i in range(n):
        if types[i] == "handoff":
            (delivers if refs[i] else assigns)[intents[i]].append(i)
    assign_t = _sorted_map(assigns)
    deliver_t = _sorted_map(delivers)

    is_assign = np.array([1 if (types[i] == "handoff" and not refs[i]) else 0 for i in range(n)])
    is_deliv = np.array([1 if (types[i] == "handoff" and refs[i]) else 0 for i in range(n)])

    # sender / receiver prefix shares
    sender_cnt, recv_cnt = defaultdict(int), defaultdict(int)
    sender_recv = defaultdict(set)
    s_share = np.zeros(n); r_share = np.zeros(n)
    s_recvs = np.zeros(n); r_frac = np.zeros(n); s_frac = np.zeros(n)
    for i in range(n):
        sender_cnt[senders[i]] += 1
        recv_cnt[receivers[i]] += 1
        sender_recv[senders[i]].add(receivers[i])
        s_share[i] = _safe_div(sender_cnt[senders[i]], i + 1)
        r_share[i] = _safe_div(recv_cnt[receivers[i]], i + 1)
        s_recvs[i] = len(sender_recv[senders[i]])
        s_frac[i] = _safe_div(sender_cnt[senders[i]], n)
        r_frac[i] = _safe_div(recv_cnt[receivers[i]], n)

    # pairwise prefix (deadlock / ping-pong)
    pair_cnt = defaultdict(int)
    pair_recip = defaultdict(int)
    pair_status = defaultdict(int)
    p_cnt = np.zeros(n); p_recip = np.zeros(n); p_status = np.zeros(n)
    for i in range(n):
        a, b = senders[i], receivers[i]
        p_cnt[i] = pair_cnt[(a, b)]
        p_recip[i] = pair_recip[(a, b)]
        p_status[i] = pair_status[(a, b)]
        pair_cnt[(a, b)] += 1
        pair_status[(a, b)] += 1 if types[i] == "status" else 0
        if pair_cnt[(b, a)] > 0:
            pair_recip[(a, b)] = pair_cnt[(b, a)]

    # status streak + reciprocal waiting
    st_streak = np.zeros(n, dtype=np.int64)
    st_recip = np.zeros(n)
    cur = 0
    status_seen = defaultdict(int)
    for i in range(n):
        cur = (cur + 1) if types[i] == "status" else 0
        st_streak[i] = cur
        if types[i] == "status":
            status_seen[(senders[i], receivers[i])] += 1
            st_recip[i] = 1.0 if status_seen[(receivers[i], senders[i])] > 0 else 0.0

    # ---------- shared state prefixes ----------
    st_t = np.asarray(sorted([int(s.get("t", 0)) for s in state]), dtype=np.int64)
    per_key = defaultdict(list)
    for s in sorted(state, key=lambda x: int(x.get("t", 0))):
        per_key[s.get("key")].append(s)
    key_prof = {}
    for k, evs in per_key.items():
        ts = np.asarray([int(e.get("t", 0)) for e in evs], dtype=np.int64)
        vals = [e.get("value") for e in evs]
        ops = [e.get("op") for e in evs]
        cnt = np.zeros(len(evs) + 1); dst = np.zeros(len(evs) + 1)
        flips = np.zeros(len(evs) + 1); alt = np.zeros(len(evs) + 1)
        ov = np.zeros(len(evs) + 1)
        for j in range(len(evs)):
            cnt[j + 1] = cnt[j] + 1
            dst[j + 1] = dst[j] + (1 if (j == 0 or vals[j] not in vals[:j]) else 0)
            flips[j + 1] = flips[j] + (1 if j > 0 and vals[j] != vals[j - 1] else 0)
            alt[j + 1] = alt[j] + (1 if j >= 2 and vals[j] == vals[j - 2]
                                    and vals[j] != vals[j - 1] else 0)
            ov[j + 1] = ov[j] + (1 if ops[j] == "override" else 0)
        key_prof[k] = (ts, cnt, dst, flips, alt, ov, vals, ops)

    st_counts = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        p = int(np.searchsorted(st_t, i, side="right"))
        allw = 0; allk = 0; allo = 0; allf = 0; alla = 0
        for k, (ts, cnt, dst, fl, al, ov, _, _) in key_prof.items():
            q = int(np.searchsorted(ts, i, side="right"))
            allw += cnt[q]; allk += dst[q]; allo += ov[q]; allf += fl[q]; alla += al[q]
        st_counts[i] = [p, allk, allo, allf, alla]

    # per-turn state-key features: the write at this turn, else the latest before
    st_key = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        best = None
        for k, (ts, cnt, dst, fl, al, ov, vals, ops) in key_prof.items():
            q = int(np.searchsorted(ts, i, side="right"))
            if q == 0:
                continue
            j = q - 1
            if best is None or ts[j] > best[0]:
                best = (int(ts[j]), k, q, j)
        if best is None:
            continue
        _, k, q, j = best
        ts, cnt, dst, fl, al, ov, vals, ops = key_prof[k]
        st_key[i] = [cnt[q], dst[q], 1.0 if ops[j] == "override" else 0.0,
                     1.0 if j > 0 and vals[j] != vals[j - 1] else 0.0,
                     1.0 if j >= 2 and vals[j] == vals[j - 2] and vals[j] != vals[j - 1] else 0.0,
                     1.0 if k == "run_status" else 0.0]

    # ---------- artifacts (causal on artifact.t) ----------
    art_t = np.asarray(sorted([int(a.get("t", 0)) for a in arts]), dtype=np.int64)
    art_by_sender = defaultdict(list)
    art_by_intent = defaultdict(list)
    for a in arts:
        art_by_sender[a.get("by")].append(int(a.get("t", 0)))
        if a.get("subtask"):
            art_by_intent[a.get("subtask")].append(int(a.get("t", 0)))
    art_by_sender = _sorted_map(art_by_sender)
    art_by_intent = _sorted_map(art_by_intent)

    # ---------- graph prefix ----------
    out_deg = defaultdict(set)
    in_deg = defaultdict(set)
    edges_seen = set()
    node_seen = set()
    g_edges = np.zeros(n); g_dens = np.zeros(n); g_recip = np.zeros(n)
    n_dist_ag = np.zeros(n)
    for i in range(n):
        s, r = senders[i], receivers[i]
        out_deg[s].add(r); in_deg[r].add(s)
        edges_seen.add((s, r))
        node_seen.add(s); node_seen.add(r)
        g_edges[i] = len(edges_seen)
        g_dens[i] = _safe_div(len(edges_seen), max(1, len(node_seen) * (len(node_seen) - 1)))
        g_recip[i] = 1.0 if any((r2, s2) in edges_seen for s2, r2 in edges_seen) else 0.0
        n_dist_ag[i] = len(node_seen)

    # ---------- run-level static ----------
    tcount = defaultdict(int)
    for t in types:
        tcount[t] += 1
    run_static = [n, n_ag, len(arts), len(state),
                  len({i for i in intents if i}),
                  len(set(types)), int(streak.max()) if n else 0,
                  _safe_div(tcount.get("handoff", 0), n),
                  _safe_div(tcount.get("status", 0), n)]

    goal_runmean = np.cumsum(goal_ov) / np.arange(1, n + 1)

    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i in range(n):
        v = []
        # position
        v += [i, _safe_div(i, max(1, n - 1)), n - i - 1] + run_static
        # types
        v += [1.0 if types[i] == t else 0.0 for t in PROTOCOL_TYPES]
        v += [is_assign[i], is_deliv[i],
              1.0 if types[i] == "assign" else 0.0]
        # payload
        v += [1.0 if intents[i] else 0.0, len(intents[i]), len(refs[i]), len(texts[i]),
              1.0 if 12 <= len(texts[i]) <= 30 else 0.0,
              1.0 if msgs[i].get("tool") else 0.0,
              1.0 if senders[i] == receivers[i] else 0.0]
        v += list(kw[i])
        # goal alignment
        prev = goal_ov[i - 1] if i > 0 else goal_ov[i]
        back3 = float(np.mean(goal_ov[max(0, i - 3):i])) if i > 0 else goal_ov[i]
        back5 = float(np.mean(goal_ov[max(0, i - 5):i])) if i > 0 else goal_ov[i]
        v += [goal_ov[i], 1.0 if goal_ov[i] == 0 else 0.0, goal_ov[i] - prev,
              goal_ov[i] - goal_runmean[i], goal_runmean[i],
              goal_ov[i] - back3, goal_ov[i] - back5]
        # roles / sender-receiver
        s, r = senders[i], receivers[i]
        v += [1.0 if roles.get(s) == "orchestrator" else 0.0,
              1.0 if roles.get(s) == "analyst" else 0.0,
              1.0 if roles.get(r) == "orchestrator" else 0.0,
              1.0 if s == r else 0.0,
              len(out_deg[s]), len(in_deg[r]), s_share[i], s_recvs[i], r_share[i],
              s_frac[i], r_frac[i]]
        # repetition
        it = intents[i]
        n_im = (int(np.searchsorted(intent_msgs[it], i, "right"))
                if it in intent_msgs else 0)
        n_ag_it = (sum(1 for _, arr in intent_agents[it].items()
                       if np.searchsorted(arr, i, "right") > 0)
                   if it in intent_agents else 0)
        n_as = (int(np.searchsorted(assign_t[it], i, "right")) if it in assign_t else 0)
        n_dl = (int(np.searchsorted(deliver_t[it], i, "right")) if it in deliver_t else 0)
        n_art_it = (int(np.searchsorted(art_by_intent[it], i, "right"))
                    if it in art_by_intent else 0)
        unans = n_as - n_dl
        v += [norm_run[i], rep_run[i],
              1.0 if i > 0 and norms[i] == norms[i - 1] else 0.0,
              1.0 if i > 0 and senders[i] == senders[i - 1] else 0.0,
              streak[i], _safe_div(streak[i], n), n_im, n_ag_it,
              _safe_div(n_ag_it, n_ag), 1.0 if n_im == 1 else 0.0,
              n_as, n_dl, 1.0 if n_dl > 0 else 0.0,
              1.0 if (is_assign[i] and n_as >= 2) else 0.0,
              1.0 if (is_assign[i] and unans > 0) else 0.0,
              _safe_div(unans, max(1, n_as))]
        # status / deadlock
        v += [st_streak[i], _safe_div(st_streak[i], n), st_recip[i],
              p_cnt[i], p_recip[i], p_status[i]]
        # shared state
        v += list(st_counts[i]) + list(st_key[i]) + [st_key[i][0] / max(1.0, n)]
        # artifacts + graph
        v += [int(np.searchsorted(art_t, i, "right")),
              int(np.searchsorted(art_by_sender[s], i, "right")) if s in art_by_sender else 0,
              n_art_it, 1.0 if n_art_it > 0 else 0.0,
              g_edges[i], g_dens[i], g_recip[i], n_dist_ag[i]]
        # windows
        for rad in RADII:
            lo, hi = max(0, i - rad), min(n, i + rad + 1)
            win = range(lo, hi)
            wtypes = [types[j] for j in win]
            v += [_safe_div(sum(1 for t_ in wtypes if t_ == "handoff"), hi - lo),
                  _safe_div(sum(1 for t_ in wtypes if t_ == "status"), hi - lo),
                  _safe_div(sum(1 for t_ in wtypes if t_ == "state_update"), hi - lo),
                  _safe_div(sum(1 for t_ in wtypes if t_ == "final"), hi - lo),
                  len({intents[j] for j in win if intents[j]}),
                  float(np.mean(goal_ov[lo:hi])), float(np.max(goal_ov[lo:hi])),
                  float(np.max(norm_run[lo:hi])),
                  _safe_div(i - lo, rad), _safe_div(hi - 1 - i, rad)]
        X[i] = v

    assert X.shape[1] == N_FEATURES, (X.shape, N_FEATURES)
    return X, np.zeros(n, dtype=np.int64)
