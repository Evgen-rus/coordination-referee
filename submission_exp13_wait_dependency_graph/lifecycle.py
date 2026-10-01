"""Experiment 05: handoff / intent lifecycle reconstruction + deadlock structure.

Why this exists
---------------
The baseline handoff accounting is deliberately coarse (``baseline/features.py``):

* ``unanswered_assign``  -> "was there ANY later message from the receiver",
  regardless of intent, ordering or topic;
* ``undelivered_assign`` -> "is the intent absent from the set of intents this
  agent EVER delivered", a set with no temporal component, so a delivery that
  happens *after* the reassignment still marks the original as delivered.

Neither models the documented lifecycle
``assignment -> ack -> work -> result/artifact -> delivery / reassignment / recovery``.

This module reconstructs that lifecycle per assignment and aggregates it to the
run level, then adds a small deadlock-specific block that separates mutual
waiting (deadlock) from a simply unanswered task (dropped_handoff).

Everything is derived from protocol structure: message ``type``, ``from``, ``to``,
``intent``, ``refs`` and turn order, plus artifacts and shared-state.  No lexicon
matching, no ``label`` / ``success`` / ``fault_turn``, no per-class hand tuning.
Division by zero yields 0.0.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Submission build: this module has no repo-relative imports left, so the
# original path hack is dropped rather than repointed.

# ---------------------------------------------------------------------------
# Feature name registry.  Two blocks: LIFECYCLE and DEADLOCK.
# ---------------------------------------------------------------------------
LIFECYCLE_NAMES = [
    # --- volume of lifecycles ---
    "lc_n_lifecycles",
    "lc_n_delivered",
    "lc_n_unresolved",
    "lc_n_unacked",
    "lc_n_acked_nodeliver",
    # --- unresolved, absolute + normalised ---
    "lc_unresolved_ratio",
    "lc_unresolved_per_msg",
    "lc_unresolved_per_agent",
    "lc_unresolved_per_intent",
    "lc_unresolved_last20pct",
    "lc_unresolved_last20pct_ratio",
    "lc_unresolved_firsthalf",
    "lc_first_unresolved_pos",
    "lc_mean_unresolved_pos",
    # --- acknowledgement ---
    "lc_acked_ratio",
    "lc_ack_lat_mean",
    "lc_ack_lat_max",
    "lc_ack_lat_per_msg",
    # --- delivery latency ---
    "lc_del_lat_mean",
    "lc_del_lat_max",
    "lc_del_lat_p90",
    "lc_del_lat_per_msg",
    "lc_late_deliveries",
    "lc_late_ratio",
    "lc_late_per_assign",
    # --- reassignment / recovery ---
    "lc_n_reassign",
    "lc_reassign_ratio",
    "lc_multi_owner_intents",
    "lc_multi_owner_ratio",
    "lc_result_after_reassign",
    "lc_recovered",
    "lc_recovered_ratio",
    "lc_dup_delivery_intents",
    "lc_dup_delivery_ratio",
    # --- silence / progress ---
    "lc_silent_after_assign",
    "lc_silent_after_assign_ratio",
    "lc_silent_other_intent",
    "lc_longest_unresolved_chain",
    "lc_no_progress_tail",
    "lc_delivered_ratio",
    "lc_orphan_assign_ratio",
]

DEADLOCK_NAMES = [
    "dl_mutual_pairs",
    "dl_mutual_pairs_per_agent",
    "dl_mutual_unresolved_pairs",
    "dl_mutual_first_pos",
    "dl_tail_after_mutual",
    "dl_status_no_result",
    "dl_status_no_result_ratio",
    "dl_status_no_result_per_msg",
    "dl_unresolved_in_scc",
    "dl_unresolved_in_scc_ratio",
    "dl_status_chain_max",
    "dl_both_ends_unresolved",
    "dl_wait_in_no_intent",
]

DEADLOCK_SET = set(DEADLOCK_NAMES)
ALL_NAMES = LIFECYCLE_NAMES + DEADLOCK_NAMES


def _d(a, b) -> float:
    try:
        return float(a) / float(b) if b else 0.0
    except (TypeError, ValueError):
        return 0.0


def _p90(xs: List[float]) -> float:
    return float(np.percentile(xs, 90)) if xs else 0.0


# ---------------------------------------------------------------------------
# lifecycle reconstruction
# ---------------------------------------------------------------------------
def build_lifecycles(msgs: List[Dict], arts: List[Dict]) -> List[Dict[str, Any]]:
    """One record per assignment (a ``handoff`` message without ``refs``).

    Matching is greedy in time order: a delivery closes the most recently opened
    still-open lifecycle with the same intent, or -- when the intent is missing
    -- the most recently opened lifecycle assigned to the delivering agent.  This
    single mechanism handles present and absent ``intent`` without heuristics
    tuned to any class.
    """
    n = len(msgs)
    lifecycles: List[Dict[str, Any]] = []
    open_by_intent: Dict[str, List[int]] = defaultdict(list)
    open_by_agent: Dict[str, List[int]] = defaultdict(list)

    art_by_sub: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for a in arts:
        sub = a.get("subtask") or ""
        if sub:
            art_by_sub[(a.get("by") or "", sub)].append(a)

    def close(lc: Dict[str, Any], t: int, kind: str, sender: str) -> None:
        lc["delivered"] = True
        lc["deliver_t"] = t
        lc["deliver_kind"] = kind
        lc["deliver_sender"] = sender
        lc["del_lat"] = t - lc["t"]
        if lc["intent"] and lc["to"] not in lc["result_senders"]:
            lc["result_senders"].add(lc["to"])
            lc["owner_delivered"] = True

    for m in msgs:
        t = int(m.get("t", 0))
        typ = m.get("type")
        refs = m.get("refs") or []
        frm, to = m.get("from") or "", m.get("to") or ""
        it = (m.get("intent") or "").strip()

        if typ == "handoff" and not refs:
            # ---- open a lifecycle ----
            prev_same = None
            for lc in reversed(lifecycles):
                if lc["intent"] and lc["intent"] == it:
                    prev_same = lc
                    break
            lc = {
                "t": t, "from": frm, "to": to, "intent": it,
                "acked": False, "ack_t": None, "delivered": False, "deliver_t": None,
                "del_lat": None, "deliver_kind": None, "deliver_sender": None,
                "owner_delivered": False, "result_senders": set(),
                "reassigned": False, "n_owner_msgs_after": 0,
                "n_other_intent_msgs_after": 0, "recovered": False,
                "has_artifact": False, "art_t": None,
                "prev_owner": prev_same["to"] if prev_same else None,
            }
            if prev_same is not None:
                lc["prev_owner"] = prev_same["to"]
            lifecycles.append(lc)
            if it:
                open_by_intent[it].append(len(lifecycles) - 1)
            open_by_agent[to].append(len(lifecycles) - 1)

        elif typ == "handoff" and refs:
            # ---- a delivery: close the most recent matching open lifecycle ----
            target = None
            if it and open_by_intent.get(it):
                idx = open_by_intent[it][-1]
                target = lifecycles[idx]
                open_by_intent[it].pop()
            elif open_by_agent.get(frm):
                # intent-free matching: same agent, most recent open assignment
                for idx in reversed(open_by_agent[frm]):
                    cand = lifecycles[idx]
                    if not cand["delivered"]:
                        target = cand
                        open_by_agent[frm].remove(idx)
                        break
            if target is not None:
                close(target, t, "refs", frm)
                target["result_senders"].add(frm)
                if not target["owner_delivered"] and frm == target["to"]:
                    target["owner_delivered"] = True

        else:
            # ---- a non-handoff message: acknowledgement or activity ----
            for lc in lifecycles:
                if lc["delivered"] or int(lc["t"]) >= t:
                    continue
                if frm == lc["to"]:
                    lc["n_owner_msgs_after"] += 1
                    if not lc["acked"] and (not lc["intent"] or
                                            not it or it == lc["intent"]):
                        lc["acked"] = True
                        lc["ack_t"] = t
                elif it and lc["intent"] and it != lc["intent"]:
                    lc["n_other_intent_msgs_after"] += 1

    # ---- artifact-based closure for lifecycles still open ----
    for lc in lifecycles:
        if not lc["intent"]:
            continue
        for key in ((lc["to"], lc["intent"]), (lc["from"], lc["intent"])):
            for a in art_by_sub.get(key, []):
                t = int(a.get("t", 0))
                if t > lc["t"]:
                    lc["has_artifact"] = True
                    if lc["art_t"] is None or t < lc["art_t"]:
                        lc["art_t"] = t
                    if not lc["delivered"]:
                        close(lc, t, "artifact", a.get("by") or "")
                    break

    # ---- reassignment / recovery / duplicate delivery ----
    by_intent: Dict[str, List[Dict]] = defaultdict(list)
    for lc in lifecycles:
        if lc["intent"]:
            by_intent[lc["intent"]].append(lc)
    for it, group in by_intent.items():
        if len(group) > 1:
            owners = {lc["to"] for lc in group}
            for i, lc in enumerate(group):
                if any(g["to"] != lc["to"] for g in group):
                    lc["reassigned"] = True
                    if lc["delivered"]:
                        lc["recovered"] = True
            if len(owners) > 1:
                for lc in group:
                    lc["multi_owner"] = True
        # result after reassignment: the original owner eventually delivered
        first = group[0]
        if first["reassigned"] and first["owner_delivered"]:
            first["result_after_reassign"] = True

    # second delivery for the same intent from a different owner
    deliveries_by_intent: Dict[str, List[Dict]] = defaultdict(list)
    for lc in lifecycles:
        if lc["delivered"] and lc["intent"]:
            deliveries_by_intent[lc["intent"]].append(lc)
    for it, ds in deliveries_by_intent.items():
        if len(ds) > 1 and len({d["deliver_sender"] or d["to"] for d in ds}) > 1:
            for d in ds:
                d["dup_delivery"] = True

    for lc in lifecycles:
        lc.setdefault("reassigned", False)
        lc.setdefault("recovered", False)
        lc.setdefault("multi_owner", False)
        lc.setdefault("result_after_reassign", False)
        lc.setdefault("dup_delivery", False)
    return lifecycles


# ---------------------------------------------------------------------------
# run-level aggregation
# ---------------------------------------------------------------------------
_MISS = object()


def _sccs(nodes: List[str], edges: List[Tuple[str, str]]) -> List[List[str]]:
    """Iterative Tarjan.

    Two separate structures are required and are NOT interchangeable: the DFS
    work list (``work``) and Tarjan's S-stack (``sstack``) of nodes that have been
    visited but not yet assigned to a component.  Conflating them makes a
    non-root frame loop forever, so they are kept distinct here.  Iterative
    rather than recursive so a large graph cannot exhaust the Python stack.
    """
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)

    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    sstack: List[str] = []
    on: set = set()
    out: List[List[str]] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        sstack.append(root)
        on.add(root)
        work = [(root, iter(adj.get(root, ())))]
        while work:
            v, it_ = work[-1]
            w = next(it_, _MISS)
            if w is not _MISS:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    sstack.append(w)
                    on.add(w)
                    work.append((w, iter(adj.get(w, ()))))
                elif w in on:
                    low[v] = min(low[v], index[w])
            else:
                # every successor of v has been processed
                work.pop()
                if low[v] == index[v]:
                    comp: List[str] = []
                    while True:
                        w = sstack.pop()
                        on.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    out.append(comp)
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
    return out


def deadlock_features(msgs: List[Dict], lifecycles: List[Dict], n_msgs: int,
                      max_turn: float) -> Dict[str, float]:
    f: Dict[str, float] = {}
    agents = sorted({m.get("from") for m in msgs if m.get("from")} |
                    {m.get("to") for m in msgs if m.get("to")})

    # directed edges of *unresolved* assignments
    unres_edges = [(lc["from"], lc["to"]) for lc in lifecycles if not lc["delivered"]]
    all_edges = [(lc["from"], lc["to"]) for lc in lifecycles]

    # mutual pairs: A->B and B->A both appear
    pair_set = set(all_edges)
    mutual = {tuple(sorted((a, b))) for a, b in all_edges if a != b and (b, a) in pair_set}
    unres_pair_set = set(unres_edges)
    mutual_unres = {tuple(sorted((a, b))) for a, b in unres_edges
                    if a != b and (b, a) in unres_pair_set}

    f["dl_mutual_pairs"] = float(len(mutual))
    f["dl_mutual_pairs_per_agent"] = _d(len(mutual), max(1, len(agents)))
    f["dl_mutual_unresolved_pairs"] = float(len(mutual_unres))

    # first turn at which some mutual dependency is fully established:
    # the later of the two directions of the first mutual pair
    first_mutual_t = None
    if mutual:
        assign_turns: Dict[Tuple[str, str], int] = {}
        for lc in lifecycles:
            key = tuple(sorted((lc["from"], lc["to"])))
            if key in mutual and lc["from"] != lc["to"]:
                # the pair is mutual only once BOTH directions have been assigned
                pair_turns = [g["t"] for g in lifecycles
                              if tuple(sorted((g["from"], g["to"]))) == key
                              and g["from"] != g["to"]]
                if len({g["from"] for g in lifecycles
                        if tuple(sorted((g["from"], g["to"]))) == key}) > 1:
                    cand = max(pair_turns)
                    first_mutual_t = cand if first_mutual_t is None else min(first_mutual_t, cand)
                    assign_turns.pop(key, None)
    f["dl_mutual_first_pos"] = _d(first_mutual_t, max_turn) if first_mutual_t is not None else 0.0
    f["dl_tail_after_mutual"] = (max(0.0, (n_msgs - 1 - first_mutual_t) / max_turn)
                                 if first_mutual_t is not None else 0.0)

    # status messages that are not followed by a delivery before the next status
    status_ts = [int(m.get("t", 0)) for m in msgs if m.get("type") == "status"]
    deliver_ts = sorted([int(m.get("t", 0)) for m in msgs
                         if m.get("type") == "handoff" and (m.get("refs") or [])])
    status_no_result = 0
    chain = 0
    best_chain = 0
    import bisect
    for i, t in enumerate(status_ts):
        nxt = status_ts[i + 1] if i + 1 < len(status_ts) else n_msgs
        j = bisect.bisect_left(deliver_ts, t)
        got = j < len(deliver_ts) and deliver_ts[j] < nxt
        if not got:
            status_no_result += 1
            chain += 1
            best_chain = max(best_chain, chain)
        else:
            chain = 0
    f["dl_status_no_result"] = float(status_no_result)
    f["dl_status_no_result_ratio"] = _d(status_no_result, len(status_ts))
    f["dl_status_no_result_per_msg"] = _d(status_no_result, n_msgs)
    f["dl_status_chain_max"] = float(best_chain)

    # unresolved handoffs inside a non-trivial SCC
    comps = [c for c in _sccs(agents, all_edges) if len(c) > 1]
    scc_of: Dict[str, int] = {}
    for i, c in enumerate(comps):
        for a in c:
            scc_of[a] = i
    in_scc = 0
    for lc in lifecycles:
        if not lc["delivered"] and lc["from"] in scc_of and lc["to"] in scc_of \
                and scc_of[lc["from"]] == scc_of[lc["to"]]:
            in_scc += 1
    unres_total = sum(1 for lc in lifecycles if not lc["delivered"])
    f["dl_unresolved_in_scc"] = float(in_scc)
    f["dl_unresolved_in_scc_ratio"] = _d(in_scc, unres_total)

    # both ends of a mutual pair are simultaneously unresolved
    f["dl_both_ends_unresolved"] = float(len(mutual_unres))

    # mutual waiting that the telemetry cannot even attribute to an intent
    unres_set = set(unres_edges)
    f["dl_wait_in_no_intent"] = float(sum(
        1 for lc in lifecycles
        if not lc["delivered"] and not lc["intent"] and lc["from"] != lc["to"]
        and (lc["from"], lc["to"]) in unres_set
        and (lc["to"], lc["from"]) in unres_set))
    return f


def lifecycle_features(run: Dict[str, Any]) -> Dict[str, float]:
    msgs = run.get("messages") or []
    arts = run.get("artifacts") or []
    n = len(msgs)
    max_turn = max(1.0, float(n - 1))
    agents = [a.get("id") for a in (run.get("agents") or []) if a.get("id")]
    lcs = build_lifecycles(msgs, arts)

    f: Dict[str, float] = {}
    n_lc = len(lcs)
    delivered = [lc for lc in lcs if lc["delivered"]]
    unres = [lc for lc in lcs if not lc["delivered"]]
    unacked = [lc for lc in lcs if not lc["acked"]]
    acked_nodeliver = [lc for lc in lcs if lc["acked"] and not lc["delivered"]]
    unres_ids = {id(lc) for lc in unres}

    f["lc_n_lifecycles"] = float(n_lc)
    f["lc_n_delivered"] = float(len(delivered))
    f["lc_n_unresolved"] = float(len(unres))
    f["lc_n_unacked"] = float(len(unacked))
    f["lc_n_acked_nodeliver"] = float(len(acked_nodeliver))

    tail_start = 0.8 * max_turn
    unres_late = [lc for lc in unres if lc["t"] >= tail_start]
    f["lc_unresolved_last20pct"] = float(len(unres_late))
    f["lc_unresolved_last20pct_ratio"] = _d(len(unres_late), n_lc)
    f["lc_unresolved_firsthalf"] = float(sum(1 for lc in unres if lc["t"] < 0.5 * max_turn))
    f["lc_unresolved_ratio"] = _d(len(unres), n_lc)
    f["lc_unresolved_per_msg"] = _d(len(unres), n)
    f["lc_unresolved_per_agent"] = _d(len(unres), len(agents))
    n_intents = len({lc["intent"] for lc in lcs if lc["intent"]})
    f["lc_unresolved_per_intent"] = _d(len(unres), n_intents)
    f["lc_first_unresolved_pos"] = (_d(min(lc["t"] for lc in unres), max_turn)
                                    if unres else 0.0)
    f["lc_mean_unresolved_pos"] = (_d(sum(lc["t"] for lc in unres), max_turn * len(unres))
                                  if unres else 0.0)

    f["lc_acked_ratio"] = _d(n_lc - len(unacked), n_lc)
    ack_lat = [lc["ack_t"] - lc["t"] for lc in lcs if lc["ack_t"] is not None]
    f["lc_ack_lat_mean"] = float(np.mean(ack_lat)) if ack_lat else 0.0
    f["lc_ack_lat_max"] = float(max(ack_lat)) if ack_lat else 0.0
    f["lc_ack_lat_per_msg"] = _d(np.mean(ack_lat) if ack_lat else 0, n)

    del_lat = [lc["del_lat"] for lc in delivered if lc["del_lat"] is not None]
    f["lc_del_lat_mean"] = float(np.mean(del_lat)) if del_lat else 0.0
    f["lc_del_lat_max"] = float(max(del_lat)) if del_lat else 0.0
    f["lc_del_lat_p90"] = _p90(del_lat)
    f["lc_del_lat_per_msg"] = _d(np.mean(del_lat) if del_lat else 0, n)

    late_thr = 0.25 * max_turn
    late = [lc for lc in delivered if (lc["del_lat"] or 0) > late_thr]
    f["lc_late_deliveries"] = float(len(late))
    f["lc_late_ratio"] = _d(len(late), n_lc)
    f["lc_late_per_assign"] = _d(len(late), n_lc)

    reass = [lc for lc in lcs if lc["reassigned"]]
    multi = [lc for lc in lcs if lc.get("multi_owner")]
    rar = [lc for lc in lcs if lc.get("result_after_reassign")]
    rec = [lc for lc in lcs if lc.get("recovered")]
    dup = [lc for lc in lcs if lc.get("dup_delivery")]
    f["lc_n_reassign"] = float(len(reass))
    f["lc_reassign_ratio"] = _d(len(reass), n_lc)
    f["lc_multi_owner_intents"] = float(len({lc["intent"] for lc in multi if lc["intent"]}))
    f["lc_multi_owner_ratio"] = _d(len(multi), n_lc)
    f["lc_result_after_reassign"] = float(len(rar))
    f["lc_recovered"] = float(len(rec))
    f["lc_recovered_ratio"] = _d(len(rec), len(reass))
    f["lc_dup_delivery_intents"] = float(len({lc["intent"] for lc in dup if lc["intent"]}))
    f["lc_dup_delivery_ratio"] = _d(len(dup), n_lc)

    # the receiver kept working (on anything) after the assignment, yet nothing arrived
    silent = [lc for lc in unres if lc["n_owner_msgs_after"] > 0]
    silent_other = [lc for lc in unres if lc["n_other_intent_msgs_after"] > 0]
    f["lc_silent_after_assign"] = float(len(silent))
    f["lc_silent_after_assign_ratio"] = _d(len(silent), n_lc)
    f["lc_silent_other_intent"] = float(len(silent_other))

    # longest run of consecutive assignments, none of which is ever delivered
    chain = best = 0
    for lc in lcs:
        if not lc["delivered"]:
            chain += 1
            best = max(best, chain)
        else:
            chain = 0
    f["lc_longest_unresolved_chain"] = float(best)

    last_deliver = max((lc["deliver_t"] for lc in delivered), default=None)
    f["lc_no_progress_tail"] = (_d(max(0, n - 1 - last_deliver), max_turn)
                                if last_deliver is not None else 0.0)
    f["lc_delivered_ratio"] = _d(len(delivered), n_lc)
    f["lc_orphan_assign_ratio"] = _d(sum(1 for lc in lcs if not lc["intent"]), n_lc)

    f.update(deadlock_features(msgs, lcs, n, max_turn))
    out = {k: float(f.get(k, 0.0)) for k in ALL_NAMES}
    return {k: (v if np.isfinite(v) else 0.0) for k, v in out.items()}


def build_matrix(runs: List[Dict[str, Any]], columns=None):
    import pandas as pd
    cols = columns or ALL_NAMES
    rows = [lifecycle_features(r) for r in runs]
    return pd.DataFrame(rows, columns=cols).fillna(0.0)
