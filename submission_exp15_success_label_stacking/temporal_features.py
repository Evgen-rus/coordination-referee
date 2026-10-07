"""Experiment 06: temporal dynamics of unresolved handoffs.

Target failure
--------------
On long runs the Exp05 lifecycle features buy nothing (long-run
``dropped_handoff`` F1 was 0.4661 for both Exp03 B and Exp05 B, 0.4764 for C).
Diagnosis: nearly every ``lc_*`` feature is an instantaneous *snapshot* (how many
unresolved, which intent, what share of the run) and not a *trajectory*.  Three
blind spots follow:

1. ``lc_unresolved_ratio`` cannot separate "three tasks assigned early, receiver
   grinding on one for the whole run" from "three tasks dropped at once";
2. ``lc_silent_after_assign`` is a raw message COUNT with no length
   normalisation, so 3 messages mean "active" in a 10-message run and "silent"
   in a 100-message run;
3. ``lc_no_progress_tail`` and ``lc_unresolved_last20pct`` both hang off the
   absolute end of the run, never off the *lifespan of the individual task*.

This module adds the two missing quantities -- per-task age and run-wide
backlog trajectory -- plus the receiver-activity and reassignment-timing
signals, all normalised by run length.

Design rules
------------
* Lifecycle semantics are NOT re-invented: ``build_lifecycles`` is imported from
  Exp05 so the intent-aware delivery matching (including the intent-free
  fallback) is exactly the one already validated by Exp05's leakage audit.
* Every feature is a ratio or a position in ``[0, 1]``; absolute counts are only
  present as a numerator, so nothing scales with run length.
* No ``label`` / ``success`` / ``fault_turn``; no keyword lexicon; no new deps.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

# Submission build: `lifecycle.py` sits next to this file, so the original
# repo-relative path hack is replaced by a plain local import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import build_lifecycles   # noqa: E402

# --------------------------------------------------------------------------
# Feature registry, split into the five ablation blocks.
# --------------------------------------------------------------------------
BLOCK_1_AGE = [
    "ua_age_max_rel",
    "ua_age_mean_rel",
    "ua_age_p90_rel",
    "ua_oldest_share_gt10",
    "ua_oldest_share_gt25",
    "ua_oldest_share_gt50",
]
BLOCK_2_BACKLOG = [
    "bd_max_rel",
    "bd_at25",
    "bd_at50",
    "bd_at75",
    "bd_at100",
    "bd_slope",
    "bd_grows_late",
    "bd_tail_growth",
    "bd_recovery_events_rel",
]
BLOCK_3_RECEIVER = [
    "ra_own_msgs_after_mean_rel",
    "ra_own_msgs_after_max_rel",
    "ra_active_to_end_frac",
    "ra_own_msgs_per_turn",
    "ra_other_arts_after_max_rel",
    "ra_other_arts_exists",
    "ra_frac_lost_while_active",
    "ra_silence_oldest",
]
BLOCK_4_REASSIGN = [
    "rt_reassign_lat_mean_rel",
    "rt_reassign_after_50",
    "rt_late_reassign_noresult",
    "rt_old_owner_result_after",
]
BLOCK_5_TAIL = [
    "ld_unresolved_after_last_art",
    "ld_silence_span_rel",
]

BLOCKS = {
    "1_age": BLOCK_1_AGE,
    "2_backlog": BLOCK_2_BACKLOG,
    "3_receiver": BLOCK_3_RECEIVER,
    "4_reassign": BLOCK_4_REASSIGN,
    "5_tail": BLOCK_5_TAIL,
}
ALL_NAMES = BLOCK_1_AGE + BLOCK_2_BACKLOG + BLOCK_3_RECEIVER + \
    BLOCK_4_REASSIGN + BLOCK_5_TAIL
assert len(ALL_NAMES) == 29, "expected 29 features, got %d" % len(ALL_NAMES)
assert len(set(ALL_NAMES)) == 29, "duplicate feature names"
# must not collide with anything Exp05 already produced
_EXP05 = {
    "lc_n_lifecycles", "lc_n_delivered", "lc_n_unresolved", "lc_n_unacked",
    "lc_n_acked_nodeliver", "lc_unresolved_ratio", "lc_unresolved_per_msg",
    "lc_unresolved_per_agent", "lc_unresolved_per_intent", "lc_unresolved_last20pct",
    "lc_unresolved_last20pct_ratio", "lc_unresolved_firsthalf", "lc_first_unresolved_pos",
    "lc_mean_unresolved_pos", "lc_acked_ratio", "lc_ack_lat_mean", "lc_ack_lat_max",
    "lc_ack_lat_per_msg", "lc_del_lat_mean", "lc_del_lat_max", "lc_del_lat_p90",
    "lc_del_lat_per_msg", "lc_late_deliveries", "lc_late_ratio", "lc_late_per_assign",
    "lc_n_reassign", "lc_reassign_ratio", "lc_multi_owner_intents", "lc_multi_owner_ratio",
    "lc_result_after_reassign", "lc_recovered", "lc_recovered_ratio",
    "lc_dup_delivery_intents", "lc_dup_delivery_ratio", "lc_silent_after_assign",
    "lc_silent_after_assign_ratio", "lc_silent_other_intent",
    "lc_longest_unresolved_chain", "lc_no_progress_tail", "lc_delivered_ratio",
    "lc_orphan_assign_ratio",
}
assert not (set(ALL_NAMES) & _EXP05), "collides with Exp05: %s" % (
    set(ALL_NAMES) & _EXP05)


def _d(a, b) -> float:
    try:
        return float(a) / float(b) if b else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def backlog_at(lcs: List[Dict], p: float) -> int:
    """Open-but-unclosed lifecycles at run position ``p``."""
    c = 0
    for lc in lcs:
        if lc["t"] <= p:
            if not lc["delivered"]:
                c += 1
            elif lc["deliver_t"] > p:
                c += 1
    return c


def backlog_recovery_events(lcs: List[Dict]) -> int:
    """Number of distinct times the open backlog shrinks (a delivery happens)."""
    times = sorted(lc["deliver_t"] for lc in lcs if lc["delivered"])
    return len(times)


def temporal_features(run: Dict[str, Any]) -> Dict[str, float]:
    msgs = run.get("messages") or []
    arts = run.get("artifacts") or []
    n = len(msgs)
    end = max(1.0, float(n - 1))
    f: Dict[str, float] = {}

    lcs = build_lifecycles(msgs, arts)
    n_lc = len(lcs)
    unres = [lc for lc in lcs if not lc["delivered"]]
    n_unres = len(unres)

    # last turn on which each agent sent anything
    last_turn: Dict[str, int] = defaultdict(lambda: -1)
    for m in msgs:
        frm = m.get("from")
        if frm:
            last_turn[frm] = max(last_turn[frm], int(m.get("t", 0)))

    # other-intent artifacts produced by an agent after a given turn
    arts_by_agent: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for a in arts:
        by = a.get("by")
        if by:
            arts_by_agent[by].append((int(a.get("t", 0)), a.get("subtask") or ""))
    for v in arts_by_agent.values():
        v.sort()

    # ---------------- BLOCK 1: age of unresolved handoffs ----------------
    ages = [end - lc["t"] for lc in unres]
    f["ua_age_max_rel"] = _d(max(ages), end) if ages else 0.0
    f["ua_age_mean_rel"] = _d(float(np.mean(ages)), end) if ages else 0.0
    f["ua_age_p90_rel"] = _d(float(np.percentile(ages, 90)), end) if ages else 0.0
    for thr, name in ((0.10, "ua_oldest_share_gt10"),
                      (0.25, "ua_oldest_share_gt25"),
                      (0.50, "ua_oldest_share_gt50")):
        k = sum(1 for a in ages if a > thr * end)
        f[name] = _d(k, n_unres) if n_unres else 0.0

    # ---------------- BLOCK 2: backlog dynamics ----------------
    b25 = backlog_at(lcs, 0.25 * end)
    b50 = backlog_at(lcs, 0.50 * end)
    b75 = backlog_at(lcs, 0.75 * end)
    b100 = backlog_at(lcs, end)
    bmax = max((backlog_at(lcs, lc["t"]) for lc in lcs), default=0) if lcs else 0
    f["bd_max_rel"] = _d(bmax, n_lc)
    f["bd_at25"] = _d(b25, n_lc)
    f["bd_at50"] = _d(b50, n_lc)
    f["bd_at75"] = _d(b75, n_lc)
    f["bd_at100"] = _d(b100, n_lc)
    # review fix 2: slope on the NORMALISED backlog, so run length cannot leak in
    f["bd_slope"] = (f["bd_at100"] - f["bd_at25"]) / 0.75
    # review fix 1: require a non-empty backlog, else 0 -> 0 would score 1
    f["bd_grows_late"] = 1.0 if (b75 > 0 and b100 >= b75) else 0.0
    f["bd_tail_growth"] = _d(b100 - b75, b75)
    f["bd_recovery_events_rel"] = _d(backlog_recovery_events(lcs), n_lc)

    # ---------------- BLOCK 3: receiver activity after the handoff -------
    own_counts = [float(lc["n_owner_msgs_after"]) for lc in unres]
    f["ra_own_msgs_after_mean_rel"] = _d(
        float(np.mean(own_counts)) if own_counts else 0.0, n)
    f["ra_own_msgs_after_max_rel"] = _d(max(own_counts) if own_counts else 0.0, n)

    # review fix 4: only assignments made before 80% of the run can be judged
    # "active to the end"; a handoff at 95% has no room to prove anything.
    late_cut = 0.8 * end
    early_unres = [lc for lc in unres if lc["t"] <= late_cut]
    if early_unres:
        k = sum(1 for lc in early_unres if last_turn.get(lc["to"], -1) > late_cut)
        f["ra_active_to_end_frac"] = _d(k, len(early_unres))
    else:
        f["ra_active_to_end_frac"] = 0.0

    # message density over the task's own lifetime, not over the whole run
    per_turn = []
    for lc in unres:
        span = end - lc["t"]
        if span >= 1:
            per_turn.append(lc["n_owner_msgs_after"] / span)
    f["ra_own_msgs_per_turn"] = float(np.mean(per_turn)) if per_turn else 0.0

    # artifacts on OTHER tasks by the same receiver after the assignment:
    # the receiver is alive and productive, so the loss is task-specific
    other_arts = []
    for lc in unres:
        mine = lc["intent"]
        k = sum(1 for (t, sub) in arts_by_agent.get(lc["to"], ())
                if t > lc["t"] and sub != mine)
        other_arts.append(float(k))
    f["ra_other_arts_after_max_rel"] = _d(max(other_arts) if other_arts else 0.0, n)
    f["ra_other_arts_exists"] = 1.0 if (other_arts and max(other_arts) > 0) else 0.0
    active = sum(1 for lc in unres if lc["n_owner_msgs_after"] > 0)
    f["ra_frac_lost_while_active"] = _d(active, n_unres) if n_unres else 0.0

    # the OLDEST unresolved task: did its receiver go quiet in the back half?
    if unres:
        oldest = max(unres, key=lambda lc: lc["t"] * -1)
        f["ra_silence_oldest"] = 1.0 if last_turn.get(oldest["to"], -1) <= 0.5 * end \
            else 0.0
    else:
        f["ra_silence_oldest"] = 0.0

    # ---------------- BLOCK 4: reassignment timing -----------------------
    by_intent: Dict[str, List[Dict]] = defaultdict(list)
    for lc in lcs:
        if lc["intent"]:
            by_intent[lc["intent"]].append(lc)

    # Results as raw protocol evidence, NOT via lifecycle attribution: Exp05's
    # greedy LIFO matching books a late delivery against the MOST RECENT
    # assignment, so when the original owner finishes after the task moved to
    # somebody else, the delivery lands on the *new* owner's lifecycle and an
    # attribution-based test can never fire.  "Did the original owner deliver
    # after reassignment" is a question about who sent the result and when, so
    # it is answered from the message log and the artifact table.
    result_msgs: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for m in msgs:
        if m.get("type") == "handoff" and (m.get("refs") or []) and m.get("intent"):
            result_msgs[(m.get("from") or "", m.get("intent") or "")] \
                .append(int(m.get("t", 0)))
    for a in arts:
        if a.get("by") and a.get("subtask"):
            result_msgs[(a["by"], a["subtask"])].append(int(a.get("t", 0)))

    reass_lats: List[float] = []
    late_reassign_noresult = 0.0
    n_reassign = 0
    old_owner_result_after = False
    for it, grp in by_intent.items():
        grp = sorted(grp, key=lambda x: x["t"])
        for i in range(1, len(grp)):
            prev, cur = grp[i - 1], grp[i]
            lat = cur["t"] - prev["t"]
            n_reassign += 1
            reass_lats.append(lat)
            if cur["t"] > 0.5 * end:
                f["rt_reassign_after_50"] = 1.0
            if lat > 0.25 * end and not prev["delivered"]:
                late_reassign_noresult += 1.0
        if len(grp) >= 2:
            orig = grp[0]["to"]
            reassign_t = grp[1]["t"]
            if any(t > reassign_t for t in result_msgs.get((orig, it), ())):
                old_owner_result_after = True
    f.setdefault("rt_reassign_after_50", 0.0)
    f["rt_reassign_lat_mean_rel"] = _d(
        float(np.mean(reass_lats)) if reass_lats else 0.0, end)
    f["rt_late_reassign_noresult"] = _d(late_reassign_noresult, n_reassign)
    f["rt_old_owner_result_after"] = 1.0 if old_owner_result_after else 0.0

    # ---------------- BLOCK 5: end-of-run behaviour ----------------------
    art_ts = [int(a.get("t", 0)) for a in arts if a.get("by")]
    last_art = max(art_ts) if art_ts else None
    if n_unres and last_art is not None:
        after = sum(1 for lc in unres if lc["t"] > last_art)
        f["ld_unresolved_after_last_art"] = _d(after, n_unres)
    else:
        f["ld_unresolved_after_last_art"] = 0.0
    if unres:
        oldest = min(unres, key=lambda lc: lc["t"])
        lt = last_turn.get(oldest["to"], -1)
        f["ld_silence_span_rel"] = _d(max(0.0, end - lt), end) if lt >= 0 else 0.0
    else:
        f["ld_silence_span_rel"] = 0.0

    out = {}
    for k in ALL_NAMES:
        v = float(f.get(k, 0.0))
        out[k] = v if np.isfinite(v) else 0.0
    return out


def build_matrix(runs: List[Dict[str, Any]], columns=None):
    import pandas as pd
    cols = columns or ALL_NAMES
    return pd.DataFrame([temporal_features(r) for r in runs],
                        columns=cols).fillna(0.0)
