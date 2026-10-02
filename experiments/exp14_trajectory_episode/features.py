"""Exp14 - explicit bounded repeat-episode representation for `runaway_loop`.

THE ONE HYPOTHESIS
------------------
    Explicit bounded repeat-episode representation improves classification of
    `runaway_loop`, because the current 312-feature Exp13 representation stores
    only order-free repeat counts (`norm_dup_max`) and adjacency streaks
    (`max_sender_repeat_streak`), but loses episode identity, chronology,
    interleaving, agent-ring structure and progress inside the episode.

The read-only diagnostic that motivated this (experiments/EXP13_ERROR_AUDIT.md,
section 9) graded `trajectory/episode` as VERDICT B - observable but genuinely
lost by the current representation - and measured episode identity at
r = 0.6158 against `norm_dup_max`: partially reconstructible, not present.
The competing direction, `assignment-delivery`, was graded VERDICT A and is
deliberately NOT touched here.

WHAT THIS MODULE IS NOT
-----------------------
It does not re-derive what the current features already encode. The audit
showed that order-free multiplicity, adjacency streaks and relative repeat
positions are already carried by `norm_dup_max`, `max_sender_repeat_streak`,
`max_consecutive_repeat` and `pos_streak_start`. Those stay. The 16 columns
below add only what is structurally absent: the episode AS AN OBJECT - how many
there are, how long the longest runs, whether agents rotate, whether other work
interleaves, and whether anything new happens inside the episode window.

It is also not a localizer. The audit measured episode-start hit@2 at 0.252
against the current L1 conditional `runaway_loop` hit@2 of 0.831, so episode
onset is a worse turn rule than the incumbent and is never used as one. This
block targets CLASSIFICATION ONLY; the L1 rule is untouched.

FROZEN DETECTOR (fixed before any model was fitted, never changed after)
------------------------------------------------------------------------
Copied verbatim from `experiments/exp13_error_audit.py`, which is the audited
artifact. The rules are:

    signature   = the EXISTING production normaliser `features._norm(text)`
                  - lowercase, digits -> '#', `art_*` and `a<n>` -> '@',
                  punctuation dropped. It is the same key the existing
                  `norm_dup_max` feature uses, so "already reconstructible?"
                  stays an honest question. NO new normaliser is invented.
    episode     = a maximal ordered run of occurrences of ONE signature whose
                  consecutive occurrence turns are at most EP_GAP_MAX apart.
                  Other signatures may interleave - unlike the adjacency-only
                  `max_sender_repeat_streak`.
    kept if     length >= EP_MIN_LEN, per the ontology's "3-14 repeats in a
                  row without a new result".
    cycle       = the episode touches >= EP_CYCLE_MIN distinct senders. The
                  ontology's runaway case is "an agent OR A RING OF 2-3 AGENTS",
                  so sender is RECORDED per occurrence, never folded into the
                  signature key: in a ring the sender legitimately rotates.
    progress    = a new artifact, a new `shared_state` write, or a delivered
                  message inside the episode window.

The normaliser is INJECTED (`norm_fn`), not imported. This module is a leaf
with no `sys.path` mutation, and injection makes it structurally impossible to
silently introduce a second normaliser variant: the caller must hand over
production `features._norm` or nothing at all.

THE THREE DEFINITIONS RESOLVED BEFORE CV
----------------------------------------
1. `ep_episodes_with_progress_share` is defined BROADLY: an episode "has
   progress" if it contains a new artifact, a new `shared_state` write, OR a
   delivered message. The audit's `no_progress` flag uses the narrower "no
   artifact and no state write", and its `episodes_with_new_result` uses the
   broad one. Taking the narrow complement would make this column a
   mathematically identical duplicate of `1 - ep_no_progress_share`, which the
   protocol forbids keeping; the broad reading is the non-degenerate one.
   Consequently this column is NOT the complement of `ep_no_progress_share`.
2. `ep_earliest_start_rel` is the MINIMUM start turn over ALL episodes. The
   audit's `longest_start_rel` keyed on the longest episode; the name says
   earliest, and `start_turn` is stored per episode, so this is a reading of
   the field, not an invention.
3. A run with no episode yields all zeros, with the two positional columns set
   to -1.0 as an explicit "no onset exists" sentinel - the same convention the
   audit used. Every `_rel` column stays inside [0, 1] otherwise.

LEAKAGE
-------
The only inputs read are `messages`, `artifacts` and `shared_state`, and only
their observable fields (`t`, `from`, `to`, `type`, `intent`, `text`, `by`).
`label`, `success` and `fault_turn` are never touched; `test_features.py` proves
it by target perturbation over 400 runs, and `run_experiment.py` re-proves it
over the same corpus before any fit.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List

import numpy as np

# --- detector constants, DECLARED BEFORE THE DETECTOR RUNS ----------------
# Ontology (`runaway_loop`): "an agent (or a ring of 2-3 agents) repeats
# practically identical actions 3-14 times in a row without a new result."
EP_MIN_LEN = 3        # ontology: "3-14 repeats in a row"
EP_GAP_MAX = 6        # max inter-repeat turn gap still counted as one episode
EP_CYCLE_MIN = 2      # >= 2 distinct senders = a ring rather than one agent

FEATURE_NAMES = (
    # A. episode amount / size
    "ep_n_episodes",
    "ep_has_episode",
    "ep_longest_len",
    "ep_longest_span_rel",
    "ep_repeat_event_share",
    # C. agent-ring structure
    "ep_max_distinct_agents",
    "ep_has_multi_agent_cycle",
    "ep_n_cycle_episodes",
    # D. interleaving
    "ep_n_interleaved_episodes",
    "ep_interleaved_share",
    # E. progress / no-progress inside the episode
    "ep_n_no_progress_episodes",
    "ep_no_progress_share",
    "ep_longest_no_progress_span_rel",
    "ep_episodes_with_progress_share",
    # F. relative onset / span
    "ep_earliest_start_rel",
    "ep_latest_end_rel",
)
N_FEATURES = len(FEATURE_NAMES)

NO_ONSET = -1.0       # sentinel for a run that has no episode at all


def _episode_row(sig, evs, art_turns, state_turns, new_intents):
    """One episode record. Verbatim from exp13_error_audit._episode_row."""
    t0, t1 = evs[0][0], evs[-1][0]
    agents = [e[1] for e in evs]
    distinct = len(set(agents))
    traj = []
    for a in agents:
        if not traj or traj[-1] != a:
            traj.append(a)
    gaps = [evs[i + 1][0] - evs[i][0] for i in range(len(evs) - 1)]
    arts_in = [t for t in art_turns if t0 <= t <= t1]
    state_in = [t for t in state_turns if t0 <= t <= t1]
    n_new_intent = sum(1 for _it, t in new_intents.items() if t0 <= t <= t1)
    delivered = sum(1 for e in evs if e[3] == "handoff" or e[3] == "inform")
    return {
        "signature_sender": "|".join(sorted(set(agents)))[:40],
        "start_turn": int(t0),
        "end_turn": int(t1),
        "length": int(len(evs)),
        "span": int(t1 - t0),
        "consecutive": int(all(g == 1 for g in gaps)),
        "interleaved": int(not all(g == 1 for g in gaps)),
        "max_gap": int(max(gaps)) if gaps else 0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "n_distinct_agents": int(distinct),
        "n_trajectory_segments": int(len(traj)),
        "is_cycle": int(distinct >= EP_CYCLE_MIN),
        "is_2cycle": int(distinct == 2),
        "is_3cycle": int(distinct == 3),
        "same_agent": int(distinct == 1),
        "agent_traj": "|".join(traj[:6]),
        "n_artifacts_during": int(len(arts_in)),
        "n_state_during": int(len(state_in)),
        "n_new_intent_during": int(n_new_intent),
        "n_deliver_msgs": int(delivered),
        # narrow definition, inherited verbatim from the audit
        "no_progress": int(not arts_in and not state_in),
        "no_progress_span": int(t1 - t0) if (not arts_in and not state_in) else 0,
        "to_agent": evs[0][2],
    }


def build_episodes(run: Dict[str, Any],
                   norm_fn: Callable[[str], str]) -> List[Dict[str, Any]]:
    """Transparent, label-blind repeat-episode detector. No ML, no semantics.

    ``norm_fn`` must be the existing production normaliser
    ``submission_exp13_wait_dependency_graph.features._norm``; it is injected
    so this module stays a leaf and no new normaliser can appear by accident.
    """
    msgs = run.get("messages") or []
    arts = run.get("artifacts") or []
    state = run.get("shared_state") or []
    if len(msgs) < EP_MIN_LEN:
        return []

    occ = defaultdict(list)          # _norm(text) -> [(turn, from, to, type)]
    for m in msgs:
        sig = norm_fn(m.get("text") or "")
        if not sig:
            continue
        occ[sig].append((int(m.get("t", 0)), m.get("from") or "",
                         m.get("to") or "", m.get("type") or ""))

    art_turns = sorted(int(a.get("t", 0)) for a in arts)
    state_turns = sorted(int(s.get("t", 0)) for s in state
                         if s.get("t") is not None)
    new_intents = {}
    for m in msgs:
        it = (m.get("intent") or "").strip()
        if it and it not in new_intents:
            new_intents[it] = int(m.get("t", 0))

    out = []
    for sig, lst in occ.items():
        if len(lst) < EP_MIN_LEN:
            continue
        cur = [lst[0]]
        for ev in lst[1:]:
            if ev[0] - cur[-1][0] <= EP_GAP_MAX:
                cur.append(ev)
            else:
                out.append(_episode_row(sig, cur, art_turns, state_turns,
                                        new_intents))
                cur = [ev]
        out.append(_episode_row(sig, cur, art_turns, state_turns, new_intents))
    out = [e for e in out if e["length"] >= EP_MIN_LEN]
    out.sort(key=lambda e: (-e["length"], e["start_turn"]))
    return out


def _safe_div(a, b) -> float:
    return float(a) / b if b else 0.0


def episode_features(run: Dict[str, Any],
                     norm_fn: Callable[[str], str]) -> Dict[str, float]:
    """The 16 frozen columns for one run, keyed by ``FEATURE_NAMES``."""
    msgs = run.get("messages") or []
    n_msgs = len(msgs)
    span_den = float(max(1, n_msgs - 1))
    f = {k: 0.0 for k in FEATURE_NAMES}

    eps = build_episodes(run, norm_fn)
    if not eps:
        f["ep_earliest_start_rel"] = NO_ONSET
        f["ep_latest_end_rel"] = NO_ONSET
        return f

    n_ep = len(eps)
    lg = eps[0]                                   # longest, earliest on a tie
    n_cycle = sum(e["is_cycle"] for e in eps)
    n_inter = sum(e["interleaved"] for e in eps)
    n_noprog = sum(e["no_progress"] for e in eps)
    n_prog = sum(1 for e in eps
                 if e["n_artifacts_during"] > 0 or e["n_state_during"] > 0
                 or e["n_deliver_msgs"] > 0)

    f["ep_n_episodes"] = float(n_ep)
    f["ep_has_episode"] = 1.0
    f["ep_longest_len"] = float(lg["length"])
    f["ep_longest_span_rel"] = _safe_div(lg["span"], span_den)
    f["ep_repeat_event_share"] = _safe_div(sum(e["length"] for e in eps),
                                           max(1, n_msgs))
    f["ep_max_distinct_agents"] = float(max(e["n_distinct_agents"] for e in eps))
    f["ep_has_multi_agent_cycle"] = float(n_cycle > 0)
    f["ep_n_cycle_episodes"] = float(n_cycle)
    f["ep_n_interleaved_episodes"] = float(n_inter)
    f["ep_interleaved_share"] = _safe_div(n_inter, n_ep)
    f["ep_n_no_progress_episodes"] = float(n_noprog)
    f["ep_no_progress_share"] = _safe_div(n_noprog, n_ep)
    f["ep_longest_no_progress_span_rel"] = _safe_div(
        max(e["no_progress_span"] for e in eps), span_den)
    f["ep_episodes_with_progress_share"] = _safe_div(n_prog, n_ep)
    f["ep_earliest_start_rel"] = _safe_div(min(e["start_turn"] for e in eps),
                                           span_den)
    f["ep_latest_end_rel"] = _safe_div(max(e["end_turn"] for e in eps), span_den)
    return f


def build_matrix(runs, norm_fn: Callable[[str], str]) -> np.ndarray:
    """``(n_runs, 16)`` float64 matrix in ``FEATURE_NAMES`` order."""
    out = np.zeros((len(runs), N_FEATURES), dtype=np.float64)
    for i, run in enumerate(runs):
        f = episode_features(run, norm_fn)
        out[i] = [f[k] for k in FEATURE_NAMES]
    if not np.isfinite(out).all():
        raise ValueError("episode block contains non-finite values")
    return out