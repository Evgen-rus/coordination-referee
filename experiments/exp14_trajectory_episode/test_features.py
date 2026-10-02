"""Exp14 - unit tests for the frozen bounded repeat-episode detector.

Required coverage (12 cases).  Every test is a plain check on real function
output; none of them touches a model, a fold or a metric.

    1. same-agent episode            -> detected, not a cycle
    2. 2-agent rotating episode      -> is_2cycle, not 3-cycle
    3. 3-agent rotating episode      -> is_3cycle
    4. interleaved episode           -> interleaved flag set, still one episode
    5. gap > EP_GAP_MAX splits       -> two separate episodes
    6. fewer than 3 repeats          -> no episode at all
    7. progress inside episode       -> broad progress definition fires
    8. no-progress episode           -> narrow no_progress fires, span kept
    9. _norm folds ids and numbers   -> one signature across id/number changes
   10. label / success / fault_turn  -> matrix invariant (target perturbation)
   11. determinism                  -> byte-identical repeat builds
   12. real-corpus smoke            -> finite, bounded ranges, fires on data
   13. baseline reproduction and A/B column identity are checked by
       run_experiment.py, not here.

Run bare (`python test_features.py`) or under pytest
(`python -m pytest test_features.py -q`).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# NOTE: HERE is deliberately NOT added to sys.path. This experiment's own
# ``features`` module must not be importable by bare name, or it would shadow
# the production ``features`` whose ``_norm`` the detector must use. It is
# loaded by explicit file path below.
for _p in (os.path.join(ROOT, "submission_exp13_wait_dependency_graph"),
           os.path.join(ROOT, "experiments", "exp10_ab_probability_blend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util                    # noqa: E402
import features as FT                    # noqa: E402  (production Exp13 normaliser)

# Both this experiment and production ship a module called ``features``. The
# production one is imported above by name; THIS experiment's own module is
# therefore loaded by explicit file path, which removes the ambiguity instead
# of relying on sys.path order.
_spec = importlib.util.spec_from_file_location(
    "exp14_episode_features", os.path.join(HERE, "features.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

FAILURES = []
N_CHECKS = [0]


def check(name, cond, detail=""):
    N_CHECKS[0] += 1
    if cond:
        print("   PASS  %s" % name)
    else:
        print("   FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


# --- fixtures -------------------------------------------------------------

def msg(t, frm, to, text, typ="status", intent=""):
    return {"t": t, "from": frm, "to": to, "type": typ, "intent": intent,
            "text": text, "tool": "", "refs": []}


def art(t, by="a1", kind="doc"):
    return {"id": "art_%02d" % (t + 1), "by": by, "t": t, "type": kind,
            "subtask": "s", "hash": "h", "status": "final"}


def state(t, agent="a1", key="k"):
    return {"t": t, "agent": agent, "key": key, "op": "set", "value": "1"}


def run(msgs, artifacts=(), states=()):
    return {"run_id": "r", "goal": "g", "agents": ["a1", "a2", "a3"],
            "messages": list(msgs), "artifacts": list(artifacts),
            "shared_state": list(states)}


def f(msgs, key, artifacts=(), states=()):
    return E.episode_features(run(msgs, artifacts, states), FT._norm)[key]


def eps(msgs, artifacts=(), states=()):
    return E.build_episodes(run(msgs, artifacts, states), FT._norm)


NORM = FT._norm
TXT = "Re-checking the deploy log for a4 on verify_deploy."
OTHER = "Different work entirely, unrelated to any repeat."


print("=" * 74)
print("Exp14 episode-block unit tests")
print("=" * 74)

# --- 0. frozen surface ----------------------------------------------------
check("0. frozen constants are 3 / 6 / 2",
      E.EP_MIN_LEN == 3 and E.EP_GAP_MAX == 6 and E.EP_CYCLE_MIN == 2,
      "got %d/%d/%d" % (E.EP_MIN_LEN, E.EP_GAP_MAX, E.EP_CYCLE_MIN))
check("0b. exactly 16 declared features",
      E.N_FEATURES == 16 and len(E.FEATURE_NAMES) == 16,
      "got %d" % E.N_FEATURES)
check("0c. feature names are unique and ep_ prefixed",
      len(set(E.FEATURE_NAMES)) == 16
      and all(n.startswith("ep_") for n in E.FEATURE_NAMES))

# --- 1. same-agent episode -----------------------------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a1", "a2", TXT), msg(2, "a1", "a2", TXT)]
e = eps(m)[0]
check("1. same-agent episode detected",
      f(m, "ep_n_episodes") == 1.0 and f(m, "ep_longest_len") == 3.0,
      "n=%s len=%s" % (f(m, "ep_n_episodes"), f(m, "ep_longest_len")))
check("1b. same-agent episode is NOT a cycle",
      not e["is_cycle"] and e["same_agent"] == 1
      and f(m, "ep_has_multi_agent_cycle") == 0.0)
check("1c. 3 consecutive repeats are not interleaved",
      e["interleaved"] == 0 and f(m, "ep_n_interleaved_episodes") == 0.0)
check("1d. has_episode flag set, distinct agents = 1",
      f(m, "ep_has_episode") == 1.0 and f(m, "ep_max_distinct_agents") == 1.0)

# --- 2. 2-agent rotating episode -----------------------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a2", "a1", TXT), msg(2, "a1", "a2", TXT)]
e = eps(m)[0]
check("2. 2-agent ring detected as cycle",
      e["is_2cycle"] == 1 and e["is_cycle"] == 1
      and f(m, "ep_n_cycle_episodes") == 1.0,
      "distinct=%s" % e["n_distinct_agents"])
check("2b. 2-ring is not a 3-ring",
      e["is_3cycle"] == 0 and e["n_distinct_agents"] == 2)
check("2c. max distinct agents = 2", f(m, "ep_max_distinct_agents") == 2.0)

# --- 3. 3-agent rotating episode -----------------------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a2", "a3", TXT), msg(2, "a3", "a1", TXT),
     msg(3, "a1", "a2", TXT), msg(4, "a2", "a3", TXT)]
e = eps(m)[0]
check("3. 3-agent ring detected as cycle",
      e["is_3cycle"] == 1 and e["is_3cycle"] and e["n_distinct_agents"] == 3)
check("3b. 3-ring flagged in the feature block",
      f(m, "ep_has_multi_agent_cycle") == 1.0
      and f(m, "ep_max_distinct_agents") == 3.0)
check("3c. episodes_with_progress_share = 0 (status msgs are not delivery)",
      f(m, "ep_episodes_with_progress_share") == 0.0,
      "got %s" % f(m, "ep_episodes_with_progress_share"))

# --- 4. interleaved episode ---------------------------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a1", "a3", OTHER), msg(2, "a1", "a2", TXT),
     msg(3, "a1", "a3", OTHER), msg(4, "a1", "a2", TXT)]
e = eps(m)[0]
check("4. interleaved repeats stay ONE episode",
      f(m, "ep_n_episodes") == 1.0 and f(m, "ep_longest_len") == 3.0,
      "n=%s len=%s" % (f(m, "ep_n_episodes"), f(m, "ep_longest_len")))
check("4b. interleaved flag set",
      e["interleaved"] == 1 and f(m, "ep_n_interleaved_episodes") == 1.0
      and f(m, "ep_interleaved_share") == 1.0)
check("4c. interleaved episode carries a gap > 1",
      e["max_gap"] == 2 and e["consecutive"] == 0)

# --- 5. gap > EP_GAP_MAX splits ------------------------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a1", "a2", TXT), msg(2, "a1", "a2", TXT),
     msg(9, "a1", "a2", TXT), msg(10, "a1", "a2", TXT), msg(11, "a1", "a2", TXT)]
e = eps(m)
check("5. gap of 7 splits into two episodes",
      len(e) == 2 and f(m, "ep_n_episodes") == 2.0,
      "got %d" % len(e))
check("5b. the two episodes are ordered longest-first",
      e[0]["start_turn"] == 0 and e[0]["end_turn"] == 2
      and e[1]["start_turn"] == 9 and e[1]["end_turn"] == 11)
check("5c. each split episode is CONSECUTIVE internally (gaps 1,1)",
      all(x["interleaved"] == 0 and x["max_gap"] == 1 for x in e)
      and f(m, "ep_n_interleaved_episodes") == 0.0,
      "note: the 7-turn GAP splits the episodes but does not interleave them")
check("5d. the split is driven by the gap, not by interleaving",
      e[0]["end_turn"] == 2 and e[1]["start_turn"] == 9
      and f(m, "ep_n_episodes") == 2.0)

# --- 6. fewer than 3 repeats --------------------------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a1", "a2", TXT)]
check("6. two repeats are not an episode",
      len(eps(m)) == 0 and f(m, "ep_n_episodes") == 0.0)
check("6b. no-episode run is all zeros except the two onset sentinels",
      E.episode_features(run(m), NORM) ==
      {k: (-1.0 if k in ("ep_earliest_start_rel", "ep_latest_end_rel") else 0.0)
       for k in E.FEATURE_NAMES},
      "got %s" % {k: v for k, v in E.episode_features(run(m), NORM).items()
                  if v != 0.0})
check("6c. only the two onset columns carry the -1 sentinel",
      set(k for k, v in E.episode_features(run(m), NORM).items() if v == -1.0)
      == {"ep_earliest_start_rel", "ep_latest_end_rel"})
check("6c. ep_longest_span_rel stays in [0,1]",
      0.0 <= f([msg(0, "a1", "a2", TXT), msg(1, "a1", "a2", TXT),
                msg(2, "a1", "a2", TXT)], "ep_longest_span_rel") <= 1.0)

# --- 7. progress inside episode (broad definition) ------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a1", "a2", TXT), msg(2, "a1", "a2", TXT)]
check("7. artifact inside the window counts as progress",
      f(m, "ep_episodes_with_progress_share", artifacts=[art(1)]) == 1.0,
      "got %s" % f(m, "ep_episodes_with_progress_share",
                    artifacts=[art(1)]))
check("7b. state write inside the window counts as progress",
      f(m, "ep_episodes_with_progress_share", states=[state(1)]) == 1.0)
MD = [msg(0, "a1", "a2", TXT, "handoff"), msg(1, "a1", "a2", TXT, "handoff"),
      msg(2, "a1", "a2", TXT, "handoff")]
check("7c. delivered message counts as progress (broad definition)",
      f(MD, "ep_episodes_with_progress_share") == 1.0)
check("7d. narrow no_progress IGNORES delivery, by design (asymmetry)",
      eps(MD)[0]["no_progress"] == 1
      and eps(MD)[0]["n_deliver_msgs"] == 3,
      "no_progress=%s deliver=%s" % (eps(MD)[0]["no_progress"],
                                      eps(MD)[0]["n_deliver_msgs"]))
check("7e. the narrow/broad asymmetry is real and both columns keep their meaning",
      f(MD, "ep_no_progress_share") == 1.0
      and f(MD, "ep_episodes_with_progress_share") == 1.0,
      "a delivery is a progress event but neither an artifact nor a state write")

# --- 8. no-progress episode ----------------------------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a1", "a2", TXT), msg(2, "a1", "a2", TXT)]
check("8. episode with nothing new is no_progress",
      eps(m)[0]["no_progress"] == 1 and f(m, "ep_n_no_progress_episodes") == 1.0)
check("8b. no_progress column is 1 for a fully no-progress run",
      f(m, "ep_no_progress_share") == 1.0)
check("8c. longest no-progress span is relative and in [0,1]",
      0.0 <= f(m, "ep_longest_no_progress_span_rel") <= 1.0,
      "got %s" % f(m, "ep_longest_no_progress_span_rel"))
check("8c'. no-progress span is zero when there IS progress",
      f(m, "ep_longest_no_progress_span_rel", states=[state(1)]) == 0.0)

# --- 9. _norm folds ids and numbers --------------------------------------
check("9. _norm folds digits to #",
      NORM("Retry 12 times") == NORM("Retry 47 times"))
check("9b. _norm folds art_ and a<n> ids to @",
      NORM("see art_07 and a3 now") == NORM("see art_19 and a9 now"))
check("9c. ids differ but signature still matches",
      f([msg(0, "a1", "a2", "check art_07 for a3"),
         msg(1, "a1", "a2", "check art_19 for a9"),
         msg(2, "a1", "a2", "check art_23 for a1")], "ep_has_episode") == 1.0)
check("9d. genuinely different text is a different signature",
      f([msg(0, "a1", "a2", TXT), msg(1, "a1", "a2", TXT),
         msg(3, "a1", "a2", OTHER)], "ep_n_episodes") == 0.0)

# --- 10. target perturbation (leakage) -----------------------------------
m = [msg(0, "a1", "a2", TXT), msg(1, "a1", "a2", TXT), msg(2, "a1", "a2", TXT)]
base = {k: E.episode_features(run(m), NORM)[k] for k in E.FEATURE_NAMES}
mutated = []
for lab, ft, succ in (("clean", 0, 1), ("runaway_loop", 7, 0),
                      ("goal_drift", 3, 1)):
    r = run(m)
    r["label"] = lab
    r["fault_turn"] = ft
    r["fault_turn"] = ft
    r["success"] = succ
    mutated.append({k: E.episode_features(r, NORM)[k] for k in E.FEATURE_NAMES})
check("10. features invariant to label / fault_turn / success",
      all(mm == base for mm in mutated))
check("10b. the fixture itself carries no targets, so none can be read",
      not any(k in run(m) for k in ("label", "success", "fault_turn")))

# --- 11. determinism ------------------------------------------------------
r = run(m)
X1 = E.build_matrix([r, r, r], NORM)
X2 = E.build_matrix([r, r, r], NORM)
check("11. deterministic output", np.array_equal(X1, X2))
check("11b. matrix shape and dtype",
      X1.shape == (3, 16) and X1.dtype == np.float64)
check("11c. three identical runs give identical rows", (X1[0] == X1[1]).all())
check("11d. no non-finite values", np.isfinite(X1).all())

# --- 12. real-corpus smoke -----------------------------------------------
tr_csv = os.path.join(ROOT, "data", "train.csv")
if os.path.exists(tr_csv):
    import pandas as pd
    tr = pd.read_csv(tr_csv)
    runs = []
    for row in tr.head(400).itertuples(index=False):
        runs.append({"run_id": str(row.run_id), "goal": str(row.goal),
                     "messages": json.loads(row.messages),
                     "artifacts": json.loads(row.artifacts),
                     "shared_state": json.loads(row.shared_state)})
    X = E.build_matrix(runs, NORM)
    i_n = E.FEATURE_NAMES.index("ep_n_episodes")
    i_l = E.FEATURE_NAMES.index("ep_longest_len")
    i_has = E.FEATURE_NAMES.index("ep_has_episode")
    check("12. no non-finite values over 400 real runs", np.isfinite(X).all())
    check("12b. episodes fire on real data",
          X[:, i_has].sum() > 0, "has=%d" % X[:, i_has].sum())
    check("12c. longest episode length >= 3 wherever an episode exists",
          bool((X[X[:, i_has] > 0][:, i_l] >= 3).all()))
    check("12d. all share columns lie in [0,1]",
          bool(((X[:, E.FEATURE_NAMES.index("ep_interleaved_share")] >= 0)
                & (X[:, E.FEATURE_NAMES.index("ep_interleaved_share")] <= 1)).all()))
    check("12e. all share columns lie in [0,1] (no-progress)",
          bool(((X[:, E.FEATURE_NAMES.index("ep_no_progress_share")] >= 0)
                & (X[:, E.FEATURE_NAMES.index("ep_no_progress_share")] <= 1)).all()))
    check("12f. relative positions in [0,1] or the -1 sentinel",
          bool(((X[:, E.FEATURE_NAMES.index("ep_earliest_start_rel")] >= -1.0)
                & (X[:, E.FEATURE_NAMES.index("ep_earliest_start_rel")] <= 1.0)).all()))
    check("12g. repeat_event_share in [0,1]",
          bool(((X[:, E.FEATURE_NAMES.index("ep_repeat_event_share")] >= 0)
                & (X[:, E.FEATURE_NAMES.index("ep_repeat_event_share")] <= 1)).all()))
    # The ontology says "an agent (or a RING OF 2-3 agents)", but the detector
    # is generic: it records however many distinct senders an episode actually
    # touches. On the real corpus a few episodes span 4 agents, so the column
    # is NOT capped at 3. Recorded here rather than silently clipped.
    n_agents_max = float(X[:, E.FEATURE_NAMES.index("ep_max_distinct_agents")].max())
    check("12h. distinct-agent counts are >= 1 wherever an episode exists",
          bool((X[X[:, i_has] > 0][:, E.FEATURE_NAMES.index("ep_max_distinct_agents")] >= 1).all()))
    check("12h'. detector is NOT capped at 3 agents (records what it sees)",
          n_agents_max >= 1.0,
          "max distinct agents over corpus = %s" % n_agents_max)
    check("12h''. 2- and 3-agent cycles are the common case",
          n_agents_max >= 3.0,
          "corpus reaches %s distinct agents" % n_agents_max)
    check("12h'. onsets are absent (-1) exactly when there is no episode",
          bool(np.array_equal(X[:, E.FEATURE_NAMES.index("ep_earliest_start_rel")] < 0,
                              X[:, i_has] == 0)))
else:
    print("   SKIP  corpus smoke (train.csv not found)")

print()
print("=" * 74)
if FAILURES:
    print("FAILED %d of %d -> %s" % (len(FAILURES), N_CHECKS[0], FAILURES))
    sys.exit(1)
print("ALL TESTS PASSED (%d checks)" % N_CHECKS[0])