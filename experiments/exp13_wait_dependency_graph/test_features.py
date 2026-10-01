"""Exp13 - unit tests for the fixed wait-dependency graph.

Required coverage (10 cases).  Every test is a plain assertion on real
function output; none of them touches a model, a fold or a metric.

    1. A waits B only                -> no reciprocal pair
    2. A waits B + B waits A          -> exactly one reciprocal pair
    3. same subtask both ways         -> same-subtask = 1
    4. different subtasks             -> reciprocal = 1, same-subtask = 0
    5. repeated waits                 -> repeat count up, unique pair count flat
    6. all six official templates     -> awaited agent parsed
    7. unrelated normal text          -> not parsed as a wait
    8. no label / fault_turn dependency
    9. deterministic output
   10. baseline reproduction is checked by run_experiment.py, not here
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import features as W  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("   PASS  %s" % name)
    else:
        print("   FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


def msg(t, frm, to, text, typ="status", intent=""):
    return {"t": t, "from": frm, "to": to, "type": typ, "intent": intent,
            "text": text, "tool": "", "refs": []}


def run(msgs):
    return {"run_id": "r", "goal": "g", "agents": ["a1", "a2"], "messages": msgs,
            "artifacts": [], "shared_state": []}


def f(msgs, key):
    return W.extract_wait_graph_features(run(msgs))[key]


print("=" * 74)
print("Exp13 wait-graph unit tests")
print("=" * 74)

# --- 0. parser shape ------------------------------------------------------
check("six template families are compiled",
      len(W.TEMPLATE_PATTERNS) == 6 and len(W.SUBTASK_PATTERNS) == 6
      and len(W.TEMPLATE_NAMES) == 6,
      "got %d/%d/%d" % (len(W.TEMPLATE_PATTERNS), len(W.SUBTASK_PATTERNS),
                         len(W.TEMPLATE_NAMES)))
check("exactly 12 declared features", W.N_FEATURES == 12, "got %d" % W.N_FEATURES)

# --- 1. one-directional wait ---------------------------------------------
m = [msg(0, "a1", "a2", "Blocked on a2: I need check_budget first."),
     msg(1, "a2", "a3", "Blocked on a3: I need check_liability first.")]
check("1. A waits B only -> no reciprocal pair",
      f(m, "wg_n_recip_pairs") == 0.0 and f(m, "wg_has_recip_pair") == 0.0,
      "pairs=%s" % f(m, "wg_n_recip_pairs"))
check("1b. one-directional wait still counts edges",
      f(m, "wg_n_wait_edges") == 2.0 and f(m, "wg_n_unique_wait_edges") == 2.0)

# --- 2. mutual wait -------------------------------------------------------
m = [msg(0, "a1", "a2", "Blocked on a2: I need check_budget first."),
     msg(1, "a2", "a1", "Waiting for a1 to deliver check_budget before I can continue.")]
check("2. A waits B + B waits A -> exactly one reciprocal pair",
      f(m, "wg_n_recip_pairs") == 1.0 and f(m, "wg_has_recip_pair") == 1.0,
      "pairs=%s" % f(m, "wg_n_recip_pairs"))
check("2b. both sides of a pair count once in unique edges",
      f(m, "wg_n_unique_wait_edges") == 2.0)
check("2c. recip_wait_share is over run length",
      abs(f(m, "wg_recip_wait_share") - 1.0) < 1e-12)
check("2d. agents in reciprocal pairs counted once",
      f(m, "wg_n_agents_in_recip_pairs") == 2.0)

# --- 3. same subtask both ways -------------------------------------------
m = [msg(0, "a1", "a2", "Blocked on a2: I need check_budget first."),
     msg(1, "a2", "a1", "Waiting for a1 to deliver check_budget before I can continue.")]
check("3. same subtask both ways -> same-subtask = 1",
      f(m, "wg_n_same_subtask_recip_pairs") == 1.0
      and f(m, "wg_has_same_subtask_recip_pair") == 1.0,
      "same=%s" % f(m, "wg_n_same_subtask_recip_pairs"))

# --- 4. different subtasks ------------------------------------------------
m = [msg(0, "a1", "a2", "Blocked on a2: I need check_budget first."),
     msg(1, "a2", "a1", "Waiting for a1 to deliver verify_fix before I can continue.")]
check("4. different subtasks -> reciprocal=1, same-subtask=0",
      f(m, "wg_n_recip_pairs") == 1.0 and f(m, "wg_n_same_subtask_recip_pairs") == 0.0,
      "pairs=%s same=%s" % (f(m, "wg_n_recip_pairs"),
                            f(m, "wg_n_same_subtask_recip_pairs")))

# --- 5. repetition -------------------------------------------------------
# The pair becomes complete at the LATER of its two first wait turns, so a
# repeat must be added AFTER that turn to count as "after onset".
one = [msg(0, "a1", "a2", "Blocked on a2: I need check_budget first."),
       msg(1, "a2", "a1", "Waiting for a1 to deliver check_budget before I can continue.")]
late = one + [msg(2, "a1", "a2", "Blocked on a2: I need check_budget first."),
              msg(3, "a2", "a1", "Waiting for a1 to deliver check_budget before I can continue."),
              msg(4, "a1", "a2", "Blocked on a2: I need check_budget first.")]
check("5. repeated waits raise repeat count, unique pairs unchanged",
      f(late, "wg_max_pair_wait_events") == 5.0
      and f(late, "wg_n_unique_wait_edges") == f(one, "wg_n_unique_wait_edges")
      and f(late, "wg_n_recip_pairs") == f(one, "wg_n_recip_pairs"),
      "events=%s unique=%s" % (f(late, "wg_max_pair_wait_events"),
                               f(late, "wg_n_unique_wait_edges")))
check("5b. onset is the later FIRST-wait turn; only later repeats counted",
      f(late, "wg_max_pair_repeat_after_onset") == 3.0,
      "got %s" % f(late, "wg_max_pair_repeat_after_onset"))
check("5c. a pair with no wait after onset has zero repeats",
      f(one, "wg_max_pair_repeat_after_onset") == 0.0,
      "got %s" % f(one, "wg_max_pair_repeat_after_onset"))

# --- 6. all six templates -------------------------------------------------
SIX = [
    ("Blocked on a4: I need check_budget first.", "a4"),
    ("Waiting for a4 to deliver check_budget before I can continue.", "a4"),
    ("Cannot proceed with my part until a4 finishes check_budget.", "a4"),
    ("On hold: check_budget from a4 has not arrived.", "a4"),
    ("Still holding for a4 on check_budget.", "a4"),
    ("My step depends on a4 closing check_budget.", "a4"),
]
for txt, agent in SIX:
    ev = W.parse_wait(msg(0, "a1", agent, txt))
    ok = ev is not None and ev[0] == "a1" and ev[1] == agent
    check("6. template parses: %s" % txt[:46], ok, "got %r" % (ev,))

for txt, agent in SIX:
    ev = W.parse_wait(msg(0, "a1", agent, txt))
    check("6b. subtask read: %s" % txt[:40],
          ev is not None and ev[2] == "check_budget", "got %r" % (ev,))

# --- 7. unrelated text ---------------------------------------------------
NEG = [
    "Starting request_quotes now.",
    "a2, please take care of request_quotes and report back.",
    "request_quotes is done, artifact art_01 produced.",
    "Acknowledged, starting check_budget.",
    "The deploy took 3 retries and then settled.",
    "s3 and h2 are filesystem names, not agents.",
]
for txt in NEG:
    check("7. not a wait: %s" % txt[:46],
          W.parse_wait(msg(0, "a1", "a2", txt)) is None)
check("7b. an inform message is never a wait even with template text",
      W.parse_wait(msg(0, "a1", "a2",
                       "Blocked on a2: I need check_budget first.",
                       typ="inform")) is None)

# --- 8. no label / fault_turn dependency ---------------------------------
msgs = [msg(0, "a1", "a2", "Blocked on a2: I need check_budget first."),
        msg(1, "a2", "a1", "Waiting for a1 to deliver check_budget before I can continue."),
        msg(2, "a1", "a2", "Cannot proceed with my part until a2 finishes check_budget.")]
base = {k: W.extract_wait_graph_features(run(msgs))[k] for k in W.FEATURE_NAMES}
mutated = []
for lab, ft, succ in (("clean", 0, 1), ("deadlock", 7, 0), ("goal_drift", 3, 1)):
    r = run(msgs)
    r["label"] = lab
    r["fault_turn"] = ft
    r["success"] = succ
    mutated.append({k: W.extract_wait_graph_features(r)[k] for k in W.FEATURE_NAMES})
check("8. features invariant to label / fault_turn / success",
      all(m == base for m in mutated))
check("8b. parser reads only type/from/text/t",
      all(W.parse_wait(m)[0] == m["from"]
          for m in msgs if W.parse_wait(m) is not None))

# --- 9. determinism ------------------------------------------------------
r = run(msgs)
X1 = W.build_matrix([r, r, r])
X2 = W.build_matrix([r, r, r])
check("9. deterministic output", np.array_equal(X1, X2))
check("9b. matrix shape and dtype",
      X1.shape == (3, 12) and X1.dtype == np.float64)
check("9c. zero-wait run yields all zeros",
      all(v == 0.0 for v in
          W.extract_wait_graph_features(
              run([msg(0, "a1", "a2", "hello there")])).values()))

# --- geometry sanity ------------------------------------------------------
m = [msg(0, "a1", "a2", "Blocked on a2: I need check_budget first."),
     msg(1, "a2", "a1", "Waiting for a1 to deliver check_budget before I can continue."),
     msg(2, "a1", "a2", "hello"), msg(3, "a1", "a2", "hi"),
     msg(4, "a1", "a2", "x"), msg(5, "a1", "a2", "y")]
check("geom. earliest_recip_pos_rel = t_complete/(n-1)",
      abs(f(m, "wg_earliest_recip_pos_rel") - 1.0 / 5.0) < 1e-12,
      "got %s" % f(m, "wg_earliest_recip_pos_rel"))
check("geom. max_pair_span_rel spans first..last wait of the pair",
      abs(f(m, "wg_max_pair_span_rel") - 1.0 / 5.0) < 1e-12,
      "got %s" % f(m, "wg_max_pair_span_rel"))
check("geom. self-wait does not create a reciprocal pair",
      f([msg(0, "a1", "a1", "Blocked on a1: I need check_budget first."),
         msg(1, "a1", "a1", "Blocked on a1: I need check_budget first.")],
        "wg_n_recip_pairs") == 0.0)

# --- real corpus smoke ----------------------------------------------------
tr_csv = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "train.csv")
if os.path.exists(tr_csv):
    import pandas as pd
    tr = pd.read_csv(tr_csv)
    runs = []
    for row in tr.head(400).itertuples(index=False):
        r = {"run_id": str(row.run_id), "goal": str(row.goal),
             "messages": json.loads(row.messages)}
        runs.append(r)
    X = W.build_matrix(runs)
    check("corpus. no non-finite values over 400 real runs", np.isfinite(X).all())
    fires = (X[:, 3] > 0).sum()
    check("corpus. reciprocal pairs fire on some real runs", fires > 0,
          "fires=%d" % fires)
    same = (X[:, 6] > 0).sum()
    check("corpus. same-subtask pairs fire on some real runs", same > 0,
          "same=%d" % same)
    check("corpus. recip share within [0,1]",
          bool((X[:, 4] >= 0).all() and (X[:, 4] <= 1).all()))
    check("corpus. relative positions within [0,1]",
          bool((X[:, 7] >= 0).all() and (X[:, 7] <= 1).all())
          and bool((X[:, 10] >= 0).all() and (X[:, 10] <= 1).all()))
else:
    print("   SKIP  corpus smoke (train.csv not found)")

print()
print("=" * 74)
if FAILURES:
    print("FAILED: %d -> %s" % (len(FAILURES), FAILURES))
    sys.exit(1)
print("ALL TESTS PASSED")
