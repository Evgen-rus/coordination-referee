"""Unit tests for Exp06 backlog / anchor-point logic on synthetic runs.

These verify the *definitions* the plan committed to, not just "it runs":
backlog at the 25/50/75/100% anchors, the normalised slope, the non-empty
guard on ``bd_grows_late``, and the 80% cut on ``ra_active_to_end_frac``.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as ft  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def handoff(t, frm, to, intent, refs=None):
    return {"t": t, "from": frm, "to": to, "type": "handoff",
            "intent": intent, "refs": refs or [], "text": ""}


def run_of(msgs, arts=None, n_agents=2):
    return {"messages": msgs, "artifacts": arts or [],
            "agents": [{"id": "a%d" % (i + 1)} for i in range(n_agents)]}


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def main():
    # 10 messages, turns 0..9, so end = 9 and anchors are 2.25 / 4.5 / 6.75 / 9
    print("1. backlog anchors: one task assigned at t=0, closed at t=8")
    msgs = [handoff(0, "a1", "a2", "task")]
    msgs += [{"t": t, "from": "a2", "to": "a1", "type": "inform",
              "intent": "other", "refs": [], "text": ""} for t in range(1, 8)]
    msgs += [handoff(8, "a2", "a1", "task", refs=["r1"])]
    msgs += [{"t": 9, "from": "a1", "to": "a2", "type": "inform",
              "intent": "task", "refs": [], "text": ""}]
    f = ft.temporal_features(run_of(msgs))
    # open from t=0 until t=8 -> open at 2.25, 4.5, 6.75; closed by t=9
    check("bd_at25 == 1", close(f["bd_at25"], 1.0), str(f["bd_at25"]))
    check("bd_at50 == 1", close(f["bd_at50"], 1.0), str(f["bd_at50"]))
    check("bd_at75 == 1", close(f["bd_at75"], 1.0), str(f["bd_at75"]))
    check("bd_at100 == 0", close(f["bd_at100"], 0.0), str(f["bd_at100"]))
    check("bd_max_rel == 1", close(f["bd_max_rel"], 1.0), str(f["bd_max_rel"]))
    check("bd_slope == (0-1)/0.75", close(f["bd_slope"], -1.0 / 0.75),
          str(f["bd_slope"]))
    check("bd_grows_late == 0 (backlog shrank)", close(f["bd_grows_late"], 0.0))
    check("no unresolved -> ua_age_max_rel == 0",
          close(f["ua_age_max_rel"], 0.0), str(f["ua_age_max_rel"]))

    print("2. review fix 1: 0 -> 0 must NOT score as growth")
    # two tasks, both opened late and both left open; at 75% there is a
    # non-empty backlog that never shrinks
    msgs2 = [handoff(0, "a1", "a2", "t%d" % i) for i in range(2)]
    msgs2 += [{"t": t, "from": "a2", "to": "a1", "type": "inform",
               "intent": "x", "refs": [], "text": ""} for t in range(1, 8)]
    f2 = ft.temporal_features(run_of(msgs2, n_agents=2))
    check("bd_at75 == 1 (2 open / 2 lifecycles)", close(f2["bd_at75"], 1.0),
          str(f2["bd_at75"]))
    check("bd_at100 == 1", close(f2["bd_at100"], 1.0), str(f2["bd_at100"]))
    check("bd_grows_late == 1 (non-empty and not shrinking)",
          close(f2["bd_grows_late"], 1.0), str(f2["bd_grows_late"]))

    # a run whose backlog is empty at 75% AND empty at 100% -> must be 0
    msgs3 = [handoff(0, "a1", "a2", "t0"), handoff(4, "a2", "a1", "t0", refs=["r"])]
    msgs3 += [{"t": t, "from": "a1", "to": "a2", "type": "inform",
               "intent": "z", "refs": [], "text": ""} for t in (5, 6, 7, 8, 9)]
    f3 = ft.temporal_features(run_of(msgs3, n_agents=2))
    check("bd_at75 == 0", close(f3["bd_at75"], 0.0), str(f3["bd_at75"]))
    check("bd_at100 == 0", close(f3["bd_at100"], 0.0), str(f3["bd_at100"]))
    check("bd_grows_late == 0 for 0 -> 0 (review fix 1)",
          close(f3["bd_grows_late"], 0.0), str(f3["bd_grows_late"]))

    print("3. review fix 4: 80% cut on ra_active_to_end_frac")
    # unresolved handoff at t=0; receiver writes at t=9 -> active to the end
    msgs4 = [handoff(0, "a1", "a2", "lost")]
    msgs4 += [{"t": t, "from": "a2", "to": "a1", "type": "inform",
               "intent": "lost", "refs": [], "text": ""} for t in range(1, 10)]
    f4 = ft.temporal_features(run_of(msgs4, n_agents=2))
    check("active-to-end == 1 when receiver writes at t=9",
          close(f4["ra_active_to_end_frac"], 1.0), str(f4["ra_active_to_end_frac"]))
    check("ra_silence_oldest == 0 (receiver still talking)",
          close(f4["ra_silence_oldest"], 0.0), str(f4["ra_silence_oldest"]))
    # same, but the handoff happens at 95% of the run -> must be EXCLUDED,
    # so the denominator is 0 and the feature stays 0
    msgs5 = [{"t": t, "from": "a1", "to": "a2", "type": "inform",
              "intent": "x", "refs": [], "text": ""} for t in range(0, 19)]
    msgs5.append(handoff(18, "a1", "a2", "late"))
    f5 = ft.temporal_features(run_of(msgs5, n_agents=2))
    check("late handoff excluded from the 80%% cut",
          close(f5["ra_active_to_end_frac"], 0.0), str(f5["ra_active_to_end_frac"]))

    print("4. age features on a single stale handoff")
    msgs6 = [handoff(0, "a1", "a2", "stale")]
    msgs6 += [{"t": t, "from": "a2", "to": "a1", "type": "inform",
               "intent": "stale", "refs": [], "text": ""} for t in range(1, 10)]
    f6 = ft.temporal_features(run_of(msgs6, n_agents=2))
    check("ua_age_max_rel == 1.0 (assigned at t=0, never closed)",
          close(f6["ua_age_max_rel"], 1.0), str(f6["ua_age_max_rel"]))
    check("ua_oldest_share_gt50 == 1.0", close(f6["ua_oldest_share_gt50"], 1.0),
          str(f6["ua_oldest_share_gt50"]))
    check("ua_oldest_share_gt10 == 1.0", close(f6["ua_oldest_share_gt10"], 1.0),
          str(f6["ua_oldest_share_gt10"]))
    check("ra_own_msgs_after_mean_rel == 0.9 (9 of 10 messages)",
          close(f6["ra_own_msgs_after_mean_rel"], 0.9),
          str(f6["ra_own_msgs_after_mean_rel"]))

    print("5. other-intent artifacts (the key dropped-handoff marker)")
    msgs7 = [handoff(0, "a1", "a2", "lost"), handoff(1, "a1", "a2", "other")]
    msgs7 += [{"t": t, "from": "a2", "to": "a1", "type": "inform",
               "intent": "other", "refs": [], "text": ""} for t in range(2, 9)]
    arts7 = [{"t": 6, "by": "a2", "subtask": "other", "hash": "h1", "status": "final"}]
    f7 = ft.temporal_features(run_of(msgs7, arts7, n_agents=2))
    # the run has 9 messages, so 1 other-intent artifact normalises to 1/9
    check("ra_other_arts_after_max_rel == 1/9 (1 artifact / 9 messages)",
          close(f7["ra_other_arts_after_max_rel"], 1 / 9),
          str(f7["ra_other_arts_after_max_rel"]))
    check("ra_other_arts_exists == 1", close(f7["ra_other_arts_exists"], 1.0))
    check("ra_frac_lost_while_active == 1 (receiver busy, tasks still open)",
          close(f7["ra_frac_lost_while_active"], 1.0),
          str(f7["ra_frac_lost_while_active"]))

    print("6. old owner delivers after reassignment")
    msgs8 = [handoff(0, "a1", "a2", "task"), handoff(3, "a1", "a3", "task")]
    msgs8 += [{"t": t, "from": "a2", "to": "a1", "type": "inform",
               "intent": "task", "refs": [], "text": ""} for t in (1, 2, 4, 5, 6, 7, 8)]
    msgs8.append(handoff(9, "a2", "a1", "task", refs=["late_result"]))
    f8 = ft.temporal_features(run_of(msgs8, n_agents=3))
    check("rt_old_owner_result_after == 1", close(f8["rt_old_owner_result_after"], 1.0),
          str(f8["rt_old_owner_result_after"]))
    check("rt_reassign_lat_mean_rel == 3/9", close(f8["rt_reassign_lat_mean_rel"], 3 / 9),
          str(f8["rt_reassign_lat_mean_rel"]))

    print("7. registry")
    check("29 features", len(ft.ALL_NAMES) == 29, str(len(ft.ALL_NAMES)))
    check("no name collisions inside", len(set(ft.ALL_NAMES)) == 29)
    check("5 ablation blocks", len(ft.BLOCKS) == 5)
    check("blocks partition the registry",
          sum(len(v) for v in ft.BLOCKS.values()) == 29)

    print("")
    if FAILS:
        print("UNIT TESTS FAILED: %s" % FAILS)
        return 1
    print("ALL UNIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
