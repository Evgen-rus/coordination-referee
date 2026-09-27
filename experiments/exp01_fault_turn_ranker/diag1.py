"""Why did the learned ranker underperform the rule baseline so badly?"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import turn_features as tf
from features import parse_run
from localize import localize

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
NAMES = tf.feature_names()


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    y = df["label"].values
    ft = df["fault_turn"].values
    runs = [parse_run(r) for r in df.to_dict("records")]

    print("=" * 72)
    print("A. Where the rule lands vs where the truth sits")
    print("=" * 72)
    for c in LABELS[1:]:
        idx = np.where(y == c)[0]
        rl = np.array([localize(runs[i], c) for i in idx])
        L = np.array([len(runs[i]["messages"]) for i in idx])
        print("  %-18s n=%4d med_true=%4.0f med_rule=%4.0f med_len=%4.0f frac=%.3f"
              % (c, len(idx), np.median(ft[idx]), np.median(rl),
                 np.median(L), np.median(ft[idx] / np.maximum(1, L))))

    print()
    print("=" * 72)
    print("B. Does the class-conditional event fire AT the fault turn?")
    print("=" * 72)
    probe = {"dropped_handoff": "is_assign",
             "duplicated_work": "is_second_assign_intent",
             "conflict": "state_is_override",
             "deadlock": "type_status",
             "runaway_loop": "streak_len>=2"}
    for c, fname in probe.items():
        idx = np.where(y == c)[0]
        hits = 0
        for i in idx:
            F, _ = tf.build_turn_matrix(runs[i])
            t = int(ft[i])
            if not (0 <= t < F.shape[0]):
                continue
            col = {n: F[t, j] for j, n in enumerate(NAMES)}
            ok = (col["streak_len"] >= 2) if fname.startswith("streak") else (col[fname] == 1)
            hits += bool(ok)
        print("  %-18s %-26s fires: %.3f" % (c, fname, hits / len(idx)))

    print()
    print("=" * 72)
    print("C. Ambiguity a causal model cannot resolve")
    print("=" * 72)
    idx = np.where(y == "dropped_handoff")[0]
    n_assigns = n_never = 0
    for i in idx:
        msgs = runs[i]["messages"]
        delivered = set()
        for m in msgs:
            if m.get("type") == "handoff" and m.get("refs"):
                delivered.add((m.get("from"), m.get("intent")))
        for a in runs[i]["artifacts"]:
            delivered.add((a.get("by"), a.get("subtask")))
        assigns = [m for m in msgs
                   if m.get("type") == "handoff" and not m.get("refs")]
        n_assigns += len(assigns)
        n_never += sum(1 for m in assigns
                       if (m.get("to"), m.get("intent")) not in delivered)
    print("  dropped_handoff: %d assigns, %d never delivered (%.1f pct)"
          % (n_assigns, n_never, 100.0 * n_never / max(1, n_assigns)))

    print()
    print("=" * 72)
    print("D. Which classes start EARLY vs LATE")
    print("=" * 72)
    for c in LABELS[1:]:
        idx = np.where(y == c)[0]
        L = np.array([len(runs[i]["messages"]) for i in idx])
        frac = ft[idx] / np.maximum(1, L)
        print("  %-18s first10=%.3f  last33=%.3f" % (c, (frac < 0.10).mean(), (frac > 0.67).mean()))


if __name__ == "__main__":
    main()
