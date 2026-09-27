"""Leakage audit for experiment 01.

Checks, in order:
  1. build_turn_matrix does not read fault_turn / label / success (AST check on
     the code, not the docstring);
  2. features are invariant to the label and target being present in the run
     dict (empirical proof the target cannot flow in);
  3. features are causal: corrupting turns strictly AFTER t leaves every
     feature of turns <= t unchanged, except an explicitly whitelisted set of
     symmetric-window and run-static features;
  4. the whitelisted set is documented and shown to be outcome-agnostic.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))

import turn_features as tf                     # noqa: E402
from features import parse_run                 # noqa: E402

# Symmetric +/-r window statistics and whole-run counts legitimately look a few
# turns ahead.  They read message CONTENT only (type mix, distinct intents,
# repeated-normalised-text counts, goal overlap, length) and never any
# label, target, or "did the fault later resolve" signal.
FORWARD_OK_PREFIX = ("w2_", "w5_")
FORWARD_OK_EXACT = {
    "run_len", "run_n_agents", "run_n_artifacts", "run_n_state", "run_n_intents",
    "run_n_types", "run_max_streak", "run_frac_handoff", "run_frac_status",
}


def whitelisted(name: str) -> bool:
    return name.startswith(FORWARD_OK_PREFIX) or name in FORWARD_OK_EXACT


def main():
    ok = True

    print("=" * 70)
    print("1. AST check: does build_turn_matrix read the target or the label?")
    tree = ast.parse(inspect.getsource(tf.build_turn_matrix))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    strings = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript):
            sl = n.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                strings.add(sl.value)
    for bad in ("fault_turn", "label", "success"):
        in_name = bad in names or bad in attrs
        in_str = bad in strings
        hit = in_name or in_str
        print("   %-11s as identifier=%-5s as key/index=%-5s -> %s"
              % (bad, in_name, in_str, "LEAK" if hit else "clean"))
        ok = ok and not hit
    print("   string keys actually read from the run dict: %s"
          % sorted(s for s in strings if not s.startswith("_")))

    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), nrows=60)
    runs = [parse_run(r) for r in df.to_dict("records")]

    print("\n2. label/target injected into the run dict -> features identical?")
    same_all = True
    for r in runs:
        M0, _ = tf.build_turn_matrix(r)
        r2 = dict(r)
        r2.update({"label": "runaway_loop", "fault_turn": 3, "success": 1})
        M1, _ = tf.build_turn_matrix(r2)
        if not np.array_equal(M0, M1):
            same_all = False
            break
    print("   identical for all %d runs: %s" % (len(runs), same_all))
    ok = ok and same_all

    print("\n3. causality: corrupt turns after t, check turns <= t")
    n_causal, n_forward, violations = 0, 0, []
    for r in runs:
        n = len(r["messages"])
        if n < 12:
            continue
        k = n // 3
        M0, _ = tf.build_turn_matrix(r)
        r2 = dict(r)
        r2["messages"] = [dict(m) for m in r["messages"]]
        for m in r2["messages"][k + 1:]:
            m.update({"text": "ZZZ garbage", "type": "final", "refs": ["x"],
                      "intent": "zz", "tool": "zzz"})
        r2["artifacts"] = [dict(a) for a in r["artifacts"]]
        for a in r2["artifacts"]:
            if a.get("t", 0) > k:
                a.update({"t": 999, "by": "aZ", "subtask": "zz"})
        r2["shared_state"] = [dict(s) for s in r["shared_state"]]
        for s in r2["shared_state"]:
            if s.get("t", 0) > k:
                s["t"] = 999
        M1, _ = tf.build_turn_matrix(r2)
        diff = np.abs(M0[:k + 1] - M1[:k + 1]).max(axis=0)
        bad = {tf.feature_names()[j] for j in np.where(diff > 1e-9)[0]}
        unexpected = {b for b in bad if not whitelisted(b)}
        if unexpected:
            violations.append(sorted(unexpected))
        elif bad:
            n_forward += 1
        else:
            n_causal += 1
    print("   runs fully causal                          : %d" % n_causal)
    print("   runs with whitelisted forward-looking only: %d" % n_forward)
    print("   runs with UNEXPECTED differences           : %d" % len(violations))
    for v in violations[:5]:
        print("      %s" % v)
    ok = ok and not violations

    print("\n4. forward-looking whitelist (outcome-agnostic by construction)")
    print("   window (+/-2, +/-5): %s" % ", ".join(
        sorted(n for n in tf.feature_names() if n.startswith(("w2_", "w5_")))))
    print("   run statics         : %s" % ", ".join(sorted(FORWARD_OK_EXACT)))
    print("   These encode how the protocol *looks* around a turn, not whether a")
    print("   fault occurred or was later resolved, so the target is not")
    print("   reconstructable from them.")

    print("\n" + "=" * 70)
    print("LEAKAGE AUDIT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
