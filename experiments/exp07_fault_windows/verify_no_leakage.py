"""Leakage audit for Exp07.  Every check must pass before results are trusted.

1. STATIC: window_features.py and the aggregation module may not read
   label / success / fault_turn as a feature source.
2. CAUSALITY: window features are bit-identical when label/success/fault_turn
   are changed in the input row.  This is the strongest form of the check - the
   function physically cannot see them.
3. DETERMINISM: same input -> same output; no NaN/inf; one row per message.
4. BASELINE UNTOUCHED: print hashes for comparison against git.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from features import parse_run                     # noqa: E402
import window_features as wf                       # noqa: E402

FORBIDDEN = ("label", "success", "fault_turn")
FAILS = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


print("=" * 78)
print("1. STATIC SCAN: no target column may be a feature source")
print("=" * 78)
# window_dataset.py is the ONE module allowed to read the targets, because it is
# the module that builds the training target.  It is scanned for the opposite
# property instead: that it emits them as a separate `y` array and never hands
# them to the feature builder.
FEATURE_BUILDERS = ("window_features.py", "aggregate.py", "grouping.py")
TARGET_BUILDERS = ("window_dataset.py", "common.py")
for fn in FEATURE_BUILDERS:
    path = os.path.join(HERE, fn)
    if not os.path.exists(path):
        print("  [SKIP] %s not present yet" % fn)
        continue
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if isinstance(node.slice, ast.Constant) and node.slice.value in FORBIDDEN:
                hits.add("subscript:%s" % node.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and node.args:
            if isinstance(node.args[0], ast.Constant) and node.args[0].value in FORBIDDEN:
                hits.add("get:%s" % node.args[0].value)
    check("%s reads no target as a feature" % fn, not hits, "found %s" % sorted(hits))
    check("%s has no absolute local path" % fn,
          "D:\\" not in src and "C:\\" not in src)

# The target builders are checked for the opposite, and stricter, property: they
# MAY read label/success/fault_turn (that is their job), but they must never hand
# a target to the feature builder, and must keep targets in a separate array.
for fn in TARGET_BUILDERS:
    path = os.path.join(HERE, fn)
    if not os.path.exists(path):
        print("  [SKIP] %s not present yet" % fn)
        continue
    src = open(path, encoding="utf-8").read()
    check("%s has no absolute local path" % fn,
          "D:\\" not in src and "C:\\" not in src)
    bad = re.search(r"run_window_features\s*\([^)]*(label|success|fault_turn|ys|ft)", src)
    check("%s never passes a target into the feature builder" % fn, not bad,
          "found %s" % (bad.group(0) if bad else ""))
    # window_dataset.py returns X and y as separate arrays; common.py never
    # returns a window matrix at all (it delegates to window_dataset).
    if fn == "window_dataset.py":
        check("%s keeps targets in a separate array from X" % fn,
              '"y":' in src and '"X":' in src)

print("=" * 78)
print("2. CAUSALITY: features unchanged when targets are altered")
print("=" * 78)
train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
recs = train.to_dict("records")[:200]
base = [wf.run_window_features(parse_run(r)) for r in recs]
alt_recs = []
for r in recs:
    q = dict(r)
    q["label"] = "runaway_loop" if r["label"] != "runaway_loop" else "clean"
    q["success"] = 1 - int(r["success"])
    q["fault_turn"] = (int(r["fault_turn"]) + 3) if int(r["fault_turn"]) >= 0 else 7
    alt_recs.append(q)
alt = [wf.run_window_features(parse_run(r)) for r in alt_recs]
d = max(float(np.abs(a - b).max()) for a, b in zip(base, alt))
check("identical features when label/success/fault_turn change", d == 0.0,
      "max abs diff = %g" % d)

print("=" * 78)
print("3. DETERMINISM, SHAPES, RANGES")
print("=" * 78)
again = [wf.run_window_features(parse_run(r)) for r in recs]
d2 = max(float(np.abs(a - b).max()) for a, b in zip(base, again))
check("deterministic across repeated calls", d2 == 0.0, "max abs diff = %g" % d2)
check("N_FEATURES in the 50-120 design range",
      50 <= wf.N_FEATURES <= 120, "N_FEATURES = %d" % wf.N_FEATURES)
check("row count == n_messages for every run",
      all(a.shape[0] == len(parse_run(r)["messages"]) for a, r in zip(base, recs)))
check("no NaN / inf in window features",
      all(np.isfinite(a).all() for a in base))
check("clean runs produce rows like any other (no special-casing at feature time)",
      all(a.shape[0] > 0 for a in base))

# clean runs must NEVER be given a synthetic fault target
import window_dataset as wd                          # noqa: E402
clean_rows = [r for r in recs if r["label"] == "clean"]
if clean_rows:
    ny = []
    for r in clean_rows:
        t = wd.window_targets(len(parse_run(r)["messages"]), "clean",
                              int(r["fault_turn"]))
        ny.append(int((t != 0).sum()))
    check("clean runs get zero non-background windows", sum(ny) == 0,
          "%d clean runs contributed %d fault windows" % (len(clean_rows), sum(ny)))
    allneg = all(int(r["fault_turn"]) == -1 for r in clean_rows)
    check("clean runs carry fault_turn == -1 (no synthetic fault turn)", allneg)
faulty = [r for r in recs if r["label"] != "clean"]
if faulty:
    ys_ = []
    for r in faulty:
        t = wd.window_targets(len(parse_run(r)["messages"]), r["label"],
                              int(r["fault_turn"]))
        ys_.append(int((t != 0).sum()))
    check("every faulty run gets at least one fault window", min(ys_) > 0,
          "min = %d" % min(ys_))
    check("fault windows respect the fixed +-2 radius",
          all(v <= 2 * wd.POS_RADIUS + 1 for v in ys_),
          "max = %d, allowed <= %d" % (max(ys_), 2 * wd.POS_RADIUS + 1))

print("=" * 78)
print("4. BASELINE UNTOUCHED")
print("=" * 78)
for f in ("baseline/features.py", "baseline/localize.py", "baseline/solution.py"):
    h = hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:12]
    print("       %-28s sha256[:12]=%s" % (f, h))
print("       `git diff --stat -- baseline/` must be empty for this to hold")

print("=" * 78)
if FAILS:
    print("LEAKAGE AUDIT FAILED: %d" % len(FAILS))
    for f in FAILS:
        print("   - %s" % f)
    sys.exit(1)
print("LEAKAGE AUDIT PASSED")
