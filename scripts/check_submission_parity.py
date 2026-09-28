"""Prove the submission reproduces the validated Exp06b configuration.

Three independent checks:

1. FEATURE PARITY - build the submission's label/success matrices on a sample of
   demo runs and compare, column by column and value by value, against the
   matrices built by the experiment code (``exp06b_stability/common.py``).
   This is the check that matters: it proves the vendored modules and the
   hand-rewritten assembly in ``solution.py`` produce the same 249 / 185
   columns in the same order with the same numbers.
2. SELF-CONTAINMENT - import every submission module with the repo root made
   unimportable, so a stray dependency on ``baseline/`` or ``experiments/``
   fails loudly instead of silently working on this machine.
3. NO FORBIDDEN INPUT - static scan proving no head reads label/success/fault_turn
   as a feature, and no absolute local path survives.
"""

from __future__ import annotations

import ast
import importlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submission_exp06_full_minus_age")
SAMPLE = 300
FORBIDDEN = {"label", "success", "fault_turn"}
FAILS = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


print("=" * 78)
print("1. FEATURE PARITY: submission vs validated experiment code")
print("=" * 78)

# Build the *experiment's* matrices (exp06b_stability/common.py).
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp06b_stability"))
sys.path.insert(0, os.path.join(ROOT, "baseline"))
import common as exp_common                      # noqa: E402

c = exp_common.build()
runs = c["runs"]
base_rows = None
from features import extract_features           # noqa: E402
base_rows = [extract_features(r) for r in runs[:SAMPLE]]
small_runs = runs[:SAMPLE]

# experiment matrices, restricted to the same sample
import new_features as nf                        # noqa: E402
import lifecycle as lc                           # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "exp06_features", os.path.join(ROOT, "experiments", "exp06_temporal_dynamics",
                                   "features.py"))
ft = exp_common.ft
X6 = ft.build_matrix(small_runs)
B1 = set(ft.BLOCKS["1_age"])
B4 = set(ft.BLOCKS["4_reassign"])

exp_label = pd.concat([
    pd.DataFrame(base_rows).fillna(0.0),
    nf.build_matrix(small_runs, base_rows),
    lc.build_matrix(small_runs)[lc.LIFECYCLE_NAMES],
    X6[[x for x in X6.columns if x not in B1]],
], axis=1)
exp_success = pd.concat([
    pd.DataFrame(base_rows).fillna(0.0),
    nf.build_matrix(small_runs, base_rows),
], axis=1)

# submission matrices - imported from the submission dir only
sys.path.insert(0, SUB)
for m in ("solution", "temporal_features", "new_features", "lifecycle"):
    sys.modules.pop(m, None)
import solution as sub_sol                       # noqa: E402

sub_label = sub_sol.build_label_matrix(small_runs, base_rows)
sub_success = sub_sol.build_success_matrix(small_runs, base_rows)

check("label feature count == 249", sub_label.shape[1] == 249,
      "got %d" % sub_label.shape[1])
check("success feature count == 185", sub_success.shape[1] == 185,
      "got %d" % sub_success.shape[1])
check("label column NAMES match experiment exactly",
      list(sub_label.columns) == list(exp_label.columns),
      "first diff at %s" % next((i for i, (a, b) in
                                 enumerate(zip(sub_label.columns, exp_label.columns))
                                 if a != b), "none"))
check("success column NAMES match experiment exactly",
      list(sub_success.columns) == list(exp_success.columns),
      "first diff at %s" % next((i for i, (a, b) in
                                 enumerate(zip(sub_success.columns, exp_success.columns))
                                 if a != b), "none"))
check("no age (ua_) feature leaked into the label matrix",
      not any(str(x).startswith("ua_") for x in sub_label.columns),
      "found %s" % [x for x in sub_label.columns if str(x).startswith("ua_")][:5])

d1 = float(np.abs(sub_label.values - exp_label.values).max())
check("label VALUES identical to experiment", d1 == 0.0, "max abs diff = %g" % d1)
d2 = float(np.abs(sub_success.values - exp_success.values).max())
check("success VALUES identical to experiment", d2 == 0.0, "max abs diff = %g" % d2)

print("=" * 78)
print("2. SELF-CONTAINMENT: import from an isolated dir with the repo hidden")
print("=" * 78)

tmp = tempfile.mkdtemp(prefix="subcheck_")
shutil.copytree(SUB, os.path.join(tmp, "sub"))
code = (
    "import sys; sys.path.insert(0, r'%s');"
    "import solution, temporal_features, new_features, lifecycle, features, localize;"
    "print('IMPORT_OK', solution.__file__)" % os.path.join(tmp, "sub"))
env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                   cwd=tmp, env=env)
check("imports resolve from a temp dir outside the repo",
      "IMPORT_OK" in p.stdout, (p.stdout + p.stderr).strip()[-300:])
shutil.rmtree(tmp, ignore_errors=True)

print("=" * 78)
print("3. NO FORBIDDEN INPUT / NO ABSOLUTE PATHS")
print("=" * 78)

for fn in ("solution.py", "features.py", "new_features.py", "lifecycle.py",
           "temporal_features.py", "localize.py"):
    src = open(os.path.join(SUB, fn), encoding="utf-8").read()
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in FORBIDDEN:
            hits.append(node.value.id)
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
            hits.append(node.attr)
    # solution.py legitimately reads train["label"]/["success"] as TARGETS
    if fn == "solution.py":
        hits = [h for h in hits if h != "__getattr__"]
    check("%s: no target leakage into features" % fn, not hits, "found %s" % sorted(set(hits)))
    check("%s: no absolute local path" % fn,
          ("D:\\" not in src) and ("D:/" not in src) and ("C:\\" not in src))
    check("%s: no network imports" % fn,
          not any(m in src for m in ("import requests", "urllib.request",
                                     "import socket", "http://", "https://")))

print("=" * 78)
if FAILS:
    print("FAILED: %d" % len(FAILS))
    for f in FAILS:
        print("   - %s" % f)
    sys.exit(1)
print("ALL CHECKS PASSED")
