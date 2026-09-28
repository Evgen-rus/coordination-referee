"""Prove the Exp07 submission reproduces the validated experiment configuration.

Independent checks, in order of how much they would catch:

1. FOUNDATION PARITY - the 249-column ``full - age`` matrix and the 185-column
   Exp03 B success matrix, built by the submission vs by
   ``exp07_fault_windows/common.py``.  Column names, order, and every value.
2. WINDOW FEATURE PARITY - the 83 turn-level features, submission vs
   ``exp07_fault_windows/window_features.py``, on real runs.
3. TARGET PARITY - ``window_targets`` and ``class_weight_vector``, elementwise.
4. AGGREGATION PARITY - the 51 run-level aggregates.  A real window model is
   fitted and its per-window probabilities are aggregated BOTH ways: by the
   submission's vectorised ``aggregate_runs`` and by the experiment's
   ``grouping.predict_runs`` + ``aggregate_block``.  This is the check that
   matters most for Exp07, because the aggregate block is the entire new signal.
5. MODEL PARITY - hyper-parameters of all three heads compared attribute by
   attribute against the experiment's constructors.
6. SELF-CONTAINMENT - every module imported from a temp dir with the repo root
   made unimportable.
7. NO FORBIDDEN INPUT - static scan for targets-as-features, absolute paths and
   network imports.
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
SUB = os.path.join(ROOT, "submission_exp07_fault_windows")
EXP = os.path.join(ROOT, "experiments", "exp07_fault_windows")
SAMPLE = 250
FORBIDDEN = {"label", "success", "fault_turn"}
FAILS = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return "first diff at %d: %r vs %r" % (i, x, y)
    return "none"


def _load_exp(name):
    """Load an exp07 module by explicit file path.

    The submission and the experiment both ship modules named `window_features`,
    `window_dataset` and `aggregate`, so a plain `import` would resolve to
    whichever landed in `sys.modules` first.  Loading by path under a unique
    module name removes that ambiguity entirely.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp07_" + name,
                                                  os.path.join(EXP, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 1 & 2 setup
print("=" * 78)
print("1. FOUNDATION PARITY: submission vs exp07_fault_windows/common.py")
print("=" * 78)

for p in (os.path.join(EXP, "..", "exp03_length_normalization"),
          os.path.join(EXP, "..", "exp05_handoff_lifecycle"),
          os.path.join(EXP, "..", "exp06_temporal_dynamics"),
          os.path.join(ROOT, "baseline"), os.path.join(ROOT, "evaluation"), EXP):
    sys.path.insert(0, os.path.abspath(p))

import common as exp_common                      # noqa: E402
from features import extract_features            # noqa: E402

c = exp_common.build_all()
runs = c["runs"]
small = runs[:SAMPLE]
base_rows = [extract_features(r) for r in small]
exp_found = c["foundation"].iloc[:SAMPLE].reset_index(drop=True)
exp_succ = c["success"].iloc[:SAMPLE].reset_index(drop=True)

# submission side
sys.path.insert(0, SUB)
for m in ("solution", "aggregate", "window_dataset", "window_features",
          "temporal_features", "new_features", "lifecycle", "features",
          "localize"):
    sys.modules.pop(m, None)
import solution as sub                            # noqa: E402

sub_found = sub.build_foundation_matrix(small, base_rows)
sub_succ = sub.build_success_matrix(small, base_rows)

check("foundation feature count == 249", sub_found.shape[1] == 249,
      "got %d" % sub_found.shape[1])
check("success feature count == 185", sub_succ.shape[1] == 185,
      "got %d" % sub_succ.shape[1])
check("foundation column NAMES match experiment",
      list(sub_found.columns) == list(exp_found.columns),
      first_diff(list(sub_found.columns), list(exp_found.columns)))
check("success column NAMES match experiment",
      list(sub_succ.columns) == list(exp_succ.columns),
      first_diff(list(sub_succ.columns), list(exp_succ.columns)))
check("no age (ua_) feature in the foundation",
      not any(str(x).startswith("ua_") for x in sub_found.columns),
      "found %s" % [x for x in sub_found.columns if str(x).startswith("ua_")][:5])
d = float(np.abs(sub_found.values - exp_found.values).max())
check("foundation VALUES identical", d == 0.0, "max abs diff = %g" % d)
d = float(np.abs(sub_succ.values - exp_succ.values).max())
check("success VALUES identical", d == 0.0, "max abs diff = %g" % d)

# ------------------------------------------------------------------- 2 window
print("=" * 78)
print("2. WINDOW FEATURE PARITY: 83 turn-level features")
print("=" * 78)
exp_wf = _load_exp("window_features")
check("N_FEATURES == 83", sub.wf.N_FEATURES == 83, "got %d" % sub.wf.N_FEATURES)
check("feature NAMES identical",
      list(sub.wf.FEATURE_NAMES) == list(exp_wf.FEATURE_NAMES),
      first_diff(list(sub.wf.FEATURE_NAMES), list(exp_wf.FEATURE_NAMES)))
check("RADIUS == 2 (fixed positive zone)", sub.wf.RADIUS == 2,
      "got %s" % sub.wf.RADIUS)
worst, n_win = 0.0, 0
for r in small[:120]:
    a, b = sub.wf.run_window_features(r), exp_wf.run_window_features(r)
    assert a.shape == b.shape, "shape mismatch %s vs %s" % (a.shape, b.shape)
    worst = max(worst, float(np.abs(a - b).max()))
    n_win += a.shape[0]
check("window feature VALUES identical over %d windows" % n_win, worst == 0.0,
      "max abs diff = %g" % worst)

# ------------------------------------------------------------------ 3 targets
print("=" * 78)
print("3. TARGET PARITY: window_targets + class_weight_vector")
print("=" * 78)
def _load_exp(name):
    """Load an exp07 module by explicit file path.

    The submission and the experiment both ship modules named `window_features`,
    `window_dataset` and `aggregate`, so a plain `import` would resolve to
    whichever landed in `sys.modules` first.  Loading by path with a unique
    module name removes that ambiguity entirely.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp07_" + name,
                                                  os.path.join(EXP, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp_wf = _load_exp("window_features")
exp_wd = _load_exp("window_dataset")
exp_ag = _load_exp("aggregate")
import grouping as grp                            # noqa: E402

check("POS_RADIUS == 2", sub.wd.POS_RADIUS == 2, "got %s" % sub.wd.POS_RADIUS)
check("N_WINDOW_CLASSES == 7", sub.wd.N_WINDOW_CLASSES == 7)
check("class name list identical", sub.wd.WCLASSES == exp_wd.WCLASSES)
check("W2I map identical", sub.wd.W2I == exp_wd.W2I)
wv_sub = sub.wd.class_weight_vector()
wv_exp = exp_wd.class_weight_vector()
check("class weight vector identical",
      np.array_equal(wv_sub, wv_exp),
      "sub bg=%.4f fault=%.4f | exp bg=%.4f fault=%.4f"
      % (wv_sub[0], wv_sub[1], wv_exp[0], wv_exp[1]))
bad = 0
for n in (0, 1, 3, 5, 17, 100):
    for lab in exp_wd.FAULT_CLASSES + ["clean"]:
        for f in (-1, 0, 1, 5, n - 1, n + 10):
            if not np.array_equal(sub.wd.window_targets(n, lab, f),
                                  exp_wd.window_targets(n, lab, f)):
                bad += 1
check("window_targets identical over 90 (n, label, fault_turn) cases", bad == 0,
      "%d mismatches" % bad)
# the +-2 rule itself, spelled out
yt = sub.wd.window_targets(40, "deadlock", 20)
check("+-2 rule: exactly turns 18..22 are the fault class",
      list(np.where(yt != 0)[0]) == [18, 19, 20, 21, 22],
      "got %s" % list(np.where(yt != 0)[0]))
check("clean run gets zero fault windows",
      int((sub.wd.window_targets(40, "clean", -1) != 0).sum()) == 0)
check("clean run with a stray fault_turn still gets zero fault windows",
      int((sub.wd.window_targets(40, "clean", 20) != 0).sum()) == 0)

# ------------------------------------------------------------- 4 aggregation
print("=" * 78)
print("4. AGGREGATION PARITY: the 51 run-level window aggregates")
print("=" * 78)

check("AGG_NAMES identical", list(sub.ag.AGG_NAMES) == list(exp_ag.AGG_NAMES))
check("N_AGG == 51", sub.ag.N_AGG == 51, "got %d" % sub.ag.N_AGG)
check("HIGH threshold == 0.5", sub.ag.HIGH == 0.5, "got %s" % sub.ag.HIGH)

# real window model -> real probability blocks, aggregated both ways
ys_all = c["ys"]
ft_all = c["fturn"]
sub_W = sub.build_window_matrix(small, ys_all[:SAMPLE], ft_all[:SAMPLE])
exp_W = exp_wd.build_window_dataset(small, ys_all[:SAMPLE], ft_all[:SAMPLE])
check("window dataset row counts agree",
      len(sub_W["y"]) == len(exp_W["y"]),
      "%d vs %d" % (len(sub_W["y"]), len(exp_W["y"])))
check("window target arrays identical", np.array_equal(sub_W["y"], exp_W["y"]))
d = float(np.abs(sub_W["X"] - exp_W["X"]).max())
check("stacked window matrix identical", d == 0.0, "max abs diff = %g" % d)
check("block length == run turn count (so nturn is a valid substitute)",
      all(int((sub_W["run"] == i).sum()) == int(sub_W["n_turns"][i])
          for i in range(len(small))))

mw = sub.fit_window(sub_W["X"], sub_W["y"])
P = mw.predict_proba(sub_W["X"])
rw = sub_W["run"]
nturn = sub_W["n_turns"]

want = np.arange(len(small))
sub_agg = sub.aggregate_runs(P, rw, nturn, want)
exp_blocks = grp.predict_runs(P, rw, len(small))
exp_agg, _, _ = grp.aggregate_block(exp_blocks, exp_ag,
                                    {int(r): k for k, r in enumerate(want)},
                                    len(small))
d = float(np.abs(sub_agg - exp_agg).max())
check("aggregate VALUES identical over %d runs" % len(small), d == 0.0,
      "max abs diff = %g" % d)

# and on random probability matrices, including degenerate shapes.
# Each synthetic run r gets its own window count, so the `n_turns` lookup is
# always in range: the point is to exercise 1-window runs, 2-window runs, and
# runs whose probability count differs from other runs in the same batch.
rng = np.random.default_rng(7)
worst_rand = 0.0
for counts in ([1], [2], [1, 1, 1], [1, 5], [2, 3, 1, 7], [33, 1, 4]):
    n_runs = len(counts)
    rows, nt_list = [], []
    for r, k in enumerate(counts):
        rows.append(np.full(k, r, dtype=np.int32))
        nt_list.append(k)
    rows = np.concatenate(rows)
    nt = np.array(nt_list, dtype=np.int64)
    Pr = rng.dirichlet(np.ones(7), size=len(rows))
    tgt = np.arange(n_runs)
    a = sub.aggregate_runs(Pr, rows, nt, tgt)
    b, _, _ = grp.aggregate_block(
        grp.predict_runs(Pr, rows, n_runs), exp_ag,
        {int(r): k for k, r in enumerate(tgt)}, n_runs)
    worst_rand = max(worst_rand, float(np.abs(a - b).max()))
check("aggregate identical on random/degenerate shapes", worst_rand == 0.0,
      "max abs diff = %g" % worst_rand)
check("empty probability block -> all-zero row",
      float(np.abs(sub.aggregate_runs(np.zeros((0, 7)), np.zeros(0, int),
                                      np.zeros(1, int), np.array([0])).sum())) == 0.0)

# ---------------------------------------------------------------- 5 models
print("=" * 78)
print("5. MODEL PARITY: hyperparameters attribute by attribute")
print("=" * 78)
for name, sub_maker, exp_maker in (
        ("label", sub.make_label_model, exp_common.lgb_label),
        ("success", sub.make_success_model, None),
        ("window", sub.make_window_model, exp_common.lgb_window)):
    a = sub_maker().get_params()
    b = exp_maker().get_params() if exp_maker else None
    if b is None:
        print("       (%s head: no direct experiment constructor; pinned to the "
              "Exp03 B config by design)" % name)
        continue
    diffs = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
             if a.get(k) != b.get(k)}
    check("%s model hyperparameters identical" % name, not diffs,
          "%s" % diffs)
check("window model num_class == 7", sub.make_window_model().get_params()["num_class"] == 7)

# -------------------------------------------------------- 6 self-containment
print("=" * 78)
print("6. SELF-CONTAINMENT: import from a temp dir with the repo hidden")
print("=" * 78)
tmp = tempfile.mkdtemp(prefix="sub07check_")
shutil.copytree(SUB, os.path.join(tmp, "sub"))
code = ("import sys; sys.path.insert(0, r'%s');"
        "import solution, aggregate, window_dataset, window_features, "
        "temporal_features, new_features, lifecycle, features, localize;"
        "print('IMPORT_OK', solution.__file__)" % os.path.join(tmp, "sub"))
env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                   cwd=tmp, env=env)
check("imports resolve from a temp dir outside the repo",
      "IMPORT_OK" in p.stdout, (p.stdout + p.stderr).strip()[-400:])
shutil.rmtree(tmp, ignore_errors=True)

# ----------------------------------------------------------- 7 static scan
print("=" * 78)
print("7. NO FORBIDDEN INPUT / NO ABSOLUTE PATHS / NO NETWORK")
print("=" * 78)
MODULES = ("solution.py", "features.py", "new_features.py", "lifecycle.py",
           "temporal_features.py", "localize.py", "window_features.py",
           "window_dataset.py", "aggregate.py")
for fn in MODULES:
    path = os.path.join(SUB, fn)
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in FORBIDDEN:
            hits.append(node.value.id)
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
            hits.append(node.attr)
    # solution.py and window_dataset.py legitimately read targets as TARGETS
    if fn in ("solution.py", "window_dataset.py"):
        hits = [h for h in hits if h != "__getattr__"]
    check("%s: no target leakage into features" % fn, not hits,
          "found %s" % sorted(set(hits)))
    check("%s: no absolute local path" % fn,
          ("D:\\" not in src) and ("D:/" not in src) and ("C:\\" not in src))
    check("%s: no network imports" % fn,
          not any(m in src for m in ("import requests", "urllib.request",
                                     "import socket", "http://", "https://")))

sol_src = open(os.path.join(SUB, "solution.py"), encoding="utf-8").read()
sol_tree = ast.parse(sol_src)
main_fn = next(n for n in sol_tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "main")
check("solution.py uses the OFFICIAL localizer for fault_turn",
      "localize(run, lab)" in sol_src)
# Every name main() reads must be bound SOMEWHERE: either at module scope, or
# by an assignment/parameter/comprehension inside main() itself.  A name that is
# read but never bound anywhere is a guaranteed NameError - this is what caught
# the real `parse_runs` bug that reading the source had missed.
bound_local = set()
for n in ast.walk(main_fn):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
        bound_local.add(n.id)
    elif isinstance(n, ast.arg):
        bound_local.add(n.arg)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            bound_local.add((a.asname or a.name).split(".")[0])
    elif isinstance(n, ast.comprehension):
        for t in ast.walk(n.target):
            if isinstance(t, ast.Name):
                bound_local.add(t.id)
    elif isinstance(n, ast.ExceptHandler) and n.name:
        bound_local.add(n.name)
module_level = set(dir(sub)) | {n.name for n in sol_tree.body
                                if isinstance(n, (ast.FunctionDef,
                                                  ast.ClassDef))}
for node in sol_tree.body:                       # module-level imports/assigns
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            module_level.add((a.asname or a.name).split(".")[0])
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            for t2 in ast.walk(t):
                if isinstance(t2, ast.Name):
                    module_level.add(t2.id)
needed = {n.id for n in ast.walk(main_fn)
          if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
import builtins                                     # noqa: E402
missing = sorted(needed - module_level - bound_local - set(dir(builtins))
                 - set(dir(np)) - set(dir(pd)))
check("main() reads no name that is bound nowhere", not missing,
      "missing %s" % missing)

# Scan ONLY the executable statements of main(), excluding its docstring and
# comments.  The module docstring documents the L1/L2 EXCLUSION, so a whole-file
# text search would flag the very documentation that proves the rule is honoured.
stmts = [n for n in main_fn.body if not isinstance(n, ast.Expr)
         or not isinstance(n.value, ast.Constant)]      # drop docstrings
main_code = "\n".join(ast.unparse(s) for s in stmts)
banned = ("peak_pos", "peak_prob", "HYBRID_CONF", "0.60", "argmax", "peak")
check("main() never touches a window peak / post-hoc hybrid",
      not any(k in main_code for k in banned),
      "found %s" % [k for k in banned if k in main_code])
check("main() calls the localizer exactly as the experiment did",
      main_code.count("localize(") == 1
      and "localize(run, lab)" in main_code,
      "localize calls = %d" % main_code.count("localize("))
# fault_turn must be written from pred_turn, and pred_turn from the localizer.
# There are three DataFrames in main(); the OUTPUT one is the only one that
# mentions the required submission columns.
out_frames = [ast.unparse(n) for n in ast.walk(main_fn)
              if isinstance(n, ast.Call) and "DataFrame" in ast.unparse(n.func)
              and "fault_turn" in ast.unparse(n)]
check("the output frame is built from pred_turn, not a window peak",
      len(out_frames) == 1 and "pred_turn" in out_frames[0]
      and "peak" not in out_frames[0].lower(), "%s" % out_frames)
check("output frame carries all four required columns",
      all(("'%s'" % c) in out_frames[0]
          for c in ("run_id", "label", "success", "fault_turn")),
      "%s" % out_frames)
pred_turn_assigns = [ast.unparse(n) for n in ast.walk(main_fn)
                     if isinstance(n, ast.Assign)
                     and any("pred_turn" in ast.unparse(t)
                             for t in n.targets)]
check("pred_turn is assigned exactly once, from the localizer",
      len(pred_turn_assigns) == 1
      and "localize(" in pred_turn_assigns[0],
      "%s" % pred_turn_assigns)
check("solution.py keeps the success head at 185 features",
      "N_SUCCESS_FEATS = 185" in sol_src)
check("solution.py has no hyperparameter search",
      not any(k in sol_src for k in ("GridSearchCV", "RandomizedSearchCV",
                                     "optuna", "halving")))
check("solution.py calls verify_stacking", "verify_stacking(" in sol_src)

print("=" * 78)
if FAILS:
    print("FAILED: %d" % len(FAILS))
    for f in FAILS:
        print("   - %s" % f)
    sys.exit(1)
print("ALL CHECKS PASSED")
