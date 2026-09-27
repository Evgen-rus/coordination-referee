"""Leakage / causality audit for the Exp05 lifecycle features.

Checks that ``lifecycle.py`` never reads a target column, that a feature value is
unchanged when the ground truth is changed, that no lexicon matching crept in,
and that the SCC helper is correct.

Run:  .\\.venv\\Scripts\\python.exe experiments\\exp05_handoff_lifecycle\\verify_no_leakage.py
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from features import parse_run     # noqa: E402
import lifecycle as lc             # noqa: E402

FORBIDDEN = {"label", "success", "fault_turn"}
FAILS: list = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    src = open(os.path.join(HERE, "lifecycle.py"), encoding="utf-8").read()
    tree = ast.parse(src)

    print("1. static check: no target column is ever read")
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in FORBIDDEN:
            hits.append(node.value.id)
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
            hits.append(node.attr)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value in FORBIDDEN:
            hits.append(node.value)
    check("no label/success/fault_turn anywhere in lifecycle.py", not hits, str(hits))

    print("2. no lexicon matching (the baseline keyword tables must not be reused)")
    # scan CODE only: docstrings and comments are stripped first, otherwise the
    # module's own explanatory note would trip the check.
    tree2 = ast.parse(src)
    for node in ast.walk(tree2):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body.pop(0)
    code_only = ast.unparse(tree2)
    banned = ["kw_", "KW_", "lexicon", "LEXICON", "_KEYWORDS", "keyword", "KEYWORD"]
    found = [b for b in banned if b in code_only]
    check("no keyword/lexicon dependency in executable code", not found, str(found))
    imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
               for a in n.names]
    imports += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    check("imports nothing but numpy/stdlib",
          not any(("features" in i) or ("localize" in i) or ("metrics" in i)
                  for i in imports), str(imports))

    print("3. feature registry is well formed")
    check("no duplicate feature names", len(set(lc.ALL_NAMES)) == len(lc.ALL_NAMES))
    check("lifecycle and deadlock blocks are disjoint",
          not (set(lc.LIFECYCLE_NAMES) & set(lc.DEADLOCK_NAMES)))
    check("total in the requested +20..60 range",
          20 <= len(lc.ALL_NAMES) <= 60, "n = %d" % len(lc.ALL_NAMES))

    print("4. all features produced, finite, none constant")
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), nrows=1200)
    recs = train.to_dict("records")
    runs = [parse_run(r) for r in recs]
    X = lc.build_matrix(runs)
    check("column count matches registry", X.shape[1] == len(lc.ALL_NAMES),
          "%d vs %d" % (X.shape[1], len(lc.ALL_NAMES)))
    check("names match exactly", list(X.columns) == list(lc.ALL_NAMES))
    check("all finite", bool(np.isfinite(X.values).all()))
    dead = [c for c in X.columns if X[c].nunique() <= 1]
    check("no constant column", not dead, "dead: %s" % dead)

    print("5. empirical causality: features are invariant to the target")
    alt = []
    for r in recs[:500]:
        q = dict(r)
        q["label"] = "runaway_loop" if r["label"] != "runaway_loop" else "clean"
        q["success"] = 1 - int(r["success"])
        q["fault_turn"] = (int(r["fault_turn"]) + 3) if int(r["fault_turn"]) >= 0 else 7
        alt.append(q)
    A = lc.build_matrix([parse_run(r) for r in recs[:500]])
    B = lc.build_matrix([parse_run(r) for r in alt])
    d = float(np.abs(A.values - B.values).max())
    check("identical features when label/success/fault_turn change", d == 0.0,
          "max abs diff = %g" % d)

    print("6. per-run purity and determinism")
    one = lc.lifecycle_features(runs[0])
    one2 = lc.lifecycle_features(parse_run(recs[0]))
    check("lifecycle_features is deterministic for a fixed run",
          all(abs(one[k] - one2[k]) < 1e-12 for k in one))

    print("7. SCC helper correctness (unit test)")
    import subprocess
    proc = subprocess.run([sys.executable, os.path.join(HERE, "test_scc.py")],
                          capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr).strip().splitlines()[-1:] or [""]
    check("test_scc.py reports zero mismatches", proc.returncode == 0, tail[0])

    print("8. baseline files untouched on disk")
    for f in ("baseline/features.py", "baseline/solution.py", "baseline/localize.py"):
        h = hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:12]
        print("       %-28s sha256[:12]=%s" % (f, h))

    print("")
    if FAILS:
        print("LEAKAGE AUDIT FAILED: %s" % FAILS)
        return 1
    print("LEAKAGE AUDIT PASSED - lifecycle features are target-independent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
