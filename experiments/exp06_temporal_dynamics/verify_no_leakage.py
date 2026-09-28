"""Leakage / causality audit for Exp06 temporal-dynamics features.

Extends the Exp05 audit: same target-independence guarantees, plus the two
Exp06-specific risks -- a feature name colliding with Exp05, and a review-mandated
guard (bd_grows_late must not fire on an empty 0 -> 0 backlog).
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(HERE, "..", "exp05_handoff_lifecycle"))

from features import parse_run   # noqa: E402
# baseline/ also has a module called `features`; load THIS experiment's module
# by explicit path so it cannot be shadowed by sys.path ordering.
import importlib.util            # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "exp06_features", os.path.join(HERE, "features.py"))
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

FORBIDDEN = {"label", "success", "fault_turn"}
FAILS: list = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def strip_docstrings(src: str) -> str:
    t = ast.parse(src)
    for node in ast.walk(t):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            b = getattr(node, "body", None)
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                b.pop(0)
    return ast.unparse(t)


def main() -> int:
    src = open(os.path.join(HERE, "features.py"), encoding="utf-8").read()
    tree = ast.parse(src)

    print("1. static: no target column is ever read")
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
    check("no label/success/fault_turn in features.py", not hits, str(hits))

    print("2. no lexicon matching in executable code")
    code = strip_docstrings(src)
    banned = ["kw_", "KW_", "lexicon", "_KEYWORDS", "keyword", "KEYWORD"]
    found = [b for b in banned if b in code]
    check("no keyword/lexicon dependency", not found, str(found))

    print("3. registry hygiene")
    check("29 features", len(ft.ALL_NAMES) == 29, str(len(ft.ALL_NAMES)))
    check("no internal duplicates", len(set(ft.ALL_NAMES)) == 29)
    check("5 blocks partition the registry",
          sum(len(v) for v in ft.BLOCKS.values()) == 29)

    print("4. no collision with Exp05 feature names")
    sys.path.insert(0, os.path.join(HERE, "..", "exp05_handoff_lifecycle"))
    import lifecycle as lc
    clash = set(ft.ALL_NAMES) & (set(lc.ALL_NAMES))
    check("no overlap with Exp05's 54 names", not clash, str(clash))

    print("5. production on real runs: finite, no constants")
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), nrows=1500)
    recs = train.to_dict("records")
    runs = [parse_run(r) for r in recs]
    X = ft.build_matrix(runs)
    check("shape matches registry", X.shape[1] == 29, str(X.shape))
    check("names match exactly", list(X.columns) == list(ft.ALL_NAMES))
    check("all finite", bool(np.isfinite(X.values).all()))
    dead = [c for c in X.columns if X[c].nunique() <= 1]
    check("no constant column", not dead, "dead: %s" % dead)

    print("6. empirical causality: invariant to all three targets")
    alt = []
    for r in recs[:600]:
        q = dict(r)
        q["label"] = "runaway_loop" if r["label"] != "runaway_loop" else "clean"
        q["success"] = 1 - int(r["success"])
        q["fault_turn"] = (int(r["fault_turn"]) + 3) if int(r["fault_turn"]) >= 0 else 7
        alt.append(q)
    A = ft.build_matrix([parse_run(r) for r in recs[:600]])
    B = ft.build_matrix([parse_run(r) for r in alt])
    d = float(np.abs(A.values - B.values).max())
    check("identical features after flipping targets", d == 0.0,
          "max abs diff = %g" % d)

    print("7. determinism")
    one = ft.temporal_features(runs[0])
    one2 = ft.temporal_features(parse_run(recs[0]))
    check("deterministic for a fixed run",
          all(abs(one[k] - one2[k]) < 1e-12 for k in one))

    print("8. review-mandated guards")
    # bd_grows_late must be 0 on a 0 -> 0 backlog
    empty = {"messages": [
        {"t": 0, "from": "a1", "to": "a2", "type": "handoff", "intent": "t",
         "refs": [], "text": ""},
        {"t": 4, "from": "a2", "to": "a1", "type": "handoff", "intent": "t",
         "refs": ["r"], "text": ""}] +
        [{"t": t, "from": "a1", "to": "a2", "type": "inform", "intent": "z",
          "refs": [], "text": ""} for t in (5, 6, 7, 8, 9)],
        "artifacts": [], "agents": [{"id": "a1"}, {"id": "a2"}]}
    g = ft.temporal_features(empty)
    check("bd_grows_late == 0 on an empty 0 -> 0 backlog", g["bd_grows_late"] == 0.0)
    # slope must be computed on the normalised backlog: a 4x longer run with
    # the same shape must give the same slope
    def scaled(k):
        return {"messages": [
            {"t": 0, "from": "a1", "to": "a2", "type": "handoff", "intent": "t",
             "refs": [], "text": ""},
            {"t": 8 * k, "from": "a2", "to": "a1", "type": "handoff",
             "intent": "t", "refs": ["r"], "text": ""}] +
            [{"t": t, "from": "a2", "to": "a1", "type": "inform", "intent": "z",
              "refs": [], "text": ""} for t in range(1, 8 * k)] +
            [{"t": t, "from": "a1", "to": "a2", "type": "inform", "intent": "z",
              "refs": [], "text": ""} for t in range(8 * k + 1, 8 * k + 2)],
            "artifacts": [], "agents": [{"id": "a1"}, {"id": "a2"}]}
    s1 = ft.temporal_features(scaled(1))["bd_slope"]
    s4 = ft.temporal_features(scaled(4))["bd_slope"]
    check("bd_slope is scale-free (1x vs 4x run)", abs(s1 - s4) < 1e-9,
          "1x=%.4f 4x=%.4f" % (s1, s4))

    print("9. unit tests")
    proc = subprocess.run([sys.executable, os.path.join(HERE, "test_backlog.py")],
                          capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr).strip().splitlines()[-1:] or [""]
    check("test_backlog.py passes", proc.returncode == 0, tail[0])

    print("10. baseline files untouched")
    for f in ("baseline/features.py", "baseline/solution.py", "baseline/localize.py"):
        h = hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:12]
        print("       %-28s sha256[:12]=%s" % (f, h))

    print("")
    if FAILS:
        print("LEAKAGE AUDIT FAILED: %s" % FAILS)
        return 1
    print("LEAKAGE AUDIT PASSED - Exp06 features are target-independent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
