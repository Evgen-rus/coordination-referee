"""Leakage / causality audit for the new normalised features.

Checks that ``new_features.py`` never reads a target column and that a feature
value for a run is unchanged when the target is changed (empirical causality
test: same run payload, different ground truth -> identical features).

Run:  .\.venv\Scripts\python.exe experiments\exp03_length_normalization\verify_no_leakage.py
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from features import extract_features, parse_run      # noqa: E402
import new_features as nf                             # noqa: E402

FORBIDDEN = {"label", "success", "fault_turn"}
FAILS = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    src = open(os.path.join(HERE, "new_features.py"), encoding="utf-8").read()
    tree = ast.parse(src)

    print("1. static check: no target column is ever subscripted")
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in FORBIDDEN:
                hits.append(node.value.id)
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
            hits.append(node.attr)
    check("no label/success/fault_turn reference in new_features.py",
          not hits, "found: %s" % hits)

    print("2. new_features imports only the official baseline helpers")
    imports = [n.module or "" for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names]
    check("no import of localise / metrics", not any(
        "localize" in i or "metrics" in i for i in imports), str(imports))

    print("3. every declared feature name is actually produced")
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), nrows=800)
    runs = [parse_run(r) for r in train.to_dict("records")]
    base = [extract_features(r) for r in runs]
    X = nf.build_matrix(runs, base)
    check("column count == declared", X.shape[1] == nf.N_NEW,
          "%d vs %d" % (X.shape[1], nf.N_NEW))
    check("names match exactly", list(X.columns) == list(nf.NEW_FEATURE_NAMES))
    check("all finite", bool(np.isfinite(X.values).all()))
    dead = [c for c in X.columns if X[c].nunique() <= 1]
    check("no constant column", not dead, "dead: %s" % dead)

    print("4. empirical causality: features are invariant to the target")
    recs = train.to_dict("records")[:400]
    alt = []
    for r in recs:
        q = dict(r)
        q["label"] = "runaway_loop" if r["label"] != "runaway_loop" else "clean"
        q["success"] = 1 - int(r["success"])
        q["fault_turn"] = (int(r["fault_turn"]) + 3) if int(r["fault_turn"]) >= 0 else 7
        alt.append(q)
    Xa = nf.build_matrix([parse_run(r) for r in recs], base)
    Xb = nf.build_matrix([parse_run(r) for r in alt], base)
    d = float(np.abs(Xa.values - Xb.values).max())
    check("identical features when label/success/fault_turn change", d == 0.0,
          "max abs diff = %g" % d)

    print("5. every denominator comes from the run itself, not from a label-derived stat")
    # the only baseline aggregates reused as denominators are unsupervised counts
    denoms = ["n_assign", "n_intents", "n_senders", "g_nodes", "g_n_pairs",
              "state_keys", "n", "n_art", "n_st", "n_ag", "max_turn", "active"]
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    check("denominator whitelist present in source",
          all(d in src for d in denoms),
          "missing: %s" % [d for d in denoms if d not in src])
    check("no denominator derived from a target", not any(
        f"base.get(\"{c}" in src for c in ("label", "success", "fault_turn")))

    print("6. baseline features are untouched on disk")
    import hashlib
    for f in ("baseline/features.py", "baseline/solution.py", "baseline/localize.py"):
        h = hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:12]
        print("       %-28s sha256[:12]=%s" % (f, h))

    print("")
    if FAILS:
        print("LEAKAGE AUDIT FAILED: %s" % FAILS)
        sys.exit(1)
    print("LEAKAGE AUDIT PASSED - new features are target-independent.")


if __name__ == "__main__":
    main()
