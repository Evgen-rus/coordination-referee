"""Parity gate for the Exp07 runner.

Compares a freshly produced mode run against the committed reference
(``results.json`` + ``oof_*.csv`` from the original ``run_experiment.py``).
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modes as MD  # noqa: E402

REF = HERE
TOL = 1e-9
KEYS = ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
        "composite")


def fail(msg, problems):
    problems.append(msg)
    print("  FAIL  %s" % msg)


def ok(msg):
    print("  ok    %s" % msg)


def compare(mode="cv", ref_dir=REF):
    new_dir = MD.outdir(mode)
    problems = []
    print("parity: reference=%s  new=%s" % (ref_dir, new_dir))
    if not os.path.exists(os.path.join(new_dir, "results.json")):
        print("  FAIL  no new results.json in %s" % new_dir)
        return False, ["missing new results"]
    ref = json.load(open(os.path.join(ref_dir, "results.json")))
    new = json.load(open(os.path.join(new_dir, "results.json")))

    for k in ("n_foundation", "n_window_feats", "n_agg_feats", "pos_radius",
              "success_f1"):
        if ref[k] != new[k]:
            fail("%s: ref=%r new=%r" % (k, ref[k], new[k]), problems)
    if not problems:
        ok("config: n_found=%s n_win=%s n_agg=%s radius=%s"
           % (new["n_foundation"], new["n_window_feats"],
              new["n_agg_feats"], new["pos_radius"]))

    for sysname in ref["systems"]:
        r = ref["systems"][sysname]
        v = new["systems"].get(sysname)
        if v is None:
            fail("system %s missing in new results" % sysname, problems)
            continue
        if v.get("partial"):
            fail("system %s is partial (early stop) - cannot compare"
                 % sysname, problems)
            continue
        if r["n_features"] != v["n_features"]:
            fail("%s n_features: ref=%s new=%s"
                 % (sysname, r["n_features"], v["n_features"]), problems)
        bad = [k for k in KEYS if abs(r[k] - v[k]) > TOL]
        if bad:
            for k in bad:
                fail("%s %s: ref=%.12f new=%.12f (d=%.3e)"
                     % (sysname, k, r[k], v[k], abs(r[k] - v[k])), problems)
        else:
            ok("%s metrics: macro=%.6f rob=%.6f hit2=%.6f comp=%.6f"
               % (sysname, v["macro_f1"], v["robustness_f1"],
                  v["fault_turn_hit2"], v["composite"]))
        df = np.abs(np.array(r["fold_macro_f1"]) -
                    np.array(v["fold_macro_f1"])).max()
        if df > TOL:
            fail("%s fold_macro_f1 max diff %.3e" % (sysname, df), problems)
        else:
            ok("%s fold_macro_f1 identical" % sysname)
        pc_r, pc_v = r["per_class"], v["per_class"]
        dp = max(abs(pc_r[c] - pc_v[c]) for c in pc_r)
        if dp > TOL:
            fail("%s per_class F1 max diff %.3e" % (sysname, dp), problems)
        else:
            ok("%s per_class F1 identical" % sysname)
        a = pd.read_csv(os.path.join(ref_dir, "oof_%s.csv" % sysname))
        b = pd.read_csv(os.path.join(new_dir, "oof_%s.csv" % sysname))
        if len(a) != len(b):
            fail("%s row count: ref=%d new=%d" % (sysname, len(a), len(b)),
                 problems)
            continue
        if not (a["run_id"].values == b["run_id"].values).all():
            fail("%s run_id order differs" % sysname, problems)
        mism = int((a["y_pred"].values != b["y_pred"].values).sum())
        if mism:
            fail("%s y_pred mismatches: %d/%d" % (sysname, mism, len(a)),
                 problems)
        else:
            ok("%s y_pred byte-identical (%d rows)" % (sysname, len(a)))

    return (not problems), problems


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    good, probs = compare(mode)
    print("\nPARITY: %s" % ("PASS" if good else "FAIL"))
    for p in probs:
        print("  - %s" % p)
    sys.exit(0 if good else 1)
