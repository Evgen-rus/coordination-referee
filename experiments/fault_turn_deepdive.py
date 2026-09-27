#!/usr/bin/env python3
"""Focused diagnostic: why is fault_turn weak?

Uses the TRUE label (oracle class) so localisation error is isolated from
classification error.  Reports absolute-error distribution per class and
tests the length dependence of each rule.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from features import extract_features, parse_run   # noqa: E402
from localize import localize                      # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]


def main():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    runs = [parse_run(r) for r in train.to_dict("records")]
    y = train["label"].values
    ft = train["fault_turn"].values
    nmsg = np.array([len(r["messages"]) for r in runs])

    print("=== LOCALISER WITH ORACLE CLASS (isolates localisation from classification) ===")
    print("%-18s %6s %7s %7s %7s %7s %7s %7s %7s"
          % ("class", "n", "hit@2", "hit@5", "hit@10", "medAE", "pred-1", "too_early", "too_late"))
    summary = {}
    for c in LABELS[1:]:
        idx = np.where(y == c)[0]
        preds = np.array([localize(runs[i], c) for i in idx])
        ae = np.abs(preds - ft[idx])
        h2 = float((ae <= 2).mean())
        summary[c] = {
            "n": len(idx), "hit@2": h2,
            "hit@5": float((ae <= 5).mean()), "hit@10": float((ae <= 10).mean()),
            "median_abs_err": float(np.median(ae)),
            "returned_minus1": int((preds < 0).sum()),
            "too_early": int((preds < ft[idx]).sum()),
            "too_late": int((preds > ft[idx]).sum()),
            "exact": float((ae == 0).mean()),
        }
        v = summary[c]
        print("%-18s %6d %7.3f %7.3f %7.3f %7.0f %7d %7d %7d"
              % (c, v["n"], v["hit@2"], v["hit@5"], v["hit@10"],
                 v["median_abs_err"], v["returned_minus1"], v["too_early"], v["too_late"]))

    all_idx = np.where(y != "clean")[0]
    allp = np.array([localize(runs[i], y[i]) for i in all_idx])
    alla = np.abs(allp - ft[all_idx])
    print("\nALL FAULTY: hit@2=%.4f  hit@5=%.4f  hit@10=%.4f  exact=%.4f  returned -1: %d/%d"
          % ((alla <= 2).mean(), (alla <= 5).mean(), (alla <= 10).mean(),
             (alla == 0).mean(), int((allp < 0).sum()), len(allp)))

    # ---- length dependence of each rule: short vs long runs ----
    print("\n=== LENGTH DEPENDENCE OF LOCALISATION (oracle class, hit@2) ===")
    med = np.median(nmsg)
    for c in LABELS[1:]:
        idx = np.where(y == c)[0]
        preds = np.array([localize(runs[i], c) for i in idx])
        ae = np.abs(preds - ft[idx])
        sh, lo = idx[nmsg[idx] <= med], idx[nmsg[idx] > med]
        h_sh = float((ae[nmsg[idx] <= med] <= 2).mean())
        h_lo = float((ae[nmsg[idx] > med] <= 2).mean())
        print("  %-18s short(n=%4d)=%.3f   long(n=%4d)=%.3f   gap=%+.3f"
              % (c, len(sh), h_sh, len(lo), h_lo, h_lo - h_sh))
        summary[c]["hit@2_short"] = h_sh
        summary[c]["hit@2_long"] = h_lo

    # ---- how often is fault_turn early in the run? (rules that guess too late) ----
    print("\n=== TRUE fault_turn POSITION (as fraction of run length) ===")
    for c in LABELS[1:]:
        idx = np.where(y == c)[0]
        frac = ft[idx] / np.maximum(1, nmsg[idx])
        print("  %-18s median_frac=%.3f  median_turn=%3.0f  median_len=%3.0f"
              % (c, float(np.median(frac)), float(np.median(ft[idx])),
                 float(np.median(nmsg[idx]))))

    # ---- goal_drift: the hard-coded 6-turn window ----
    idx = np.where(y == "goal_drift")[0]
    preds = np.array([localize(runs[i], "goal_drift") for i in idx])
    ae = np.abs(preds - ft[idx])
    print("\n=== goal_drift RULE DETAIL (window `zeros[i+2]-zeros[i] <= 6`) ===")
    print("  hit@2=%.3f  median_abs_err=%.0f  predicted too early %d / too late %d"
          % ((ae <= 2).mean(), float(np.median(ae)),
             int((preds < ft[idx]).sum()), int((preds > ft[idx]).sum())))
    print("  share of drift runs where rule returns the FIRST zero-overlap turn: %.3f"
          % float((preds == np.array([z for z in preds])[0]).mean() * 0 +
               np.mean([preds[i] == _first_zero(runs[idx[i]]) for i in range(len(idx))])))

    # ---- runaway_loop: 'hot' text requires >=3 exact normalised repeats ----
    idx = np.where(y == "runaway_loop")[0]
    cnt_lt3 = 0
    for i in idx:
        norms = [__import__("features")._norm(m.get("text", "")) for m in runs[i]["messages"]]
        c2 = Counter(norms)
        if not any(v >= 3 for v in c2.values()):
            cnt_lt3 += 1
    print("\n=== runaway_loop RULE DETAIL ===")
    print("  runs with NO normalised text repeated >=3x (rule falls back): %d/%d (%.1f%%)"
          % (cnt_lt3, len(idx), 100.0 * cnt_lt3 / len(idx)))

    # ---- clean contract ----
    cl = np.where(y == "clean")[0]
    bad = sum(1 for i in cl if localize(runs[i], "clean") != -1)
    print("\nclean runs returning -1: %d/%d (contract violations: %d)"
          % (len(cl) - bad, len(cl), bad))

    with open(os.path.join(HERE, "fault_turn_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nwrote experiments/fault_turn_diagnostics.json")


def _first_zero(run):
    from features import _tokens
    goal_tok = _tokens(run.get("goal", ""))
    for m in run["messages"]:
        if not (goal_tok & _tokens(m.get("text", ""))):
            return int(m.get("t", 0))
    return -1


if __name__ == "__main__":
    main()
