"""Exp07 diagnostics: window-model behaviour, fault-turn localisation, significance.

Answers the three separate questions the experiment has to keep apart:

  1. Does the window model actually find the ROOT-CAUSE REGION, or does it just
     recognise the overall run type?  (hit@0/1/2, peak distance, per-class)
  2. Would a window-based localiser beat the official one?  (L0 / L1 / L2)
  3. Is the label gain real?  (paired bootstrap on composite and macro)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from metrics import macro_f1                                  # noqa: E402
import window_dataset as wd                                   # noqa: E402
from localize import localize                                 # noqa: E402

N_BOOT = 20000
RNG = np.random.default_rng(20260928)
HYBRID_CONF = 0.60          # fixed in advance, NOT tuned; see L2 note below
LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
A_KEY, B_KEY = "A_foundation", "B_plus_window"


def main():
    res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
    oofA = pd.read_csv(os.path.join(HERE, "oof_%s.csv" % A_KEY))
    oofB = pd.read_csv(os.path.join(HERE, "oof_%s.csv" % B_KEY))
    peak_pos = np.load(os.path.join(HERE, "window_peak_pos.npy"))
    peak_prob = np.load(os.path.join(HERE, "window_peak_prob.npy"))
    wy = np.load(os.path.join(HERE, "window_oof_y.npy"))
    wp = np.load(os.path.join(HERE, "window_oof_pred.npy"))
    wr = np.load(os.path.join(HERE, "window_oof_run.npy"))

    ys = oofA["y_true"].values
    ft = oofA["fault_turn"].values
    pa, pb = oofA["y_pred"].values, oofB["y_pred"].values
    n = len(ys)
    out = {"localisation": {}, "window_model": {}, "significance": {}}

    # ---------- 1. WINDOW MODEL (validation windows, out-of-fold) ----------
    bg = wy == 0
    out["window_model"] = {
        "n_windows": int(len(wy)),
        "bg_precision": float(((wp == 0) & bg).sum() / max(1, (wp == 0).sum())),
        "bg_recall": float(((wp == 0) & bg).sum() / max(1, bg.sum())),
        "overall_acc": float((wp == wy).mean()),
        "per_class": {},
    }
    for ci in range(1, wd.N_WINDOW_CLASSES):
        m = wy == ci
        out["window_model"]["per_class"][wd.WCLASSES[ci]] = {
            "recall": float(((wp == ci) & m).sum() / max(1, m.sum())),
            "precision": float(((wp == ci) & m).sum() / max(1, (wp == ci).sum())),
            "n": int(m.sum()),
        }

    # ---------- 2. LOCALISATION: does the peak sit on the true fault turn? ----
    # All three turn sources are evaluated on the OFFICIAL denominator: every
    # truly-faulty run counts, and a misclassified label is a miss.  Arrays are
    # index-aligned with the full run list, not filtered.
    loc = {"L0_baseline": {}, "L1_window": {}, "L2_hybrid": {}}
    for tag, pred in (("A_foundation", pa), ("B_plus_window", pb)):
        label_ok = pred == ys
        L0 = np.full(n, -1, dtype=np.int64)
        L1 = np.full(n, -1, dtype=np.int64)
        L2 = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            lab = pred[i]
            if lab == "clean":
                continue
            base = int(localize(oofA_run(i), lab))
            L0[i] = base
            ci = wd.W2I[lab]
            L1[i] = int(peak_pos[i, ci])
            # L2: take the window turn only when it is confident enough,
            # otherwise defer to the official localiser.  The 0.60 cut is a
            # POST-HOC DIAGNOSTIC - it was fixed after seeing L1, so it is
            # reported separately and never touches the headline result.
            L2[i] = int(peak_pos[i, ci]) if peak_prob[i, ci] >= HYBRID_CONF else base
        loc["L0_baseline"][tag] = hit_table(ys, ft, L0, label_ok)
        loc["L1_window"][tag] = hit_table(ys, ft, L1, label_ok)
        loc["L2_hybrid"][tag] = hit_table(ys, ft, L2, label_ok)
    out["localisation"] = loc
    out["hybrid_conf_threshold"] = HYBRID_CONF
    out["localisation_note"] = (
        "L0/L1/L2 are all on the official hit@k definition: denominator = every "
        "truly-faulty run (7200), a misclassified label counts as a miss. "
        "L2's 0.60 cut is post-hoc and diagnostic only.")

    # peak distance distribution (B predictions, window turn chosen)
    dist = []
    for i in range(n):
        if pb[i] == "clean":
            continue
        dist.append(int(peak_pos[i, wd.W2I[pb[i]]]) - int(ft[i]))
    dist = np.array(dist)
    out["peak_offset"] = {
        "median": float(np.median(dist)),
        "mean": float(dist.mean()),
        "frac_exact": float((dist == 0).mean()),
        "frac_within_2": float((np.abs(dist) <= 2).mean()),
        "frac_within_5": float((np.abs(dist) <= 5).mean()),
        "frac_after_fault": float((dist > 2).mean()),
        "hist": {str(k): int(v) for k, v in
                 zip(*np.unique(np.clip(dist, -20, 20), return_counts=True))},
    }

    # ---------- 3. SIGNIFICANCE: paired bootstrap on the label metrics ------
    def sl_rob(mask):
        return macro_f1(list(ys[mask]), list(pb[mask]))

    import common as C
    c = C.build_all()
    hard = c["SL"]["hard/robustness"]

    def comp(pred, succ, i):
        m = macro_f1([ys[i]], [pred[i]])
        r = macro_f1([ys[j] for j in np.where(hard)[0]],
                     [pred[j] for j in np.where(hard)[0]])
        return 0.50 * m + 0.25 * r + 0.15 * succ + 0.10 * 0.0

    # bootstrap the paired macro/composite difference over runs
    idx = np.arange(n)
    d_macro = np.empty(N_BOOT)
    d_comp = np.empty(N_BOOT)
    labs = LABELS
    hard_idx = np.where(hard)[0]

    def mac(yb, p, sel):
        q = p[sel]
        return float(np.mean([(2 * ((yb == c_) & (q == c_)).sum()) /
                              max(1, ((yb == c_) | (q == c_)).sum())
                              for c_ in labs]))

    for b in range(N_BOOT):
        ii = RNG.integers(0, n, n)
        yb = ys[ii]
        mA, mB = mac(yb, pa, ii), mac(yb, pb, ii)
        d_macro[b] = mB - mA
        yh, ha, hb = yb[hard[ii]], pa[ii][hard[ii]], pb[ii][hard[ii]]
        s = c["success_f1"]
        d_comp[b] = (0.50 * mB + 0.25 * mac(yh, ha, np.arange(len(yh)))
                     + 0.15 * s) - (0.50 * mA + 0.25 * mac(yh, hb, np.arange(len(yh)))
                                     + 0.15 * s)
    out["significance"] = {
        "d_macro": {"delta": float(d_macro.mean()),
                    "ci95": [float(np.percentile(d_macro, 2.5)),
                             float(np.percentile(d_macro, 97.5))],
                    "p": float(2 * min((d_macro <= 0).mean(), (d_macro >= 0).mean()))},
        "d_composite": {"delta": float(d_comp.mean()),
                        "ci95": [float(np.percentile(d_comp, 2.5)),
                                 float(np.percentile(d_comp, 97.5))],
                        "p": float(2 * min((d_comp <= 0).mean(), (d_comp >= 0).mean()))},
        "n_boot": N_BOOT,
    }

    with open(os.path.join(HERE, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


_RUNS = None


def oofA_run(i):
    global _RUNS
    if _RUNS is None:
        import common as C
        _RUNS = C.build_all()["runs"]
    return _RUNS[i]


def hit_table(ys, ft, turns, label_ok):
    """hit@k over truly-faulty runs, on the OFFICIAL denominator.

    Two things this must get right, and the first draft got wrong:

      1. a run whose LABEL was misclassified counts as a miss, exactly as in
         ``metrics.fault_turn_hit_at_k``; so the denominator is all truly-faulty
         runs, not "runs the system called faulty";
      2. ``turns`` is indexed by the same ``i`` as ``ys``/``ft``.  Building it by
         skipping clean runs and then zipping positionally silently misaligns
         every element after the first clean row, which is what produced the
         impossible 0.18 figure in the first run of this script.
    """
    out = {"hit@0": 0, "hit@1": 0, "hit@2": 0, "n_faulty": 0,
           "n_label_correct": 0, "per_class": {}}
    per = {}
    for i, yt in enumerate(ys):
        if yt == "clean":
            continue
        out["n_faulty"] += 1
        d = per.setdefault(yt, {"n": 0, "h0": 0, "h1": 0, "h2": 0})
        d["n"] += 1
        if not label_ok[i]:
            continue                      # misclassified -> miss
        out["n_label_correct"] += 1
        tp = turns[i]                     # aligned by index, not position
        if tp is not None and int(tp) >= 0:
            dd = abs(int(tp) - int(ft[i]))
            for tag, lim in (("h0", 0), ("h1", 1), ("h2", 2)):
                if dd <= lim:
                    out["hit@%s" % tag[-1]] += 1
                    d[tag] += 1
    n = max(1, out["n_faulty"])
    for k in ("hit@0", "hit@1", "hit@2"):
        out[k + "_rate"] = out[k] / n
    for c, d in per.items():
        out["per_class"][c] = {k: d[k] / max(1, d["n"]) for k in ("h0", "h1", "h2")}
        out["per_class"][c]["n"] = d["n"]
    return out


if __name__ == "__main__":
    main()
