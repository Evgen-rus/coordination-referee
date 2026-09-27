"""Exp04 follow-up analysis: fold-level blends, significance, complementarity.

Run after ``run_experiment.py``.  All headline metrics are produced there; this
script answers the remaining questions:

  * per-fold behaviour of every system (the "2 of 3 folds" success criterion),
  * paired significance of the blend against the Exp03 B tabular foundation,
  * whether the text model is *complementary* (can it fix tabular's mistakes?),
  * why ``dropped_handoff`` resists the text signal.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from metrics import f1_per_class, macro_f1   # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
KEYS = ["baseline_flat", "exp03_B", "text_word", "text_char", "text_wc",
        "blend_a0.25", "blend_a0.50", "blend_a0.75", "blend_honest"]


def mf1(t, p):
    return macro_f1(list(t), list(p))


def main():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    yi = train["label"].map(L2I).values
    ys = np.array([LABELS[i] for i in yi], dtype=object)
    n = len(yi)
    with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
        res = json.load(f)

    preds = {}
    for key in KEYS:
        d = pd.read_csv(os.path.join(HERE, "oof_%s.csv" % key))
        assert (d["run_id"].values == train["run_id"].values).all()
        preds[key] = d["y_pred"].values

    from sklearn.model_selection import StratifiedKFold
    splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(n), yi))

    print("=" * 84)
    print("FOLD-BY-FOLD Macro F1  (d vs exp03_B)")
    print("=" * 84)
    tbl = {k: [mf1([ys[i] for i in va], preds[k][va]) for _, va in splits] for k in KEYS}
    base = tbl["exp03_B"]
    for k in KEYS:
        r = tbl[k]
        d = [r[i] - base[i] for i in range(3)]
        wins = sum(1 for x in d if x > 0)
        print("  %-14s %.4f  %.4f  %.4f   |  d = %+.4f %+.4f %+.4f   wins %d/3"
              % (k, r[0], r[1], r[2], d[0], d[1], d[2], wins))
    res["fold_macro_all"] = {k: [float(x) for x in v] for k, v in tbl.items()}

    # ---------------- significance ----------------
    print("")
    print("PAIRED BOOTSTRAP: blend_honest vs exp03_B")
    y, A, B = ys, preds["exp03_B"], preds["blend_honest"]
    rng = np.random.default_rng(0)
    boot = np.empty(2000)
    for i in range(2000):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        boot[i] = mf1(y[idx], B[idx]) - mf1(y[idx], A[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    point = mf1(y, B) - mf1(y, A)
    print("  delta macro_f1 = %+.4f   95%%CI [%+.4f, %+.4f]   P(delta>0) = %.3f"
          % (point, lo, hi, (boot > 0).mean()))

    from scipy.stats import binomtest
    fix = int(((B == y) & (A != y)).sum())
    brk = int(((A == y) & (B != y)).sum())
    pv = float(binomtest(min(fix, brk), fix + brk, 0.5, alternative="two-sided").pvalue)
    print("  McNemar: fixed %d, broke %d, net %+d, two-sided p = %.4f" % (fix, brk, fix - brk, pv))

    print("")
    print("  same test for the better fixed alpha (a=0.50) and drop of robustness")
    A50 = preds["blend_a0.50"]
    f2 = int(((A50 == y) & (A != y)).sum())
    b2 = int(((A == y) & (A50 != y)).sum())
    pv2 = float(binomtest(min(f2, b2), f2 + b2, 0.5, alternative="two-sided").pvalue)
    print("  a=0.50: fixed %d, broke %d, net %+d, p = %.4f" % (f2, b2, f2 - b2, pv2))
    res["significance"] = {
        "honest": {"delta_macro": float(point), "ci95": [float(lo), float(hi)],
                   "p_delta_gt0": float((boot > 0).mean()),
                   "fixed": fix, "broke": brk, "mcnemar_p": pv},
        "a050": {"fixed": f2, "broke": b2, "net": f2 - b2, "mcnemar_p": pv2}}

    # ---------------- complementarity ----------------
    print("")
    print("COMPLEMENTARITY: is the text model right where the tabular model is wrong?")
    wrongA = A != y
    for k in ("text_word", "text_char", "text_wc"):
        T = preds[k]
        hit = int(((T == y) & wrongA).sum())
        orc = mf1(y, np.where(A == y, A, T))
        print("  %-10s correct on %4d / %4d tabular misses (%.1f%%)   oracle(A,T) macro = %.4f"
              % (k, hit, int(wrongA.sum()), 100 * hit / max(1, int(wrongA.sum())), orc))
        res.setdefault("complementarity", {})[k] = {
            "hits_on_tabular_misses": hit, "n_tabular_misses": int(wrongA.sum()),
            "oracle_macro": float(orc)}
    print("  mean agreement with exp03_B: %.1f%%"
          % (100 * np.mean([np.mean(preds[k] == A) for k in ("text_word", "text_char", "text_wc")])))

    # ---------------- per class, blend ----------------
    print("")
    print("PER-CLASS F1: exp03_B -> blend_honest, and text_wc for reference")
    fa = f1_per_class(list(y), list(A), LABELS)
    fb = f1_per_class(list(y), list(B), LABELS)
    fw = f1_per_class(list(y), list(preds["text_wc"]), LABELS)
    for c in LABELS:
        print("  %-18s %.4f -> %.4f (%+.4f)   text_wc=%.4f" % (c, fa[c], fb[c], fb[c] - fa[c], fw[c]))
    res["per_class_blend"] = {c: {"exp03_B": fa[c], "blend": fb[c],
                                  "text_wc": fw[c], "delta": fb[c] - fa[c]} for c in LABELS}

    # ---------------- dropped_handoff ----------------
    print("")
    print("DROPPED_HANDOFF (n=%d) recall by system" % int((ys == "dropped_handoff").sum()))
    dh = ys == "dropped_handoff"
    res["dropped_handoff_recall"] = {}
    for k in ("baseline_flat", "exp03_B", "text_word", "text_char", "text_wc", "blend_honest"):
        r = float((preds[k][dh] == "dropped_handoff").mean())
        res["dropped_handoff_recall"][k] = r
        print("  %-14s recall = %.4f" % (k, r))

    # ---------------- no_intent deep dive ----------------
    print("")
    print("NO_INTENT deep dive")
    oof = pd.read_csv(os.path.join(HERE, "oof_exp03_B.csv"))
    noint = oof["n_intents"].values <= 1
    idx = np.where(noint)[0]
    print("  n = %d   label distribution:" % len(idx))
    print("   ", pd.Series(ys[idx]).value_counts().to_dict())
    for k in ("exp03_B", "text_word", "text_wc", "blend_honest"):
        print("  %-14s macro = %.4f" % (k, mf1([ys[i] for i in idx], [preds[k][i] for i in idx])))
    res["no_intent"] = {"n": int(noint.sum()),
                        "label_mix": pd.Series(ys[idx]).value_counts().to_dict()}

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=lambda o: float(o))
    print("")
    print("updated results.json")


if __name__ == "__main__":
    main()
