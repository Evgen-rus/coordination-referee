"""Exp06 significance: paired bootstrap + McNemar vs the Exp05 B foundation.

The long-run dropped_handoff slice contains ONLY true dropped_handoff runs, so
precision there is 1.0 by construction and its F1 is a monotone transform of
recall.  The long-slice number is therefore a recall diagnostic, bootstrapped as
recall rather than over-read as an F1.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "evaluation"))
from metrics import macro_f1                                # noqa: E402
from stat_common import KEYS, load, long_f1, long_recall, recall_at   # noqa: E402


def mf1(t, p):
    return macro_f1(list(t), list(p))


def main():
    from scipy.stats import binomtest
    _t, y, yi, P, NM = load()
    n = len(y)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=0)
                  .split(np.zeros(n), yi))
    q66 = np.quantile(NM["0_B_exp05_B"], 2 / 3)
    long_m = (y == "dropped_handoff") & (NM["0_B_exp05_B"] > q66)
    rng = np.random.default_rng(0)
    A = P["0_B_exp05_B"]
    allidx = np.arange(n)
    out = {}

    print("=" * 84)
    print("PAIRED BOOTSTRAP on Macro F1 vs foundation (2000 resamples)")
    print("=" * 84)
    for k in KEYS[1:]:
        B = P[k]
        bt = np.empty(2000)
        for i in range(2000):
            ix = rng.choice(allidx, size=n, replace=True)
            bt[i] = mf1(y[ix], B[ix]) - mf1(y[ix], A[ix])
        lo, hi = np.percentile(bt, [2.5, 97.5])
        print("  %-20s d=%+.4f  95%%CI [%+.4f,%+.4f]  P(>0)=%.3f"
              % (k, mf1(y, B) - mf1(y, A), lo, hi, (bt > 0).mean()))
        out.setdefault(k, {})["macro"] = {"d": float(mf1(y, B) - mf1(y, A)),
                                          "ci": [float(lo), float(hi)],
                                          "p": float((bt > 0).mean())}

    print("")
    print("McNemar (exact two-sided) vs foundation")
    for k in KEYS[1:]:
        B = P[k]
        fx = int(((B == y) & (A != y)).sum())
        bk = int(((A == y) & (B != y)).sum())
        pv = float(binomtest(min(fx, bk), fx + bk, 0.5,
                             alternative="two-sided").pvalue)
        print("  %-20s fixed=%4d broke=%4d net=%+4d p=%.4f" % (k, fx, bk, fx - bk, pv))
        out.setdefault(k, {})["mcnemar"] = {"fixed": fx, "broke": bk, "p": pv}

    print("")
    print("LONG-SLICE dh (n=%d true dh; P==1.0 by construction)"
          % int(long_m.sum()))
    print("  %-20s %6s %8s %8s %8s" % ("variant", "P", "R", "F1", "dF1"))
    b0 = long_f1(A, long_m)
    for k in KEYS:
        r = long_recall(P[k], long_m)
        f1 = 2 * r / (1 + r) if r else 0.0
        print("  %-20s %6.1f %8.4f %8.4f %+8.4f" % (k, 1.0, r, f1, f1 - b0))
        out.setdefault(k, {})["dh_long"] = {"recall": float(r), "f1": float(f1),
                                             "d_f1": float(f1 - b0)}

    print("")
    print("LONG-SLICE RECALL bootstrap (slice F1 is monotone in recall)")
    idx = np.where(long_m)[0]
    m = len(idx)
    for k in ["6_B_ALL", "7_full_minus_age", "8_full_minus_backlog", "3_B_receiver"]:
        d = np.empty(2000)
        for i in range(2000):
            b = idx[rng.choice(m, size=m, replace=True)]
            d[i] = recall_at(P[k], b) - recall_at(A, b)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  %-20s d_recall=%+.4f 95%%CI [%+.4f,%+.4f] P(>0)=%.3f"
              % (k, d.mean(), lo, hi, (d > 0).mean()))
        out.setdefault(k, {})["dh_recall_boot"] = {
            "d": float(d.mean()), "ci": [float(lo), float(hi)],
            "p": float((d > 0).mean())}

    print("")
    print("IS THE AGE BLOCK HARMFUL?  full(6) vs full-minus-age(7)")
    six, sev = P["6_B_ALL"], P["7_full_minus_age"]
    fx = int(((sev == y) & (six != y)).sum())
    bk = int(((six == y) & (sev != y)).sum())
    pv = float(binomtest(min(fx, bk), fx + bk, 0.5, alternative="two-sided").pvalue)
    bt = np.empty(2000)
    for i in range(2000):
        ix = rng.choice(allidx, size=n, replace=True)
        bt[i] = mf1(y[ix], sev[ix]) - mf1(y[ix], six[ix])
    lo, hi = np.percentile(bt, [2.5, 97.5])
    print("  d(7-6) macro = %+.4f  95%%CI [%+.4f,%+.4f]  P(>0)=%.3f"
          % (mf1(y, sev) - mf1(y, six), lo, hi, (bt > 0).mean()))
    print("  McNemar: 7 fixes %d, breaks %d, p=%.4f" % (fx, bk, pv))
    out["_age"] = {"d_macro": float(mf1(y, sev) - mf1(y, six)),
                   "ci": [float(lo), float(hi)], "p": float((bt > 0).mean()),
                   "fixed": fx, "broke": bk, "mcnemar_p": pv}

    print("")
    print("PER-FOLD d(macro) vs foundation")
    for k in KEYS[1:]:
        d = [mf1(y[va], P[k][va]) - mf1(y[va], A[va]) for _, va in splits]
        print("  %-20s %s  wins %d/3"
              % (k, ", ".join("%+.4f" % x for x in d), sum(1 for x in d if x > 0)))

    with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    res["significance"] = out
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=lambda o: float(o))
    print("")
    print("updated results.json")


if __name__ == "__main__":
    main()
