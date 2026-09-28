"""Experiment 06 driver: 12 variants, 3-fold OOF, on the Exp05 B foundation.

Variants 0-6 are incremental (each block added to the clean 226-col foundation)
and carry the honest attribution; 7-11 are drop-one, reported for completeness
only because bd_slope / bd_grows_late / bd_tail_growth are mutually
substitutable.  Success F1 is the saved Exp05 B OOF; fault_turn uses the official
rule-based localiser on each variant's own label.
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (ALL6, DH, INCR, LABELS, build_all, composite,  # noqa: E402
                    f1_per_class, fault_turn_hit_at_k, ft, lgb_label,
                    localize, log, mf1, prf)
import report  # noqa: E402


def main():
    c = build_all()
    runs, mats, yi, ys = c["runs"], c["mats"], c["yi"], c["ys"]
    n, nmsg = c["n"], c["nmsg"]

    splits = list(StratifiedKFold(3, shuffle=True, random_state=0)
                  .split(np.zeros(n), yi))
    oof = {k: np.zeros(n, dtype=int) for k in mats}
    fold_m = {k: [] for k in mats}
    gain = {k: np.zeros(m.shape[1]) for k, m in mats.items()}

    for f, (tr, va) in enumerate(splits):
        log("FOLD %d train=%d val=%d" % (f, len(tr), len(va)))
        for key, X in mats.items():
            m = lgb_label().fit(X.iloc[tr], yi[tr])
            p = m.predict(X.iloc[va])
            oof[key][va] = p
            fold_m[key].append(mf1([ys[i] for i in va], [LABELS[j] for j in p]))
            gain[key] += m.booster_.feature_importance(importance_type="gain")
            log("  %-20s fold macro_f1=%.4f" % (key, fold_m[key][-1]))
            del m
        gc.collect()

    names = {k: [LABELS[i] for i in oof[k]] for k in oof}
    objs = {k: np.array(v, dtype=object) for k, v in names.items()}
    SL = c["SL"]

    def sl_macro(key, m):
        idx = np.where(m)[0]
        return None if len(idx) < 30 else mf1([ys[i] for i in idx],
                                              [names[key][i] for i in idx])

    res = {"n_runs": int(n), "n_new_feats": len(ALL6),
           "success_f1": c["success_f1"], "systems": {}}
    for key in mats:
        mf = mf1(ys, names[key])
        rf = sl_macro(key, SL["hard/robustness"])
        turns = [localize(runs[i], names[key][i]) for i in range(n)]
        h2 = fault_turn_hit_at_k(list(ys), names[key], list(c["fturn"]), turns)
        res["systems"][key] = {
            "n_features": int(mats[key].shape[1]),
            "macro_f1": mf, "robustness_f1": rf, "success_f1": c["success_f1"],
            "fault_turn_hit2": h2,
            "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                    "success_f1": c["success_f1"],
                                    "fault_turn_hit2": h2}),
            "per_class": f1_per_class(list(ys), names[key], LABELS),
            "fold_macro_f1": [float(x) for x in fold_m[key]],
            "slices": {s: sl_macro(key, m) for s, m in SL.items()},
            "dh_overall": prf(ys, objs[key], DH),
            "dh_subslices": {s: prf(ys, objs[key], DH, m)
                             for s, m in c["DH_SL"].items()},
        }
    ref = res["systems"]["0_B_exp05_B"]
    for key in mats:
        d = res["systems"][key]
        d["d_macro"] = d["macro_f1"] - ref["macro_f1"]
        d["d_rob"] = d["robustness_f1"] - ref["robustness_f1"]
        d["d_comp"] = d["composite"] - ref["composite"]
        d["d_dh"] = d["dh_overall"]["f1"] - ref["dh_overall"]["f1"]
        d["d_dh_long"] = (d["dh_subslices"]["long(>p66)"]["f1"]
                          - ref["dh_subslices"]["long(>p66)"]["f1"])

    for key in oof:
        pd.DataFrame({"run_id": c["train"]["run_id"], "y_true": ys,
                      "y_pred": names[key], "n_messages": nmsg}).to_csv(
            os.path.join(HERE, "oof_%s.csv" % key), index=False)
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    log("saved results.json + %d oof files" % len(oof))

    g = gain["6_B_ALL"] / gain["6_B_ALL"].sum()
    s = pd.Series(g, index=mats["6_B_ALL"].columns).sort_values(ascending=False)
    new = set(ALL6)
    top = s.reindex(ALL6).fillna(0.0).sort_values(ascending=False).head(20)
    res["importance"] = {
        "gain_share": float(s.reindex(ALL6).fillna(0.0).sum()),
        "n_used": int((s.reindex(ALL6).fillna(0.0) > 0).sum()),
        "in_top20": sum(1 for x in s.head(20).index if x in new),
        "in_top50": sum(1 for x in s.head(50).index if x in new),
        "by_block": {b: float(s.reindex(cols).fillna(0.0).sum())
                     for b, cols in ft.BLOCKS.items()},
        "top_new": {x: float(s[x]) for x in top.index}}
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    report.show(res, s, objs, ys, c)


if __name__ == "__main__":
    main()
