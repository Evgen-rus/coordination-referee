"""Exp06b driver: 4 pre-registered systems x 5 seeds x 3-fold OOF.

Nothing is selected here.  The systems were fixed before any of these splits
were run, and the output table is the whole study.  The one thing this script
deliberately does NOT do is pick a fifth subset out of the results - that is
explicitly out of scope, because choosing a subset on the same splits used to
score it would recreate exactly the post-hoc bias this study exists to measure.
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
from common import (CANDIDATES, COLS, DH, N_FEATURES, N_FOLDS,  # noqa: E402
                    REFERENCE, SEEDS, build, lgb_label, log, macro_f1, prf)
from metrics import composite, fault_turn_hit_at_k          # noqa: E402
from localize import localize                              # noqa: E402
from common import LABELS                                  # noqa: E402

KEYS = list(COLS.keys())


def sl_macro(ys, pred, mask, min_n=30):
    idx = np.where(mask)[0]
    if len(idx) < min_n:
        return None
    return macro_f1([ys[i] for i in idx], [pred[i] for i in idx])


def main():
    c = build()
    runs, mats, yi, ys = c["runs"], c["mats"], c["yi"], c["ys"]
    n, nmsg, fturn = c["n"], c["nmsg"], c["fturn"]
    success_f1, SL = c["success_f1"], c["SL"]
    rob_m, mesh_m, dhlong_m = SL["hard/robustness"], SL["topo_mesh"], \
        c["DH_SL"]["long(>p66)"]

    out = {"n_runs": int(n), "n_folds": N_FOLDS, "seeds": SEEDS,
           "success_f1": success_f1, "n_features": N_FEATURES,
           "reference": REFERENCE, "runs": {}}

    for seed in SEEDS:
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
                      .split(np.zeros(n), yi))
        oof = {k: np.zeros(n, dtype=int) for k in KEYS}
        fold_m = {k: [] for k in KEYS}

        for f, (tr, va) in enumerate(splits):
            log("seed=%d FOLD %d train=%d val=%d" % (seed, f, len(tr), len(va)))
            for key in KEYS:
                m = lgb_label().fit(mats[key].iloc[tr], yi[tr])
                p = m.predict(mats[key].iloc[va])
                oof[key][va] = p
                fold_m[key].append(macro_f1([ys[i] for i in va],
                                           [LABELS[j] for j in p]))
                del m
            gc.collect()
            log("  " + "  ".join("%s=%.4f" % (k, fold_m[k][-1]) for k in KEYS))

        seed_res = {}
        for key in KEYS:
            pred = [LABELS[i] for i in oof[key]]
            po = np.array(pred, dtype=object)
            mf = macro_f1(ys, pred)
            rf = sl_macro(ys, po, rob_m)
            turns = [localize(runs[i], pred[i]) for i in range(n)]
            h2 = fault_turn_hit_at_k(list(ys), pred, list(fturn), turns)
            long = prf(ys, po, DH, dhlong_m)
            # Guard the Exp06 finding that this slice has precision 1.0 by
            # construction, so the "F1" is really a recall proxy.  If that ever
            # stops holding, the printed number would silently mean something
            # else, so assert it instead of assuming it.
            assert long["precision"] == 1.0, \
                "long slice precision %.4f != 1.0; not a recall proxy" % long["precision"]
            seed_res[key] = {
                "macro_f1": mf,
                "robustness_f1": rf,
                "success_f1": success_f1,
                "fault_turn_hit2": h2,
                "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                        "success_f1": success_f1,
                                        "fault_turn_hit2": h2}),
                "dropped_handoff_f1": prf(ys, po, DH)["f1"],
                "long_dh_recall": long["recall"],
                "deadlock_f1": prf(ys, po, "deadlock")["f1"],
                "topo_mesh_macro_f1": sl_macro(ys, po, mesh_m),
                "fold_macro_f1": [float(x) for x in fold_m[key]],
            }
            pd.DataFrame({"run_id": c["train"]["run_id"], "y_true": ys,
                          "y_pred": pred, "n_messages": nmsg}).to_csv(
                os.path.join(HERE, "oof_%s_seed%d.csv" % (key, seed)), index=False)
        out["runs"][str(seed)] = seed_res
        log("seed=%d done" % seed)
        with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)

    report(out)
    return out


METRICS = [("macro_f1", "Macro F1"), ("robustness_f1", "Robustness F1"),
           ("composite", "composite"), ("dropped_handoff_f1", "dh F1 overall"),
           ("long_dh_recall", "long dh RECALL"),
           ("deadlock_f1", "deadlock F1"), ("topo_mesh_macro_f1", "topo_mesh Macro F1")]


def report(out):
    runs = out["runs"]
    lines = []
    add = lines.append

    add("## Exp06b - per-seed OOF metrics (3-fold, seed is the CV split only)\n")
    for mk, mname in METRICS:
        add("### %s\n" % mname)
        add("| system | features | " + " | ".join("s%d" % s for s in SEEDS)
            + " | mean | std | min | max |")
        add("|---|---|" + "---|" * (len(SEEDS) + 4))
        for k in KEYS:
            v = [runs[str(s)][k][mk] for s in SEEDS]
            add("| %s | %d | %s | **%.4f** | %.4f | %.4f | %.4f |"
                % (k, out["n_features"][k],
                   " | ".join("%.4f" % x for x in v),
                   np.mean(v), np.std(v, ddof=1), np.min(v), np.max(v)))
        add("")

    add("### Delta vs Exp05 B (same seed, same folds)\n")
    ref = REFERENCE
    for mk, mname in METRICS:
        add("**%s**\n" % mname)
        add("| system | " + " | ".join("s%d" % s for s in SEEDS)
            + " | mean | std | wins |")
        add("|---|" + "---|" * (len(SEEDS) + 3))
        for k in KEYS:
            if k == ref:
                continue
            d = [runs[str(s)][k][mk] - runs[str(s)][ref][mk] for s in SEEDS]
            w = sum(1 for x in d if x > 0)
            add("| %s | %s | **%+.4f** | %.4f | %d/%d |"
                % (k, " | ".join("%+.4f" % x for x in d),
                   np.mean(d), np.std(d, ddof=1), w, len(d)))
        add("")

    with open(os.path.join(HERE, "SEED_TABLE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
