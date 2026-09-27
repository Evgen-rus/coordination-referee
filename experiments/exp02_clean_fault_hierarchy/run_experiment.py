"""Experiment 02 driver: clean-vs-fault hierarchy vs the flat baseline.

Same 3 folds as the baseline configuration, only the official 122 features,
so the comparison isolates the architecture and nothing else.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from features import extract_features, parse_run
from localize import localize
from metrics import composite, f1_per_class, fault_turn_hit_at_k
from models import (BASE, BASE_COMPOSITE, FAULTY, LABELS, VARIANTS, make_binary,
                    make_flat, make_type, mf1, tune_threshold)

_T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - _T0, m), flush=True)


def main():
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    y = train["label"].values
    ft = train["fault_turn"].values
    is_fault = (y != "clean").astype(int)

    log("extracting the official 122 features")
    runs = [parse_run(r) for r in train.to_dict("records")]
    X = pd.DataFrame([extract_features(r) for r in runs]).fillna(0.0)
    assert X.shape[1] == 122, X.shape

    qs = np.quantile(X["n_messages"].values, [0.33, 0.66])
    tert = {"short": X["n_messages"].values <= qs[0],
            "mid": (X["n_messages"].values > qs[0]) & (X["n_messages"].values <= qs[1]),
            "long": X["n_messages"].values > qs[1]}
    hard = ((X["n_messages"] > X["n_messages"].median())
            & (X["topo_mesh"] + X["topo_blackboard"] > 0)).values
    for k, v in tert.items():
        log("  %-6s n=%d" % (k, int(v.sum())))
    log("  hard   n=%d" % int(hard.sum()))

    splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    oof = {v: np.empty(len(y), dtype=object) for v in VARIANTS}
    thr_used = {"B_hier_thr_tuned": [], "C2_hier_bal_thr_tuned": []}
    fold_macro = {v: [] for v in VARIANTS}

    for k, (tr, va) in enumerate(splits):
        log("FOLD %d  train=%d val=%d" % (k, len(tr), len(va)))
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ftr = is_fault[tr]

        oof["baseline_flat"][va] = make_flat().fit(Xtr, y[tr]).predict(Xva)
        fold_macro["baseline_flat"].append(mf1(y[va], oof["baseline_flat"][va]))

        for weighted in (False, True):
            gate = make_binary(weighted).fit(Xtr, ftr)
            pf = gate.predict_proba(Xva)[:, 1]
            fr = tr[ftr == 1]
            tm = make_type().fit(X.iloc[fr], y[fr])
            # class order comes from the fitted model (alphabetical for string
            # labels); using the FAULTY constant would permute every class
            cls = tm.classes_[np.argmax(tm.predict_proba(Xva), axis=1)]
            assert set(tm.classes_) == set(FAULTY), tm.classes_
            k05, kt = ("A_hier_thr0.5", "B_hier_thr_tuned") if not weighted \
                else ("C1_hier_bal_thr0.5", "C2_hier_bal_thr_tuned")
            oof[k05][va] = np.where(pf >= 0.5, cls, "clean")
            best, bf, f05 = tune_threshold(X, y, ftr, tr, weighted)
            thr_used[kt].append(best)
            oof[kt][va] = np.where(pf >= best, cls, "clean")
            for v in (k05, kt):
                fold_macro[v].append(mf1(y[va], oof[v][va]))
            log("  %-22s thr=%.2f (inner %.4f vs 0.5->%.4f) fold macro=%.4f"
                % (kt, best, bf, f05, fold_macro[kt][-1]))
        log("  baseline_flat         fold macro=%.4f" % fold_macro["baseline_flat"][-1])
        log("  A_hier_thr0.5         fold macro=%.4f" % fold_macro["A_hier_thr0.5"][-1])

    summary, real = summarise(oof, y, hard, tert, fold_macro, thr_used, ft, runs)

    out = {"baseline_reference": BASE, "baseline_composite_isolated": BASE_COMPOSITE,
           "variants": summary, "realistic_composite": real, "thresholds": thr_used,
           "runtime_s": time.time() - _T0, "hard_n": int(hard.sum()),
           "strata_n": {k: int(v.sum()) for k, v in tert.items()}}
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    for v in VARIANTS:
        pd.DataFrame({"run_id": train["run_id"], "true": y, "pred": oof[v]}
                     ).to_csv(os.path.join(HERE, "oof_%s.csv" % v), index=False)
    log("")
    log("wrote results.json + oof_*.csv; total %.1fs" % (time.time() - _T0))


def summarise(oof, y, hard, tert, fold_macro, thr_used, ft, runs):
    log("=" * 76)
    log("OVERALL (3-fold OOF, identical splits)")
    log("=" * 76)
    summary = {}
    for v in VARIANTS:
        p = oof[v]
        mf, rf = mf1(y, p), mf1(y[hard], p[hard])
        comp = composite({"macro_f1": mf, "robustness_f1": rf,
                          "success_f1": BASE["success_f1"],
                          "fault_turn_hit2": BASE["fault_turn_hit2"]})
        cmask = p == "clean"
        faulty = y != "clean"
        summary[v] = {
            "macro_f1": mf, "robustness_f1": rf, "composite": comp,
            "d_macro": mf - BASE["macro_f1"], "d_rob": rf - BASE["robustness_f1"],
            "d_comp": comp - BASE_COMPOSITE,
            "clean_recall": float((p[y == "clean"] == "clean").mean()),
            "clean_precision": float((y[cmask] == "clean").mean()),
            "false_clean_total": int((p[faulty] == "clean").sum()),
            "clean_pred_fault": int((p[cmask] != "clean").sum()),
            "false_clean_per_class": {c: int((p[y == c] == "clean").sum()) for c in FAULTY},
            "per_class": f1_per_class(list(y), list(p), LABELS),
            "fold_macro_f1": [float(x) for x in fold_macro[v]],
            "fold_macro_min": float(np.min(fold_macro[v])),
            "fold_macro_std": float(np.std(fold_macro[v])),
            "strata": {k: mf1(y[m], p[m]) for k, m in tert.items()},
        }

    log("")
    log("%-24s %8s %8s %9s %8s %11s" % ("variant", "MacroF1", "dMacro", "RobustF1",
                                        "dRobust", "dComposite"))
    for v in VARIANTS:
        s = summary[v]
        log("%-24s %8.4f %+8.4f %9.4f %+8.4f %+11.4f"
            % (v, s["macro_f1"], s["d_macro"], s["robustness_f1"], s["d_rob"],
               s["d_comp"]))

    log("")
    log("%-24s %11s %11s %11s %12s %9s %8s"
        % ("variant", "clean_recall", "clean_prec", "false_clean", "clean->fault",
           "fold_min", "fold_std"))
    for v in VARIANTS:
        s = summary[v]
        log("%-24s %11.4f %11.4f %11d %12d %9.4f %8.4f"
            % (v, s["clean_recall"], s["clean_precision"], s["false_clean_total"],
               s["clean_pred_fault"], s["fold_macro_min"], s["fold_macro_std"]))

    log("")
    log("false-clean per class (true -> predicted clean):")
    for v in VARIANTS:
        log("  %-24s %s" % (v, " ".join("%s=%d" % (c[:8], summary[v]["false_clean_per_class"][c])
                                        for c in FAULTY)))
    log("")
    log("thresholds from inner CV (train part only):")
    for k, v in thr_used.items():
        log("  %-22s %s" % (k, ["%.2f" % t for t in v]))

    log("")
    log("PER-CLASS F1")
    log("%-18s %s" % ("class", " ".join("%10s" % v[:10] for v in VARIANTS)))
    for c in LABELS:
        log("%-18s %s" % (c, " ".join("%10.4f" % summary[v]["per_class"][c]
                                      for v in VARIANTS)))

    log("")
    log("MACRO F1 BY RUN LENGTH (tertiles) + HARD SLICE")
    log("%-24s %s" % ("variant", " ".join("%10s" % v[:10] for v in VARIANTS)))
    for k in ("short", "mid", "long"):
        log("%-24s %s" % ("len_" + k, " ".join("%10.4f" % summary[v]["strata"][k]
                                                for v in VARIANTS)))
    log("%-24s %s" % ("hard_slice", " ".join("%10.4f" % summary[v]["robustness_f1"]
                                             for v in VARIANTS)))

    for v in VARIANTS:
        cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(oof[v], name="pred"))
        cm = cm.reindex(index=LABELS, columns=LABELS, fill_value=0)
        log("")
        log("CONFUSION MATRIX: %s" % v)
        print(cm.to_string())

    log("")
    log("=" * 76)
    log("SECONDARY: hit@2 recomputed per variant (not part of the verdict)")
    log("=" * 76)
    real, ref = {}, None
    for v in VARIANTS:
        turns = [localize(runs[i], oof[v][i]) for i in range(len(y))]
        h2 = fault_turn_hit_at_k(list(y), list(oof[v]), list(ft), turns)
        s = summary[v]
        c = composite({"macro_f1": s["macro_f1"], "robustness_f1": s["robustness_f1"],
                       "success_f1": BASE["success_f1"], "fault_turn_hit2": h2})
        real[v] = {"hit2": h2, "composite": c}
        if v == "baseline_flat":
            ref = c
    for v in VARIANTS:
        log("  %-24s hit@2=%.4f composite=%.4f (d %+.4f vs flat)"
            % (v, real[v]["hit2"], real[v]["composite"],
               real[v]["composite"] - ref))

    return summary, real


if __name__ == "__main__":
    main()
