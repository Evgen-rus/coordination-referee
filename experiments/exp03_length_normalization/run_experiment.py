"""Experiment 03 driver: length-normalised features on top of the 122 baseline.

Three variants on the SAME 3 folds, identical LightGBM hyperparameters, no
tuning, no architecture change:

  A  baseline       : the official 122 features
  B  baseline + new : 122 + 63 normalised / positional features   (main candidate)
  C  norm only      : 122 minus the absolute volume features, plus the new
                      ones (ablation for causality only, never a candidate)

Success head is refit per variant and reported separately. fault_turn uses the
official rule-based localiser unchanged; its value changes only as a
consequence of the predicted label.
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

from features import extract_features, parse_run          # noqa: E402
from localize import localize                             # noqa: E402
from metrics import binary_f1, composite, f1_per_class     # noqa: E402
from metrics import fault_turn_hit_at_k, macro_f1         # noqa: E402
import new_features as nf                                 # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
VARIANTS = ["A_baseline122", "B_baseline_plus_norm", "C_norm_only"]
SHORT = ["A_baseline", "B_plus_norm", "C_norm_only"]

_T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - _T0, m), flush=True)


def mf1(a, b):
    assert len(a) == len(b), "alignment mismatch %d vs %d" % (len(a), len(b))
    return macro_f1(list(a), list(b))


def names(idx=None, arr=None):
    """Integer label ids -> label strings (works for scalars, lists, arrays)."""
    if arr is None:
        return [LABELS[i] for i in idx]
    return [LABELS[int(i)] for i in arr[slice(None)]]


def lgb_label():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def lgb_success():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05,
                              num_leaves=63, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                              random_state=42, verbose=-1)


def main():
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    yi = train["label"].map(L2I).values            # integer labels, as baseline
    ys = np.array([LABELS[i] for i in yi], dtype=object)
    s = train["success"].values
    ft = train["fault_turn"].values

    log("parsing runs + official 122 features")
    runs = [parse_run(r) for r in train.to_dict("records")]
    base_rows = [extract_features(r) for r in runs]
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    log("baseline features: %d" % Xb.shape[1])

    log("building new normalised / positional features")
    t0 = time.time()
    Xn = nf.build_matrix(runs, base_rows)
    build_s = time.time() - t0
    log("new features: %d  (build %.1fs)" % (Xn.shape[1], build_s))

    vol = [c for c in nf.VOLUME_BASELINE if c in Xb.columns]
    mats = {
        "A_baseline122": Xb.copy(),
        "B_baseline_plus_norm": pd.concat([Xb, Xn], axis=1),
        "C_norm_only": Xb.drop(columns=vol).join(Xn),
    }
    log("volume feats dropped in C: %d | columns  A=%d  B=%d  C=%d"
        % (len(vol), *(mats[v].shape[1] for v in VARIANTS)))

    splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(mats["A_baseline122"], yi))
    oof = {v: np.zeros(len(yi), dtype=int) for v in VARIANTS}
    oof_s = {v: np.zeros(len(yi), dtype=int) for v in VARIANTS}
    fold_m = {v: [] for v in VARIANTS}
    gain = {v: np.zeros(m.shape[1]) for v, m in mats.items()}

    for k, (tr, va) in enumerate(splits):
        log("FOLD %d  train=%d  val=%d" % (k, len(tr), len(va)))
        for v, X in mats.items():
            m = lgb_label().fit(X.iloc[tr], yi[tr])
            oof[v][va] = m.predict(X.iloc[va])
            oof_s[v][va] = lgb_success().fit(X.iloc[tr], s[tr]).predict(X.iloc[va])
            gain[v] += m.booster_.feature_importance(importance_type="gain")
            fold_m[v].append(mf1([ys[i] for i in va],
                                 names(arr=oof[v][va])))
            log("   %-20s fold macro_f1=%.4f" % (v, fold_m[v][-1]))

    out = {"n_new_features": int(Xn.shape[1]),
           "n_volume_dropped_in_C": len(vol),
           "new_feature_build_s": build_s,
           "n_runs": int(len(yi)), "variants": {}}

    nmsg = mats["A_baseline122"]["n_messages"].values
    q33, q66 = np.quantile(nmsg, [0.33, 0.66])
    med = np.median(nmsg)
    tert = {"short (p0-33)": nmsg <= q33,
            "mid (p33-66)": (nmsg > q33) & (nmsg <= q66),
            "long (p66-100)": nmsg > q66}
    hard = ((nmsg > med)
            & ((mats["A_baseline122"]["topo_mesh"].values > 0)
               | (mats["A_baseline122"]["topo_blackboard"].values > 0)))
    XA = mats["A_baseline122"]
    dec = np.quantile(nmsg, np.linspace(0, 1, 11))
    slices = {"topo_star": XA["topo_star"].values > 0,
              "topo_pipeline": XA["topo_pipeline"].values > 0,
              "topo_mesh": XA["topo_mesh"].values > 0,
              "topo_hierarchical": XA["topo_hierarchical"].values > 0,
              "topo_blackboard": XA["topo_blackboard"].values > 0,
              "rare_topo(mesh+bb)": (XA["topo_mesh"].values > 0) | (XA["topo_blackboard"].values > 0),
              "big_team_top20%": XA["n_agents"].values >= np.quantile(XA["n_agents"].values, 0.8),
              "no_intent(n_intents<=1)": XA["n_intents"].values <= 1,
              "longest_20%": nmsg >= np.quantile(nmsg, 0.8)}

    log("")
    log("=" * 82)
    log("HEADLINE (3-fold OOF, identical folds and hyperparameters)")
    log("=" * 82)
    log("%-20s %8s %9s %9s %8s %9s %9s" % ("variant", "MacroF1", "RobustF1",
                                           "SuccessF1", "hit@2", "composite", "dComp"))
    base_ref = None
    for v in VARIANTS:
        p = names(arr=oof[v])
        hidx = np.where(hard)[0]
        mf = mf1(ys, p)
        rf = mf1([ys[i] for i in hidx], [p[i] for i in hidx])
        sf = binary_f1(list(s), list(oof_s[v]))
        turns = [localize(runs[i], p[i]) for i in range(len(yi))]
        h2 = fault_turn_hit_at_k(ys, p, list(ft), turns)
        comp = composite({"macro_f1": mf, "robustness_f1": rf,
                          "success_f1": sf, "fault_turn_hit2": h2})
        if base_ref is None:
            base_ref = (mf, rf, sf, h2, comp)
        ci = np.where(oof[v] == L2I["clean"])[0]
        out["variants"][v] = {
            "macro_f1": mf, "robustness_f1": rf, "success_f1": sf,
            "fault_turn_hit2": h2, "composite": comp,
            "d_macro": mf - base_ref[0], "d_rob": rf - base_ref[1],
            "d_success": sf - base_ref[2], "d_hit2": h2 - base_ref[3],
            "d_composite": comp - base_ref[4],
            "per_class": f1_per_class(ys, p, LABELS),
            "fold_macro_f1": [float(x) for x in fold_m[v]],
            "fold_std": float(np.std(fold_m[v])),
            "clean_precision": float(np.mean(ys[ci] == "clean")) if len(ci) else 0.0,
            "clean_recall": float(np.mean(oof[v][yi == L2I["clean"]] == L2I["clean"])),
        }
        log("%-20s %8.4f %9.4f %9.4f %8.4f %9.4f %+9.4f"
            % (v, mf, rf, sf, h2, comp, comp - base_ref[4]))

    log("")
    log("PER-CLASS F1")
    log("%-18s %s" % ("class", "".join("%13s" % t for t in SHORT)))
    for c in LABELS:
        log("%-18s %s" % (c, "".join("%13.4f" % out["variants"][v]["per_class"][c]
                                     for v in VARIANTS)))
    log("")
    log("fold macro_f1  " + "   ".join(
        "%s=%s(std %.4f)" % (SHORT[i], ",".join("%.4f" % x for x in out["variants"][v]["fold_macro_f1"]),
                             out["variants"][v]["fold_std"]) for i, v in enumerate(VARIANTS)))
    log("clean precision " + "  ".join(
        "%s=%.4f" % (SHORT[i], out["variants"][v]["clean_precision"]) for i, v in enumerate(VARIANTS)))
    log("clean recall    " + "  ".join(
        "%s=%.4f" % (SHORT[i], out["variants"][v]["clean_recall"]) for i, v in enumerate(VARIANTS)))

    def strat(title, masks):
        log("")
        log(title)
        log("%-24s %6s %s" % ("slice", "n", "".join("%13s" % t for t in SHORT)))
        rows = {}
        for key, m in masks.items():
            if m.sum() < 30:
                log("%-24s %6d  (too small, skipped)" % (key, int(m.sum())))
                continue
            idx = np.where(m)[0]
            vals = [mf1([ys[i] for i in idx], [names(arr=oof[v])[i] for i in idx])
                    for v in VARIANTS]
            rows[key] = {"n": int(m.sum()), "macro_f1": dict(zip(SHORT, vals))}
            log("%-24s %6d %s" % (key, int(m.sum()), "".join("%13.4f" % x for x in vals)))
        return rows

    out["strata"] = strat("MACRO F1 BY RUN LENGTH (tertiles, identical masks for all variants)", tert)
    hidx = np.where(hard)[0]
    out["strata"]["hard/robustness"] = {"n": int(hard.sum()),
                                        "macro_f1": dict(zip(SHORT, [mf1(
                                            [ys[i] for i in hidx],
                                            [p[i] for i in hidx])
                                            for v, p in
                                            ((v, names(arr=oof[v])) for v in VARIANTS)]))}
    log("%-24s %6d %s" % ("hard/robustness", int(hard.sum()), "".join(
        "%13.4f" % mf1([ys[i] for i in hidx], [names(arr=oof[v])[i] for i in hidx])
        for v in VARIANTS)))
    out["shift_slices"] = strat("DISTRIBUTION-SHIFT PROXIES (diagnostic only, no tuning)", slices)

    log("")
    log("LENGTH DECILES -> Macro F1")
    log("%-8s %6s %12s %12s %12s %10s" % ("decile", "n", "range", "A_baseline",
                                          "B_plus_norm", "delta"))
    dec_rows = []
    for i in range(10):
        lo, hi = dec[i], dec[i + 1]
        m = ((nmsg >= lo) & (nmsg <= hi)) if i == 9 else ((nmsg >= lo) & (nmsg < hi))
        if m.sum() < 30:
            continue
        idx = np.where(m)[0]
        a = mf1([ys[j] for j in idx], [names(arr=oof["A_baseline122"])[j] for j in idx])
        b = mf1([ys[j] for j in idx], [names(arr=oof["B_baseline_plus_norm"])[j] for j in idx])
        c = mf1([ys[j] for j in idx], [names(arr=oof["C_norm_only"])[j] for j in idx])
        dec_rows.append({"decile": i + 1, "n": int(m.sum()), "lo": float(lo), "hi": float(hi),
                         "A": a, "B": b, "C": c, "delta_BA": b - a, "delta_CA": c - a})
        log("%-8d %6d %12s %12.4f %12.4f %+10.4f   (C %.4f, dC %+.4f)"
            % (i + 1, int(m.sum()), "%d-%d" % (lo, hi), a, b, b - a, c, c - a))
    out["deciles"] = dec_rows

    # ---------------- feature importance for variant B ----------------
    log("")
    log("=" * 82)
    log("FEATURE IMPORTANCE, variant B (gain share, sum of 3 folds)")
    log("=" * 82)
    XB = mats["B_baseline_plus_norm"]
    newcols = list(Xn.columns)
    g = gain["B_baseline_plus_norm"] / gain["B_baseline_plus_norm"].sum()
    sB = pd.Series(g, index=XB.columns).sort_values(ascending=False)
    gA = gain["A_baseline122"] / gain["A_baseline122"].sum()
    sA = pd.Series(gA, index=XA.columns)

    new_share = float(sB.reindex(newcols).fillna(0.0).sum())
    vol_share_B = float(sB.reindex(vol).fillna(0.0).sum())
    vol_share_A = float(sA.reindex(vol).fillna(0.0).sum())
    used_new = int((sB.reindex(newcols).fillna(0.0) > 0).sum())
    top20 = list(sB.head(20).index)
    top50 = list(sB.head(50).index)
    in20 = sum(1 for c in top20 if c in newcols)
    in50 = sum(1 for c in top50 if c in newcols)
    log("  new features (n=%d) gain share           : %6.2f%%" % (len(newcols), 100 * new_share))
    log("  absolute volume feats (n=%d) share in A   : %6.2f%%" % (len(vol), 100 * vol_share_A))
    log("  absolute volume feats (n=%d) share in B   : %6.2f%%  (delta %+.2f pp)"
        % (len(vol), 100 * vol_share_B, 100 * (vol_share_B - vol_share_A)))
    log("  new features with non-zero gain           : %d / %d" % (used_new, len(newcols)))
    log("  new features in top-20 / top-50           : %d / %d" % (in20, in50))
    log("")
    log("  top 15 NEW features by gain:")
    snew = sB.reindex(newcols).fillna(0.0).sort_values(ascending=False)
    for c in snew.head(15).index:
        log("    %-42s %6.3f%%  (A-share %.3f%%)"
            % (c, 100 * sB[c], 100 * (sA.get(c, 0.0))))
    log("")
    log("  top 15 overall by gain (variant B):")
    for c in list(sB.head(15).index):
        log("    %-42s %6.3f%%   %s" % (c, 100 * sB[c], "NEW" if c in newcols else "baseline"))
    out["importance"] = {
        "new_gain_share": new_share,
        "volume_gain_share_A": vol_share_A, "volume_gain_share_B": vol_share_B,
        "n_new_used": used_new, "n_new_in_top20": in20, "n_new_in_top50": in50,
        "top_new": {c: float(sB[c]) for c in snew.head(25).index},
        "top_overall_B": {c: float(sB[c]) for c in list(sB.head(30).index)},
        "top_overall_A": {c: float(sA[c]) for c in list(sA.head(30).index)},
        "n_new_top1_20": [c for c in list(sA.head(20).index) if c in newcols],
    }

    out["runtime_s"] = time.time() - _T0
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    for v in VARIANTS:
        pd.DataFrame({"run_id": train["run_id"], "y_true": ys,
                      "y_pred": names(arr=oof[v]),
                      "success_true": s, "success_pred": oof_s[v],
                      "n_messages": nmsg}).to_csv(
            os.path.join(HERE, "oof_%s.csv" % v), index=False)
    log("")
    log("wrote results.json + 3 oof_*.csv; total %.1fs" % (time.time() - _T0))


if __name__ == "__main__":
    main()
