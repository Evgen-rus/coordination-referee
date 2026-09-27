"""Experiment 05 driver: handoff lifecycle features on top of the Exp03 B foundation.

Variants, on the SAME 3 folds and with UNCHANGED LightGBM hyperparameters:

  A  exp03_B              122 baseline + 63 normalised                (185 cols)
  B  A + lifecycle        + 40 lifecycle aggregates                   (225 cols)
  C  B + deadlock         + 14 deadlock-structure features             (239 cols)

Success F1 is the saved Exp03 B OOF (no lifecycle features target success, as
instructed).  fault_turn uses the official rule-based localiser on each variant's
own predicted label, so composite is honest.

Confusion pairs, per-slice Macro F1, no_intent / long-run behaviour and both
group ablations (drop the whole lifecycle block, drop the whole deadlock block)
are computed from the same OOF predictions.

Run:  .\\.venv\\Scripts\\python.exe experiments\\exp05_handoff_lifecycle\\run_experiment.py
"""

from __future__ import annotations

import gc
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
sys.path.insert(0, os.path.join(HERE, "..", "exp03_length_normalization"))

from features import extract_features, parse_run      # noqa: E402
from localize import localize                         # noqa: E402
from metrics import binary_f1, composite, f1_per_class  # noqa: E402
from metrics import fault_turn_hit_at_k, macro_f1     # noqa: E402
import lifecycle as lc                                # noqa: E402
import new_features as nf                             # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
VARIANTS = ["A_exp03_B", "B_lifecycle", "C_lifecycle_deadlock"]
SHORT = ["A_exp03B", "B_lifecycle", "C_life+dead"]

_T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - _T0, m), flush=True)


def mf1(t, p):
    assert len(t) == len(p)
    return macro_f1(list(t), list(p))


def lgb_label():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def prf(y_true: np.ndarray, y_pred: np.ndarray, cls: str) -> dict:
    tp = int(((y_true == cls) & (y_pred == cls)).sum())
    fp = int(((y_true != cls) & (y_pred == cls)).sum())
    fn = int(((y_true == cls) & (y_pred != cls)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


def main():
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    yi = train["label"].map(L2I).values
    ys = np.array([LABELS[i] for i in yi], dtype=object)
    s = train["success"].values
    ft = train["fault_turn"].values
    n = len(yi)

    log("parsing runs + official 122 features")
    runs = [parse_run(r) for r in train.to_dict("records")]
    base_rows = [extract_features(r) for r in runs]
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    t0 = time.time()
    log("building exp03 normalised features")
    Xn = nf.build_matrix(runs, base_rows)
    log("building lifecycle features")
    Xl = lc.build_matrix(runs)
    life_s = time.time() - t0
    log("  normalised %d + lifecycle %d cols in %.1fs" % (Xn.shape[1], Xl.shape[1], life_s))
    del base_rows
    gc.collect()

    XA = pd.concat([Xb, Xn], axis=1)
    XB = pd.concat([XA, Xl[lc.LIFECYCLE_NAMES]], axis=1)
    XC = pd.concat([XB, Xl[lc.DEADLOCK_NAMES]], axis=1)
    mats = {"A_exp03_B": XA, "B_lifecycle": XB, "C_lifecycle_deadlock": XC}
    log("columns: " + "  ".join("%s=%d" % (k, m.shape[1]) for k, m in mats.items()))

    # group-ablation variants (drop a whole block, keep the rest)
    mats["ablate_lifecycle"] = pd.concat([XA, Xl[lc.DEADLOCK_NAMES]], axis=1)
    mats["ablate_deadlock"] = pd.concat([XA, Xl[lc.LIFECYCLE_NAMES]], axis=1)
    for k, m in mats.items():
        log("  %-22s -> %d cols" % (k, m.shape[1]))

    exp03 = pd.read_csv(os.path.join(HERE, "..", "exp03_length_normalization",
                                     "oof_B_baseline_plus_norm.csv"))
    assert (exp03["run_id"].values == train["run_id"].values).all()
    succ_b = exp03["success_pred"].values
    SUCCESS_F1 = float(binary_f1(list(s), list(succ_b)))
    log("Exp03 B Success F1 (held fixed) = %.4f" % SUCCESS_F1)

    splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(XA, yi))
    oof = {k: np.zeros(n, dtype=int) for k in mats}
    fold_m = {k: [] for k in mats}
    gain = {k: np.zeros(m.shape[1]) for k, m in mats.items()}

    for k, (tr, va) in enumerate(splits):
        log("FOLD %d  train=%d val=%d" % (k, len(tr), len(va)))
        for key, X in mats.items():
            m = lgb_label().fit(X.iloc[tr], yi[tr])
            p = m.predict(X.iloc[va])
            oof[key][va] = p
            fold_m[key].append(mf1([ys[i] for i in va], [LABELS[j] for j in p]))
            gain[key] += m.booster_.feature_importance(importance_type="gain")
            log("  %-22s fold macro_f1=%.4f" % (key, fold_m[key][-1]))
            del m
        gc.collect()

    # ---------------- masks ----------------
    nmsg = Xb["n_messages"].values
    hard = (nmsg > np.median(nmsg)) & ((Xb["topo_mesh"].values > 0) |
                                       (Xb["topo_blackboard"].values > 0))
    no_intent = Xb["n_intents"].values <= 1
    q66 = np.quantile(nmsg, 2 / 3)
    q80 = np.quantile(nmsg, 0.8)
    MASKS = {
        "no_intent": no_intent,
        "has_intent": ~no_intent,
        "long(>p66)": nmsg > q66,
        "longest_20%": nmsg >= q80,
        "hard/robustness": hard,
        "big_team_top20%": Xb["n_agents"].values >= np.quantile(Xb["n_agents"].values, .8),
        "topo_star": Xb["topo_star"].values > 0,
        "topo_pipeline": Xb["topo_pipeline"].values > 0,
        "topo_mesh": Xb["topo_mesh"].values > 0,
        "topo_hierarchical": Xb["topo_hierarchical"].values > 0,
        "topo_blackboard": Xb["topo_blackboard"].values > 0,
    }

    def sl(pred, m):
        idx = np.where(m)[0]
        if len(idx) < 30:
            return None
        return mf1([ys[i] for i in idx], [LABELS[pred[i]] for i in idx])

    # ---------------- score ----------------
    res = {"n_runs": int(n), "n_lifecycle_feats": len(lc.LIFECYCLE_NAMES),
           "n_deadlock_feats": len(lc.DEADLOCK_NAMES),
           "success_f1_fixed_from_exp03B": SUCCESS_F1,
           "lifecycle_build_s": life_s, "systems": {}}
    pnames = {k: [LABELS[i] for i in oof[k]] for k in oof}
    for key in VARIANTS:
        p = pnames[key]
        mf = mf1(ys, p)
        rf = sl(oof[key], hard)
        turns = [localize(runs[i], p[i]) for i in range(n)]
        h2 = fault_turn_hit_at_k(list(ys), p, list(ft), turns)
        comp = composite({"macro_f1": mf, "robustness_f1": rf,
                          "success_f1": SUCCESS_F1, "fault_turn_hit2": h2})
        res["systems"][key] = {
            "macro_f1": mf, "robustness_f1": rf, "success_f1": SUCCESS_F1,
            "fault_turn_hit2": h2, "composite": comp,
            "per_class": f1_per_class(list(ys), p, LABELS),
            "fold_macro_f1": [float(x) for x in fold_m[key]],
            "slices": {k: sl(oof[key], m) for k, m in MASKS.items()},
            "dropped_handoff_prf": prf(ys, np.array(p, dtype=object), "dropped_handoff"),
            "deadlock_prf": prf(ys, np.array(p, dtype=object), "deadlock"),
        }
    A = res["systems"]["A_exp03_B"]
    for key in VARIANTS:
        d = res["systems"][key]
        d["d_macro"] = d["macro_f1"] - A["macro_f1"]
        d["d_rob"] = d["robustness_f1"] - A["robustness_f1"]
        d["d_comp"] = d["composite"] - A["composite"]
        d["d_dh_f1"] = d["per_class"]["dropped_handoff"] - A["per_class"]["dropped_handoff"]
        d["d_deadlock_f1"] = d["per_class"]["deadlock"] - A["per_class"]["deadlock"]

    # ---------------- save before fragile analysis ----------------
    for key in oof:
        pd.DataFrame({"run_id": train["run_id"], "y_true": ys, "y_pred": pnames[key],
                      "n_messages": nmsg, "n_intents": Xb["n_intents"].values,
                      "no_intent": no_intent}).to_csv(
            os.path.join(HERE, "oof_%s.csv" % key), index=False)
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    log("saved results.json + %d oof files" % len(oof))

    # ---------------- reporting ----------------
    log("")
    log("=" * 96)
    log("HEADLINE  (3-fold OOF; Success F1 fixed from Exp03 B; fault_turn = official localiser)")
    log("=" * 96)
    log("%-18s %8s %9s %9s %9s %9s %8s %8s"
        % ("variant", "MacroF1", "RobustF1", "composite", "dMacro", "dRobust",
           "dh_F1", "dl_F1"))
    for key in VARIANTS:
        d = res["systems"][key]
        log("%-18s %8.4f %9.4f %9.4f %+9.4f %+9.4f %8.4f %8.4f"
            % (key, d["macro_f1"], d["robustness_f1"], d["composite"],
               d["d_macro"], d["d_rob"], d["per_class"]["dropped_handoff"],
               d["per_class"]["deadlock"]))

    log("")
    log("PER-CLASS F1")
    log("%-18s %s" % ("class", "".join("%14s" % s for s in SHORT)))
    for c in LABELS:
        log("%-18s %s" % (c, "".join("%14.4f" % res["systems"][k]["per_class"][c]
                                     for k in VARIANTS)))
    log("")
    for c in ("dropped_handoff", "deadlock"):
        log("%s precision/recall:" % c)
        log("%-18s %s" % ("", "".join("%14s" % s for s in SHORT)))
        for m_ in ("precision", "recall", "f1"):
            log("  %-16s %s" % (m_, "".join(
                "%14.4f" % res["systems"][k]["%s_prf" % c][m_] for k in VARIANTS)))

    log("")
    log("FOLD-BY-FOLD Macro F1 (d vs A)")
    for key in VARIANTS:
        f = res["systems"][key]["fold_macro_f1"]
        d = [f[i] - A["fold_macro_f1"][i] for i in range(3)]
        log("  %-18s %s | d = %s | wins %d/3"
            % (key, ", ".join("%.4f" % x for x in f),
               ", ".join("%+.4f" % x for x in d),
               sum(1 for x in d if x > 0)))

    log("")
    log("SLICE TABLE (Macro F1)")
    sk = VARIANTS
    log("%-19s %6s %s" % ("slice", "n", "".join("%14s" % k for k in sk)))
    for name, m in MASKS.items():
        if m.sum() < 30:
            continue
        log("%-19s %6d %s" % (name, int(m.sum()), "".join(
            "%14.4f" % (res["systems"][k]["slices"][name] or 0.0) for k in sk)))

    # ---------------- confusion pairs ----------------
    log("")
    log("=" * 96)
    log("CONFUSION PAIRS (count of true class -> predicted class)")
    log("=" * 96)
    PAIRS = [("dropped_handoff", "clean"), ("dropped_handoff", "deadlock"),
             ("deadlock", "clean"), ("deadlock", "dropped_handoff"),
             ("duplicated_work", "clean"), ("duplicated_work", "dropped_handoff")]
    conf = {}
    log("%-22s %6s %s" % ("true -> pred", "n", "".join("%16s" % k for k in VARIANTS)))
    for a, b in PAIRS:
        row = []
        for k in VARIANTS:
            v = int(((ys == a) & (np.array(pnames[k], dtype=object) == b)).sum())
            row.append(v)
        conf["%s->%s" % (a, b)] = dict(zip([k for k in VARIANTS], row))
        log("%-22s %6d %s" % ("%s -> %s" % (a, b), int((ys == a).sum()),
                              "".join("%16d" % v for v in row)))
    res["confusion_pairs"] = conf

    # false-clean overall
    log("")
    for k in VARIANTS:
        p = np.array(pnames[k], dtype=object)
        fc = int(((p == "clean") & (ys != "clean")).sum())
        wrong_faulty = int(((p != "clean") & (ys != "clean") &
                            (p != ys)).sum())
        log("  %-22s false-clean=%4d  wrong-faulty-class=%4d" % (k, fc, wrong_faulty))
        res["systems"][k]["false_clean"] = fc
        res["systems"][k]["wrong_faulty_class"] = wrong_faulty

    # ---------------- dropped_handoff sub-slices ----------------
    log("")
    log("DROPPED_HANDOFF sub-slices (only structurally defined categories)")
    # lifecycles for the sub-slice split; reuse the PARSED runs (raw CSV rows
    # still hold messages/artifacts as JSON strings).
    lcs_all = [lc.build_lifecycles(r.get("messages") or [],
                                   r.get("artifacts") or [])
               for r in runs]
    dh_mask = ys == "dropped_handoff"
    groups = {
        "all dropped_handoff": dh_mask,
        "acknowledged-but-undelivered": np.array(
            [any(x["acked"] and not x["delivered"] for x in L) for L in lcs_all]) & dh_mask,
        "unacknowledged": np.array(
            [any(not x["acked"] for x in L) for L in lcs_all]) & dh_mask,
        "reassigned": np.array(
            [any(x["reassigned"] for x in L) for L in lcs_all]) & dh_mask,
        "no-intent run": no_intent & dh_mask,
        "long runs": (nmsg > q66) & dh_mask,
        "all lifecycles delivered": np.array(
            [all(x["delivered"] for x in L) for L in lcs_all]) & dh_mask,
    }
    dh_tbl = {}
    log("%-30s %6s %s" % ("sub-slice", "n", "".join("%16s" % k for k in VARIANTS)))
    for name, m in groups.items():
        n_sub = int(m.sum())
        if n_sub < 30:
            log("%-30s %6d  (too small)" % (name, n_sub))
            continue
        row = {}
        for k in VARIANTS:
            idx = np.where(m)[0]
            p = np.array(pnames[k], dtype=object)
            tp = int(((ys[idx] == "dropped_handoff") &
                      (p[idx] == "dropped_handoff")).sum())
            fp = int(((ys[idx] != "dropped_handoff") &
                      (p[idx] == "dropped_handoff")).sum())
            fn = n_sub - tp
            pr = tp / (tp + fp) if (tp + fp) else 0.0
            rc = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            row[k] = {"precision": pr, "recall": rc, "f1": f1}
        dh_tbl[name] = {"n": n_sub, **{k: row[k] for k in VARIANTS}}
        log("%-30s %6d %s" % (name, n_sub, "".join(
            "%16.4f" % row[k]["f1"] for k in VARIANTS)))
    log("  (columns above are F1 for dropped_handoff inside that sub-slice)")
    res["dropped_handoff_subslices"] = dh_tbl

    # ---------------- feature importance + group ablation ----------------
    log("")
    log("=" * 96)
    log("FEATURE IMPORTANCE (variant C) and GROUP ABLATION")
    log("=" * 96)
    gC = gain["C_lifecycle_deadlock"] / gain["C_lifecycle_deadlock"].sum()
    sC = pd.Series(gC, index=XC.columns).sort_values(ascending=False)
    newcols = lc.LIFECYCLE_NAMES + lc.DEADLOCK_NAMES
    share = float(sC.reindex(newcols).fillna(0.0).sum())
    used = int((sC.reindex(newcols).fillna(0.0) > 0).sum())
    top20 = sum(1 for c in list(sC.head(20).index) if c in newcols)
    top50 = sum(1 for c in list(sC.head(50).index) if c in newcols)
    log("  new features (n=%d): gain share %.2f%% | non-zero gain %d | in top-20 %d | in top-50 %d"
        % (len(newcols), 100 * share, used, top20, top50))
    log("")
    log("  top 20 NEW features by gain:")
    for c in sC.reindex(newcols).fillna(0.0).sort_values(ascending=False).head(20).index:
        log("    %-34s %6.3f%%" % (c, 100 * sC[c]))
    log("")
    log("  group ablation (drop the whole block, keep the rest):")
    abl = {}
    for key, dropped in (("A_exp03_B", "both"), ("B_lifecycle", "deadlock"),
                         ("C_lifecycle_deadlock", "none"),
                         ("ablate_lifecycle", "lifecycle (40)"),
                         ("ablate_deadlock", "deadlock (14)")):
        f = fold_m[key]
        ov = mf1(ys, pnames[key])
        abl[key] = {"oof_macro": ov, "fold": [float(x) for x in f]}
        log("    %-22s dropped=%-18s oof=%.4f  folds=%s"
            % (key, dropped, ov, ", ".join("%.4f" % x for x in f)))
    res["ablation"] = abl
    res["importance"] = {"new_gain_share": share, "n_new_used": used,
                         "n_new_in_top20": top20, "n_new_in_top50": top50,
                         "top_new": {c: float(sC[c]) for c in
                                     sC.reindex(newcols).fillna(0.0)
                                     .sort_values(ascending=False).head(25).index}}

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    log("")
    log("updated results.json; total %.0fs" % (time.time() - _T0))


if __name__ == "__main__":
    main()
