#!/usr/bin/env python3
"""Diagnostic analysis of the reproduced reference baseline.

Read-only with respect to ``baseline/`` — imports the official feature
extractor, localiser and metric definitions without modifying them.

    python experiments/diagnostics.py

Produces:
  * out-of-fold confusion matrix + worst confused class pairs
  * fault_turn analysis: per-class hit@2, signed error, localisation ceiling
  * LightGBM feature importance, aggregated by feature group
  * robustness probes: length / topology / lexicon dependence via group ablation
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from features import extract_features, parse_run          # noqa: E402
from localize import localize                             # noqa: E402
from metrics import binary_f1, composite, f1_per_class, fault_turn_hit_at_k, macro_f1  # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
SHORT = {c: c for c in LABELS}

# ---------------------------------------------------------------- feature groups
GROUPS = {
    "volume": ["n_messages", "n_agents", "n_artifacts", "n_state_writes",
               "msgs_per_agent", "arts_per_msg", "state_per_msg", "n_tools_declared"],
    "msg_types": ["type_handoff", "type_inform", "type_status", "type_state_update",
                  "type_final", "type_system", "share_handoff", "share_inform",
                  "share_status", "share_state_update", "share_final", "share_system",
                  "has_final_msg", "has_system_msg"],
    "senders": ["n_senders", "sender_entropy", "sender_max_share", "silent_agent_ratio"],
    "repetition": ["dup_text_max", "dup_text_ratio", "norm_dup_max", "norm_dup_ratio",
                   "norm_unique_ratio", "max_consecutive_repeat", "max_sender_repeat_streak",
                   "n_intents", "intent_entropy", "intent_max_count", "intent_max_share"],
    "lexicon": [],  # filled dynamically: kw_*
    "goal_alignment": ["goal_ov_mean", "goal_ov_head", "goal_ov_tail", "goal_ov_delta",
                       "goal_ov_zero_ratio", "goal_ov_max", "intent_head_tail_jaccard",
                       "new_intents_in_tail"],
    "handoff_accounting": ["n_assign", "n_result", "assign_result_gap", "unanswered_assign",
                           "unanswered_ratio", "undelivered_assign", "undelivered_ratio",
                           "reply_delay_mean", "reply_delay_max", "reassigned_intents",
                           "reassigned_ratio", "max_assign_per_intent",
                           "intents_with_multiple_owners"],
    "artifacts": ["n_unique_hashes", "dup_hash_pairs", "dup_hash_max", "dup_hash_ratio",
                  "max_arts_per_subtask", "subtasks_with_2plus_arts", "art_partial",
                  "art_copy", "has_final_artifact", "coverage", "art_types"],
    "shared_state": ["state_keys", "state_overrides", "state_override_ratio",
                     "state_max_writes_per_key", "state_max_distinct_per_key",
                     "state_flips", "state_alternating_flips", "state_keys_3plus",
                     "status_completed", "status_failed", "status_missing"],
    "topology": ["topo_star", "topo_pipeline", "topo_mesh", "topo_hierarchical",
                 "topo_blackboard", "topo_edges", "topo_edge_per_agent"],
    "tail": ["tail_status_share", "tail_inform_share", "tail_handoff_share", "tail_has_final"],
    "graph": ["g_nodes", "g_edges", "g_density", "g_self_loops", "g_reciprocity",
              "g_two_cycles", "g_triangles", "g_max_out_deg", "g_mean_out_deg",
              "g_max_in_deg", "g_largest_scc", "g_scc_share", "g_n_pairs", "g_max_pair",
              "g_top_pair_share", "g_pingpong", "g_pingpong_share"],
}
LENGTH_FEATURES = ["n_messages", "n_agents", "n_artifacts", "n_state_writes",
                   "msgs_per_agent", "arts_per_msg", "state_per_msg", "n_tools_declared",
                   "g_nodes", "g_edges", "g_max_pair", "g_n_pairs", "max_sender_repeat_streak",
                   "dup_text_max", "norm_dup_max", "intent_max_count",
                   "unanswered_assign", "reply_delay_max", "tail_status_share",
                   "tail_inform_share", "tail_handoff_share", "n_intents", "topo_edges",
                   "topo_edge_per_agent", "g_max_out_deg", "g_max_in_deg", "g_largest_scc",
                   "state_keys", "state_flips", "state_overrides", "dup_hash_pairs",
                   "art_partial", "art_copy", "state_max_writes_per_key", "unanswered_ratio",
                   "undelivered_assign", "dup_hash_max", "art_types", "state_per_msg",
                   "reply_delay_mean", "max_arts_per_subtask", "subtasks_with_2plus_arts",
                   "n_result", "n_assign", "assign_result_gap", "state_max_distinct_per_key",
                   "dup_text_ratio", "norm_dup_ratio", "new_intents_in_tail",
                   "reassigned_intents", "state_keys_3plus", "coverage", "g_self_loops",
                   "g_top_pair_share", "g_pingpong", "n_senders", "art_copy",
                   "state_alternating_flips", "g_reciprocity", "arts_per_msg",
                   "reassigned_ratio", "norm_unique_ratio", "dup_hash_ratio",
                   "sender_max_share", "n_unique_hashes", "art_partial"]
TOPO_FEATURES = GROUPS["topology"]


def group_of(col: str) -> str:
    for g, cols in GROUPS.items():
        if col in cols:
            return g
    return "lexicon" if col.startswith("kw_") else "other"


def log(msg):
    print("[%6.1fs] %s" % (time.time() - _T0, msg), flush=True)


_T0 = time.time()


def build(df):
    runs = [parse_run(r) for r in df.to_dict("records")]
    X = pd.DataFrame([extract_features(r) for r in runs]).fillna(0.0)
    return runs, X


def make_clf(n_classes):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=n_classes,
                              n_estimators=600, learning_rate=0.05, num_leaves=63,
                              min_child_samples=20, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                              random_state=42, verbose=-1)


def make_suc():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05,
                              num_leaves=63, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                              random_state=42, verbose=-1)


def main():
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))

    log("extracting features (train %d)" % len(train))
    runs, X = build(train)
    log("extracting features (test %d)" % len(test))
    _, Xte = build(test)

    y = train["label"].values
    s = train["success"].values
    ft_true = train["fault_turn"].values

    # feature shift probe: train vs test distribution
    shift_rows = []
    for c in X.columns:
        a, b = X[c].values, Xte[c].values
        shift_rows.append({
            "feature": c,
            "train_mean": float(a.mean()), "test_mean": float(b.mean()),
            "train_med": float(np.median(a)), "test_med": float(np.median(b)),
            "rel_shift": float(abs(b.mean() - a.mean()) / (abs(a.mean()) + 1e-9)),
        })
    shift = pd.DataFrame(shift_rows).sort_values("rel_shift", ascending=False)

    # ---- OOF with the exact solution.py model configuration ----
    log("3-fold OOF with solution.py config")
    oof_lab = np.empty(len(train), dtype=object)
    oof_suc = np.zeros(len(train), dtype=int)
    oof_proba = np.zeros((len(train), len(LABELS)))
    imp = np.zeros(X.shape[1])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    for tr, te in folds:
        m = make_clf(len(LABELS)).fit(X.iloc[tr], y[tr])
        oof_lab[te] = m.predict(X.iloc[te])
        oof_proba[te] = m.predict_proba(X.iloc[te])
        imp += m.feature_importances_
        oof_suc[te] = make_suc().fit(X.iloc[tr], s[tr]).predict(X.iloc[te])

    macro = macro_f1(list(y), list(oof_lab))
    sf1 = binary_f1(list(s), list(oof_suc))
    turns = [localize(r, l) for r, l in zip(runs, oof_lab)]
    hit2 = fault_turn_hit_at_k(list(y), list(oof_lab), list(ft_true), turns)
    hard = ((X["n_messages"] > X["n_messages"].median())
            & (X["topo_mesh"] + X["topo_blackboard"] > 0)).values
    rob = macro_f1(list(y[hard]), list(oof_lab[hard]))
    score = composite({"macro_f1": macro, "robustness_f1": rob,
                       "success_f1": sf1, "fault_turn_hit2": hit2})

    out = {"oof_solcfg": {"macro_f1": macro, "robustness_f1": rob, "success_f1": sf1,
                          "fault_turn_hit2": hit2, "score": score,
                          "per_class": f1_per_class(list(y), list(oof_lab), LABELS)}}

    # ---------------- 1. confusion matrix ----------------
    log("confusion matrix")
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(oof_lab, name="pred"))
    cm = cm.reindex(index=LABELS, columns=LABELS, fill_value=0)
    print("\n=== CONFUSION MATRIX (rows=true, cols=pred) ===")
    print(cm.to_string())
    pairs = []
    for i, a in enumerate(LABELS):
        for b in LABELS:
            if a != b and cm.iloc[i, LABELS.index(b)] > 0:
                tot = cm.iloc[i].sum()
                pairs.append({"true": a, "pred": b, "n": int(cm.iloc[i, LABELS.index(b)]),
                              "share_of_true": float(cm.iloc[i, LABELS.index(b)] / tot)})
    pairs = sorted(pairs, key=lambda r: -r["n"])
    print("\n=== TOP CONFUSED PAIRS ===")
    for p in pairs[:12]:
        print("  %-18s -> %-18s n=%4d (%.1f%% of true %s)"
              % (p["true"], p["pred"], p["n"], 100 * p["share_of_true"], p["true"]))
    out["confusion_pairs"] = pairs[:12]

    # clean recall / precision detail
    clean_mask = y == "clean"
    out["clean_recall"] = float((oof_lab[clean_mask] == "clean").mean())
    out["clean_precision"] = float((oof_lab == "clean")[y != "clean"].sum() == 0)  # placeholder
    out["clean_precision"] = float((y[oof_lab == "clean"] == "clean").mean())
    print("\nclean recall=%.4f  precision=%.4f" % (out["clean_recall"], out["clean_precision"]))

    # ---------------- 2. fault_turn ----------------
    log("fault_turn analysis")
    # oracle: correct class known -> how good is the rule alone?
    turns_oracle = [localize(r, l) for r, l in zip(runs, y)]
    hit2_oracle = fault_turn_hit_at_k(list(y), list(y), list(ft_true), turns_oracle)
    out["hit2_oracle_class"] = hit2_oracle

    per_class_ft = {}
    for c in LABELS[1:]:
        idx = np.where(y == c)[0]
        hit = sum(1 for i in idx
                  if oof_lab[i] == c and int(turns[i]) >= 0
                  and abs(int(turns[i]) - int(ft_true[i])) <= 2)
        cls_ok = int((oof_lab[idx] == c).sum())
        hit_or = sum(1 for i in idx
                     if int(turns_oracle[i]) >= 0
                     and abs(int(turns_oracle[i]) - int(ft_true[i])) <= 2)
        errs = [int(turns[i]) - int(ft_true[i]) for i in idx
                if oof_lab[i] == c and int(turns[i]) >= 0]
        per_class_ft[c] = {
            "n_true": len(idx), "class_correct": cls_ok,
            "class_acc": cls_ok / len(idx),
            "hit2_pred": hit / len(idx),
            "hit2_if_class_oracle": hit_or / len(idx),
            "median_signed_err": float(np.median(errs)) if errs else None,
            "returned_minus1": sum(1 for i in idx if int(turns[i]) < 0),
        }
    print("\n=== fault_turn PER CLASS ===")
    print("%-18s %5s %8s %8s %8s %8s" %
          ("class", "n", "cls_acc", "hit@2", "hit@2_orc", "med_err"))
    for c, v in per_class_ft.items():
        me = "  n/a" if v["median_signed_err"] is None else "%+6.0f" % v["median_signed_err"]
        print("%-18s %5d %8.3f %8.3f %11.3f %8s"
              % (c, v["n_true"], v["class_acc"], v["hit2_pred"],
                 v["hit2_if_class_oracle"], me))
    print("\noverall hit@2            = %.4f" % hit2)
    print("hit@2 with ORACLE class  = %.4f  <- ceiling of the rule-based localiser"
          % hit2_oracle)
    out["per_class_fault_turn"] = per_class_ft

    # ---------------- 3. feature importance ----------------
    log("feature importance")
    imp_s = pd.Series(imp / imp.sum(), index=X.columns).sort_values(ascending=False)
    by_group = defaultdict(float)
    for c, v in imp_s.items():
        by_group[group_of(c)] += v
    bg = pd.Series(by_group).sort_values(ascending=False)
    print("\n=== FEATURE IMPORTANCE BY GROUP (gain-weighted, LightGBM split count) ===")
    for g, v in bg.items():
        n = sum(1 for c in X.columns if group_of(c) == g)
        print("  %-20s %6.2f%%   (%d features)" % (g, 100 * v, n))
    print("\n=== TOP 25 FEATURES ===")
    for c, v in imp_s.head(25).items():
        print("  %-28s %6.2f%%  [%s]" % (c, 100 * v, group_of(c)))
    out["importance_by_group"] = {k: float(v) for k, v in bg.items()}
    out["importance_top25"] = {c: float(v) for c, v in imp_s.head(25).items()}

    # ---------------- 4. length / topology / lexicon dependence ----------------
    log("ablation probes")
    probes = {}
    variants = {
        "full": [],
        "no_length": sorted(set(LENGTH_FEATURES) & set(X.columns)),
        "no_topology": TOPO_FEATURES,
        "no_lexicon": [c for c in X.columns if c.startswith("kw_")],
        "no_goal_alignment": GROUPS["goal_alignment"],
        "no_graph": GROUPS["graph"],
        "no_state": GROUPS["shared_state"],
        "no_handoff_acct": GROUPS["handoff_accounting"],
        "no_tail": GROUPS["tail"],
    }
    for name, drop in variants.items():
        cols = [c for c in X.columns if c not in set(drop)]
        Xv = X[cols]
        oof = np.empty(len(train), dtype=object)
        for tr, te in folds:
            m = make_clf(len(LABELS)).fit(Xv.iloc[tr], y[tr])
            oof[te] = m.predict(Xv.iloc[te])
        mf = macro_f1(list(y), list(oof))
        probes[name] = {"macro_f1": mf, "delta": mf - macro, "n_features": len(cols)}
        log("  ablation %-18s macro_f1=%.4f (delta %+.4f) features=%d"
            % (name, mf, mf - macro, len(cols)))
    out["ablations"] = probes

    # ---------------- 5. length / topology stratified performance ----------------
    log("stratified performance")
    strat = {}
    qs = np.quantile(X["n_messages"].values, [0.33, 0.66])
    for name, mask in [
            ("len_q1_short", X["n_messages"].values <= qs[0]),
            ("len_q2_mid", (X["n_messages"].values > qs[0]) & (X["n_messages"].values <= qs[1])),
            ("len_q3_long", X["n_messages"].values > qs[1]),
            ("topo_star", X["topo_star"].values > 0),
            ("topo_pipeline", X["topo_pipeline"].values > 0),
            ("topo_mesh", X["topo_mesh"].values > 0),
            ("topo_hierarchical", X["topo_hierarchical"].values > 0),
            ("topo_blackboard", X["topo_blackboard"].values > 0),
            ("no_intent_telemetry", (X["n_intents"].values <= 1)),
    ]:
        if mask.sum() < 50:
            strat[name] = {"n": int(mask.sum()), "macro_f1": None}
            continue
        strat[name] = {"n": int(mask.sum()),
                       "macro_f1": macro_f1(list(y[mask]), list(oof_lab[mask]))}
    print("\n=== STRATIFIED MACRO F1 (OOF) ===")
    for k, v in strat.items():
        print("  %-24s n=%5d  macro_f1=%s"
              % (k, v["n"], "n/a" if v["macro_f1"] is None else "%.4f" % v["macro_f1"]))
    out["stratified"] = strat

    # ---------------- 6. distribution shift train vs test ----------------
    print("\n=== TOP 20 FEATURE SHIFTS train -> test (|mean(test)-mean(train)|) ===")
    for _, r in shift.head(20).iterrows():
        print("  %-28s train=%9.3f test=%9.3f  rel=%+.3f"
              % (r["feature"], r["train_mean"], r["test_mean"], r["rel_shift"]))
    out["top_shift"] = [{"feature": r["feature"], "train_mean": r["train_mean"],
                         "test_mean": r["test_mean"], "rel_shift": r["rel_shift"]}
                        for _, r in shift.head(20).iterrows()]

    print("\n=== TOPOLOGY / LENGTH DISTRIBUTION train vs test ===")
    for t in TOPO_FEATURES[:5]:
        print("  %-24s train=%.3f test=%.3f" % (t, X[t].mean(), Xte[t].mean()))
    print("  %-24s train=%.1f test=%.1f" % ("n_messages mean", X["n_messages"].mean(),
                                            Xte["n_messages"].mean()))
    print("  %-24s train=%.1f test=%.1f" % ("n_agents mean", X["n_agents"].mean(),
                                            Xte["n_agents"].mean()))
    out["dist"] = {
        "topology_train": {t: float(X[t].mean()) for t in TOPO_FEATURES[:5]},
        "topology_test": {t: float(Xte[t].mean()) for t in TOPO_FEATURES[:5]},
        "n_messages_train_mean": float(X["n_messages"].mean()),
        "n_messages_test_mean": float(Xte["n_messages"].mean()),
        "n_agents_train_mean": float(X["n_agents"].mean()),
        "n_agents_test_mean": float(Xte["n_agents"].mean()),
    }

    # predicted-label distribution train OOF vs test predictions
    pred = pd.read_csv(os.path.join(ROOT, "predictions.csv"))
    cmp = pd.DataFrame({
        "train_oof_share": pd.Series(oof_lab).value_counts(normalize=True),
        "test_pred_share": pred["label"].value_counts(normalize=True),
        "train_true_share": pd.Series(y).value_counts(normalize=True),
    }).reindex(LABELS)
    print("\n=== LABEL SHARE: true train / OOF pred train / pred test ===")
    print(cmp.to_string(float_format=lambda v: "%.3f" % v))
    out["label_shares"] = json.loads(cmp.to_json())

    with open(os.path.join(HERE, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    log("wrote experiments/diagnostics.json")


if __name__ == "__main__":
    main()
