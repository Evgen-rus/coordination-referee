#!/usr/bin/env python3
"""Distribution-shift probes: how much does the baseline depend on run length,
topology and the hard-coded keyword lexicon?

Each probe trains on one slice of train and evaluates on another, which mimics
the test-time shift (longer runs, more agents, rarer topologies, different
wording) without touching baseline/ code.

NOTE: evaluation/metrics.f1_per_class zips its inputs positionally and only
counts t == p, so misaligned arrays silently score 1.0. Every comparison here
goes through _mf1(), which asserts equal lengths first.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)

from features import extract_features, parse_run   # noqa: E402
from metrics import macro_f1                       # noqa: E402
from diagnostics import group_of                   # noqa: E402

_T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - _T0, m), flush=True)


def _mf1(y_true, y_pred):
    """Order-safe macro F1."""
    assert len(y_true) == len(y_pred), (len(y_true), len(y_pred))
    return macro_f1(list(y_true), list(y_pred))


def mk():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def main():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    runs = [parse_run(r) for r in train.to_dict("records")]
    X = pd.DataFrame([extract_features(r) for r in runs]).fillna(0.0)
    y = train["label"].values
    nmsg = X["n_messages"].values
    nagt = X["n_agents"].values
    med, med_agt = np.median(nmsg), np.median(nagt)
    short, long_ = nmsg <= med, nmsg > med
    few, many = nagt <= med_agt, nagt > med_agt
    rare_topo = (X["topo_mesh"] + X["topo_blackboard"]).values > 0
    common_topo = ~rare_topo
    allidx = np.arange(len(y))
    res = {}

    # sanity: _mf1 must be honest
    assert abs(_mf1(y, y) - 1.0) < 1e-12
    assert _mf1(y, y[::-1]) < 0.5
    log("sanity check on _mf1 passed (perfect=1.0, reversed<0.5)")

    print("slice sizes: short=%d long=%d few_agents=%d many_agents=%d rare_topo=%d common_topo=%d"
          % (short.sum(), long_.sum(), few.sum(), many.sum(), rare_topo.sum(), common_topo.sum()))

    # ---------------- A. cross-slice generalisation ----------------
    log("probe A: train on one slice, test on another")
    combos = [(allidx, "all", long_, "long"),
              (allidx, "all", short, "short"),
              (short, "short", long_, "long"),
              (long_, "long", short, "short"),
              (short, "short", allidx, "all"),
              (long_, "long", allidx, "all"),
              (allidx, "all", many, "many_agents"),
              (few, "few_agents", many, "many_agents"),
              (allidx, "all", rare_topo, "rare_topo"),
              (common_topo, "common_topo", rare_topo, "rare_topo")]
    models = {}
    for tr_m, tr_n, te_m, te_n in combos:
        tr = np.where(tr_m)[0] if tr_m.dtype == bool else tr_m
        te = np.where(te_m)[0] if te_m.dtype == bool else te_m
        key = (tr_n, te_n)
        if key not in models:
            models[key] = mk().fit(X.iloc[tr], y[tr])
        p = models[key].predict(X.iloc[te])
        f = _mf1(y[te], p)
        k = "train_%s__test_%s" % (tr_n, te_n)
        res[k] = {"n_train": len(tr), "n_test": len(te), "macro_f1": f}
        log("  %-38s n_tr=%5d n_te=%5d macro_f1=%.4f" % (k, len(tr), len(te), f))

    res["length_shift_penalty"] = (res["train_short__test_long"]["macro_f1"]
                                   - res["train_all__test_long"]["macro_f1"])
    res["agent_shift_penalty"] = (res["train_few_agents__test_many_agents"]["macro_f1"]
                                  - res["train_all__test_many_agents"]["macro_f1"])
    res["topology_shift_penalty"] = (res["train_common_topo__test_rare_topo"]["macro_f1"]
                                     - res["train_all__test_rare_topo"]["macro_f1"])
    log("  LENGTH  shift penalty (train_short vs train_all, test=long): %+.4f"
        % res["length_shift_penalty"])
    log("  AGENTS  shift penalty (train_few   vs train_all, test=many): %+.4f"
        % res["agent_shift_penalty"])
    log("  TOPOLOGY shift penalty (train_common vs train_all, test=rare): %+.4f"
        % res["topology_shift_penalty"])

    # ---------------- B/C/D. feature- and telemetry-level degradation at test time ----
    log("probe B-D: train on clean data, evaluate on degraded copies")

    m_all = models[("all", "long")] if ("all", "long") in models else None
    m_full = mk().fit(X, y)
    base = _mf1(y, m_full.predict(X))
    res["baseline_in_sample_macro_f1"] = base
    log("  reference: model trained on all, scored in-sample macro_f1=%.4f" % base)

    def evaluate_degraded(name, Xd):
        Xd = pd.DataFrame(Xd).fillna(0.0).reindex(columns=X.columns, fill_value=0.0)
        f = _mf1(y, m_full.predict(Xd))
        res[name] = f
        log("  %-34s macro_f1=%.4f  (delta %+.4f)" % (name, f, f - base))
        return f

    # B. lexicon: keyword features blanked
    kw = [c for c in X.columns if c.startswith("kw_")]
    Xkw = X.copy(); Xkw[kw] = 0.0
    evaluate_degraded("kw_features_zeroed_at_test", Xkw)

    # B. lexicon: goal-alignment features blanked
    ga = ["goal_ov_mean", "goal_ov_head", "goal_ov_tail", "goal_ov_delta",
          "goal_ov_zero_ratio", "goal_ov_max", "intent_head_tail_jaccard",
          "new_intents_in_tail"]
    Xga = X.copy(); Xga[ga] = 0.0
    evaluate_degraded("goal_alignment_zeroed_at_test", Xga)

    # C. real text corruption: every message text replaced by an opaque token
    runs_masked = []
    for r in runs:
        r2 = dict(r)
        r2["messages"] = [dict(mm, text="item %d" % mm.get("t", 0)) for mm in r["messages"]]
        r2["goal"] = "objective"
        runs_masked.append(r2)
    evaluate_degraded("all_texts_masked_at_test",
                      [extract_features(r) for r in runs_masked])

    # C2. only keyword groups removed from text (lexicon probe on real strings)
    def strip_words(r, words):
        r2 = dict(r)
        ms = []
        for mm in r["messages"]:
            t = mm.get("text", "").lower()
            for w in words:
                if " " in w:
                    t = t.replace(w, " ")
                else:
                    t = t.replace(w, " ")
            ms.append(dict(mm, text=t))
        r2["messages"] = ms
        return r2

    drift_words = ["side note", "expanding scope", "pivot", "new direction",
                   "switching focus", "deprioritis", "deprioritiz", "also look into",
                   "matters more"]
    evaluate_degraded("drift_keywords_removed_from_text",
                      [extract_features(strip_words(r, drift_words)) for r in runs])

    # D. telemetry degradation
    runs_ni = [dict(r, messages=[dict(mm, intent="") for mm in r["messages"]]) for r in runs]
    evaluate_degraded("intents_blanked_at_test",
                      [extract_features(r) for r in runs_ni])

    runs_nr = [dict(r, messages=[dict(mm, refs=[]) for mm in r["messages"]]) for r in runs]
    evaluate_degraded("refs_stripped_at_test",
                      [extract_features(r) for r in runs_nr])

    # ---------------- E. gain-based importance ----------------
    log("probe E: gain-based feature importance")
    gain = m_full.booster_.feature_importance(importance_type="gain")
    g = pd.Series(gain / gain.sum(), index=X.columns).sort_values(ascending=False)
    by_group = {}
    for c, v in g.items():
        by_group[group_of(c)] = by_group.get(group_of(c), 0.0) + float(v)
    print("\n=== GAIN-BASED IMPORTANCE BY GROUP (in-sample model) ===")
    for k, v in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print("  %-20s %6.2f%%" % (k, 100 * v))
    print("\n=== TOP 20 FEATURES BY GAIN ===")
    for c, v in g.head(20).items():
        print("  %-28s %6.2f%%  [%s]" % (c, 100 * v, group_of(c)))
    res["gain_by_group"] = by_group
    res["gain_top20"] = {c: float(v) for c, v in g.head(20).items()}

    lf = {"n_messages", "n_agents", "n_artifacts", "n_state_writes", "msgs_per_agent",
          "arts_per_msg", "state_per_msg", "n_tools_declared", "g_nodes", "g_edges",
          "g_max_pair", "g_n_pairs", "max_sender_repeat_streak", "dup_text_max",
          "norm_dup_max", "intent_max_count", "reply_delay_max", "topo_edges",
          "topo_edge_per_agent", "g_max_out_deg", "g_max_in_deg", "g_largest_scc",
          "g_density", "g_pingpong", "g_pingpong_share", "g_top_pair_share",
          "unanswered_assign", "sender_max_share", "g_mean_out_deg", "g_scc_share",
          "dup_text_ratio", "norm_dup_ratio", "norm_unique_ratio", "n_result", "n_assign",
          "assign_result_gap", "unanswered_ratio", "undelivered_assign", "n_senders",
          "reply_delay_mean", "state_keys", "state_flips", "state_overrides",
          "dup_hash_pairs", "art_partial", "art_copy", "max_arts_per_subtask",
          "subtasks_with_2plus_arts", "state_max_writes_per_key", "coverage", "art_types",
          "state_max_distinct_per_key", "state_keys_3plus", "state_alternating_flips",
          "state_override_ratio", "g_self_loops", "g_reciprocity", "intent_max_share",
          "n_unique_hashes", "dup_hash_max", "dup_hash_ratio", "arts_per_msg",
          "reassigned_intents", "reassigned_ratio", "intents_with_multiple_owners",
          "max_assign_per_intent", "sender_entropy", "silent_agent_ratio",
          "has_final_msg", "has_system_msg", "new_intents_in_tail", "tail_status_share",
          "tail_inform_share", "tail_handoff_share", "tail_has_final"} & set(X.columns)
    res["gain_share_length_related"] = float(g[list(lf)].sum())
    lex = set([c for c in X.columns if c.startswith("kw_")] + ga)
    res["gain_share_lexicon"] = float(g[list(lex)].sum())
    topo = [c for c in X.columns if c.startswith("topo_")]
    res["gain_share_topology"] = float(g[topo].sum())
    log("\n  GAIN SHARE length/volume/size features : %6.2f%%" % (100 * res["gain_share_length_related"]))
    log("  GAIN SHARE keyword + goal-lexicon      : %6.2f%%" % (100 * res["gain_share_lexicon"]))
    log("  GAIN SHARE explicit topology one-hots  : %6.2f%%" % (100 * res["gain_share_topology"]))

    with open(os.path.join(HERE, "shift_probes.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    log("wrote experiments/shift_probes.json")


if __name__ == "__main__":
    main()
