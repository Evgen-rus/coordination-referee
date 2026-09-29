"""Exp07 runner with FAST / CV / FINAL modes, safe caching and honest stacking.

The model mathematics is identical in every mode.  Only the depth of
verification changes.  See ``modes.py``.
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

import aggregate as ag          # noqa: E402
import cache as ca              # noqa: E402
import grouping as grp          # noqa: E402
import modes as MD              # noqa: E402
import parallel as par          # noqa: E402
import window_dataset as wd      # noqa: E402
from common import (LABELS, build_all, build_windows, composite,  # noqa: E402
                    f1_per_class, fault_turn_hit_at_k, lgb_label, log,
                    localize, prf, sl_macro)

N_OUTER = 3
N_INNER = 3
SEED = 0
A_KEY, B_KEY = "A_foundation", "B_plus_window"

WINDOW_PARAMS = dict(objective="multiclass", num_class=wd.N_WINDOW_CLASSES,
                     n_estimators=300, learning_rate=0.08, num_leaves=63,
                     min_child_samples=50, subsample=0.9, subsample_freq=1,
                     colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
                     verbose=-1)


def _assert_params_match_original():
    """The runner builds its estimators from these dicts instead of calling
    ``common.lgb_label`` / ``common.fit_window``.  That is only safe while the
    dicts are identical, so the equivalence is asserted, not assumed."""
    import lightgbm as lgb
    from common import lgb_label, lgb_window
    ref = lgb_window().get_params()
    probe = lgb.LGBMClassifier(**WINDOW_PARAMS, n_jobs=-1).get_params()
    diff = {k: (ref[k], probe[k]) for k in ref
            if k in probe and ref[k] != probe[k]}
    if diff:
        raise AssertionError("window params drifted from common.lgb_window(): %s"
                             % diff)
    # the label model is still taken from common, so nothing to assert there


def verify_stacking(n_runs, assign, n_inner, tag):
    """Every outer-train run must be held out exactly once, every inner fold
    must be non-empty, and the model that predicted a held-out run must have
    been trained on a strictly disjoint set."""
    if len(assign) != n_runs:
        raise AssertionError("assign length %d != n_runs %d" % (len(assign), n_runs))
    if assign.min() < 0 or assign.max() >= n_inner:
        raise AssertionError("assign has values outside [0, %d)" % n_inner)
    held_counts = np.zeros(n_runs, dtype=int)
    for f in range(n_inner):
        held = np.where(assign == f)[0]
        trained = np.where(assign != f)[0]
        if len(held) == 0:
            raise AssertionError("inner fold %d of %s is empty" % (f, tag))
        if len(trained) == 0:
            raise AssertionError("inner fold %d of %s has no training runs" % (f, tag))
        if len(np.intersect1d(held, trained)):
            raise AssertionError("inner fold %d of %s: held-out runs appear in "
                                 "its own training set" % (f, tag))
        held_counts[held] += 1
    if not (held_counts == 1).all():
        raise AssertionError("inner folds of %s do not partition the outer-train "
                             "runs exactly once" % tag)
    log("  [%s] stacking verified: %d outer-train runs, each cross-fitted from a "
        "model excluding it (%d inner folds, no overlap)" % (tag, n_runs, n_inner))


def confusion(ys, pred):
    out = {}
    for a, b in (("dropped_handoff", "clean"), ("dropped_handoff", "deadlock"),
                 ("deadlock", "dropped_handoff"), ("duplicated_work", "clean"),
                 ("runaway_loop", "clean"), ("goal_drift", "clean")):
        m = ys == a
        out["%s->%s" % (a, b)] = int((pred[m] == b).sum()) if m.any() else 0
    out["_n"] = {a: int((ys == a).sum()) for a in
                 ("dropped_handoff", "deadlock", "duplicated_work",
                  "runaway_loop", "goal_drift", "conflict")}
    return out


def _prepare(use_cache=True):
    """build_all() and build_windows(), each memoised on a content hash."""
    ca.set_enabled(use_cache)
    c = ca.get_or_build("build_all", build_all)
    W = ca.get_or_build("windows",
                        lambda: wd.build_window_dataset(c["runs"], c["ys"],
                                                        c["fturn"]))
    return c, W


def run(mode="cv", use_cache=True, threads=par.DEFAULT_THREADS, workers=1,
        early_stop=True, quiet=False):
    """Execute Exp07 in ``mode``.  Returns the results dict."""
    mode = MD.resolve(mode)
    _assert_params_match_original()
    fold_ids = MD.folds_for(mode)
    outdir = MD.outdir(mode)
    os.makedirs(outdir, exist_ok=True)

    c, W = _prepare(use_cache=use_cache)
    runs, yi, ys, n = c["runs"], c["yi"], c["ys"], c["n"]
    found, SL = c["foundation"], c["SL"]
    Xw, yw, rw = W["X"], W["y"], W["run"]
    log("mode=%s  folds=%s  windows=%s  threads=%d workers=%d cache=%s"
        % (mode, list(fold_ids), Xw.shape, threads, workers, ca.enabled()))

    splits = list(StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
                  .split(np.zeros(n), yi))
    oof = {A_KEY: np.zeros(n, dtype=int), B_KEY: np.zeros(n, dtype=int)}
    oof_proba = {A_KEY: np.zeros((n, len(LABELS))),
                 B_KEY: np.zeros((n, len(LABELS)))}
    fold_m = {A_KEY: [], B_KEY: []}
    peak_pos = np.full((n, wd.N_WINDOW_CLASSES), -1, dtype=np.int32)
    peak_prob = np.zeros((n, wd.N_WINDOW_CLASSES))
    win_oof_y, win_oof_pred, win_oof_run = [], [], []
    honest = {"mode": mode, "seed": SEED, "n_outer": N_OUTER,
              "n_inner": N_INNER, "folds": [], "verified": True,
              "statement": "every outer-train run received its window "
                           "aggregates from a window model fitted without it"}
    stopped_early = False
    fold_sizes = {f: splits[f][1] for f in fold_ids}

    for f in fold_ids:
        tr, va = splits[f]
        log("OUTER FOLD %d  train=%d  val=%d" % (f, len(tr), len(va)))

        # ---- system A: foundation only ----
        mA = lgb_label().fit(found.iloc[tr], yi[tr])
        pA = mA.predict_proba(found.iloc[va])
        oof[A_KEY][va] = pA.argmax(1)
        oof_proba[A_KEY][va] = pA
        fold_m[A_KEY].append(_fold_macro(ys[va], oof[A_KEY][va], LABELS))

        # ---- honest inner cross-fit for the outer-train rows ----
        tr_runs = np.array(sorted(set(tr.tolist())))
        inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=SEED)
                     .split(tr_runs, yi[tr_runs]))
        assign = np.zeros(len(tr_runs), dtype=int)
        for j, (_, hl) in enumerate(inner):
            assign[hl] = j
        verify_stacking(len(tr_runs), assign, N_INNER, "fold %d" % f)
        honest["folds"].append({"fold": f, "n_train": int(len(tr)),
                                "n_val": int(len(va)),
                                "inner_holdout_sizes": [int((assign == j).sum())
                                                        for j in range(N_INNER)]})

        CW = wd.class_weight_vector()
        payloads = []
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            keep = tr_runs[assign != j]
            payloads.append((f, j, Xw, yw, CW, np.isin(rw, keep),
                             np.isin(rw, hold), WINDOW_PARAMS))
        probs = par.run_inner_fits(payloads, threads=threads, workers=workers)
        del payloads
        gc.collect()

        Xagg_tr = np.zeros((len(tr), ag.N_AGG))
        slot = {int(r): k for k, r in enumerate(tr_runs)}
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            P = probs[(f, j)]
            blocks = grp.predict_runs(P, rw[np.isin(rw, hold)], len(tr_runs))
            sub, _, _ = grp.aggregate_block(blocks, ag, slot, len(tr_runs))
            Xagg_tr[assign == j] = sub[assign == j]
        del probs
        gc.collect()

        # ---- outer-valid features from a model fitted on ALL outer-train ----
        import lightgbm as lgb
        mw = lgb.LGBMClassifier(n_jobs=threads, **WINDOW_PARAMS)
        mw.fit(Xw[np.isin(rw, tr)], yw[np.isin(rw, tr)],
               sample_weight=CW[yw[np.isin(rw, tr)]])
        Pva = mw.predict_proba(Xw[np.isin(rw, va)])
        win_oof_y.append(yw[np.isin(rw, va)])
        win_oof_pred.append(Pva.argmax(1))
        win_oof_run.append(rw[np.isin(rw, va)])
        blocks = grp.predict_runs(Pva, rw[np.isin(rw, va)], n)
        slot_va = {int(r): k for k, r in enumerate(va)}
        Xagg_va, pk, pp = grp.aggregate_block(blocks, ag, slot_va, len(va))
        peak_prob[va] = pk
        peak_pos[va] = pp
        del mw
        gc.collect()

        # ---- system B: foundation + window aggregates ----
        cols = list(found.columns) + list(ag.AGG_NAMES)
        Xb_tr = pd.DataFrame(np.hstack([found.iloc[tr].values, Xagg_tr]),
                             columns=cols)
        Xb_va = pd.DataFrame(np.hstack([found.iloc[va].values, Xagg_va]),
                             columns=cols)
        mB = lgb_label().fit(Xb_tr, yi[tr])
        pB = mB.predict_proba(Xb_va)
        oof[B_KEY][va] = pB.argmax(1)
        oof_proba[B_KEY][va] = pB
        fold_m[B_KEY].append(_fold_macro(ys[va], oof[B_KEY][va], LABELS))
        log("  fold macro:  A=%.4f   B=%.4f" % (fold_m[A_KEY][-1],
                                                fold_m[B_KEY][-1]))
        del mA, mB, Xb_tr, Xb_va, Xagg_tr, Xagg_va
        gc.collect()

        if early_stop and mode == MD.FAST:
            a_m = _partial(ys, oof[A_KEY], SL, runs, c, va)
            b_m = _partial(ys, oof[B_KEY], SL, runs, c, va)
            dc = b_m["composite"] - a_m["composite"]
            dm = b_m["macro_f1"] - a_m["macro_f1"]
            if dc < MD.EARLY_COMPOSITE_DELTA or dm < MD.EARLY_MACRO_DELTA:
                stopped_early = True
                log("FAST early stop: d_comp=%+.4f d_macro=%+.4f "
                    "(threshold %+.3f / %+.3f) - screening only, not a verdict"
                    % (dc, dm, MD.EARLY_COMPOSITE_DELTA, MD.EARLY_MACRO_DELTA))
                break

    return _finalise(c, mode, outdir, n, runs, ys, found, SL, oof, oof_proba,
                     fold_m, fold_ids, peak_pos, peak_prob, win_oof_y,
                     win_oof_pred, win_oof_run, honest, stopped_early,
                     fold_sizes)


def _fold_macro(ys_va, pred_va, labels):
    return sl_macro(ys_va, np.array([labels[j] for j in pred_va], dtype=object),
                    np.ones(len(ys_va), bool)) or 0.0


def _partial(ys, oof_int, SL, runs, c, va):
    """Partial metrics over the rows seen so far - FAST screening only."""
    m = np.zeros(len(ys), bool)
    m[va] = True
    names = [LABELS[i] for i in oof_int[m]]
    obj = np.array(names, dtype=object)
    sub_ys = ys[m]
    mf = sl_macro(sub_ys, obj, np.ones(m.sum(), bool))
    rf = sl_macro(sub_ys, obj, SL["hard/robustness"][m])
    turns = [localize(runs[i], LABELS[oof_int[i]])
             for i in np.where(m)[0]]
    h2 = fault_turn_hit_at_k(list(sub_ys), names,
                             list(c["fturn"][m]), turns)
    return {"macro_f1": mf, "robustness_f1": rf, "fault_turn_hit2": h2,
            "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                    "success_f1": c["success_f1"],
                                    "fault_turn_hit2": h2})}


def _finalise(c, mode, outdir, n, runs, ys, found, SL, oof, oof_proba, fold_m,
              fold_ids, peak_pos, peak_prob, win_oof_y, win_oof_pred,
              win_oof_run, honest, stopped_early):
    # a row is scored only if it appeared in a validation split
    names = {k: [LABELS[i] for i in oof[k]] for k in oof}
    objs = {k: np.array(v, dtype=object) for k, v in names.items()}

    res = {"n_runs": int(n), "mode": mode, "n_outer_folds": N_OUTER,
           "n_inner_folds": N_INNER, "seed": SEED, "success_f1": c["success_f1"],
           "n_foundation": int(found.shape[1]), "n_window_feats": int(XN()),
           "n_agg_feats": int(ag.N_AGG), "pos_radius": wd.POS_RADIUS,
           "folds_evaluated": list(fold_ids), "early_stopped": stopped_early,
           "partial": bool(stopped_early), "systems": {}}

    if stopped_early:
        for k in oof:
            res["systems"][k] = {"partial": True,
                                 "note": "FAST screening stopped early; "
                                         "these rows were never scored"}
    else:
        for k in oof:
            mf = sl_macro(ys, objs[k], np.ones(n, bool))
            rf = sl_macro(ys, objs[k], SL["hard/robustness"])
            turns = [localize(runs[i], names[k][i]) for i in range(n)]
            h2 = fault_turn_hit_at_k(list(ys), names[k], list(c["fturn"]), turns)
            res["systems"][k] = {
                "n_features": int(found.shape[1] +
                                  (ag.N_AGG if k == B_KEY else 0)),
                "macro_f1": mf, "robustness_f1": rf,
                "success_f1": c["success_f1"], "fault_turn_hit2": h2,
                "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                        "success_f1": c["success_f1"],
                                        "fault_turn_hit2": h2}),
                "per_class": f1_per_class(list(ys), names[k], LABELS),
                "fold_macro_f1": [float(x) for x in fold_m[k]],
                "slices": {s: sl_macro(ys, objs[k], m) for s, m in SL.items()},
                "confusion_pairs": confusion(ys, objs[k]),
            }
        ref = res["systems"][A_KEY]
        for k in res["systems"]:
            d = res["systems"][k]
            d["d_macro"] = d["macro_f1"] - ref["macro_f1"]
            d["d_rob"] = d["robustness_f1"] - ref["robustness_f1"]
            d["d_comp"] = d["composite"] - ref["composite"]
            d["d_hit2"] = d["fault_turn_hit2"] - ref["fault_turn_hit2"]
            d["d_dh"] = (prf(ys, objs[k], DH)["f1"]
                         - prf(ys, objs[A_KEY], DH)["f1"])

    # ---------------- artifacts ----------------
    for k in oof:
        pd.DataFrame({"run_id": c["train"]["run_id"], "y_true": ys,
                      "y_pred": names[k], "fault_turn": c["fturn"],
                      "n_messages": c["nmsg"]}).to_csv(
            os.path.join(outdir, "oof_%s.csv" % k), index=False)
        np.save(os.path.join(outdir, "oof_proba_%s.npy" % k),
                oof_proba[k].astype(np.float32))
    with open(os.path.join(outdir, "oof_runid.json"), "w") as fh:
        json.dump([str(x) for x in c["train"]["run_id"]], fh)
    # record what the artifacts ARE, so a consumer can verify rather than
    # trust them (see reuse.py)
    honest["proba_sha256"] = {k: ca.sha256_array(oof_proba[k].astype(np.float32))
                              for k in oof}
    honest["label_sha256"] = {k: ca.sha256_array(oof[k]) for k in oof}
    honest["outer_val_sizes"] = [int(len(fold_sizes[f])) for f in _folds_run]
    honest["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if win_oof_y:
        np.save(os.path.join(outdir, "window_oof_y.npy"),
                np.concatenate(win_oof_y))
        np.save(os.path.join(outdir, "window_oof_pred.npy"),
                np.concatenate(win_oof_pred))
        np.save(os.path.join(outdir, "window_oof_run.npy"),
                np.concatenate(win_oof_run))
        np.save(os.path.join(outdir, "window_peak_pos.npy"), peak_pos)
        np.save(os.path.join(outdir, "window_peak_prob.npy"), peak_prob)
    with open(os.path.join(outdir, "honest_manifest.json"), "w") as fh:
        json.dump(honest, fh, indent=2)
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    log("wrote %s" % outdir)
    return res


def XN():
    import window_features as _wf
    return _wf.N_FEATURES


DH = "dropped_handoff"


def main(argv=None):
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Exp07 runner (FAST/CV/FINAL)")
    ap.add_argument("--mode", default="cv", choices=list(MD.MODES))
    ap.add_argument("--no-cache", action="store_true",
                    help="rebuild every artifact from scratch (parity path)")
    ap.add_argument("--threads", type=int, default=par.DEFAULT_THREADS)
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the independent inner window fits. "
                         "1 (default) is the parity-safe setting: each fit "
                         "already uses all cores via n_jobs=-1, so extra "
                         "processes only add contention on a 6C/12T laptop.")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    a = ap.parse_args(argv)
    if a.clear_cache:
        ca.clear()
    t0 = time.time()
    res = run(a.mode, use_cache=not a.no_cache, threads=a.threads,
              workers=a.workers, early_stop=not a.no_early_stop)
    log("MODE %s done in %.1fs (%.2f min)" % (a.mode, time.time() - t0,
                                              (time.time() - t0) / 60))
    for k, v in res["systems"].items():
        if v.get("partial"):
            print("%-16s partial (early stop)" % k)
        else:
            print("%-16s macro=%.6f rob=%.6f hit2=%.6f comp=%.6f"
                  % (k, v["macro_f1"], v["robustness_f1"],
                     v["fault_turn_hit2"], v["composite"]))
    return res


if __name__ == "__main__":
    main()
