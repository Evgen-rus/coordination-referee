"""Exp18: one frozen train-only telemetry-loss augmentation CV."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP07 = ROOT / "experiments" / "exp07_fault_windows"
EXP13 = ROOT / "experiments" / "exp13_wait_dependency_graph"
EXP15_DIR = ROOT / "experiments" / "exp15_success_label_stacking"
EXP16_DIR = ROOT / "experiments" / "exp16_semantic_sequence"
AUGMENT_SEED = 20261008
N_FOLDS = 3
FOLD_SEED = 0
INNER_SEED = 0
MAX_BENCHMARK_SECONDS = 7200.0
LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
BASELINE_METRICS = {
    "macro_f1": 0.8129545586738590,
    "robustness_f1": 0.7599843692318033,
    "success_f1": 0.8949086161879896,
    "fault_turn_hit2": 0.6315277777777778,
    "composite": 0.7938624418508566,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_helpers():
    sys.path.insert(0, str(EXP15_DIR))
    exp15 = importlib.import_module("run_experiment")
    metrics = importlib.import_module("metrics")
    common = importlib.import_module("common")
    if Path(metrics.__file__).resolve() != (ROOT / "evaluation" / "metrics.py").resolve():
        raise RuntimeError("metrics import did not resolve to the official evaluator")
    if Path(common.__file__).resolve() != (EXP07 / "common.py").resolve():
        raise RuntimeError("common import did not resolve to Exp07 feature builder")
    return exp15, metrics, common


def generate_augmented_views(runs):
    """Make at most one deterministic corrupted view per source run."""
    views, parents = [], []
    selected = {"intent": 0, "refs": 0, "artifact_subtask": 0, "text": 0}
    changed = {"intent": 0, "refs": 0, "artifact_subtask": 0, "text": 0}
    rng = np.random.default_rng(AUGMENT_SEED)
    for parent, original in enumerate(runs):
        flags = rng.random(4) < np.asarray([0.12, 0.08, 0.10, 0.15])
        if not flags.any():
            continue
        view = copy.deepcopy(original)
        messages = view.get("messages", []) or []
        if flags[0]:
            selected["intent"] += 1
            before = [m.get("intent") for m in messages]
            for m in messages:
                m["intent"] = ""
            changed["intent"] += int(any(v not in (None, "") for v in before))
        if flags[1]:
            selected["refs"] += 1
            n_pick = len(messages) // 2
            picks = rng.permutation(len(messages))[:n_pick]
            did_change = False
            for i in picks:
                if messages[int(i)].get("refs"):
                    did_change = True
                messages[int(i)]["refs"] = []
            changed["refs"] += int(did_change)
        if flags[2]:
            selected["artifact_subtask"] += 1
            arts = view.get("artifacts", []) or []
            did_change = any(a.get("subtask") not in (None, "") for a in arts)
            for a in arts:
                a["subtask"] = ""
            changed["artifact_subtask"] += int(did_change)
        if flags[3]:
            selected["text"] += 1
            n_pick = len(messages) // 2
            picks = rng.permutation(len(messages))[:n_pick]
            did_change = False
            for i in picks:
                m = messages[int(i)]
                old = str(m.get("text") or "")
                limit = int(rng.integers(12, 31))
                clipped = old[:limit]
                did_change |= clipped != old
                m["text"] = clipped
            changed["text"] += int(did_change)

        _assert_only_allowed_fields_changed(original, view)
        views.append(view)
        parents.append(parent)
    if any(len(v.get("messages", []) or []) != len(runs[p].get("messages", []) or [])
           for v, p in zip(views, parents)):
        raise AssertionError("augmentation changed the number of events")
    return views, np.asarray(parents, dtype=np.int64), {
        "seed": AUGMENT_SEED,
        "view_count": len(views),
        "selected_runs_by_transform": selected,
        "runs_with_effective_change_by_transform": changed,
    }


def _assert_only_allowed_fields_changed(original, view):
    """Make target/event identity fields an executable augmentation guard."""
    original_messages = original.get("messages", []) or []
    view_messages = view.get("messages", []) or []
    if len(original_messages) != len(view_messages):
        raise AssertionError("augmentation changed message count")
    for before, after in zip(original_messages, view_messages):
        probe = copy.deepcopy(after)
        for key in ("intent", "refs", "text"):
            if key in before:
                probe[key] = copy.deepcopy(before[key])
            else:
                probe.pop(key, None)
        if probe != before:
            raise AssertionError("augmentation changed a non-approved message field")
    original_arts = original.get("artifacts", []) or []
    view_arts = view.get("artifacts", []) or []
    if len(original_arts) != len(view_arts):
        raise AssertionError("augmentation changed artifact count")
    for before, after in zip(original_arts, view_arts):
        probe = copy.deepcopy(after)
        if "subtask" in before:
            probe["subtask"] = before["subtask"]
        else:
            probe.pop("subtask", None)
        if probe != before:
            raise AssertionError("augmentation changed a non-approved artifact field")
    for key in ("goal", "agents", "topology", "shared_state"):
        if view.get(key) != original.get(key):
            raise AssertionError("augmentation changed run field %s" % key)


def build_foundation(runs, common, columns):
    base_rows = [common.extract_features(run) for run in runs]
    base = pd.DataFrame(base_rows).fillna(0.0)
    normalized = common.nf.build_matrix(runs, base_rows)
    lifecycle = common.lc.build_matrix(runs)[common.lc.LIFECYCLE_NAMES]
    first = pd.concat([base, normalized, lifecycle], axis=1)
    temporal = common.ft.build_matrix(runs)
    temporal = temporal[[col for col in temporal.columns if col not in common.B1]]
    foundation = pd.concat([first, temporal], axis=1)
    if foundation.columns.tolist() != list(columns):
        raise AssertionError("foundation feature order differs from Exp15")
    values = foundation.to_numpy()
    if values.shape[1] != 249 or not np.isfinite(values).all():
        raise AssertionError("augmented foundation is not finite 249-column data")
    return foundation


def append_windows(original, added, offset):
    if not len(added["y"]):
        return original
    return {
        "X": np.vstack([original["X"], added["X"]]),
        "y": np.concatenate([original["y"], added["y"]]),
        "run": np.concatenate([original["run"], added["run"] + offset]),
        "turn": np.concatenate([original["turn"], added["turn"]]),
    }


def slice_baseline(c, wd, wg, rows, exp15):
    rows = np.asarray(rows, dtype=np.int64)
    runs = [c["runs"][int(i)] for i in rows]
    y = np.asarray(c["yi"])[rows]
    ys = np.asarray(c["ys"], dtype=object)[rows]
    ft = np.asarray(c["fturn"], dtype=np.int64)[rows]
    return ({"n": len(rows), "yi": y,
             "foundation": c["foundation"].iloc[rows].reset_index(drop=True),
             "runs": runs},
            exp15.wd.build_window_dataset(runs, ys, ft),
            np.asarray(wg)[rows], ys, ft)


def make_augmented_data(c, wd, wg, aug_views, aug_parents, exp15, common):
    n = int(c["n"])
    aug_yi = np.asarray(c["yi"])[aug_parents]
    aug_ys = np.asarray(c["ys"], dtype=object)[aug_parents]
    aug_ft = np.asarray(c["fturn"], dtype=np.int64)[aug_parents]
    aug_foundation = build_foundation(aug_views, common, c["foundation"].columns)
    aug_wg = exp15.W.build_matrix(aug_views)
    aug_wd = exp15.wd.build_window_dataset(aug_views, aug_ys, aug_ft)
    c_aug = dict(c)
    c_aug["runs"] = list(c["runs"]) + list(aug_views)
    c_aug["n"] = n + len(aug_views)
    c_aug["yi"] = np.concatenate([np.asarray(c["yi"], dtype=np.int64), aug_yi])
    c_aug["ys"] = np.concatenate([np.asarray(c["ys"], dtype=object), aug_ys])
    c_aug["fturn"] = np.concatenate([np.asarray(c["fturn"], dtype=np.int64), aug_ft])
    c_aug["foundation"] = pd.concat(
        [c["foundation"].reset_index(drop=True), aug_foundation], ignore_index=True)
    wg_aug = np.vstack([np.asarray(wg), aug_wg])
    wd_aug = append_windows(wd, aug_wd, n)
    parent_of = np.concatenate([np.arange(n, dtype=np.int64), aug_parents])
    if (len(c_aug["runs"]) != len(parent_of) or
            len(c_aug["foundation"]) != len(parent_of) or
            wg_aug.shape != (len(parent_of), 12)):
        raise AssertionError("augmented run-level matrices lost row alignment")
    if not np.array_equal(c_aug["yi"][n:], c_aug["yi"][aug_parents]):
        raise AssertionError("augmented view label differs from its parent")
    if not np.array_equal(c_aug["fturn"][n:], c_aug["fturn"][aug_parents]):
        raise AssertionError("augmented view fault_turn differs from its parent")
    return c_aug, wd_aug, wg_aug, parent_of


def corrected_agg_block_grouped(tr_idx, va_idx, c, wd, parent_of, tag, exp15,
                                log=print):
    """Exp15 corrected aggregation with parent-grouped inner folds, serial fits."""
    yi, n = c["yi"], c["n"]
    Xw, yw, rw = wd["X"], wd["y"], wd["run"]
    cw = exp15.wd.class_weight_vector()
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    va_idx = np.asarray(va_idx, dtype=np.int64)
    if not len(tr_idx) or not len(va_idx):
        raise ValueError("%s: empty train or validation split" % tag)
    if len(np.unique(tr_idx)) != len(tr_idx) or np.any(np.diff(tr_idx) <= 0):
        raise AssertionError("%s: train rows must be unique and ascending" % tag)
    if np.intersect1d(tr_idx, va_idx).size:
        raise AssertionError("%s: train and validation rows overlap" % tag)
    tr_runs = tr_idx.copy()
    train_parents = np.unique(parent_of[tr_runs])
    if np.intersect1d(train_parents, parent_of[va_idx]).size:
        raise AssertionError("%s: parent view crosses train/validation" % tag)

    inner = list(StratifiedKFold(3, shuffle=True, random_state=INNER_SEED)
                 .split(train_parents, yi[train_parents]))
    assign = np.full(len(tr_runs), -1, dtype=np.int8)
    for fold, (_, hold_parent_pos) in enumerate(inner):
        hold_parents = train_parents[hold_parent_pos]
        assign[np.isin(parent_of[tr_runs], hold_parents)] = fold
    if np.any(assign < 0):
        raise AssertionError("%s: parent-group inner folds did not cover rows" % tag)
    for fold in range(3):
        fit_parents = set(parent_of[tr_runs[assign != fold]].tolist())
        hold_parents = set(parent_of[tr_runs[assign == fold]].tolist())
        if fit_parents & hold_parents:
            raise AssertionError("%s: parent copy leaked across inner fold" % tag)
    exp15.R.verify_stacking(len(tr_runs), assign, 3, tag)
    g2l = exp15.M.global_to_local_map(tr_runs)

    payloads = []
    for fold in range(3):
        fit_runs = tr_runs[assign != fold]
        hold_runs = tr_runs[assign == fold]
        payloads.append((tag, fold, Xw, yw, cw,
                         np.isin(rw, fit_runs), np.isin(rw, hold_runs),
                         exp15.R.WINDOW_PARAMS))
    probs = exp15.par.run_inner_fits(payloads, workers=1)
    Xagg_tr = np.zeros((len(tr_runs), exp15.ag.N_AGG), dtype=np.float64)
    for fold in range(3):
        hold = tr_runs[assign == fold]
        sel = rw[np.isin(rw, hold)]
        blocks, slot = exp15.M.group_fixed(probs[(tag, fold)], sel, g2l,
                                            len(tr_runs))
        sub, _, _ = exp15.grp.aggregate_block(blocks, exp15.ag, slot,
                                               len(tr_runs))
        Xagg_tr[assign == fold] = sub[assign == fold]
    del probs, payloads
    gc.collect()

    fit_mask = np.isin(rw, tr_runs)
    val_mask = np.isin(rw, va_idx)
    window_model = exp15.lgb.LGBMClassifier(
        n_jobs=exp15.par.DEFAULT_THREADS, **exp15.R.WINDOW_PARAMS)
    window_model.fit(Xw[fit_mask], yw[fit_mask], sample_weight=cw[yw[fit_mask]])
    p_val = window_model.predict_proba(Xw[val_mask])
    blocks = exp15.grp.predict_runs(p_val, rw[val_mask], n)
    slot_val = {int(run): pos for pos, run in enumerate(va_idx)}
    Xagg_va, _, _ = exp15.grp.aggregate_block(blocks, exp15.ag, slot_val,
                                               len(va_idx))
    del window_model
    gc.collect()
    log("%s: %d parent groups -> %d train views, %d validation runs" %
        (tag, len(train_parents), len(tr_runs), len(va_idx)))
    return Xagg_tr, Xagg_va, {"n_train_parents": int(len(train_parents)),
                              "n_train_views": int(len(tr_runs)),
                              "n_validation": int(len(va_idx)),
                              "inner_holdout_sizes": [int((assign == j).sum())
                                                      for j in range(3)]}


def score_rows(rows, probabilities, d, success_pred, robust, metrics):
    rows = np.asarray(rows, dtype=np.int64)
    pred = np.asarray(probabilities).argmax(1)
    true = np.asarray(d["yi"])[rows]
    yt = [LABELS[int(i)] for i in true]
    yp = [LABELS[int(i)] for i in pred]
    class_f1 = metrics.f1_per_class(yt, yp, LABELS)
    macro = metrics.macro_f1(yt, yp, LABELS)
    robust_local = np.asarray(robust, dtype=bool)[rows]
    robust_rows = np.flatnonzero(robust_local)
    robust_f1 = (metrics.macro_f1([yt[i] for i in robust_rows],
                                  [yp[i] for i in robust_rows], LABELS)
                 if len(robust_rows) else 0.0)
    s_true = np.asarray(d["train"]["success"], dtype=np.int64)[rows]
    success_f1 = metrics.binary_f1(s_true.tolist(),
                                   np.asarray(success_pred)[rows].tolist())
    turns = np.asarray(d["peak_pos"])[rows][np.arange(len(rows)), pred]
    faulty = np.flatnonzero(true != 0)
    hit2 = metrics.fault_turn_hit_at_k(
        [yt[i] for i in faulty], [yp[i] for i in faulty],
        np.asarray(d["train"]["fault_turn"], dtype=np.int64)[rows][faulty].tolist(),
        np.asarray(turns, dtype=np.int64)[faulty].tolist(), k=2)
    composite = metrics.composite({"macro_f1": macro,
                                   "robustness_f1": robust_f1,
                                   "success_f1": success_f1,
                                   "fault_turn_hit2": hit2})
    return {"macro_f1": macro, "robustness_f1": robust_f1,
            "success_f1": success_f1, "fault_turn_hit2": hit2,
            "composite": composite, "per_class_f1": class_f1}


def validate_baseline(d, exp15, metrics, success_pred):
    baseline_path = EXP16_DIR / "baseline_exp15_oof.csv"
    manifest = json.loads((EXP16_DIR / "baseline.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(baseline_path)
    train_ids = [str(v) for v in d["train"]["run_id"]]
    if baseline["run_id"].astype(str).tolist() != train_ids:
        raise AssertionError("frozen baseline predictions are misaligned")
    if sha256_file(baseline_path) != manifest["baseline_oof_prediction_sha256"]:
        raise AssertionError("frozen Exp15 OOF file hash mismatch")
    if sha256_file(ROOT / "data" / "train.csv") != manifest["train_csv_sha256"]:
        raise AssertionError("train input differs from the frozen Exp15 input")
    zip_path = ROOT / "submission_exp15_success_label_stacking.zip"
    if sha256_file(zip_path) != manifest["baseline_zip_sha256"]:
        raise AssertionError("Exp15 production ZIP changed")
    p_base = np.load(EXP13 / "oof_D_corrected_wg.npy", allow_pickle=False)
    if sha256_file(EXP13 / "oof_D_corrected_wg.npy") != manifest[
            "exp13_oof_probability_sha256"]:
        raise AssertionError("Exp13 OOF probability hash mismatch")
    pred = p_base.argmax(1)
    if not np.array_equal(np.asarray(baseline["label"], dtype=object),
                          np.asarray(LABELS, dtype=object)[pred]):
        raise AssertionError("frozen Exp15 labels differ from Exp13 OOF")
    if not np.array_equal(np.asarray(baseline["success"], dtype=np.int64),
                          np.asarray(success_pred, dtype=np.int64)):
        raise AssertionError("frozen Exp15 success OOF differs from saved array")
    peak_turns = np.asarray(d["peak_pos"])[np.arange(len(pred)), pred]
    if not np.array_equal(np.asarray(baseline["fault_turn"], dtype=np.int64),
                          peak_turns.astype(np.int64)):
        raise AssertionError("frozen Exp15 fault_turn differs from Exp13 L1")
    base_metrics = score_rows(np.arange(len(pred)), p_base, d, success_pred,
                              exp15.ENS.robustness_mask(), metrics)
    for name, expected in BASELINE_METRICS.items():
        if abs(float(base_metrics[name]) - expected) > 1e-8:
            raise AssertionError("Exp15 baseline %s mismatch" % name)
    return p_base, baseline, base_metrics


def run_baseline_control(c, wd, wg, d, exp15, metrics, success_pred, robust,
                         folds, log):
    n = int(c["n"])
    all_rows = np.arange(n, dtype=np.int64)
    probabilities = np.full((n, len(LABELS)), np.nan, dtype=np.float32)
    fold_rows = []
    t0 = time.time()
    for fold, valid in enumerate(folds):
        valid = np.asarray(valid, dtype=np.int64)
        train = all_rows[~np.isin(all_rows, valid)]
        started = time.time()
        xagg_train, xagg_valid, _ = corrected_agg_block_grouped(
            train, valid, c, wd, all_rows, "exp18 control fold %d" % fold,
            exp15, log=log)
        x_train = exp15.label_matrix_312(train, xagg_train, c, wg)
        x_valid = exp15.label_matrix_312(valid, xagg_valid, c, wg)
        model = exp15.lgb.LGBMClassifier(
            **exp15.ENS.label_params(42)).fit(x_train, np.asarray(c["yi"])[train])
        p_val = model.predict_proba(x_valid).astype(np.float32)
        probabilities[valid] = p_val
        expected = np.load(EXP13 / "oof_D_corrected_wg.npy", allow_pickle=False)
        if not np.array_equal(p_val.argmax(1), expected[valid].argmax(1)):
            raise AssertionError("unaugmented control failed Exp13 label parity")
        fold_metrics = score_rows(valid, p_val, d, success_pred, robust, metrics)
        fold_rows.append({"fold": fold,
                          "composite": fold_metrics["composite"],
                          "seconds": round(time.time() - started, 2)})
        log("control fold %d reproduced labels; %.1fs" %
            (fold, fold_rows[-1]["seconds"]))
        del model, x_train, x_valid, xagg_train, xagg_valid
        gc.collect()
    if not np.isfinite(probabilities).all():
        raise AssertionError("unaugmented control OOF is incomplete")
    control_metrics = score_rows(all_rows, probabilities, d, success_pred,
                                 robust, metrics)
    for name, expected in BASELINE_METRICS.items():
        if abs(float(control_metrics[name]) - expected) > 1e-8:
            raise AssertionError("unaugmented control %s mismatch" % name)
    return probabilities, control_metrics, fold_rows, round(time.time() - t0, 2)


def benchmark(sample_size, c, wd, wg, d, exp15, metrics, common,
              aug_views, aug_parents, log):
    n = int(c["n"])
    y = np.asarray(c["yi"], dtype=np.int64)
    sample_folds = StratifiedKFold(20, shuffle=True, random_state=AUGMENT_SEED)
    sample = np.sort(next(sample_folds.split(np.zeros(n), y))[1])[:sample_size]
    # The first sealed 1/20 stratified fold is the fixed 500-run timing sample.
    if len(sample) != sample_size:
        raise AssertionError("benchmark sample does not contain 500 runs")
    global_to_local = {int(row): pos for pos, row in enumerate(sample)}
    local_aug_ix = [j for j, parent in enumerate(aug_parents)
                    if int(parent) in global_to_local]
    sample_aug = [aug_views[j] for j in local_aug_ix]
    sample_aug_parents = np.asarray(
        [global_to_local[int(aug_parents[j])] for j in local_aug_ix], dtype=np.int64)

    t_prep = time.time()
    base_small, wd_base, wg_base, ys_small, ft_small = slice_baseline(
        c, wd, wg, sample, exp15)
    aug_foundation = build_foundation(sample_aug, common,
                                     c["foundation"].columns) if sample_aug else \
        c["foundation"].iloc[:0].copy()
    aug_labels = ys_small[sample_aug_parents]
    aug_ft = ft_small[sample_aug_parents]
    aug_wg = exp15.W.build_matrix(sample_aug) if sample_aug else \
        np.zeros((0, 12), dtype=np.float64)
    wd_added = exp15.wd.build_window_dataset(sample_aug, aug_labels, aug_ft) \
        if sample_aug else {"X": np.zeros((0, wd_base["X"].shape[1])),
                            "y": np.zeros(0, dtype=np.int32),
                            "run": np.zeros(0, dtype=np.int32),
                            "turn": np.zeros(0, dtype=np.int32)}
    wd_small = append_windows(wd_base, wd_added, len(sample))
    c_small = dict(base_small)
    c_small["runs"] = list(base_small["runs"]) + sample_aug
    c_small["n"] = len(sample) + len(sample_aug)
    c_small["yi"] = np.concatenate([base_small["yi"], y[sample][sample_aug_parents]])
    c_small["ys"] = np.concatenate([ys_small, aug_labels])
    c_small["fturn"] = np.concatenate([ft_small, aug_ft])
    c_small["foundation"] = pd.concat(
        [base_small["foundation"], aug_foundation], ignore_index=True)
    wg_small = np.vstack([wg_base, aug_wg])
    parent_small = np.concatenate([np.arange(len(sample)), sample_aug_parents])
    t_prep_sec = time.time() - t_prep

    splits = list(StratifiedKFold(5, shuffle=True, random_state=AUGMENT_SEED)
                  .split(np.arange(len(sample)), y[sample]))
    train_parent, valid_parent = splits[0]
    t_control = time.time()
    xa, xb, _ = corrected_agg_block_grouped(
        train_parent, valid_parent, base_small, wd_base, np.arange(len(sample)),
        "exp18 benchmark control", exp15, log=log)
    xt = exp15.label_matrix_312(train_parent, xa, base_small, wg_base)
    xv = exp15.label_matrix_312(valid_parent, xb, base_small, wg_base)
    exp15.lgb.LGBMClassifier(**exp15.ENS.label_params(42)).fit(
        xt, y[sample][train_parent]).predict_proba(xv)
    control_sec = time.time() - t_control

    train_rows = np.flatnonzero(np.isin(parent_small, train_parent))
    t_candidate = time.time()
    xa, xb, _ = corrected_agg_block_grouped(
        train_rows, valid_parent, c_small, wd_small, parent_small,
        "exp18 benchmark candidate", exp15, log=log)
    xt = exp15.label_matrix_312(train_rows, xa, c_small, wg_small)
    xv = exp15.label_matrix_312(valid_parent, xb, c_small, wg_small)
    exp15.lgb.LGBMClassifier(**exp15.ENS.label_params(42)).fit(
        xt, np.asarray(c_small["yi"])[train_rows]).predict_proba(xv)
    candidate_sec = time.time() - t_candidate

    full_valid = np.asarray(d["folds"][0], dtype=np.int64)
    full_train_mask = np.ones(n, dtype=bool)
    full_train_mask[full_valid] = False
    base_window_counts = np.bincount(wd["run"], minlength=n)
    sample_base_counts = np.bincount(wd_base["run"], minlength=len(sample))
    sample_aug_counts = np.bincount(wd_added["run"],
                                    minlength=len(sample_aug))
    full_aug_counts = np.asarray([
        len(run.get("messages", []) or []) for run in aug_views], dtype=np.int64)
    full_aug_train = np.isin(aug_parents, np.flatnonzero(full_train_mask))
    full_base_train_windows = int(base_window_counts[full_train_mask].sum())
    full_candidate_train_windows = full_base_train_windows + int(
        full_aug_counts[full_aug_train].sum())
    sample_base_train_windows = int(sample_base_counts[train_parent].sum())
    local_train_mask = np.isin(sample_aug_parents, train_parent)
    sample_candidate_train_windows = (sample_base_train_windows + int(
        sample_aug_counts[local_train_mask].sum()))
    base_scale = full_base_train_windows / max(1, sample_base_train_windows)
    candidate_scale = full_candidate_train_windows / max(1,
                                                       sample_candidate_train_windows)
    prep_scale = len(aug_views) / max(1, len(sample_aug))
    estimated_prep = t_prep_sec * prep_scale * 1.25
    estimated = (estimated_prep + 3.0 * 1.25 *
                 (control_sec * base_scale + candidate_sec * candidate_scale))
    result = {
        "runs_in_timing_sample": int(sample_size),
        "augmented_views_in_sample": int(len(sample_aug)),
        "augmented_views_total": int(len(aug_views)),
        "sample_input_feature_and_window_seconds": round(t_prep_sec, 3),
        "control_one_fold_seconds": round(control_sec, 3),
        "candidate_one_fold_seconds": round(candidate_sec, 3),
        "control_train_window_scale_to_full_fold": round(base_scale, 3),
        "candidate_train_window_scale_to_full_fold": round(candidate_scale, 3),
        "estimated_full_augmented_input_prep_seconds": round(estimated_prep, 2),
        "estimated_control_plus_candidate_3fold_seconds": round(estimated, 2),
        "estimated_full_cv_minutes": round(estimated / 60.0, 2),
        "max_full_cv_seconds": MAX_BENCHMARK_SECONDS,
        "ready_for_full_cv": bool(estimated <= MAX_BENCHMARK_SECONDS),
    }
    return result


def run_full_cv(c, wd, wg, d, exp15, metrics, success_pred, robust,
                folds, aug_views, aug_parents, log):
    n = int(c["n"])
    t_prep = time.time()
    common = importlib.import_module("common")
    c_aug, wd_aug, wg_aug, parent_of = make_augmented_data(
        c, wd, wg, aug_views, aug_parents, exp15, common)
    prep_sec = round(time.time() - t_prep, 2)
    log("full augmented views built: %d; prep %.1fs" % (len(aug_views), prep_sec))

    p_control, control_metrics, control_folds, control_sec = run_baseline_control(
        c, wd, wg, d, exp15, metrics, success_pred, robust, folds, log)
    all_rows = np.arange(n, dtype=np.int64)
    p_candidate = np.full((n, len(LABELS)), np.nan, dtype=np.float32)
    candidate_folds = []
    candidate_sec = 0.0
    for fold, valid in enumerate(folds):
        valid = np.asarray(valid, dtype=np.int64)
        train_parents = all_rows[~np.isin(all_rows, valid)]
        train_rows = np.flatnonzero(np.isin(parent_of, train_parents))
        started = time.time()
        xagg_train, xagg_valid, meta = corrected_agg_block_grouped(
            train_rows, valid, c_aug, wd_aug, parent_of,
            "exp18 outer %d" % fold, exp15, log=log)
        x_train = exp15.label_matrix_312(train_rows, xagg_train, c_aug, wg_aug)
        x_valid = exp15.label_matrix_312(valid, xagg_valid, c_aug, wg_aug)
        model = exp15.lgb.LGBMClassifier(
            **exp15.ENS.label_params(42)).fit(x_train, c_aug["yi"][train_rows])
        p_candidate[valid] = model.predict_proba(x_valid).astype(np.float32)
        base_fold = score_rows(valid, p_control[valid], d, success_pred,
                               robust, metrics)
        candidate_fold = score_rows(valid, p_candidate[valid], d, success_pred,
                                    robust, metrics)
        row = {"fold": fold, "n_train_parents": int(len(train_parents)),
               "n_train_views": int(len(train_rows)),
               "n_augmented_train_views": int(len(train_rows) - len(train_parents)),
               "baseline": base_fold, "candidate": candidate_fold,
               "composite_delta": candidate_fold["composite"] -
                                  base_fold["composite"],
               "grouping": meta,
               "seconds": round(time.time() - started, 2)}
        candidate_folds.append(row)
        candidate_sec += row["seconds"]
        log("Exp18 fold %d: composite %.6f -> %.6f (%+.6f), DH F1 %.4f -> %.4f, %.1fs" %
            (fold, base_fold["composite"], candidate_fold["composite"],
             row["composite_delta"], base_fold["per_class_f1"]["dropped_handoff"],
             candidate_fold["per_class_f1"]["dropped_handoff"], row["seconds"]))
        del model, x_train, x_valid, xagg_train, xagg_valid
        gc.collect()
    if not np.isfinite(p_candidate).all():
        raise AssertionError("candidate OOF predictions are incomplete")

    candidate_metrics = score_rows(all_rows, p_candidate, d, success_pred,
                                   robust, metrics)
    class_deltas = {label: candidate_metrics["per_class_f1"][label] -
                    control_metrics["per_class_f1"][label] for label in LABELS}
    folds_won = sum(row["composite_delta"] > 0 for row in candidate_folds)
    deltas = {name: candidate_metrics[name] - control_metrics[name]
              for name in BASELINE_METRICS}
    checks = {
        "composite_delta_at_least_0_008": deltas["composite"] >= 0.008,
        "macro_f1_does_not_fall": deltas["macro_f1"] >= 0.0,
        "robustness_f1_does_not_fall": deltas["robustness_f1"] >= 0.0,
        "dropped_handoff_f1_improves": class_deltas["dropped_handoff"] > 0.0,
        "composite_wins_at_least_2_of_3_folds": folds_won >= 2,
        "no_class_f1_drop_over_0_02": all(delta >= -0.02
                                           for delta in class_deltas.values()),
    }
    passed = all(checks.values())
    result = {
        "decision": "PROMOTE" if passed else "STOP",
        "hypothesis": "fixed train-only telemetry-loss augmentation improves Robustness and dropped_handoff classification",
        "baseline": control_metrics,
        "candidate": candidate_metrics,
        "deltas": deltas,
        "per_class_f1_delta": class_deltas,
        "folds_won": int(folds_won),
        "folds": candidate_folds,
        "baseline_control_folds": control_folds,
        "augmentation": {"seed": AUGMENT_SEED,
                         "views": int(len(aug_views)),
                         "parents": int(len(np.unique(aug_parents))),
                         "transform_rates": {"intent": 0.12, "refs": 0.08,
                                             "artifact_subtask": 0.10,
                                             "text": 0.15},
                         "training_input_prep_seconds": prep_sec},
        "gate": {"frozen": {"composite_delta_min": 0.008,
                             "macro_f1_delta_min": 0.0,
                             "robustness_f1_delta_min": 0.0,
                             "target_class": "dropped_handoff",
                             "target_class_f1_delta_min_exclusive": 0.0,
                             "folds_won_min": 2,
                             "per_class_f1_drop_max": 0.02},
                 "checks": checks, "passed": bool(passed)},
        "promotion_gate": {"passed": bool(passed)},
        "runtime_seconds": {"baseline_control_3fold": control_sec,
                             "candidate_3fold": round(candidate_sec, 2),
                             "feature_window_prep": prep_sec},
        "baseline_provenance": {
            "exp15_commit": json.loads((EXP16_DIR / "baseline.json").read_text(
                encoding="utf-8"))["baseline_commit"],
            "production_changed": False},
    }
    np.save(HERE / "structural_label_proba.npy", p_control)
    np.save(HERE / "candidate_label_proba.npy", p_candidate)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                        encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    log_lines = []

    def log(message=""):
        print(message, flush=True)
        log_lines.append(str(message))

    t0 = time.time()
    exp15, metrics, common = load_helpers()
    d = exp15.ENS.load_verified()
    c, wd = exp15.ENS._prepare()
    wg = exp15.W.build_matrix(c["runs"])
    n = int(c["n"])
    if n != 10000 or len(d["train"]) != n:
        raise AssertionError("unexpected training row count")
    y = np.asarray(c["yi"], dtype=np.int64)
    folds = [np.asarray(v, dtype=np.int64) for v in d["folds"]]
    sealed = [np.asarray(valid, dtype=np.int64) for _, valid in
              StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
              .split(np.zeros(n), y)]
    if any(not np.array_equal(a, b) for a, b in zip(folds, sealed)):
        raise AssertionError("Exp15 outer folds changed")
    success_pred = np.load(EXP15_DIR / "success_pred_B.npy", allow_pickle=False)
    p_base, baseline, baseline_metrics = validate_baseline(
        d, exp15, metrics, success_pred)
    robust = np.asarray(exp15.ENS.robustness_mask(), dtype=bool)
    log("Exp15 baseline verified: composite=%.10f; Exp13 label/turn and Exp15 success exact" %
        baseline_metrics["composite"])
    log("outer folds are the sealed seed-0 StratifiedKFold(3)")

    t_aug = time.time()
    aug_views, aug_parents, aug_manifest = generate_augmented_views(c["runs"])
    generation_sec = round(time.time() - t_aug, 2)
    log("generated %d fixed views from %d original runs in %.1fs" %
        (len(aug_views), n, generation_sec))

    benchmark_result = benchmark(500, c, wd, wg, d, exp15, metrics, common,
                                 aug_views, aug_parents, log)
    benchmark_result["augmentation_generation_full_seconds"] = generation_sec
    benchmark_result["augmentation_manifest"] = aug_manifest
    benchmark_result["baseline_composite"] = baseline_metrics["composite"]
    benchmark_result["baseline_parity"] = True
    benchmark_result["benchmark_time_seconds"] = round(time.time() - t0, 2)
    (HERE / "cpu_benchmark.json").write_text(
        json.dumps(benchmark_result, indent=2) + "\n", encoding="utf-8")
    log("CPU estimate: %.2f minutes; readiness=%s" %
        (benchmark_result["estimated_full_cv_minutes"],
         benchmark_result["ready_for_full_cv"]))
    if args.benchmark_only:
        (HERE / "run.log").write_text("\n".join(log_lines) + "\n",
                                      encoding="utf-8")
        return 0
    if not benchmark_result["ready_for_full_cv"]:
        log("STOP before full CV: estimate exceeds two hours")
        (HERE / "run.log").write_text("\n".join(log_lines) + "\n",
                                      encoding="utf-8")
        return 0

    result = run_full_cv(c, wd, wg, d, exp15, metrics, success_pred,
                         robust, folds, aug_views, aug_parents, log)
    result["augmentation_manifest"] = aug_manifest
    result["cpu_benchmark"] = benchmark_result
    result["baseline_metric_source"] = "Exp15 frozen OOF + official evaluation.metrics"
    result["public_leaderboard_used"] = False
    result["total_runtime_seconds"] = round(time.time() - t0, 2)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                        encoding="utf-8")
    log("Exp18 %s; composite delta %+.6f" %
        (result["decision"], result["deltas"]["composite"]))
    (HERE / "run.log").write_text("\n".join(log_lines) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
