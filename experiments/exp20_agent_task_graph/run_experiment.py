from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for path in (ROOT, os.path.join(ROOT, "experiments", "exp12_fix_window_run_mapping"),
             os.path.join(ROOT, "experiments", "exp13_wait_dependency_graph"),
             os.path.join(ROOT, "experiments", "exp07_fault_windows"),
             os.path.join(ROOT, "experiments", "exp10_ab_probability_blend"),
             os.path.join(ROOT, "experiments", "exp08_window_localizer"),
             os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "baseline")):
    if path not in sys.path:
        sys.path.insert(0, path)

import lightgbm as lgb  # noqa: E402
from baseline.features import parse_run  # noqa: E402
from experiments.exp07_fault_windows.common import lgb_label  # noqa: E402
from experiments.exp12_fix_window_run_mapping import mapping_fix as M  # noqa: E402
from experiments.exp13_wait_dependency_graph import production_confirmation as E13C  # noqa: E402
from experiments.exp20_agent_task_graph.features import (  # noqa: E402
    FEATURE_NAMES, build_matrix,
)
from experiments.exp20_agent_task_graph.scoring import (  # noqa: E402
    l1_turns, official_scores,
)
from experiments.exp10_ab_probability_blend.blend import LABELS  # noqa: E402

EXP13 = os.path.join(ROOT, "experiments", "exp13_wait_dependency_graph")
EXP15 = os.path.join(ROOT, "experiments", "exp15_success_label_stacking")
TRAIN_CSV = os.path.join(ROOT, "data", "train.csv")
TEST_CSV = os.path.join(ROOT, "data", "test.csv")
BASELINE_RUNTIME_SEC = 593.4
RUNTIME_LIMIT_SEC = 1800.0
RAM_LIMIT_MB = 16 * 1024
LOG_LINES = []


def log(message=""):
    print(message, flush=True)
    LOG_LINES.append(str(message))


def digest_json(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rss_mb():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(Counters),
                                               wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)
    except Exception as exc:
        raise RuntimeError("peak RSS measurement is unavailable") from exc


def benchmark_runtime():
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        raise FileNotFoundError("train.csv or test.csv is absent; runtime gate cannot be measured")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    if len(train) != 10000 or len(test) != 4000:
        raise AssertionError("runtime benchmark requires 10,000 train and 4,000 test rows")

    sample = pd.concat([
        train.sample(n=500, random_state=42),
        test.sample(n=500, random_state=43),
    ], ignore_index=True)
    runs = [parse_run(row) for row in sample.to_dict("records")]
    event_count = sum(len(r.get("messages") or []) + len(r.get("artifacts") or [])
                      + len(r.get("shared_state") or []) for r in runs)
    t0 = time.perf_counter()
    graph = build_matrix(runs)
    extraction_sec = time.perf_counter() - t0
    if graph.shape != (1000, len(FEATURE_NAMES)):
        raise AssertionError("benchmark extraction dropped runs or changed feature width")

    # Measure only the model overhead added by 24 columns against the current
    # 312-column model size. Training is sequential; no two fits overlap.
    rng = np.random.RandomState(42)
    y = np.asarray([LABELS.index(x) for x in train["label"].values], dtype=np.int64)
    X312 = rng.normal(size=(10000, 312)).astype(np.float32)
    Xb_extra = rng.normal(size=(10000, len(FEATURE_NAMES))).astype(np.float32)
    Xt312 = rng.normal(size=(4000, 312)).astype(np.float32)
    Xt_extra = rng.normal(size=(4000, len(FEATURE_NAMES))).astype(np.float32)
    params = lgb_label().get_params()
    fit = {}
    for name, Xtr, Xte in (
            ("control_312", X312, Xt312),
            ("candidate_336", np.hstack([X312, Xb_extra]),
             np.hstack([Xt312, Xt_extra]))):
        start = time.perf_counter()
        model = lgb.LGBMClassifier(**params).fit(Xtr, y)
        fit_start = time.perf_counter()
        pred = model.predict(Xte)
        infer_end = time.perf_counter()
        fit[name] = {"fit_sec": fit_start - start,
                     "infer_4000_sec": infer_end - fit_start,
                     "rows_train": len(Xtr), "rows_test": len(Xte),
                     "features": Xtr.shape[1],
                     "predicted_rows": int(len(pred))}
        del model, pred

    graph_full_estimate = extraction_sec * (14000.0 / len(runs))
    model_full_delta = max(0.0, (fit["candidate_336"]["fit_sec"]
                                 + fit["candidate_336"]["infer_4000_sec"])
                            - (fit["control_312"]["fit_sec"]
                               + fit["control_312"]["infer_4000_sec"]))
    total_estimate = BASELINE_RUNTIME_SEC + graph_full_estimate + model_full_delta
    peak = rss_mb()
    result = {
        "baseline_production_runtime_sec": BASELINE_RUNTIME_SEC,
        "source": "Exp15 packaged smoke record in experiments/results.csv",
        "logical_cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "sample_runs": len(runs), "sample_events": event_count,
        "feature_extraction_sec": extraction_sec,
        "runs_per_sec": len(runs) / extraction_sec if extraction_sec else None,
        "events_per_sec": event_count / extraction_sec if extraction_sec else None,
        "estimated_extraction_14000_sec": graph_full_estimate,
        "lightgbm_batch": fit,
        "estimated_incremental_model_sec": model_full_delta,
        "estimated_total_production_sec": total_estimate,
        "runtime_limit_sec": RUNTIME_LIMIT_SEC,
        "peak_rss_mb": peak,
        "ram_limit_mb": RAM_LIMIT_MB,
        "ready": bool(total_estimate <= RUNTIME_LIMIT_SEC and peak <= RAM_LIMIT_MB),
        "warning": "Model timing uses a deterministic full-size synthetic matrix; it estimates incremental LightGBM cost only. No research CV time is treated as production time.",
    }
    with open(os.path.join(HERE, "preflight.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    with open(os.path.join(HERE, "preflight.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")
    return result


def verify_fold_partition(folds, n):
    owner = np.zeros(n, dtype=np.int8)
    all_rows = np.arange(n, dtype=np.int64)
    for fold_id, val in enumerate(folds):
        val = np.asarray(val, dtype=np.int64)
        if len(np.unique(val)) != len(val):
            raise AssertionError("fold %d has duplicate validation rows" % fold_id)
        train = np.setdiff1d(all_rows, val, assume_unique=False)
        if np.intersect1d(train, val).size:
            raise AssertionError("fold %d train/validation overlap" % fold_id)
        owner[val] += 1
    if len(folds) != 3 or not np.all(owner == 1):
        raise AssertionError("folds do not cover every row exactly once")
    return owner


def target_leakage_probe(runs, n=200):
    probe = runs[:n]
    base = build_matrix(probe)
    mutated = []
    for i, run in enumerate(probe):
        item = copy.deepcopy(run)
        item["label"] = LABELS[i % len(LABELS)]
        item["success"] = i % 2
        item["fault_turn"] = i * 3
        mutated.append(item)
    changed = build_matrix(mutated)
    return {"rows": n, "identical": bool(np.array_equal(base, changed)),
            "max_abs_diff": float(np.max(np.abs(base - changed))) if base.size else 0.0}


def rename_agent_ids(run):
    item = copy.deepcopy(run)
    ids = set()
    for agent in item.get("agents") or []:
        if agent.get("id") is not None:
            ids.add(str(agent["id"]))
    for msg in item.get("messages") or []:
        for key in ("from", "to"):
            if msg.get(key) is not None:
                ids.add(str(msg[key]))
    for art in item.get("artifacts") or []:
        if art.get("by") is not None:
            ids.add(str(art["by"]))
    for state in item.get("shared_state") or []:
        if state.get("agent") is not None:
            ids.add(str(state["agent"]))
    mapping = {old: "a%d" % (100000 + i) for i, old in enumerate(sorted(ids))}
    for agent in item.get("agents") or []:
        if agent.get("id") is not None:
            agent["id"] = mapping[str(agent["id"])]
    for msg in item.get("messages") or []:
        for key in ("from", "to"):
            if msg.get(key) is not None:
                msg[key] = mapping[str(msg[key])]
        if msg.get("text"):
            text = str(msg["text"])
            for old, new in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
                text = re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(old),
                              new, text, flags=re.IGNORECASE)
            msg["text"] = text
    for art in item.get("artifacts") or []:
        if art.get("by") is not None:
            art["by"] = mapping[str(art["by"])]
    for state in item.get("shared_state") or []:
        if state.get("agent") is not None:
            state["agent"] = mapping[str(state["agent"])]
    return item


def id_rename_probe(runs, n=200):
    original = build_matrix(runs[:n])
    renamed = build_matrix([rename_agent_ids(run) for run in runs[:n]])
    return {"rows": n, "identical": bool(np.array_equal(original, renamed)),
            "max_abs_diff": float(np.max(np.abs(original - renamed))) if original.size else 0.0}


def duplicate_diagnostic(base, graph):
    exact = []
    correlations = []
    for j, name in enumerate(FEATURE_NAMES):
        values = graph[:, j]
        dup = [int(i) for i in range(base.shape[1])
               if np.array_equal(values, base[:, i])]
        if dup:
            exact.append({"feature": name, "base_columns": dup})
        if len(np.unique(values)) < 2:
            correlations.append({"feature": name, "max_abs_r": None,
                                 "base_feature": None,
                                 "unique_values": int(len(np.unique(values)))})
            continue
        best_r, best_name = 0.0, None
        for i in range(base.shape[1]):
            old = base[:, i]
            if len(np.unique(old)) < 2:
                continue
            corr = float(np.corrcoef(values, old)[0, 1])
            if np.isfinite(corr) and abs(corr) > abs(best_r):
                best_r, best_name = corr, int(i)
        correlations.append({"feature": name, "max_abs_r": abs(best_r),
                             "base_feature_index": best_name,
                             "unique_values": int(len(np.unique(values)))})
    has_novel_variable = any(len(np.unique(graph[:, j])) > 1
                             and not any(x["feature"] == FEATURE_NAMES[j] for x in exact)
                             for j in range(graph.shape[1]))
    return {"exact_duplicates": exact, "correlations": correlations,
            "has_novel_variable": bool(has_novel_variable)}


def predict_fold_oof(fold_matrices, yi, params, tag, graph=None):
    n = len(yi)
    out = np.zeros((n, len(LABELS)), dtype=np.float64)
    for fold_id, train, val, Xtrain, Xval in fold_matrices:
        train = np.asarray(train, dtype=np.int64)
        val = np.asarray(val, dtype=np.int64)
        if np.intersect1d(train, val).size:
            raise AssertionError("train/validation overlap in fold %d" % fold_id)
        if graph is not None:
            names = list(Xtrain.columns) + list(FEATURE_NAMES)
            Xtrain = pd.DataFrame(np.hstack([Xtrain.values, graph[train]]),
                                  columns=names)
            Xval = pd.DataFrame(np.hstack([Xval.values, graph[val]]),
                                columns=names)
        t0 = time.perf_counter()
        model = lgb.LGBMClassifier(**params).fit(Xtrain, yi[train])
        out[val] = model.predict_proba(Xval)
        log("%s fold %d complete: train=%d val=%d elapsed=%.1fs"
            % (tag, fold_id, len(train), len(val), time.perf_counter() - t0))
        del model
    return out


def fold_metrics(y_label, pred, y_success, success_pred, true_turn, pred_turn,
                 robust_mask, folds):
    out = []
    for fold_id, val in enumerate(folds):
        idx = np.asarray(val, dtype=np.int64)
        scores = official_scores(y_label[idx], pred[idx], y_success[idx],
                                 success_pred[idx], true_turn[idx],
                                 pred_turn[idx], robust_mask[idx])
        out.append({"fold": fold_id, "n_val": int(len(idx)), "metrics": scores})
    return out


def write_summary(result, text):
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")
    with open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def append_history(verdict, comment, metrics_a=None, metrics_b=None):
    path = os.path.join(ROOT, "experiments", "results.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if any(row and row[0].strip() == "exp20" for row in rows):
        raise RuntimeError("Exp20 already exists in experiments/results.csv")
    def value(metrics, key):
        return "" if metrics is None else "%.10f" % metrics[key]
    row = ["exp20", "agent_task_graph",
           value(metrics_a, "macro_f1"), value(metrics_b, "macro_f1"),
           ("" if metrics_a is None or metrics_b is None else
            "%.10f" % (metrics_b["macro_f1"] - metrics_a["macro_f1"])),
           value(metrics_a, "robustness_f1"), value(metrics_b, "robustness_f1"),
           ("" if metrics_a is None or metrics_b is None else
            "%.10f" % (metrics_b["robustness_f1"] - metrics_a["robustness_f1"])),
           verdict, comment, "experiments/exp20_agent_task_graph/RESULTS.md"]
    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(row)


def stop_preflight(reason, details=None):
    log("STOP before candidate CV: " + reason)
    details = details or {}
    control_a = details.get("control_A", {})
    result = {"experiment": "exp20", "decision": "STOP_PREFLIGHT",
              "reason": reason, "details": details,
              "control_A_cv_completed": bool(control_a.get("completed", False)),
              "candidate_B_cv_started": False, "candidate_B_trained": False,
              "promotion_gate": {"passed": False, "evaluated": False,
                                 "reason": "baseline parity failed before Candidate B"},
              "production_changed": False, "zip_created": False,
              "leaderboard_used": False}
    report = ["# Exp20 — preflight STOP", "", reason, ""]
    if control_a.get("completed"):
        report += ["Control A was fitted on the fixed outer folds. Candidate B was not trained.", "",
                   "| Metric | Control A | Frozen Exp15 | Absolute difference |", "|---|---:|---:|---:|"]
        for name, values in control_a["metrics"].items():
            report.append("| %s | %.12f | %.12f | %.12g |" %
                          (name, values["got"], values["expected"], values["abs_diff"]))
        report += ["", "- Label prediction mismatches: %d / 10000." % control_a["label_mismatches"],
                   "- Mismatch counts by outer fold: %s." % control_a["mismatches_by_fold"],
                   "- L1 fault-turn outputs identical: %s." % control_a["turns_identical"], ""]
    report += ["No candidate CV was run. Exp15 production remains unchanged."]
    runtime = details.get("runtime")
    if runtime:
        report += ["", "## Runtime", "",
                   "- Corrected incumbent matrix build: %.1f s." % runtime["matrix_build_sec"],
                   "- Control A fits: %.1f s total." % runtime["control_A_fit_sec"],
                   "- CPU production estimate: %.1f s / 1800 s; benchmark peak RSS %.1f MB." %
                   (runtime["production_estimate_sec"], runtime["benchmark_peak_rss_mb"]),
                   "- This research preflight timing is not a full candidate CV or production run."]
    write_summary(result, "\n".join(report))
    history_path = os.path.join(ROOT, "experiments", "results.csv")
    with open(history_path, newline="", encoding="utf-8") as fh:
        history = list(csv.reader(fh))
    if not any(row and row[0].strip() == "exp20" for row in history):
        append_history("not_promoting",
                       "STOP before Candidate B: %s; no candidate metrics; Exp15 unchanged." % reason)
    return 2


def run_cv():
    preflight_path = os.path.join(HERE, "preflight.json")
    if not os.path.exists(preflight_path):
        return stop_preflight("runtime benchmark is missing")
    preflight = read_json(preflight_path)
    if not preflight.get("ready"):
        return stop_preflight("runtime/RAM gate failed", preflight)

    t0 = time.perf_counter()
    log("EXP20 - Temporal Agent-Task Graph; fixed plan committed before CV")
    d = M.load_verified()
    train = d["train"]
    ids = [str(x) for x in train["run_id"].values]
    with open(os.path.join(EXP15, "results.json"), encoding="utf-8") as fh:
        exp15 = json.load(fh)
    rid_sha = M.ca.sha256_run_ids(ids)
    if rid_sha != exp15["run_id_alignment"]["run_id_sha256"]:
        return stop_preflight("train run_id order differs from frozen Exp15 outputs",
                              {"got_sha256": rid_sha,
                               "expected_sha256": exp15["run_id_alignment"]["run_id_sha256"]})
    if len(ids) != 10000 or len(set(ids)) != len(ids):
        return stop_preflight("train row count or run_id uniqueness check failed")
    folds = d["folds"]
    verify_fold_partition(folds, len(ids))

    c, Wd = M._prepare()
    if len(c["runs"]) != len(ids):
        return stop_preflight("parsed train runs are not aligned to train.csv")
    graph = build_matrix(c["runs"])
    if graph.shape != (len(ids), len(FEATURE_NAMES)):
        return stop_preflight("graph feature matrix dropped rows or changed width",
                              {"shape": list(graph.shape)})
    leakage = target_leakage_probe(c["runs"])
    rename = id_rename_probe(c["runs"])
    if not leakage["identical"] or not rename["identical"]:
        return stop_preflight("target leakage or agent-ID dependence detected",
                              {"leakage": leakage, "id_rename": rename})

    build_start = time.perf_counter()
    Xwg = E13C.W.build_matrix(c["runs"])
    if Xwg.shape != (len(ids), 12):
        return stop_preflight("frozen Exp13 wait-graph matrix has wrong shape",
                              {"shape": list(Xwg.shape)})
    fold_matrices = []
    wg_names = list(E13C.W.FEATURE_NAMES)
    base_oof = np.zeros((len(ids), 312), dtype=np.float64)
    base_cols = None
    base_names = None
    for fold_id, tr, va, Xtr, Xva in E13C.corrected_fold_matrices(folds, c, Wd):
        cols = list(Xtr.columns)
        if len(cols) != 300 or list(Xva.columns) != cols:
            return stop_preflight("corrected incumbent builder did not return 300 ordered columns",
                                  {"fold": fold_id, "train_cols": len(cols),
                                   "val_cols": len(Xva.columns)})
        if base_cols is not None and base_cols != cols:
            return stop_preflight("incumbent feature order changes across folds")
        base_cols = cols
        base_names = cols + wg_names
        Xtr_array = Xtr.values.astype(np.float64, copy=False)
        Xva_array = Xva.values.astype(np.float64, copy=False)
        Xtr_312 = pd.DataFrame(np.hstack([Xtr_array, Xwg[tr]]), columns=base_names)
        Xva_312 = pd.DataFrame(np.hstack([Xva_array, Xwg[va]]), columns=base_names)
        base_oof[va] = Xva_312.values
        # Reuse the exact corrected 312-column matrices for both A and B.
        fold_matrices.append((fold_id, np.asarray(tr, dtype=np.int64),
                              np.asarray(va, dtype=np.int64), Xtr_312, Xva_312))
    if len(fold_matrices) != 3 or base_oof.shape != (len(ids), 312):
        return stop_preflight("corrected incumbent matrices are incomplete",
                              {"folds": len(fold_matrices), "shape": list(base_oof.shape)})
    if len(base_names) != 312:
        return stop_preflight("incumbent feature matrix is not exactly 312 columns",
                              {"names": len(base_names)})
    overlap = duplicate_diagnostic(base_oof, graph)
    if overlap["exact_duplicates"] or not overlap["has_novel_variable"]:
        return stop_preflight("graph block has exact duplicates or no variable novel column",
                              overlap)
    log("corrected fold-local incumbent 312 + graph %d matrices ready in %.1fs"
        % (graph.shape[1], time.perf_counter() - build_start))

    success_pred = np.load(os.path.join(EXP15, "success_pred_B.npy"), allow_pickle=False)
    success_true = train["success"].to_numpy(dtype=np.int64)
    y_label = train["label"].to_numpy(dtype=object)
    y_code = d["yi"].astype(np.int64)
    true_turn = train["fault_turn"].to_numpy(dtype=np.int64)
    robust = M.robustness_mask()
    frozen_D = np.load(os.path.join(EXP13, "oof_D_corrected_wg.npy"), allow_pickle=False)
    if (success_pred.shape != (len(ids),) or frozen_D.shape != (len(ids), len(LABELS))
            or not np.isin(success_pred, [0, 1]).all()):
        return stop_preflight("frozen Exp15 success or Exp13 label OOF shape invalid")
    saved_pred = frozen_D.argmax(1).astype(np.int64)
    params = M.label_params()

    log("Control A: fitting unchanged Exp15 312-feature model across the fixed folds")
    p_a = predict_fold_oof(fold_matrices, y_code, params, "Control A")
    pred_a = p_a.argmax(1).astype(np.int64)
    turns_a = l1_turns(pred_a, d["peak_pos"])
    scores_a = official_scores(y_label, np.asarray(LABELS)[pred_a], success_true,
                               success_pred, true_turn, turns_a, robust)
    expected = exp15["metrics_B"]
    parity_metrics = {key: {"got": float(scores_a[key]), "expected": float(expected[key]),
                            "abs_diff": abs(float(scores_a[key]) - float(expected[key])),
                            "pass": abs(float(scores_a[key]) - float(expected[key])) <= 1e-12}
                      for key in ("macro_f1", "robustness_f1", "success_f1",
                                  "fault_turn_hit2", "composite")}
    parity_labels = bool(np.array_equal(pred_a, saved_pred))
    baseline_turns = l1_turns(saved_pred, d["peak_pos"])
    parity_turns = bool(np.array_equal(turns_a, baseline_turns))
    if not parity_labels or not parity_turns or not all(x["pass"] for x in parity_metrics.values()):
        control_a = {"completed": True, "metrics": parity_metrics,
                     "label_mismatches": int(np.sum(pred_a != saved_pred)),
                     "mismatches_by_fold": {
                         str(f): int(np.sum((pred_a != saved_pred) &
                                            np.isin(np.arange(len(ids)), va)))
                         for f, va in enumerate(folds)},
                     "turns_identical": parity_turns}
        np.savez_compressed(os.path.join(HERE, "control_a_debug.npz"),
                            run_id=np.asarray(ids),
                            baseline_pred=saved_pred, control_pred=pred_a,
                            baseline_proba=frozen_D, control_proba=p_a,
                            fold_of=np.argmax(np.stack([
                                np.isin(np.arange(len(ids)), f) for f in folds]), axis=0))
        return stop_preflight("Control A did not reproduce frozen Exp15 baseline",
                              {"metrics": parity_metrics,
                               "labels_identical": parity_labels,
                               "control_A": control_a,
                               "turns_identical": parity_turns})
    log("Control A parity PASS; all five metrics, classes, and L1 turns match")

    log("Candidate B: fixed 336-column model; same folds and parameters")
    p_b = predict_fold_oof(fold_matrices, y_code, params, "Candidate B", graph=graph)
    pred_b = p_b.argmax(1).astype(np.int64)
    turns_b = l1_turns(pred_b, d["peak_pos"])
    scores_b = official_scores(y_label, np.asarray(LABELS)[pred_b], success_true,
                               success_pred, true_turn, turns_b, robust)
    classes_a = {name: scores_a["per_class_f1"][name] for name in LABELS}
    classes_b = {name: scores_b["per_class_f1"][name] for name in LABELS}
    fold_a = fold_metrics(y_label, np.asarray(LABELS)[pred_a], success_true,
                          success_pred, true_turn, turns_a, robust, folds)
    fold_b = fold_metrics(y_label, np.asarray(LABELS)[pred_b], success_true,
                          success_pred, true_turn, turns_b, robust, folds)
    folds_out = []
    for a, b in zip(fold_a, fold_b):
        folds_out.append({"fold": a["fold"], "n_val": a["n_val"],
                          "control_A": a["metrics"], "candidate_B": b["metrics"],
                          "composite_delta": b["metrics"]["composite"]
                          - a["metrics"]["composite"]})
    class_delta = {name: classes_b[name] - classes_a[name] for name in LABELS}
    folds_won = sum(item["composite_delta"] > 0 for item in folds_out)
    gate = {
        "composite_delta_ge_0_005": scores_b["composite"] - scores_a["composite"] >= 0.005,
        "macro_not_lower": scores_b["macro_f1"] >= scores_a["macro_f1"],
        "robustness_not_lower": scores_b["robustness_f1"] >= scores_a["robustness_f1"],
        "dropped_handoff_delta_ge_0_01": class_delta["dropped_handoff"] >= 0.01,
        "composite_folds_won_at_least_2": folds_won >= 2,
        "no_class_drop_over_0_02": min(class_delta.values()) >= -0.02,
        "no_leakage_fold_overlap_baseline_or_success_change": bool(
            leakage["identical"] and rename["identical"] and parity_labels and parity_turns),
    }
    decision = "PROMOTE" if all(gate.values()) else "STOP"
    elapsed = time.perf_counter() - t0
    peak = rss_mb()
    np.savez_compressed(os.path.join(HERE, "oof_predictions.npz"),
                        run_id=np.asarray(ids), y_true=y_label,
                        pred_A=np.asarray(LABELS)[pred_a],
                        pred_B=np.asarray(LABELS)[pred_b],
                        success_pred=success_pred, fault_turn_A=turns_a,
                        fault_turn_B=turns_b, fold=np.argmax(
                            np.stack([np.isin(np.arange(len(ids)), f) for f in folds]), axis=0))
    result = {
        "experiment": "exp20", "decision": decision, "cv_started": True,
        "plan_commits": ["51843be", "e492f8f", "c73691a", "ca723f7"],
        "run_id_sha256": rid_sha, "n_rows": len(ids),
        "fold_validation_sha256": d["fold_sha"],
        "features": list(FEATURE_NAMES), "n_graph_features": len(FEATURE_NAMES),
        "novelty_diagnostic": overlap, "target_leakage_probe": leakage,
        "agent_id_rename_probe": rename,
        "control_A_parity": {"passed": True, "metrics": parity_metrics,
                             "labels_identical": parity_labels,
                             "l1_turns_identical": parity_turns},
        "metrics_A": scores_a, "metrics_B": scores_b,
        "deltas": {key: scores_b[key] - scores_a[key]
                   for key in ("macro_f1", "robustness_f1", "success_f1",
                               "fault_turn_hit2", "composite")},
        "class_f1_A": classes_a, "class_f1_B": classes_b,
        "class_f1_delta": class_delta, "folds": folds_out,
        "folds_won": int(folds_won), "gate": gate,
        "gate_passed": bool(all(gate.values())),
        "runtime": {"research_cv_sec": elapsed,
                    "production_estimate": preflight,
                    "peak_rss_mb": peak},
        "params": {key: value for key, value in params.items()
                   if isinstance(value, (int, float, str, bool))},
        "success_predictions_frozen": True,
        "production_changed": False, "zip_created": False,
        "leaderboard_used": False,
    }
    with open(os.path.join(HERE, "oof_predictions.csv"), "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "y_true", "pred_A", "pred_B", "success_pred",
                         "fault_turn_A", "fault_turn_B", "fold"])
        fold_of = np.empty(len(ids), dtype=np.int8)
        for f, val in enumerate(folds):
            fold_of[np.asarray(val, dtype=np.int64)] = f
        for i in range(len(ids)):
            writer.writerow([ids[i], y_label[i], LABELS[pred_a[i]], LABELS[pred_b[i]],
                             int(success_pred[i]), int(turns_a[i]), int(turns_b[i]),
                             int(fold_of[i])])

    details = ["# Exp20 — Temporal Agent-Task Graph", "",
               "**Decision:** %s. Exp15 production remains unchanged." % decision,
               "", "## OOF metrics", "",
               "| Metric | Exp15 / Control A | Candidate B | Delta |",
               "|---|---:|---:|---:|"]
    for key in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2", "composite"):
        details.append("| %s | %.10f | %.10f | %+.10f |" %
                       (key, scores_a[key], scores_b[key], scores_b[key] - scores_a[key]))
    details += ["", "## F1 by class", "", "| Class | Control A | Candidate B | Delta |",
                "|---|---:|---:|---:|"]
    for name in LABELS:
        details.append("| %s | %.6f | %.6f | %+.6f |" %
                       (name, classes_a[name], classes_b[name], class_delta[name]))
    details += ["", "## Fold metrics", "", "| Fold | N | A composite | B composite | Delta |",
                "|---:|---:|---:|---:|---:|"]
    for item in folds_out:
        details.append("| %d | %d | %.6f | %.6f | %+.6f |" %
                       (item["fold"], item["n_val"],
                        item["control_A"]["composite"],
                        item["candidate_B"]["composite"],
                        item["composite_delta"]))
    details += ["", "## Frozen gate", ""]
    details += ["- %s: %s" % (key, "PASS" if value else "FAIL")
                for key, value in gate.items()]
    details += ["", "## Runtime and diagnostics", "",
                "- Research CV: %.1f s; production estimate is reported separately below." % elapsed,
                "- Production estimate: %.1f s / 1800 s." % preflight["estimated_total_production_sec"],
                "- Peak working set: %.1f MB / %d MB." % (peak, RAM_LIMIT_MB),
                "- Exact graph/base duplicates: %d." % len(overlap["exact_duplicates"]),
                "- Artifact/state event order: message → state → artifact at equal `t`.",
                "- No encoder, Exp19, production ZIP, leaderboard, or model search used."]
    comment = ("%s; composite %.10f (%+.10f), dropped_handoff F1 %+.6f; "
               "fixed 24-column temporal agent-task graph; no production package." %
               (decision, scores_b["composite"], scores_b["composite"] - scores_a["composite"],
                class_delta["dropped_handoff"]))
    write_summary(result, "\n".join(details))
    append_history("promoting" if decision == "PROMOTE" else "not_promoting",
                   comment, scores_a, scores_b)
    log("Exp20 %s; composite delta=%+.6f; research CV=%.1fs; RSS=%.1fMB"
        % (decision, scores_b["composite"] - scores_a["composite"], elapsed, peak))
    with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")
    return 0 if decision == "PROMOTE" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        result = benchmark_runtime()
        log(json.dumps(result, indent=2, sort_keys=True))
        with open(os.path.join(HERE, "preflight.log"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(LOG_LINES) + "\n")
        return 0 if result["ready"] else 2
    return run_cv()


if __name__ == "__main__":
    raise SystemExit(main())
