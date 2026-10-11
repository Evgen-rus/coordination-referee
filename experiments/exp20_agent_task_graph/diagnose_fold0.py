"""Isolated, fixed-condition Control A replay for Exp20 fold 0 only."""
from __future__ import annotations

import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "diagnostics", "control_a_fold0_20261011")
for path in (ROOT, os.path.join(ROOT, "experiments", "exp12_fix_window_run_mapping"),
             os.path.join(ROOT, "experiments", "exp13_wait_dependency_graph"),
             os.path.join(ROOT, "experiments", "exp07_fault_windows"),
             os.path.join(ROOT, "experiments", "exp10_ab_probability_blend"),
             os.path.join(ROOT, "experiments", "exp08_window_localizer"),
             os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "baseline")):
    if path not in sys.path:
        sys.path.insert(0, path)

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from experiments.exp10_ab_probability_blend.blend import LABELS  # noqa: E402
from experiments.exp12_fix_window_run_mapping import mapping_fix as M  # noqa: E402
from experiments.exp13_wait_dependency_graph import production_confirmation as E13  # noqa: E402
from experiments.exp20_agent_task_graph import run_experiment as E20  # noqa: E402


def sha_array(value):
    return M.ca.sha256_array(np.ascontiguousarray(np.asarray(value)))


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=str).encode("utf-8")).hexdigest()


def main():
    if os.path.exists(OUT):
        raise FileExistsError("refusing to overwrite diagnostic directory: " + OUT)
    os.makedirs(OUT)
    log_path = os.path.join(OUT, "diagnostic.log")
    record_path = os.path.join(OUT, "diagnostics.json")
    prob_path = os.path.join(OUT, "fold0_probabilities.npz")
    mismatch_path = os.path.join(OUT, "fold0_mismatches.csv")
    exp13 = os.path.join(ROOT, "experiments", "exp13_wait_dependency_graph")
    exp20 = os.path.join(HERE, "control_a_debug.npz")

    if not os.path.isfile(exp20):
        raise FileNotFoundError("the saved Exp20 control diagnostic is missing")
    d = M.load_verified()
    manifest = d["manifest"]
    ids = np.asarray(d["run_ids"], dtype=str)
    folds = [np.asarray(x, dtype=np.int64) for x in d["folds"]]
    saved = np.load(exp20, allow_pickle=False)
    frozen_debug = saved["baseline_proba"]
    old_control = saved["control_proba"]
    frozen = np.load(os.path.join(exp13, "oof_D_corrected_wg.npy"), allow_pickle=False)
    if not np.array_equal(ids, saved["run_id"]):
        raise AssertionError("train run_id order differs from the saved Exp20 diagnostic")
    if not np.array_equal(frozen, frozen_debug):
        raise AssertionError("Exp20 frozen probabilities differ from saved Exp13 OOF")
    if not np.array_equal(folds[0], np.flatnonzero(saved["fold_of"] == 0)):
        raise AssertionError("saved fold-0 membership differs from the verified seed-0 fold")

    cache_paths = [M.ca._path("build_all"), M.ca._path("windows")]
    if not all(os.path.isfile(p) for p in cache_paths):
        raise FileNotFoundError("verified current Exp07 input caches are absent; refusing to rebuild")

    def log(message):
        line = str(message)
        print(line, flush=True)

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file):
        log("Exp20 Control A isolated fold-0 diagnostic; no Candidate B")
        log("Started UTC: " + dt.datetime.now(dt.timezone.utc).isoformat())
        log("Train CSV SHA256: " + M.ca.data_sha256())
        log("Verified manifest data SHA256: " + str(manifest["data_sha256"]))
        log("Run ID SHA256: " + M.ca.sha256_run_ids(ids))
        log("Fold 0 validation SHA256: " + M.ca.sha256_indices(folds[0]))
        log("Outer fold sizes: " + repr([len(x) for x in folds]))
        log("Verified manifest code SHA256: " + str(manifest["code_sha256"]))
        log("Current Exp07 code SHA256: " + M.ca.code_sha256())
        log("Current cache fingerprint: " + M.ca._fingerprint())
        log("Cache files: " + repr(cache_paths))
        log("Python=%s LightGBM=%s NumPy=%s scikit-learn=%s joblib=%s" %
            (platform.python_version(), lgb.__version__, np.__version__,
             sklearn.__version__, joblib.__version__))
        log("CPU logical=%s joblib=%s" % (os.cpu_count(), joblib.cpu_count()))
        omp = {k: os.environ.get(k) for k in
               ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "LOKY_MAX_CPU_COUNT")}
        log("Thread environment: " + json.dumps(omp, sort_keys=True))
        log("Fold construction: seed-0 StratifiedKFold(3); window inner fits: workers=3 default, threads=-1")
        log("Current corrected_fold_matrices source: " + M.ca.sha256_file(E13.__file__))
        log("Current Exp20 source: " + M.ca.sha256_file(E20.__file__))

        # Load only the existing content-addressed feature caches; no cache rebuild.
        c, Wd = M._prepare()
        log("Foundation matrix: shape=%s dtype=%s sha256=%s" %
            (c["foundation"].shape, c["foundation"].to_numpy().dtype,
             sha_array(c["foundation"].to_numpy())))
        log("Foundation column names SHA256: " + sha_json(list(c["foundation"].columns)))
        log("Window dataset X: shape=%s dtype=%s sha256=%s" %
            (Wd["X"].shape, np.asarray(Wd["X"]).dtype, sha_array(Wd["X"])))
        log("Window targets/run-index SHA256: %s / %s" %
            (sha_array(Wd["y"]), sha_array(Wd["run"])))
        Xwg = E13.W.build_matrix(c["runs"])
        log("Frozen Exp13 wg_* matrix: shape=%s dtype=%s sha256=%s" %
            (Xwg.shape, Xwg.dtype, sha_array(Xwg)))

        runs = []
        inner_prob_arrays = {}
        original_inner = E13.par.run_inner_fits
        original_aggregate = E13.M.grp.aggregate_block
        original_group = E13.M.group_fixed
        original_classifier = lgb.LGBMClassifier

        for repeat in (1, 2):
            repeat_start = time.perf_counter()
            stage = {"repeat": repeat, "aggregate_calls": [],
                     "inner_probability_hashes": {}, "outer_window_probability": None}

            def capture_inner(payloads, *args, **kwargs):
                payloads = list(payloads)
                out = original_inner(payloads, *args, **kwargs)
                for key, probs in sorted(out.items()):
                    arr = np.asarray(probs)
                    name = "r%d_inner_f%d_j%d" % (repeat, int(key[0]), int(key[1]))
                    inner_prob_arrays[name] = arr.copy()
                    stage["inner_probability_hashes"][name] = {
                        "shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": sha_array(arr)}
                return out

            def capture_group(probs, rw_sel, g2l, n_tr_runs):
                blocks, slot = original_group(probs, rw_sel, g2l, n_tr_runs)
                block_summary = [(int(r), int(k), sha_array(p))
                                 for r, (p, k) in sorted(blocks.items())]
                stage.setdefault("group_fixed_calls", []).append({
                    "selected_window_run_sha256": sha_array(rw_sel),
                    "blocks_sha256": sha_json(block_summary),
                    "slot_sha256": sha_json(slot), "n_runs": int(n_tr_runs)})
                return blocks, slot

            def capture_aggregate(blocks, agg_mod, slot, n_out):
                result = original_aggregate(blocks, agg_mod, slot, n_out)
                matrix = np.asarray(result[0])
                stage["aggregate_calls"].append({
                    "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                    "sha256": sha_array(matrix)})
                return result

            class TrackedLGBM(original_classifier):
                def fit(self, X, y, *args, **kwargs):
                    self._exp20_fit_width = int(X.shape[1])
                    self._exp20_fit_rows = int(X.shape[0])
                    return super().fit(X, y, *args, **kwargs)

                def predict_proba(self, X):
                    probs = super().predict_proba(X)
                    if getattr(self, "_exp20_fit_width", 0) == Wd["X"].shape[1]:
                        arr = np.asarray(probs)
                        stage["outer_window_probability"] = {
                            "shape": list(arr.shape), "dtype": str(arr.dtype),
                            "sha256": sha_array(arr)}
                        inner_prob_arrays["r%d_outer_window_val" % repeat] = arr.copy()
                    return probs

            E13.par.run_inner_fits = capture_inner
            E13.M.group_fixed = capture_group
            E13.M.grp.aggregate_block = capture_aggregate
            lgb.LGBMClassifier = TrackedLGBM
            try:
                matrix_generator = E13.corrected_fold_matrices(
                    folds, c, Wd, log=lambda m: log("matrix: " + str(m)))
                fold_id, tr, va, Xtr300, Xva300 = next(matrix_generator)
                matrix_generator.close()
            finally:
                E13.par.run_inner_fits = original_inner
                E13.M.group_fixed = original_group
                E13.M.grp.aggregate_block = original_aggregate
                lgb.LGBMClassifier = original_classifier

            if fold_id != 0:
                raise AssertionError("isolated builder did not return fold 0")
            feature_names = list(Xtr300.columns) + list(E13.W.FEATURE_NAMES)
            Xtr312 = pd.DataFrame(np.hstack([Xtr300.values, Xwg[tr]]), columns=feature_names)
            Xva312 = pd.DataFrame(np.hstack([Xva300.values, Xwg[va]]), columns=feature_names)
            stage["fold"] = int(fold_id)
            stage["train_index_sha256"] = sha_array(tr)
            stage["validation_index_sha256"] = sha_array(va)
            stage["train_300_sha256"] = sha_array(Xtr300.to_numpy())
            stage["validation_300_sha256"] = sha_array(Xva300.to_numpy())
            stage["train_312_sha256"] = sha_array(Xtr312.to_numpy())
            stage["validation_312_sha256"] = sha_array(Xva312.to_numpy())
            stage["train_matrix_dtype"] = str(Xtr312.to_numpy().dtype)
            stage["validation_matrix_dtype"] = str(Xva312.to_numpy().dtype)
            params = M.label_params()
            if params != E13.ENS.label_params(42):
                raise AssertionError("Exp20 and Exp13 Control A LightGBM params differ")
            stage["label_params"] = {k: v for k, v in params.items()
                                     if isinstance(v, (str, int, float, bool)) or v is None}
            fold_matrices = [(0, tr, va, Xtr312, Xva312)]
            with contextlib.redirect_stdout(io.StringIO()):
                P_all = E20.predict_fold_oof(
                    fold_matrices, d["yi"].astype(np.int64), params,
                    "Control A isolated fold 0 repeat %d" % repeat)
            control = P_all[va].copy()
            frozen_fold = frozen[va].copy()
            prior_control = old_control[va].copy()
            pred_frozen = frozen_fold.argmax(1)
            pred_control = control.argmax(1)
            stage["classifier_probability"] = {
                "shape": list(control.shape), "dtype": str(control.dtype),
                "sha256": sha_array(control)}
            stage["prior_debug_control_sha256"] = sha_array(prior_control)
            stage["frozen_probability_sha256"] = sha_array(frozen_fold)
            stage["label_mismatches_vs_frozen"] = int(np.sum(pred_frozen != pred_control))
            stage["label_mismatches_vs_prior_control"] = int(
                np.sum(pred_control != prior_control.argmax(1)))
            stage["probability_identical_to_frozen"] = bool(np.array_equal(control, frozen_fold))
            stage["probability_identical_to_prior_control"] = bool(np.array_equal(control, prior_control))
            stage["float32_label_mismatches_vs_frozen"] = int(
                np.sum(pred_frozen != control.astype(np.float32).argmax(1)))
            stage["max_abs_probability_diff_vs_frozen"] = float(
                np.max(np.abs(control.astype(np.float64) - frozen_fold.astype(np.float64))))
            stage["elapsed_sec"] = round(time.perf_counter() - repeat_start, 3)
            stage["control_probability"] = control
            stage["frozen_probability"] = frozen_fold
            stage["prior_control_probability"] = prior_control
            stage["run_id"] = ids[va]
            runs.append(stage)
            log("repeat %d: labels differ vs frozen=%d/%d; vs previous Exp20 Control A=%d; max|dP|=%.9f" %
                (repeat, stage["label_mismatches_vs_frozen"], len(va),
                 stage["label_mismatches_vs_prior_control"],
                 stage["max_abs_probability_diff_vs_frozen"]))

        arrays = {}
        for stage in runs:
            repeat = stage["repeat"]
            arrays["run_id"] = stage["run_id"]
            arrays["frozen_probability"] = stage["frozen_probability"]
            arrays["prior_control_probability"] = stage["prior_control_probability"]
            arrays["control_probability_repeat_%d" % repeat] = stage["control_probability"]
        arrays.update(inner_prob_arrays)
        np.savez_compressed(prob_path, **arrays)

        mismatch_rows = []
        for local_i, run_id in enumerate(runs[0]["run_id"]):
            old_class = int(runs[0]["frozen_probability"][local_i].argmax())
            new_class = int(runs[0]["control_probability"][local_i].argmax())
            if old_class == new_class:
                continue
            b = runs[0]["frozen_probability"][local_i]
            p = runs[0]["control_probability"][local_i]
            top_b = np.argsort(b)[-2:][::-1]
            top_p = np.argsort(p)[-2:][::-1]
            mismatch_rows.append({
                "run_id": str(run_id), "frozen_class": LABELS[old_class],
                "repeat1_class": LABELS[new_class],
                "repeat2_class": LABELS[int(runs[1]["control_probability"][local_i].argmax())],
                "frozen_top1": LABELS[int(top_b[0])], "frozen_top1_p": float(b[top_b[0]]),
                "frozen_top2": LABELS[int(top_b[1])], "frozen_top2_p": float(b[top_b[1]]),
                "frozen_top_margin": float(b[top_b[0]] - b[top_b[1]]),
                "repeat1_top1": LABELS[int(top_p[0])], "repeat1_top1_p": float(p[top_p[0]]),
                "repeat1_top2": LABELS[int(top_p[1])], "repeat1_top2_p": float(p[top_p[1]]),
                "repeat1_top_margin": float(p[top_p[0]] - p[top_p[1]]),
                "max_abs_probability_diff": float(np.max(np.abs(
                    b.astype(np.float64) - p.astype(np.float64))))})
        with open(mismatch_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "run_id", "frozen_class", "repeat1_class", "repeat2_class",
                "frozen_top1", "frozen_top1_p", "frozen_top2", "frozen_top2_p",
                "frozen_top_margin", "repeat1_top1", "repeat1_top1_p", "repeat1_top2",
                "repeat1_top2_p", "repeat1_top_margin", "max_abs_probability_diff"])
            writer.writeheader()
            writer.writerows(mismatch_rows)

        json_runs = []
        for stage in runs:
            json_runs.append({k: v for k, v in stage.items()
                              if k not in {"control_probability", "frozen_probability",
                                           "prior_control_probability", "run_id"}})
        result = {
            "experiment": "exp20_control_a_fold0_diagnostic",
            "candidate_b_started": False,
            "full_cv_started": False,
            "input": {"data_sha256": M.ca.data_sha256(),
                      "manifest_data_sha256": manifest["data_sha256"],
                      "run_id_sha256": M.ca.sha256_run_ids(ids),
                      "fold_sha256": [M.ca.sha256_indices(x) for x in folds],
                      "exp13_oof_sha256": sha_array(frozen),
                      "debug_frozen_equals_exp13": True,
                      "cache_fingerprint": M.ca._fingerprint()},
            "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                         "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                         "joblib": joblib.__version__, "cpu_count": os.cpu_count(),
                         "joblib_cpu_count": joblib.cpu_count(), "thread_env": omp},
            "runs": json_runs,
            "mismatch_count_repeat1_vs_frozen": runs[0]["label_mismatches_vs_frozen"],
            "mismatch_count_repeat2_vs_frozen": runs[1]["label_mismatches_vs_frozen"],
            "repeat_probabilities_bit_identical": bool(np.array_equal(
                runs[0]["control_probability"], runs[1]["control_probability"])),
            "mismatch_csv_rows": len(mismatch_rows),
            "outputs": {"probabilities": os.path.basename(prob_path),
                        "mismatches": os.path.basename(mismatch_path),
                        "log": os.path.basename(log_path)},
        }
        with open(record_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
        log("Repeat probabilities bit-identical: " + str(result["repeat_probabilities_bit_identical"]))
        log("Saved probabilities, intermediate hashes, and mismatch CSV under " + OUT)

    print("Diagnostic outputs: " + OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
