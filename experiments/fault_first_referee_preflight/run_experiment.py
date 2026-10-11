"""One preregistered joint root-class and fault-onset CV, based on event order."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from event_inputs import build_dataset
from model import FaultFirstReferee

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP15_DIR = ROOT / "experiments" / "exp15_success_label_stacking"
EXP16_DIR = ROOT / "experiments" / "exp16_semantic_sequence"
TRAIN_CSV = ROOT / "data" / "train.csv"
BASELINE_CSV = EXP16_DIR / "baseline_exp15_oof.csv"
BASELINE_JSON = EXP16_DIR / "baseline.json"
LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
FOLDS = 3
FOLD_SEED = 0
EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ONSET_LOSS_WEIGHT = 0.5
SEED = 42
BASELINE_METRICS = {
    "macro_f1": 0.8129545586738590,
    "robustness_f1": 0.7599843692318033,
    "success_f1": 0.8949086161879896,
    "fault_turn_hit2": 0.6315277777777778,
    "composite": 0.7938624418508566,
}
GATE = {
    "composite_delta_min": 0.025,
    "macro_f1_delta_min_exclusive": 0.0,
    "robustness_f1_delta_min_exclusive": 0.0,
    "folds_won_min": 2,
    "per_class_f1_drop_max": 0.02,
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_helpers():
    exp15_spec = importlib.util.spec_from_file_location(
        "exp21_exp15_helpers", EXP15_DIR / "run_experiment.py")
    exp15 = importlib.util.module_from_spec(exp15_spec)
    exp15_spec.loader.exec_module(exp15)
    metrics_spec = importlib.util.spec_from_file_location(
        "exp21_official_metrics", ROOT / "evaluation" / "metrics.py")
    metrics = importlib.util.module_from_spec(metrics_spec)
    metrics_spec.loader.exec_module(metrics)
    if Path(metrics.__file__).resolve() != \
            (ROOT / "evaluation" / "metrics.py").resolve():
        raise RuntimeError("official evaluation.metrics did not load")
    return type("Helpers", (), {"exp15": exp15, "metrics": metrics})


def _scores(metrics, true_label, pred_label, true_success, pred_success,
            true_turn, pred_turn, robust):
    per_class = metrics.f1_per_class(true_label, pred_label, LABELS)
    macro = metrics.macro_f1(true_label, pred_label, LABELS)
    robust_rows = np.flatnonzero(robust)
    robustness = metrics.macro_f1(
        [true_label[i] for i in robust_rows],
        [pred_label[i] for i in robust_rows], LABELS)
    success = metrics.binary_f1(true_success, pred_success)
    hit2 = metrics.fault_turn_hit_at_k(true_label, pred_label,
                                       true_turn, pred_turn, k=2)
    values = {"macro_f1": macro, "robustness_f1": robustness,
              "success_f1": success, "fault_turn_hit2": hit2,
              "per_class_f1": per_class}
    values["composite"] = metrics.composite(values)
    return values


def verify_frozen_baseline(train):
    if sha256_file(TRAIN_CSV) != json.loads(BASELINE_JSON.read_text(
            encoding="utf-8"))["train_csv_sha256"]:
        raise RuntimeError("demo train differs from the frozen Exp15 baseline")
    manifest = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    if sha256_file(BASELINE_CSV) != manifest["baseline_oof_prediction_sha256"]:
        raise RuntimeError("frozen Exp15 OOF file SHA mismatch")
    baseline = pd.read_csv(BASELINE_CSV)
    if baseline["run_id"].astype(str).tolist() != \
            train["run_id"].astype(str).tolist():
        raise RuntimeError("frozen Exp15 OOF run_id order differs from train")
    helpers = _load_helpers()
    metrics = helpers.metrics
    robust = np.asarray(helpers.exp15.ENS.robustness_mask(), dtype=bool)
    if len(robust) != len(train):
        raise RuntimeError("frozen robustness mask is not row-aligned")
    base = _scores(
        metrics, train.label.astype(str).tolist(), baseline.label.astype(str).tolist(),
        train.success.astype(int).tolist(), baseline.success.astype(int).tolist(),
        train.fault_turn.astype(int).tolist(), baseline.fault_turn.astype(int).tolist(),
        robust)
    for name, expected in BASELINE_METRICS.items():
        if abs(base[name] - expected) > 1e-12:
            raise RuntimeError(f"frozen Exp15 {name} parity failure: {base[name]}")
    return baseline, robust, helpers


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(model, data, rows, device):
    model.eval()
    probabilities = []
    predicted_turns = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            batch = data.batch(batch_rows)
            label_logits, turn_scores, _ = model(batch)
            probs = torch.softmax(label_logits, dim=1).cpu().numpy()
            pred_ids = probs.argmax(1)
            probabilities.append(probs.astype(np.float32))
            for batch_index, row in enumerate(batch_rows):
                pred = int(pred_ids[batch_index])
                positions = data.message_positions[int(row)]
                turns = data.message_turns[int(row)]
                if pred == 0 or not positions:
                    predicted_turns.append(-1)
                    continue
                scores = turn_scores[batch_index,
                                     torch.as_tensor(positions, device=device),
                                     pred - 1]
                chosen = int(torch.argmax(scores).item())
                predicted_turns.append(int(turns[chosen]))
    return np.concatenate(probabilities, axis=0), np.asarray(predicted_turns,
                                                              dtype=np.int64)


def _batch_loss(model, batch, batch_rows, labels, true_turns, device):
    label_logits, turn_scores, _ = model(batch)
    target_labels = torch.as_tensor(labels[batch_rows], dtype=torch.long,
                                    device=device)
    label_loss = F.cross_entropy(label_logits, target_labels)
    onset_losses = []
    for batch_index, row in enumerate(batch_rows):
        label_id = int(labels[row])
        if label_id == 0:
            continue
        turns = batch["message_turns"][batch_index]
        positions = batch["message_positions"][batch_index]
        target_turn = int(true_turns[row])
        if not positions or target_turn not in turns:
            raise ValueError("fault_turn does not map to a message event")
        target_position = turns.index(target_turn)
        turn_logits = turn_scores[
            batch_index,
            torch.as_tensor(positions, dtype=torch.long, device=device),
            label_id - 1]
        onset_losses.append(F.cross_entropy(
            turn_logits.unsqueeze(0),
            torch.tensor([target_position], dtype=torch.long, device=device)))
    onset_loss = (torch.stack(onset_losses).mean() if onset_losses else
                  label_loss.new_zeros(()))
    return label_logits, turn_scores, label_loss + ONSET_LOSS_WEIGHT * onset_loss


def _train_fold(data, labels, true_turns, train_rows, valid_rows, fold):
    seed = SEED + fold
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultFirstReferee().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_rows)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch_rows = order[start:start + BATCH_SIZE]
            batch = data.batch(batch_rows)
            _, _, loss = _batch_loss(model, batch, batch_rows, labels,
                                     true_turns, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_steps += 1
        print(f"fold={fold} epoch={epoch + 1}/{EPOCHS} "
              f"loss={epoch_loss / max(1, n_steps):.5f}", flush=True)
    probabilities, turns = _predict(model, data, valid_rows, device)
    del optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probabilities, turns


def run_cv(output_dir=None):
    output_dir = Path(output_dir or HERE)
    preflight_path = HERE / "diagnostics" / "preflight.json"
    if not (HERE / "PLAN.md").is_file() or not preflight_path.is_file():
        raise RuntimeError("CV requires a committed preregistered PLAN and preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("single_model_train_plus_infer_estimate_under_1800"):
        raise RuntimeError("preflight STOP: Exp21 CV is disabled by the 1800 s gate")
    if (output_dir / "results.json").exists() or \
            (output_dir / "oof_predictions.csv").exists():
        raise FileExistsError("refusing to overwrite an existing Exp21 result")
    train = pd.read_csv(TRAIN_CSV)
    baseline, robust, helpers = verify_frozen_baseline(train)
    print("Exp15 frozen OOF parity: PASS (5 official metrics within 1e-12)",
          flush=True)
    print(f"rows={len(train)} baseline_sha={sha256_file(BASELINE_CSV)}",
          flush=True)
    input_records = train[["goal", "agents", "messages", "artifacts",
                           "shared_state", "topology"]].to_dict("records")
    t0 = time.perf_counter()
    data = build_dataset(input_records)
    input_build_sec = time.perf_counter() - t0
    print(f"input events={int(data.event_offsets[-1])} "
          f"input build sec={input_build_sec:.1f}", flush=True)
    labels = np.asarray([LABEL_TO_ID[x] for x in train.label.astype(str)],
                        dtype=np.int64)
    true_turns = train.fault_turn.to_numpy(dtype=np.int64)
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=FOLD_SEED)
    probabilities = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    turns = np.full(len(train), -1, dtype=np.int64)
    folds = np.full(len(train), -1, dtype=np.int8)
    fold_metrics = []
    fit_start = time.perf_counter()
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        if np.intersect1d(tr, va).size:
            raise AssertionError("outer fold train/validation overlap")
        p, t = _train_fold(data, labels, true_turns, tr, va, fold)
        probabilities[va] = p
        turns[va] = t
        folds[va] = fold
        cand_ids = p.argmax(1)
        cand_label = [LABELS[int(x)] for x in cand_ids]
        cand = _scores(
            helpers.metrics, [train.label.iloc[i] for i in va], cand_label,
            [int(train.success.iloc[i]) for i in va],
            [int(baseline.success.iloc[i]) for i in va],
            [int(train.fault_turn.iloc[i]) for i in va], t,
            robust[va])
        base_fold = _scores(
            helpers.metrics, [train.label.iloc[i] for i in va],
            [baseline.label.iloc[i] for i in va],
            [int(train.success.iloc[i]) for i in va],
            [int(baseline.success.iloc[i]) for i in va],
            [int(train.fault_turn.iloc[i]) for i in va],
            [int(baseline.fault_turn.iloc[i]) for i in va], robust[va])
        fold_metrics.append({"fold": fold, "n_valid": len(va),
                             "baseline": base_fold, "candidate": cand,
                             "composite_delta": cand["composite"] -
                                                base_fold["composite"]})
        print(f"fold={fold} complete rows={len(va)} "
              f"composite_delta={fold_metrics[-1]['composite_delta']:+.6f}",
              flush=True)
    fit_sec = time.perf_counter() - fit_start
    if np.any(folds < 0) or np.any(turns[folds >= 0] < -1):
        raise AssertionError("OOF coverage or turn output is invalid")
    base = _scores(
        helpers.metrics, train.label.astype(str).tolist(),
        baseline.label.astype(str).tolist(), train.success.astype(int).tolist(),
        baseline.success.astype(int).tolist(), train.fault_turn.astype(int).tolist(),
        baseline.fault_turn.astype(int).tolist(), robust)
    candidate_ids = probabilities.argmax(1)
    candidate_labels = [LABELS[int(x)] for x in candidate_ids]
    candidate = _scores(
        helpers.metrics, train.label.astype(str).tolist(), candidate_labels,
        train.success.astype(int).tolist(), baseline.success.astype(int).tolist(),
        train.fault_turn.astype(int).tolist(), turns.tolist(), robust)
    deltas = {name: candidate[name] - base[name]
              for name in ("composite", "macro_f1", "robustness_f1",
                           "success_f1", "fault_turn_hit2")}
    class_deltas = {name: candidate["per_class_f1"][name] -
                    base["per_class_f1"][name] for name in LABELS}
    passed = (deltas["composite"] >= GATE["composite_delta_min"] and
              deltas["macro_f1"] > 0 and deltas["robustness_f1"] > 0 and
              sum(f["composite_delta"] > 0 for f in fold_metrics) >=
              GATE["folds_won_min"] and
              min(class_deltas.values()) >= -GATE["per_class_f1_drop_max"])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "oof_probabilities.npz",
                        probabilities=probabilities, fold_of=folds)
    np.save(output_dir / "oof_fault_turn.npy", turns, allow_pickle=False)
    output = pd.DataFrame({"run_id": train.run_id.astype(str),
                           "label": candidate_labels,
                           "success": baseline.success.astype(int),
                           "fault_turn": turns,
                           "fold": folds})
    output.to_csv(output_dir / "oof_predictions.csv", index=False)
    results = {
        "experiment": "exp21_fault_first_referee",
        "decision": "PASS_GATE" if passed else "STOP",
        "baseline": base, "candidate": candidate, "deltas": deltas,
        "per_class_f1_delta": class_deltas, "folds": fold_metrics,
        "gate": GATE, "gate_passed": passed,
        "cv_seconds": fit_sec,
        "input_build_seconds": input_build_sec,
        "rows": len(train), "events": int(data.event_offsets[-1]),
        "baseline_prediction_sha256": sha256_file(BASELINE_CSV),
        "train_sha256": sha256_file(TRAIN_CSV),
        "candidate_B_production_changed": False,
        "leaderboard_used": False,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2),
                                               encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == "__main__":
    run_cv()
