"""Frozen Exp19 StepFinder-inspired fault-turn CV; Exp15 labels stay fixed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from stepfinder_model import StepFinderLocalizer, compute_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP15_DIR = ROOT / "experiments" / "exp15_success_label_stacking"
EXP16_DIR = ROOT / "experiments" / "exp16_semantic_sequence"
EXP16_SCRIPT = EXP16_DIR / "run_experiment.py"
CACHE_DIR = ROOT / ".cache" / "exp16_semantic_sequence"
BASELINE_CSV = EXP16_DIR / "baseline_exp15_oof.csv"
BASELINE_JSON = EXP16_DIR / "baseline.json"
LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
FAULT_CLASSES = LABELS[1:]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
EVENT_KIND_ORDER = {"message": 0, "state": 1, "artifact": 2}
EVENT_KIND_META = {"goal": 0, "message": 1, "artifact": 2, "state": 3}
SEED = 42
FOLD_SEED = 0
N_FOLDS = 3
EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_CV_SECONDS = 7200.0
BENCHMARK_N = 256
BASELINE_METRICS = {
    "macro_f1": 0.8129545586738590,
    "robustness_f1": 0.7599843692318033,
    "success_f1": 0.8949086161879896,
    "fault_turn_hit2": 0.6315277777777778,
    "composite": 0.7938624418508566,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exp16_module():
    spec = importlib.util.spec_from_file_location("exp16_sequence_inputs",
                                                  EXP16_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode(exp16, value, default):
    return exp16._json_field(value, default)


def _event_actor_role(event, kind, roles):
    if kind == "message":
        raw_id = event.get("from")
    elif kind == "artifact":
        raw_id = event.get("by", event.get("agent"))
    else:
        raw_id = event.get("agent", event.get("by"))
    key = "" if raw_id is None else str(raw_id).strip()
    role = roles.get(key, "[unknown]")
    role = " ".join(str(role).split()).casefold()
    return role or "[unknown]"


def build_event_map(record, prepared, exp16):
    """Recreate Exp16's stable event order and bind messages to their raw t."""
    agents = _decode(exp16, record.get("agents"), [])
    messages = _decode(exp16, record.get("messages"), [])
    states = _decode(exp16, record.get("shared_state"), [])
    artifacts = _decode(exp16, record.get("artifacts"), [])
    if not all(isinstance(value, list) for value in
               (agents, messages, states, artifacts)):
        raise ValueError("event fields must decode to lists")
    roles = {str(agent.get("id", "")): agent.get("role", "[unknown]")
             for agent in agents if isinstance(agent, dict)}
    events = []
    for kind, values in (("message", messages), ("state", states),
                         ("artifact", artifacts)):
        for source_index, event in enumerate(values):
            if not isinstance(event, dict):
                raise ValueError("%s event is not an object" % kind)
            events.append((exp16._event_turn(event), EVENT_KIND_ORDER[kind],
                           source_index, kind, event))
    events.sort(key=lambda item: (item[0] is None,
                                  item[0] if item[0] is not None else 0,
                                  item[1], item[2]))
    if len(prepared["event_texts"]) != len(events) + 1:
        raise AssertionError("Exp16 event count and turn map differ")

    count = len(events) + 1
    event_to_turn = np.full(count, -1, dtype=np.int32)
    role_names = ["[unknown]"] * count
    message_positions = []
    turn_to_event = np.full(len(messages), -1, dtype=np.int32)
    meta = prepared["event_meta"]
    if meta.shape != (count, exp16.META_DIM):
        raise AssertionError("event metadata shape differs from Exp16 sequence")
    if meta[0, EVENT_KIND_META["goal"]] != 1:
        raise AssertionError("event zero is not the separate goal")
    for position, (turn, _, source_index, kind, event) in enumerate(events,
                                                                   start=1):
        if meta[position, EVENT_KIND_META[kind]] != 1:
            raise AssertionError("event type ordering differs from Exp16 cache")
        expected_relpos = position / (position + 1.0)
        if not np.isclose(meta[position, 4], expected_relpos, atol=0, rtol=0):
            raise AssertionError("relative event position differs from Exp16")
        expected_time = (0.0 if turn is None else
                         float(turn) / (1.0 + abs(float(turn))))
        if (meta[position, 5] != expected_time or
                meta[position, 6] != float(turn is not None)):
            raise AssertionError("source t shifted or differs from Exp16 metadata")
        role_names[position] = _event_actor_role(event, kind, roles)
        if kind == "message":
            if turn is None or turn != source_index:
                raise AssertionError("message t must equal its dense source index")
            if turn < 0 or turn >= len(messages) or turn_to_event[turn] != -1:
                raise AssertionError("message t is missing, duplicate, or out of range")
            turn_to_event[turn] = position
            event_to_turn[position] = turn
            message_positions.append(position)
    if len(messages) and not np.array_equal(turn_to_event >= 0,
                                            np.ones(len(messages), dtype=bool)):
        raise AssertionError("not every message turn maps to an event")
    if len(message_positions) != len(messages):
        raise AssertionError("message event count differs from raw turns")
    return {"event_to_turn": event_to_turn,
            "turn_to_event": turn_to_event,
            "message_positions": np.asarray(message_positions, dtype=np.int32),
            "role_names": tuple(role_names),
            "length": count}


def verify_cache_and_prepare_inputs(frame: pd.DataFrame, exp16):
    """Verify every pinned cache component; never rebuild or encode on failure."""
    cache_dir = CACHE_DIR
    manifest_path = cache_dir / "manifest.json"
    required = [manifest_path, cache_dir / "embeddings.npy",
                cache_dir / "event_meta.npy", cache_dir / "event_offsets.npy"]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Exp16 embedding cache is incomplete; encoder pass forbidden")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_csv = ROOT / "data" / "train.csv"
    encoder_weights = cache_dir / "encoder" / "model.safetensors"
    if not encoder_weights.is_file():
        raise RuntimeError("pinned MiniLM weight file is missing")
    if sha256_file(encoder_weights) != exp16.ENCODER_WEIGHTS_SHA256:
        raise RuntimeError("pinned MiniLM weight SHA-256 mismatch")
    input_sha = sha256_file(train_csv)
    if input_sha != manifest.get("input_csv_sha256"):
        raise RuntimeError("train CSV SHA does not match Exp16 cache")

    offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    meta_parts = []
    maps = []
    text_hash = hashlib.sha256()
    first_text = True
    for row_number, record in enumerate(frame.to_dict("records")):
        prepared = exp16.prepare_run(record)
        sequence_map = build_event_map(record, prepared, exp16)
        if sequence_map["length"] != len(prepared["event_texts"]):
            raise AssertionError("event map lost an observable event")
        offsets[row_number + 1] = offsets[row_number] + sequence_map["length"]
        meta_parts.append(prepared["event_meta"])
        for text in prepared["event_texts"]:
            if not first_text:
                text_hash.update(b"\0")
            text_hash.update(text.encode("utf-8"))
            first_text = False
        maps.append(sequence_map)

    expected_manifest = {
        "format_version": exp16.INPUT_FORMAT_VERSION,
        "encoder_id": exp16.ENCODER_ID,
        "encoder_revision": exp16.ENCODER_REVISION,
        "encoder_weights_sha256": exp16.ENCODER_WEIGHTS_SHA256,
        "input_csv_sha256": input_sha,
        "rows": len(frame),
        "events": int(offsets[-1]),
        "embedding_dim": exp16.EMBEDDING_DIM,
        "max_wordpieces_per_event": exp16.MAX_WORDPIECES,
        "semantic_input_sha256": text_hash.hexdigest(),
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise RuntimeError("Exp16 cache manifest/input digest mismatch")

    embedding_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "event_meta.npy"
    offsets_path = cache_dir / "event_offsets.npy"
    if sha256_file(embedding_path) != manifest.get("embedding_cache_sha256"):
        raise RuntimeError("Exp16 embedding array SHA-256 mismatch")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    cached_meta = np.load(meta_path, mmap_mode="r", allow_pickle=False)
    cached_offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    expected_meta = np.concatenate(meta_parts, axis=0)
    if (embeddings.shape != (int(offsets[-1]), exp16.EMBEDDING_DIM) or
            embeddings.dtype != np.float16):
        raise RuntimeError("Exp16 cached embedding shape/dtype mismatch")
    if (not np.array_equal(cached_offsets, offsets) or
            not np.array_equal(cached_meta, expected_meta)):
        raise RuntimeError("Exp16 offsets or event metadata do not match input rows")
    if frame["run_id"].astype(str).duplicated().any():
        raise RuntimeError("train run_ids are not unique")

    cache = exp16.EmbeddingCache(cache_dir, len(frame), offsets, cached_meta)
    return cache, maps, manifest


def verify_target_alignment(frame, sequences, true_labels):
    turns = frame["fault_turn"].to_numpy(dtype=np.int64)
    message_count = 0
    for row, (label, target, sequence) in enumerate(
            zip(true_labels, turns, sequences)):
        count = len(sequence["turn_to_event"])
        message_count += count
        if label == "clean":
            if target != -1:
                raise AssertionError("clean run has a nonnegative fault_turn")
        elif (target < 0 or target >= count or
              sequence["turn_to_event"][target] < 0 or
              sequence["event_to_turn"][sequence["turn_to_event"][target]] != target):
            raise AssertionError("fault_turn does not map to its source message event")
    if message_count != 457203:
        raise AssertionError("Exp16 message event count changed: %d" % message_count)
    return {"message_events": message_count,
            "faulty_targets": int(np.count_nonzero(true_labels != "clean")),
            "all_targets_mapped": True}


def frozen_folds(labels, exp15_folds):
    n = len(labels)
    expected = [np.asarray(valid, dtype=np.int64) for _, valid in
                StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
                .split(np.zeros(n), labels)]
    if len(exp15_folds) != N_FOLDS:
        raise AssertionError("Exp15 fold count changed")
    actual = [np.asarray(valid, dtype=np.int64) for valid in exp15_folds]
    if any(not np.array_equal(a, b) for a, b in zip(expected, actual)):
        raise AssertionError("outer folds differ from frozen Exp15 folds")
    if sum(len(rows) for rows in actual) != n or len(np.unique(np.concatenate(actual))) != n:
        raise AssertionError("outer validation folds do not partition all rows")
    return actual


def split_rows(n_rows, valid_rows):
    valid_rows = np.asarray(valid_rows, dtype=np.int64)
    train_mask = np.ones(n_rows, dtype=bool)
    train_mask[valid_rows] = False
    train_rows = np.flatnonzero(train_mask)
    if np.intersect1d(train_rows, valid_rows).size:
        raise AssertionError("outer train and validation overlap")
    return train_rows


def faulty_training_rows(train_rows, true_label_indices, fault_turns):
    rows = np.asarray(train_rows, dtype=np.int64)
    y = np.asarray(true_label_indices, dtype=np.int64)
    turns = np.asarray(fault_turns, dtype=np.int64)
    if np.any((y[rows] != 0) & (turns[rows] < 0)):
        raise AssertionError("faulty training rows must have labeled fault_turn")
    return rows[(y[rows] != 0) & (turns[rows] >= 0)]


def role_vocabulary(rows, sequences):
    roles = {role for row in rows for role in sequences[int(row)]["role_names"]
             if role and role != "[unknown]"}
    return {"[unknown]": 0, **{role: i + 1 for i, role in
                               enumerate(sorted(roles))}}


def collate_rows(rows, class_ids, cache, sequences, role_vocab,
                 target_turns=None):
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    classes = np.asarray(class_ids, dtype=np.int64)
    if len(rows) == 0 or classes.shape != (len(rows),):
        raise ValueError("nonempty rows and one class id per run are required")
    if np.any(classes < 0) or np.any(classes >= len(FAULT_CLASSES)):
        raise ValueError("class conditioning must be one of the six fault classes")
    semantic, metadata, lengths = cache.batch(rows)
    batch, width, _ = semantic.shape
    role_ids = np.zeros((batch, width), dtype=np.int64)
    valid = np.zeros((batch, width), dtype=bool)
    message = np.zeros((batch, width), dtype=bool)
    target_event = None
    if target_turns is not None:
        target_turns = np.asarray(target_turns, dtype=np.int64)
        if target_turns.shape != (len(rows),):
            raise ValueError("one fault_turn is required per training run")
        target_event = np.empty(len(rows), dtype=np.int64)
    for slot, row in enumerate(rows):
        sequence = sequences[int(row)]
        length = int(lengths[slot])
        if length != sequence["length"]:
            raise AssertionError("embedding offsets and event map misaligned")
        valid[slot, :length] = True
        role_ids[slot, :length] = [role_vocab.get(role, 0)
                                   for role in sequence["role_names"]]
        message[slot, sequence["message_positions"]] = True
        if target_event is not None:
            turn = int(target_turns[slot])
            if turn < 0 or turn >= len(sequence["turn_to_event"]):
                raise ValueError("training target is not a message turn")
            target_event[slot] = int(sequence["turn_to_event"][turn])
            if not message[slot, target_event[slot]]:
                raise AssertionError("fault target points to a non-message event")
    return {
        "content": torch.from_numpy(semantic),
        "metadata": torch.from_numpy(metadata),
        "role_ids": torch.from_numpy(role_ids),
        "class_ids": torch.as_tensor(classes, dtype=torch.long),
        "valid_mask": torch.from_numpy(valid),
        "message_mask": torch.from_numpy(message),
        "target_event": (None if target_event is None else
                         torch.as_tensor(target_event, dtype=torch.long)),
        "lengths": lengths,
    }


def seed_everything(torch, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def train_fold(cache, sequences, rows, true_labels, fault_turns, role_vocab,
               epochs=EPOCHS, log=print):
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    y = np.asarray(true_labels, dtype=np.int64)
    turns = np.asarray(fault_turns, dtype=np.int64)
    if not len(rows) or np.any(y[rows] == 0) or np.any(turns[rows] < 0):
        raise AssertionError("localizer fit accepts only annotated faulty runs")
    seed_everything(torch, SEED)
    model = StepFinderLocalizer(len(role_vocab)).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    losses = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(rows)
        batch_losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch_rows = order[start:start + BATCH_SIZE]
            batch = collate_rows(batch_rows, y[batch_rows] - 1, cache,
                                 sequences, role_vocab, turns[batch_rows])
            optimizer.zero_grad(set_to_none=True)
            logits, temporal_loss = model(
                batch["content"], batch["metadata"], batch["role_ids"],
                batch["class_ids"], batch["valid_mask"], batch["message_mask"])
            loss = compute_loss(logits, batch["target_event"], temporal_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        mean_loss = float(np.mean(batch_losses))
        losses.append(mean_loss)
        log("epoch %d/%d train_loss=%.6f; no validation scoring" %
            (epoch + 1, epochs, mean_loss))
    model.eval()
    del optimizer
    return model, losses


def predict_turns(model, rows, predicted_label_indices, cache, sequences,
                  role_vocab):
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    predicted = np.asarray(predicted_label_indices, dtype=np.int64)
    if predicted.shape != (len(rows),):
        raise ValueError("frozen predicted classes must align to validation rows")
    turns = np.full(len(rows), -1, dtype=np.int64)
    # clean predictions stay -1 and are never sent through the fault localizer.
    active_slots = [i for i, row in enumerate(rows)
                    if predicted[i] != 0 and
                    len(sequences[int(row)]["message_positions"]) > 0]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(active_slots), BATCH_SIZE):
            slots = active_slots[start:start + BATCH_SIZE]
            batch_rows = rows[slots]
            class_ids = predicted[slots] - 1
            batch = collate_rows(batch_rows, class_ids, cache, sequences,
                                 role_vocab)
            logits, _ = model(batch["content"], batch["metadata"],
                              batch["role_ids"], batch["class_ids"],
                              batch["valid_mask"], batch["message_mask"])
            for local, slot in enumerate(slots):
                event_pos = int(logits[local].argmax())
                sequence = sequences[int(rows[slot])]
                if event_pos not in sequence["message_positions"]:
                    raise AssertionError("localizer selected a non-message event")
                turn = int(sequence["event_to_turn"][event_pos])
                if turn < 0:
                    raise AssertionError("selected event has no raw message t")
                turns[slot] = turn
    return turns


def _score(rows, predicted_turns, predicted_labels, success_pred, d, robust,
           metrics):
    rows = np.asarray(rows, dtype=np.int64)
    y_true = np.asarray(d["yi"], dtype=np.int64)[rows]
    y_pred = np.asarray(predicted_labels, dtype=np.int64)[rows]
    true_names = [LABELS[int(i)] for i in y_true]
    pred_names = [LABELS[int(i)] for i in y_pred]
    per_class_f1 = metrics.f1_per_class(true_names, pred_names, LABELS)
    macro = metrics.macro_f1(true_names, pred_names, LABELS)
    robust_rows = np.flatnonzero(np.asarray(robust, dtype=bool)[rows])
    robust_f1 = metrics.macro_f1(
        [true_names[i] for i in robust_rows],
        [pred_names[i] for i in robust_rows], LABELS)
    success_true = np.asarray(d["train"]["success"], dtype=np.int64)[rows]
    success_f1 = metrics.binary_f1(success_true.tolist(),
                                   np.asarray(success_pred)[rows].tolist())
    target_turns = np.asarray(d["train"]["fault_turn"], dtype=np.int64)[rows]
    turns = np.asarray(predicted_turns, dtype=np.int64)[rows]
    faulty = y_true != 0
    hit2 = metrics.fault_turn_hit_at_k(
        [true_names[i] for i in np.flatnonzero(faulty)],
        [pred_names[i] for i in np.flatnonzero(faulty)],
        target_turns[faulty].tolist(), turns[faulty].tolist(), k=2)
    composite = metrics.composite({"macro_f1": macro,
                                   "robustness_f1": robust_f1,
                                   "success_f1": success_f1,
                                   "fault_turn_hit2": hit2})
    class_hit2 = {}
    for cls_index, cls in enumerate(LABELS[1:], start=1):
        mask = y_true == cls_index
        class_hit2[cls] = metrics.fault_turn_hit_at_k(
            [true_names[i] for i in np.flatnonzero(mask)],
            [pred_names[i] for i in np.flatnonzero(mask)],
            target_turns[mask].tolist(), turns[mask].tolist(), k=2)
    return {"macro_f1": macro, "robustness_f1": robust_f1,
            "success_f1": success_f1, "fault_turn_hit2": hit2,
            "composite": composite, "per_class_f1": per_class_f1,
            "per_class_hit2": class_hit2}


def _work_units(rows, sequences, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(rows, dtype=np.int64))
    work = 0
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        lengths = [sequences[int(row)]["length"] for row in batch]
        max_len = max(lengths)
        # Attention is quadratic in the padded sequence length; also count LSTM work.
        work += len(batch) * max_len * max_len + sum(lengths)
    return work


def _peak_rss_mb(exp16):
    return exp16._peak_rss_mb()


def run_benchmark(cache, sequences, true_labels, predicted_labels, fault_turns,
                  folds, exp16,
                  log=print):
    y = np.asarray(true_labels, dtype=np.int64)
    turns = np.asarray(fault_turns, dtype=np.int64)
    all_rows = np.arange(len(y), dtype=np.int64)
    fold_train_rows = split_rows(len(y), folds[0])
    fold_faulty = faulty_training_rows(fold_train_rows, y, turns)
    ordered = fold_faulty[np.argsort(
        [sequences[int(row)]["length"] for row in fold_faulty], kind="stable")]
    quartiles = np.array_split(ordered, 4)
    rng = np.random.default_rng(SEED)
    sample = np.sort(np.concatenate([
        rng.choice(group, size=BENCHMARK_N // 4, replace=False)
        for group in quartiles]))
    vocab = role_vocabulary(sample, sequences)
    start = time.time()
    model, losses = train_fold(cache, sequences, sample, y, turns, vocab,
                               epochs=1,
                               log=lambda m: log("benchmark " + m))
    sample_seconds = time.time() - start
    sample_work = _work_units(sample, sequences, SEED)
    fold_estimates = []
    inference_work = 0
    inference_started = time.time()
    predicted_class = np.asarray(predicted_labels, dtype=np.int64)[sample]
    prediction = predict_turns(model, sample, predicted_class, cache,
                               sequences, vocab)
    inference_seconds = time.time() - inference_started
    if len(prediction) != BENCHMARK_N:
        raise AssertionError("benchmark inference output count changed")
    for fold, valid in enumerate(folds):
        train_rows = split_rows(len(y), valid)
        train_faulty = faulty_training_rows(train_rows, y, turns)
        train_work = sum(_work_units(train_faulty, sequences,
                                     SEED + epoch) for epoch in range(EPOCHS))
        train_est = sample_seconds * train_work / max(1, sample_work)
        valid_rows = np.asarray(valid, dtype=np.int64)
        # Forecast only the rows the frozen Exp15 model would send to a fault head.
        baseline_labels = np.asarray(predicted_labels, dtype=np.int64)[valid_rows]
        nonclean = valid_rows[baseline_labels != 0]
        valid_work = _work_units(nonclean, sequences, SEED) if len(nonclean) else 0
        active_sample = sample[predicted_class != 0]
        full_inference_work = (_work_units(active_sample, sequences, SEED)
                               if len(active_sample) else 0)
        infer_est = inference_seconds * valid_work / max(1, full_inference_work)
        inference_work += valid_work
        fold_estimates.append({"fold": fold,
                               "train_faulty_runs": int(len(train_faulty)),
                               "estimated_train_seconds": round(train_est, 2),
                               "predicted_fault_validation_runs": int(len(nonclean)),
                               "estimated_inference_seconds": round(infer_est, 2)})
    del model
    total_est = sum(x["estimated_train_seconds"] +
                    x["estimated_inference_seconds"] for x in fold_estimates)
    return {"device": "cpu", "torch_version": __import__("torch").__version__,
            "torch_threads": __import__("torch").get_num_threads(),
            "benchmark_faulty_runs": int(len(sample)),
            "benchmark_event_lengths": {"min": int(min(
                sequences[int(r)]["length"] for r in sample)),
                "median": float(np.median([sequences[int(r)]["length"]
                                           for r in sample])),
                "max": int(max(sequences[int(r)]["length"] for r in sample))},
            "one_epoch_train_seconds": round(sample_seconds, 2),
            "one_pass_inference_seconds": round(inference_seconds, 2),
            "one_epoch_loss": losses[0],
            "estimated_fold_runtime": fold_estimates,
            "estimated_full_3fold_seconds": round(total_est, 2),
            "estimated_full_3fold_minutes": round(total_est / 60, 2),
            "peak_process_rss_mb": _peak_rss_mb(exp16),
            "max_events_per_run": int(max(s["length"] for s in sequences)),
            "max_full_turns_retained": True,
            "max_full_cv_seconds": MAX_CV_SECONDS,
            "ready_for_full_cv": bool(total_est <= MAX_CV_SECONDS)}


def run_cv(args):
    import torch

    exp16 = load_exp16_module()
    helpers = exp16._load_helpers()
    exp15, metrics = helpers.exp15, helpers.metrics
    d = exp15.ENS.load_verified()
    train = d["train"]
    n = len(train)
    if n != 10000:
        raise AssertionError("expected frozen 10,000-row training set")
    labels = train["label"].astype(str).to_numpy()
    true_label_indices = np.asarray(d["yi"], dtype=np.int64)
    if not np.array_equal(np.asarray([LABEL_TO_INDEX[x] for x in labels]),
                          true_label_indices):
        raise AssertionError("training labels differ from verified Exp15 rows")
    robust = np.asarray(exp15.ENS.robustness_mask(), dtype=bool)
    baseline, _, baseline_metrics = exp16._validate_baseline(d, robust, helpers)
    if (abs(baseline_metrics["composite"] - BASELINE_METRICS["composite"]) > 1e-8 or
            abs(baseline_metrics["fault_turn_hit2"] -
                BASELINE_METRICS["fault_turn_hit2"]) > 1e-8):
        raise AssertionError("Exp15 Composite or hit@2 parity failed before fit")
    frozen_labels = np.asarray([LABEL_TO_INDEX[str(x)]
                                 for x in baseline["label"]], dtype=np.int64)
    frozen_success = baseline["success"].to_numpy(dtype=np.int64)
    if not np.array_equal(frozen_labels,
                          np.load(exp16.EXP13_DIR / "oof_D_corrected_wg.npy",
                                  allow_pickle=False).argmax(1)):
        raise AssertionError("frozen Exp15 labels differ from Exp13 OOF")
    if len(np.unique(train["run_id"].astype(str))) != n:
        raise AssertionError("run_id values must be unique")
    folds = frozen_folds(true_label_indices, d["folds"])
    sequences_cache, sequences, cache_manifest = verify_cache_and_prepare_inputs(
        train, exp16)
    alignment = verify_target_alignment(train, sequences, labels)
    if not np.array_equal(train["run_id"].astype(str).to_numpy(),
                          baseline["run_id"].astype(str).to_numpy()):
        raise AssertionError("Exp15 OOF run_id order differs from cache input")

    log_lines = []

    def log(message):
        print(message, flush=True)
        log_lines.append(str(message))

    log("Exp19 cache verified; no encoder pass; 384d frozen MiniLM reused")
    log("baseline parity exact: Composite=%.10f hit@2=%.10f" %
        (baseline_metrics["composite"], baseline_metrics["fault_turn_hit2"]))
    log("events=%d messages=%d; event order goal / t / message->state->artifact" %
        (cache_manifest["events"], alignment["message_events"]))
    log("folds=StratifiedKFold(3, seed=0); localizer only fits faulty train rows")

    if args.check_only:
        log("Exp15 baseline verified: composite=%.10f hit@2=%.10f; no model fit" %
            (baseline_metrics["composite"], baseline_metrics["fault_turn_hit2"]))
        (HERE / "run.log").write_text("\n".join(log_lines) + "\n",
                                      encoding="utf-8")
        return 0

    benchmark = run_benchmark(sequences_cache, sequences, true_label_indices,
                              frozen_labels,
                              train["fault_turn"].to_numpy(dtype=np.int64),
                              folds, exp16, log=log)
    benchmark["cache_sha256"] = cache_manifest["embedding_cache_sha256"]
    benchmark["input_csv_sha256"] = cache_manifest["input_csv_sha256"]
    benchmark["alignment"] = alignment
    (HERE / "cpu_benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    log("CPU estimate %.2f min; ready=%s" %
        (benchmark["estimated_full_3fold_minutes"],
         benchmark["ready_for_full_cv"]))
    if args.benchmark_only:
        (HERE / "run.log").write_text("\n".join(log_lines) + "\n",
                                      encoding="utf-8")
        return 0
    if not benchmark["ready_for_full_cv"]:
        log("STOP before full CV: estimate exceeds 120 minutes")
        (HERE / "run.log").write_text("\n".join(log_lines) + "\n",
                                      encoding="utf-8")
        return 3

    fault_turns = train["fault_turn"].to_numpy(dtype=np.int64)
    success_before = frozen_success.copy()
    labels_before = frozen_labels.copy()
    baseline_turns = np.asarray(d["peak_pos"], dtype=np.int64)[
        np.arange(n), frozen_labels]
    candidate_turns = np.full(n, -1, dtype=np.int64)
    fold_results = []
    seen = np.zeros(n, dtype=np.int8)
    start_cv = time.time()
    for fold, valid in enumerate(folds):
        fold_started = time.time()
        train_rows = split_rows(n, valid)
        train_fault_rows = faulty_training_rows(train_rows,
                                                true_label_indices,
                                                fault_turns)
        role_vocab = role_vocabulary(train_fault_rows, sequences)
        if np.intersect1d(train_rows, valid).size:
            raise AssertionError("outer fold leakage")
        if np.intersect1d(train_fault_rows, valid).size:
            raise AssertionError("validation row entered localizer training")
        log("fold %d: fitting %d faulty train runs; roles=%d" %
            (fold, len(train_fault_rows), len(role_vocab) - 1))
        model, losses = train_fold(
            sequences_cache, sequences, train_fault_rows,
            true_label_indices, fault_turns, role_vocab, epochs=EPOCHS,
            log=lambda m: log("fold %d %s" % (fold, m)))
        # Prediction accepts only frozen Exp15 classes, never validation targets.
        turns = predict_turns(model, valid, frozen_labels[valid],
                              sequences_cache, sequences, role_vocab)
        candidate_turns[valid] = turns
        seen[valid] += 1
        base = _score(valid, baseline_turns, frozen_labels, frozen_success,
                      d, robust, metrics)
        candidate = _score(valid, candidate_turns, frozen_labels,
                           frozen_success, d, robust, metrics)
        fold_results.append({"fold": fold,
                             "n_train_runs": int(len(train_rows)),
                             "n_train_faulty_runs": int(len(train_fault_rows)),
                             "n_validation_runs": int(len(valid)),
                             "baseline": base, "candidate": candidate,
                             "hit2_delta": candidate["fault_turn_hit2"] -
                                           base["fault_turn_hit2"],
                             "composite_delta": candidate["composite"] -
                                                base["composite"],
                             "train_loss_by_epoch": losses,
                             "role_vocabulary_size": len(role_vocab),
                             "seconds": round(time.time() - fold_started, 2)})
        log("fold %d hit@2 %.6f -> %.6f (%+.6f), composite delta %+.6f, %.1fs" %
            (fold, base["fault_turn_hit2"], candidate["fault_turn_hit2"],
             fold_results[-1]["hit2_delta"],
             fold_results[-1]["composite_delta"],
             fold_results[-1]["seconds"]))
        del model

    if not np.all(seen == 1) or np.any(candidate_turns < -1):
        raise AssertionError("OOF turn predictions are incomplete")
    if not np.array_equal(frozen_labels, labels_before) or not np.array_equal(
            frozen_success, success_before):
        raise AssertionError("label or success predictions changed")
    baseline_metrics = _score(np.arange(n), baseline_turns, frozen_labels,
                              frozen_success, d, robust, metrics)
    candidate_metrics = _score(np.arange(n), candidate_turns, frozen_labels,
                                frozen_success, d, robust, metrics)
    if abs(baseline_metrics["composite"] - BASELINE_METRICS["composite"]) > 1e-8:
        raise AssertionError("full Exp15 baseline Composite parity failed")
    if abs(baseline_metrics["fault_turn_hit2"] -
           BASELINE_METRICS["fault_turn_hit2"]) > 1e-8:
        raise AssertionError("full Exp15 baseline hit@2 parity failed")
    if not np.array_equal(frozen_labels,
                          np.asarray([LABEL_TO_INDEX[x] for x in
                                      baseline["label"].astype(str)])):
        raise AssertionError("OOF class predictions changed")
    if not np.array_equal(frozen_success,
                          baseline["success"].to_numpy(dtype=np.int64)):
        raise AssertionError("success predictions changed")

    deltas = {name: candidate_metrics[name] - baseline_metrics[name]
              for name in ("fault_turn_hit2", "composite")}
    class_hit_deltas = {
        label: candidate_metrics["per_class_hit2"][label] -
        baseline_metrics["per_class_hit2"][label] for label in FAULT_CLASSES}
    folds_won = sum(row["hit2_delta"] > 0 for row in fold_results)
    checks = {
        "hit2_improves_at_least_0_05": deltas["fault_turn_hit2"] >= 0.05,
        "composite_improves_at_least_0_005": deltas["composite"] >= 0.005,
        "hit2_improves_on_at_least_2_of_3_folds": folds_won >= 2,
        "no_class_hit2_drop_over_0_05": all(
            delta >= -0.05 for delta in class_hit_deltas.values()),
        "label_predictions_unchanged": np.array_equal(
            frozen_labels, labels_before),
        "success_predictions_unchanged": np.array_equal(
            frozen_success, success_before),
        "no_validation_based_model_selection": True,
        "no_fold_overlap": True,
    }
    passed = all(checks.values())
    result = {
        "decision": "PROMOTE" if passed else "STOP",
        "hypothesis": "full-sequence agent-aware class-conditioned scoring improves primary fault localization over Exp13 L1",
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "deltas": deltas,
        "per_class_hit2_delta": class_hit_deltas,
        "folds_won": int(folds_won),
        "folds": fold_results,
        "gate": {"frozen": {"hit2_delta_min": 0.05,
                             "composite_delta_min": 0.005,
                             "fold_wins_min": 2,
                             "max_per_class_hit2_drop": 0.05},
                 "checks": checks, "passed": bool(passed)},
        "runtime_seconds": {"cv": round(time.time() - start_cv, 2),
                             "peak_process_rss_mb": _peak_rss_mb(exp16)},
        "cache": {"embedding_sha256": cache_manifest[
                      "embedding_cache_sha256"],
                  "input_csv_sha256": cache_manifest["input_csv_sha256"],
                  "rows": cache_manifest["rows"],
                  "events": cache_manifest["events"],
                  "message_events": alignment["message_events"],
                  "encoder_recomputed": False},
        "architecture": {"content": "frozen Exp16 MiniLM 384 -> 128",
                         "role_embedding_dim": 32,
                         "bilstm": "2-layer bidirectional hidden 64 per direction",
                         "attention": "agent-aware 2 x 32",
                         "scales": [1, 2], "alpha": 0.1, "beta": 0.9,
                         "gamma": 0.0, "epochs": EPOCHS,
                         "batch_size": BATCH_SIZE,
                         "learning_rate": LEARNING_RATE,
                         "weight_decay": WEIGHT_DECAY,
                         "dropout": 0.5,
                         "lambda_temporal": 0.9,
                         "seed": SEED},
        "baseline_provenance": {"exp15_commit": json.loads(
            BASELINE_JSON.read_text(encoding="utf-8"))["baseline_commit"],
            "production_changed": False},
        "public_leaderboard_used": False,
    }
    np.save(HERE / "baseline_fault_turn.npy", baseline_turns,
            allow_pickle=False)
    np.save(HERE / "candidate_fault_turn.npy", candidate_turns,
            allow_pickle=False)
    output = pd.DataFrame({"run_id": train["run_id"].astype(str),
                           "label": baseline["label"].astype(str),
                           "success": frozen_success,
                           "baseline_fault_turn": baseline_turns,
                           "candidate_fault_turn": candidate_turns})
    output.to_csv(HERE / "oof_predictions.csv", index=False)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                       encoding="utf-8")
    log("gate %s; hit@2 delta %+.6f; composite delta %+.6f" %
        ("PASS" if passed else "STOP", deltas["fault_turn_hit2"],
         deltas["composite"]))
    (HERE / "run.log").write_text("\n".join(log_lines) + "\n",
                                  encoding="utf-8")
    return 0 if passed else 2


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args(argv)
    return run_cv(args)


if __name__ == "__main__":
    raise SystemExit(main())
