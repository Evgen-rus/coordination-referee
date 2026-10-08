"""One frozen semantic-sequence experiment against the Exp15 label baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP15_DIR = ROOT / "experiments" / "exp15_success_label_stacking"
EXP13_DIR = ROOT / "experiments" / "exp13_wait_dependency_graph"
CACHE_DIR = ROOT / ".cache" / "exp16_semantic_sequence"
BASELINE_CSV = HERE / "baseline_exp15_oof.csv"
BASELINE_JSON = HERE / "baseline.json"

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]

# Frozen before CV. Keep this snapshot local and load with local_files_only=True.
ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "46605decb5369335a3847c9f41bb0b896c07dd1a"
ENCODER_WEIGHTS_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
EMBEDDING_DIM = 384
MAX_WORDPIECES = 256
ENCODER_BATCH_SIZE = 128
INPUT_FORMAT_VERSION = 2

# Sequence model and CV are also frozen before CV; no early stopping or search.
N_FOLDS = 3
FOLD_SEED = 0
FUSION_WEIGHT = 0.5
EPOCHS = 10
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 128
DROPOUT = 0.15
FOLD_MODEL_SEEDS = (1701, 1702, 1703)
META_NAMES = ("event_goal", "event_message", "event_artifact", "event_state",
              "relative_position", "relative_time", "time_known",
              "intent_present", "refs_present", "log_refs_count",
              "log_artifact_count_so_far", "log_state_count_so_far")
META_DIM = len(META_NAMES)
EVENT_ORDER = {"message": 0, "state": 1, "artifact": 2}

BASELINE_METRICS = {
    "macro_f1": 0.8129545586738590,
    "robustness_f1": 0.7599843692318033,
    "success_f1": 0.8949086161879896,
    "fault_turn_hit2": 0.6315277777777778,
    "composite": 0.7938624418508566,
}
GATE = {
    "composite_delta_min": 0.005,
    "macro_f1_delta_min_exclusive": 0.0,
    "robustness_f1_delta_min_exclusive": 0.0,
    "folds_won_min": 2,
    "per_class_f1_drop_max": 0.02,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_field(value, default):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return default
    value = value.strip()
    return json.loads(value) if value else default


def _text(value, fallback="[empty]"):
    if value is None:
        return fallback
    text = " ".join(str(value).split())
    return text or fallback


def _without_agent_ids(value, agent_ids, fallback="[empty]"):
    text = _text(value, fallback)
    for agent_id in agent_ids:
        text = re.sub(r"(?<![\w])%s(?![\w])" % re.escape(agent_id),
                      "[agent]", text, flags=re.IGNORECASE)
    return text


def _event_turn(event):
    value = event.get("t")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("event timestamp is not an integer: %r" % value) from exc


def prepare_run(record):
    """Build goal-first semantic events and causal metadata, without targets."""
    agents = _json_field(record.get("agents"), [])
    messages = _json_field(record.get("messages"), [])
    artifacts = _json_field(record.get("artifacts"), [])
    states = _json_field(record.get("shared_state"), [])
    if not isinstance(agents, list) or not isinstance(messages, list):
        raise ValueError("agents and messages must decode to JSON lists")
    if not isinstance(artifacts, list) or not isinstance(states, list):
        raise ValueError("artifacts and shared_state must decode to JSON lists")

    roles = {str(agent.get("id", "")): _text(agent.get("role"), "unknown")
             for agent in agents if isinstance(agent, dict)}
    agent_ids = sorted((key for key in roles if key), key=len, reverse=True)
    safe = lambda value, fallback="[none]": _without_agent_ids(  # noqa: E731
        value, agent_ids, fallback)
    events = []

    def add_events(kind, records):
        for source_index, event in enumerate(records):
            if not isinstance(event, dict):
                raise ValueError("%s %d is not an object" % (kind, source_index))
            turn = _event_turn(event)
            if kind == "message":
                sender = _text(event.get("from"), "")
                receiver = _text(event.get("to"), "")
                intent = safe(event.get("intent"))
                refs = event.get("refs") or []
                if not isinstance(refs, list):
                    refs = [refs]
                has_intent = intent != "[none]"
                text = (
                    "event=message; type=%s; sender role=%s; receiver role=%s; "
                    "intent=%s; tool=%s; text=%s" % (
                        safe(event.get("type"), "[unknown]"),
                        safe(roles.get(sender, "[unknown]"), "[unknown]"),
                        safe(roles.get(receiver, "[unknown]"), "[unknown]"),
                        intent, safe(event.get("tool")),
                        safe(event.get("text"), "[empty]")))
                item = {"kind": kind, "time": turn, "source_index": source_index,
                        "text": text, "intent_present": has_intent,
                        "refs_present": bool(refs), "refs_count": len(refs)}
            elif kind == "artifact":
                producer = _text(event.get("by", event.get("agent")), "")
                text = (
                    "event=artifact; producer role=%s; subtask=%s; type=%s; "
                    "status=%s" % (
                        safe(roles.get(producer, "[unknown]"), "[unknown]"),
                        safe(event.get("subtask")), safe(event.get("type")),
                        safe(event.get("status"))))
                item = {"kind": kind, "time": turn,
                        "source_index": source_index, "text": text}
            else:
                writer = _text(event.get("agent", event.get("by")), "")
                text = (
                    "event=state; writer role=%s; key=%s; op=%s; value=%s" % (
                        safe(roles.get(writer, "[unknown]"), "[unknown]"),
                        safe(event.get("key")), safe(event.get("op")),
                        safe(event.get("value"))))
                item = {"kind": kind, "time": turn,
                        "source_index": source_index, "text": text}
            events.append(item)

    add_events("message", messages)
    add_events("state", states)
    add_events("artifact", artifacts)
    events.sort(key=lambda event: (
        event["time"] is None,
        event["time"] if event["time"] is not None else 0,
        EVENT_ORDER[event["kind"]], event["source_index"]))

    event_texts = ["event=goal; text=%s" % safe(record.get("goal"), "[empty]")]
    metadata = [np.eye(4, dtype=np.float32)[0].tolist() +
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    artifact_count = state_count = 0
    for index, event in enumerate(events, start=1):
        if event["kind"] == "artifact":
            artifact_count += 1
        elif event["kind"] == "state":
            state_count += 1
        kind_idx = {"goal": 0, "message": 1, "artifact": 2, "state": 3}[
            event["kind"]]
        one_hot = np.eye(4, dtype=np.float32)[kind_idx]
        turn = event["time"]
        relative_time = (0.0 if turn is None else
                         float(turn) / (1.0 + abs(float(turn))))
        metadata.append(one_hot.tolist() + [
            index / (index + 1.0), relative_time, float(turn is not None),
            float(event.get("intent_present", False)),
            float(event.get("refs_present", False)),
            math.log1p(event.get("refs_count", 0)),
            math.log1p(artifact_count), math.log1p(state_count)])
        event_texts.append(event["text"])

    return {"event_texts": event_texts,
            "event_meta": np.asarray(metadata, dtype=np.float32).reshape(
                -1, META_DIM)}


class EmbeddingCache:
    def __init__(self, path: Path, n_runs: int, offsets, meta):
        self.path = path
        self.n_runs = n_runs
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.meta = np.asarray(meta, dtype=np.float32)
        self.embeddings = np.load(path / "embeddings.npy", mmap_mode="r",
                                  allow_pickle=False)

    def batch(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        lengths = self.offsets[rows + 1] - self.offsets[rows]
        width = int(lengths.max())
        semantic = np.zeros((len(rows), width, EMBEDDING_DIM), dtype=np.float32)
        metadata = np.zeros((len(rows), width, META_DIM), dtype=np.float32)
        for slot, row in enumerate(rows):
            lo, hi = int(self.offsets[row]), int(self.offsets[row + 1])
            semantic[slot, :hi - lo] = self.embeddings[lo:hi]
            metadata[slot, :hi - lo] = self.meta[lo:hi]
        return semantic, metadata, lengths.astype(np.int64)


def build_embeddings(frame: pd.DataFrame, model_dir: Path, cache_dir: Path,
                     csv_path: Path, device: str, log=print):
    """Cache frozen mean-pooled event embeddings; preserve the full sequence."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable; no encoder pass or fit started") from exc
    from transformers import AutoModel, AutoTokenizer

    required = ("config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("pinned local encoder files missing: %s" % missing)
    weight_sha = sha256_file(model_dir / "model.safetensors")
    if weight_sha != ENCODER_WEIGHTS_SHA256:
        raise ValueError("encoder weights SHA-256 mismatch: %s" % weight_sha)

    sequences = [prepare_run(row) for row in frame.to_dict("records")]
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    for i, seq in enumerate(sequences):
        offsets[i + 1] = offsets[i] + len(seq["event_texts"])
    meta = (np.concatenate([s["event_meta"] for s in sequences], axis=0)
            if offsets[-1] else np.zeros((0, META_DIM), dtype=np.float32))
    texts = [text for seq in sequences for text in seq["event_texts"]]

    manifest = {
        "format_version": INPUT_FORMAT_VERSION,
        "encoder_id": ENCODER_ID,
        "encoder_revision": ENCODER_REVISION,
        "encoder_weights_sha256": weight_sha,
        "input_csv_sha256": sha256_file(csv_path),
        "rows": len(frame),
        "events": int(offsets[-1]),
        "embedding_dim": EMBEDDING_DIM,
        "max_wordpieces_per_event": MAX_WORDPIECES,
        "semantic_input_sha256": hashlib.sha256(
            "\0".join(texts).encode("utf-8")).hexdigest(),
    }
    manifest_path = cache_dir / "manifest.json"
    embedding_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "event_meta.npy"
    offsets_path = cache_dir / "event_offsets.npy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if all(p.is_file() for p in (manifest_path, embedding_path, meta_path,
                                 offsets_path)):
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            arr = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            cached_meta = np.load(meta_path, mmap_mode="r", allow_pickle=False)
            cached_offsets = np.load(offsets_path, mmap_mode="r",
                                     allow_pickle=False)
            same_inputs = all(old.get(key) == value
                              for key, value in manifest.items())
            same_shapes = (arr.shape ==
                           (int(offsets[-1]), EMBEDDING_DIM)
                           and np.array_equal(cached_meta, meta)
                           and np.array_equal(cached_offsets, offsets))
            same_bytes = (old.get("embedding_cache_sha256") ==
                          sha256_file(embedding_path))
            if same_inputs and same_shapes and same_bytes:
                return EmbeddingCache(cache_dir, len(frame), offsets, cached_meta)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, use_fast=True)
    encoder = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True)
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    embeddings = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float16,
        shape=(int(offsets[-1]), EMBEDDING_DIM))
    start_time = time.time()
    with torch.inference_mode():
        for start in range(0, len(texts), ENCODER_BATCH_SIZE):
            batch = texts[start:start + ENCODER_BATCH_SIZE]
            tokens = tokenizer(batch, padding=True, truncation=True,
                               max_length=MAX_WORDPIECES,
                               return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings[start:start + len(batch)] = pooled.cpu().numpy().astype(
                np.float16)
            if start == 0 or start + len(batch) == len(texts):
                log("encoder: %d/%d goal+event strings" %
                    (min(start + len(batch), len(texts)), len(texts)))
    embeddings.flush()
    del encoder, tokenizer
    np.save(meta_path, meta, allow_pickle=False)
    np.save(offsets_path, offsets, allow_pickle=False)
    manifest["embedding_cache_sha256"] = sha256_file(embedding_path)
    manifest["encode_seconds"] = round(time.time() - start_time, 2)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    log("encoder pass: %.1fs; events kept=%d (no event truncation)" %
        (manifest["encode_seconds"], int(offsets[-1])))
    return EmbeddingCache(cache_dir, len(frame), offsets, meta)


def _build_network(torch):
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    class TurnClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.meta = nn.Sequential(nn.Linear(META_DIM, 16), nn.GELU())
            self.project = nn.Sequential(
                nn.Linear(EMBEDDING_DIM + 16, 192), nn.GELU())
            self.gru = nn.GRU(192, HIDDEN_SIZE, batch_first=True,
                              bidirectional=True)
            self.dropout = nn.Dropout(DROPOUT)
            self.label = nn.Linear(2 * HIDDEN_SIZE, len(LABELS))

        def forward(self, semantic, metadata, lengths):
            side = self.meta(metadata)
            turns = self.project(torch.cat((semantic, side), dim=-1))
            packed = pack_padded_sequence(turns, lengths.cpu(),
                                          batch_first=True,
                                          enforce_sorted=False)
            _, state = self.gru(packed)
            pooled = torch.cat((state[-2], state[-1]), dim=1)
            return self.label(self.dropout(pooled))

    return TurnClassifier


def train_semantic_fold(cache, y, train_rows, valid_rows, seed, device, log=print):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    Model = _build_network(torch)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    def tensors(rows):
        semantic, metadata, lengths = cache.batch(rows)
        return (torch.from_numpy(semantic).to(device),
                torch.from_numpy(metadata).to(device),
                torch.as_tensor(lengths, dtype=torch.long),
                torch.as_tensor(y[rows], dtype=torch.long, device=device))

    rng = np.random.default_rng(seed)
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), TRAIN_BATCH_SIZE):
            rows = order[start:start + TRAIN_BATCH_SIZE]
            semantic, metadata, lengths, target = tensors(rows)
            optimizer.zero_grad(set_to_none=True)
            logits = model(semantic, metadata, lengths)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        log("semantic fold seed=%d epoch=%d/%d loss=%.5f" %
            (seed, epoch + 1, EPOCHS, float(np.mean(losses))))

    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(valid_rows), TRAIN_BATCH_SIZE):
            rows = valid_rows[start:start + TRAIN_BATCH_SIZE]
            semantic, metadata, lengths, _ = tensors(rows)
            outputs.append(torch.softmax(model(semantic, metadata, lengths),
                                         dim=1).cpu().numpy())
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _score_rows(rows, probabilities, d, robust, baseline, helpers):
    metrics = helpers.metrics
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
    success_true = np.asarray(d["train"]["success"], dtype=np.int64)[rows]
    success_f1 = metrics.binary_f1(success_true.tolist(),
                                   np.asarray(baseline["success"])[rows].tolist())
    # Exp15's class-conditioned Exp13 L1 peaks are unchanged; only the class
    # chosen by the label branch selects a column from the verified fold cache.
    turns = np.asarray(d["peak_pos"])[rows][np.arange(len(rows)), pred]
    faulty = np.flatnonzero(true != 0)
    fault_hit2 = metrics.fault_turn_hit_at_k(
        [yt[i] for i in faulty], [yp[i] for i in faulty],
        np.asarray(d["train"]["fault_turn"], dtype=np.int64)[rows][faulty].tolist(),
        np.asarray(turns, dtype=np.int64)[faulty].tolist(), k=2)
    composite = metrics.composite({"macro_f1": macro,
                                   "robustness_f1": robust_f1,
                                   "success_f1": success_f1,
                                   "fault_turn_hit2": fault_hit2})
    return {"macro_f1": macro, "robustness_f1": robust_f1,
            "success_f1": success_f1, "fault_turn_hit2": fault_hit2,
            "composite": composite, "per_class_f1": class_f1}


def _load_helpers():
    if str(EXP15_DIR) not in sys.path:
        sys.path.insert(0, str(EXP15_DIR))
    exp15 = importlib.import_module("run_experiment")
    metrics = importlib.import_module("metrics")
    expected = (ROOT / "evaluation" / "metrics.py").resolve()
    if Path(metrics.__file__).resolve() != expected:
        raise RuntimeError("metrics import did not resolve to official evaluator")
    return type("Helpers", (), {"exp15": exp15, "metrics": metrics})


def _validate_baseline(d, robust, helpers):
    baseline = pd.read_csv(BASELINE_CSV)
    train_ids = [str(value) for value in d["train"]["run_id"]]
    if baseline["run_id"].astype(str).tolist() != train_ids:
        raise AssertionError("frozen Exp15 predictions are not row-aligned")
    manifest = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    if manifest["baseline_commit"] != "fdd588c4412c7cc5cb4f6a1f18bc1d17cbeb264a":
        raise AssertionError("frozen production baseline commit changed")
    if sha256_file(BASELINE_CSV) != manifest["baseline_oof_prediction_sha256"]:
        raise AssertionError("frozen Exp15 OOF prediction file changed")
    if sha256_file(ROOT / "data" / "train.csv") != manifest["train_csv_sha256"]:
        raise AssertionError("the local train input differs from the frozen baseline")
    if sha256_file(ROOT / "submission_exp15_success_label_stacking.zip") != \
            manifest["baseline_zip_sha256"]:
        raise AssertionError("Exp15 production ZIP changed after baseline freeze")
    p_base = np.load(EXP13_DIR / "oof_D_corrected_wg.npy", allow_pickle=False)
    if p_base.shape != (len(train_ids), len(LABELS)):
        raise AssertionError("Exp13 OOF probabilities have unexpected shape")
    expected_hash = manifest["exp13_oof_probability_sha256"]
    if sha256_file(EXP13_DIR / "oof_D_corrected_wg.npy") != expected_hash:
        raise AssertionError("Exp13 OOF probability artifact changed")
    pred = p_base.argmax(1)
    if not np.array_equal(np.asarray(baseline["label"], dtype=object),
                          np.asarray(LABELS, dtype=object)[pred]):
        raise AssertionError("frozen Exp15 labels differ from Exp13 OOF")
    turns = np.asarray(d["peak_pos"])[np.arange(len(pred)), pred]
    if not np.array_equal(np.asarray(baseline["fault_turn"], dtype=np.int64),
                          turns.astype(np.int64)):
        raise AssertionError("frozen Exp15 fault_turn differs from Exp13 L1")
    scored = _score_rows(np.arange(len(pred)), p_base, d, robust,
                         {"success": baseline["success"].to_numpy()}, helpers)
    for name, expected_value in BASELINE_METRICS.items():
        if abs(float(scored[name]) - expected_value) > 1e-8:
            raise AssertionError("Exp15 baseline %s mismatch: %.12f vs %.12f" %
                                 (name, scored[name], expected_value))
    return baseline, p_base, scored


def run_cv(args):
    helpers = _load_helpers()
    exp15 = helpers.exp15
    d = exp15.ENS.load_verified()
    y = np.asarray(d["yi"], dtype=np.int64)
    robust = np.asarray(exp15.ENS.robustness_mask(), dtype=bool)
    baseline, p_base, baseline_metrics = _validate_baseline(d, robust, helpers)
    log_lines = []

    def log(message=""):
        print(message, flush=True)
        log_lines.append(str(message))

    log("EXP16 frozen semantic turn sequence; no production changes")
    log("baseline commit=%s composite=%.10f Macro=%.10f Robustness=%.10f" %
        (json.loads(BASELINE_JSON.read_text(encoding="utf-8"))["baseline_commit"],
         baseline_metrics["composite"], baseline_metrics["macro_f1"],
         baseline_metrics["robustness_f1"]))
    log("encoder=%s@%s; frozen 384d; mean pool; max_wordpieces=%d" %
        (ENCODER_ID, ENCODER_REVISION, MAX_WORDPIECES))
    log("event order=goal, then t / message->state->artifact; no truncation")
    log("fusion=%.2f structural + %.2f semantic probabilities" %
        (1.0 - FUSION_WEIGHT, FUSION_WEIGHT))

    train = d["train"]
    model_dir = Path(args.encoder_dir).resolve()
    cache = build_embeddings(train, model_dir, Path(args.cache_dir).resolve(),
                             ROOT / "data" / "train.csv", args.device, log=log)
    if cache.n_runs != len(train):
        raise AssertionError("embedding cache row count mismatch")

    c, window_data = exp15.ENS._prepare()
    wg = exp15.W.build_matrix(c["runs"])
    all_rows = np.arange(len(y), dtype=np.int64)
    folds = [np.asarray(valid, dtype=np.int64) for valid in d["folds"]]
    sealed = [np.asarray(valid, dtype=np.int64) for _, valid in
              StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
              .split(np.zeros(len(y)), y)]
    if any(not np.array_equal(a, b) for a, b in zip(folds, sealed)):
        raise AssertionError("outer folds differ from Exp15 seed-0 folds")

    p_structural = np.full_like(p_base, np.nan, dtype=np.float32)
    p_semantic = np.full_like(p_base, np.nan, dtype=np.float32)
    p_candidate = np.full_like(p_base, np.nan, dtype=np.float32)
    seen = np.zeros(len(y), dtype=np.int8)
    fold_results = []
    t_cv = time.time()
    for fold, valid in enumerate(folds):
        t_fold = time.time()
        train_rows = all_rows[~np.isin(all_rows, valid)]
        if np.intersect1d(train_rows, valid).size:
            raise AssertionError("outer fit/hold overlap")
        xagg_train, xagg_valid, _ = exp15.corrected_agg_block(
            train_rows, valid, c, window_data, "exp16 outer %d" % fold,
            log=log)
        x_train = exp15.label_matrix_312(train_rows, xagg_train, c, wg)
        x_valid = exp15.label_matrix_312(valid, xagg_valid, c, wg)
        if x_train.shape != (len(train_rows), 312) or x_valid.shape != (len(valid), 312):
            raise AssertionError("Exp13 structural matrix is not 312 columns")

        structural = exp15.lgb.LGBMClassifier(
            **exp15.ENS.label_params(42)).fit(x_train, y[train_rows])
        if not np.array_equal(structural.classes_, np.arange(len(LABELS))):
            raise AssertionError("structural class order changed")
        p_struct = structural.predict_proba(x_valid).astype(np.float32)
        if not np.array_equal(p_struct.argmax(1), p_base[valid].argmax(1)):
            raise AssertionError("Exp13 312-feature OOF baseline did not reproduce")

        p_semantic_fold = train_semantic_fold(
            cache, y, train_rows, valid, FOLD_MODEL_SEEDS[fold], args.device,
            log=log)
        p_structural[valid] = p_struct
        p_semantic[valid] = p_semantic_fold
        p_candidate[valid] = ((1.0 - FUSION_WEIGHT) * p_struct +
                              FUSION_WEIGHT * p_semantic_fold)
        seen[valid] += 1
        base_fold = _score_rows(valid, p_base[valid], d, robust,
                                {"success": baseline["success"].to_numpy()}, helpers)
        structural_fold = _score_rows(
            valid, p_struct, d, robust,
            {"success": baseline["success"].to_numpy()}, helpers)
        semantic_fold = _score_rows(
            valid, p_semantic_fold, d, robust,
            {"success": baseline["success"].to_numpy()}, helpers)
        candidate_fold = _score_rows(valid, p_candidate[valid], d, robust,
                                     {"success": baseline["success"].to_numpy()}, helpers)
        fold_results.append({"fold": fold, "n_train": len(train_rows),
                             "n_valid": len(valid), "baseline": base_fold,
                             "structural_diagnostic": structural_fold,
                             "semantic_diagnostic": semantic_fold,
                             "candidate": candidate_fold,
                             "composite_delta": candidate_fold["composite"] -
                                               base_fold["composite"],
                             "seconds": round(time.time() - t_fold, 2)})
        log("fold %d composite %.6f -> %.6f (%+.6f), %.1fs" %
            (fold, base_fold["composite"], candidate_fold["composite"],
             fold_results[-1]["composite_delta"], fold_results[-1]["seconds"]))
        del structural, x_train, x_valid, xagg_train, xagg_valid

    if (not np.all(seen == 1) or not np.isfinite(p_structural).all() or
            not np.isfinite(p_semantic).all() or
            not np.isfinite(p_candidate).all()):
        raise AssertionError("outer OOF coverage is incomplete or non-finite")
    rows = all_rows
    structural_metrics = _score_rows(
        rows, p_structural, d, robust,
        {"success": baseline["success"].to_numpy()}, helpers)
    semantic_metrics = _score_rows(
        rows, p_semantic, d, robust,
        {"success": baseline["success"].to_numpy()}, helpers)
    candidate_metrics = _score_rows(rows, p_candidate, d, robust,
                                    {"success": baseline["success"].to_numpy()}, helpers)
    deltas = {key: candidate_metrics[key] - baseline_metrics[key]
              for key in ("macro_f1", "robustness_f1", "success_f1",
                          "fault_turn_hit2", "composite")}
    class_deltas = {label: candidate_metrics["per_class_f1"][label] -
                    baseline_metrics["per_class_f1"][label] for label in LABELS}
    fold_wins = sum(row["composite_delta"] > 0 for row in fold_results)
    checks = {
        "composite_at_least_plus_0_005": deltas["composite"] >= 0.005,
        "macro_f1_improves": deltas["macro_f1"] > 0,
        "robustness_f1_improves": deltas["robustness_f1"] > 0,
        "at_least_2_of_3_folds_win": fold_wins >= 2,
        "no_class_drops_more_than_0_02": min(class_deltas.values()) >= -0.02,
    }
    passed = all(checks.values())
    result = {
        "decision": "CONTINUE_TO_FAULT_TURN" if passed else "STOP",
        "hypothesis": "frozen semantic turn representations add information missing from Exp13 312 structural features",
        "baseline": baseline_metrics,
        "structural_diagnostic": structural_metrics,
        "semantic_diagnostic": semantic_metrics,
        "candidate": candidate_metrics,
        "deltas": deltas,
        "per_class_f1_delta": class_deltas,
        "folds_won": fold_wins,
        "folds": fold_results,
        "gate": {"frozen": GATE, "checks": checks, "passed": passed},
        "runtime_seconds": round(time.time() - t_cv, 2),
        "encoder_revision": ENCODER_REVISION,
        "event_truncation": "none; tokenizer max length applies per event",
        "fusion_weight_semantic": FUSION_WEIGHT,
        "fault_turn_head": "unchanged Exp13 L1 peaks; no semantic localizer trained",
    }
    np.save(HERE / "structural_label_proba.npy", p_structural,
            allow_pickle=False)
    np.save(HERE / "semantic_label_proba.npy", p_semantic,
            allow_pickle=False)
    np.save(HERE / "candidate_label_proba.npy", p_candidate,
            allow_pickle=False)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n",
                                       encoding="utf-8")
    log("gate: %s (%s)" % ("PASS" if passed else "STOP",
                           ", ".join(k for k, value in checks.items() if not value)
                           or "all checks passed"))
    log("decision: %s; elapsed %.1fs" % (result["decision"],
                                          result["runtime_seconds"]))
    (HERE / "run.log").write_text("\n".join(log_lines) + "\n",
                                   encoding="utf-8")
    return 0 if passed else 2


def _peak_rss_mb():
    if os.name == "nt":
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged_pool",
                    "paged_pool", "peak_nonpaged_pool", "nonpaged_pool",
                    "pagefile", "peak_pagefile")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
            return round(counters.peak_working_set / (1024 ** 2), 1)
        return None
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def run_cpu_benchmark(args):
    import torch

    if not 500 <= args.benchmark_runs <= 1000:
        raise ValueError("CPU benchmark must use 500 to 1000 runs")
    torch.set_num_threads(8)
    helpers = _load_helpers()
    d = helpers.exp15.ENS.load_verified()
    robust = np.asarray(helpers.exp15.ENS.robustness_mask(), dtype=bool)
    _validate_baseline(d, robust, helpers)
    train = d["train"]
    sample_rows = np.random.default_rng(1616).choice(
        len(train), size=args.benchmark_runs, replace=False)
    sample = train.iloc[sample_rows].reset_index(drop=True)
    model_dir = Path(args.encoder_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() / (
        "cpu_benchmark_%d" % args.benchmark_runs)
    started = time.time()
    cache = build_embeddings(sample, model_dir, cache_dir,
                             ROOT / "data" / "train.csv", "cpu")
    build_seconds = time.time() - started
    manifest = json.loads((cache_dir / "manifest.json").read_text(
        encoding="utf-8"))
    encode_seconds = float(manifest.get("encode_seconds", build_seconds))

    model = _build_network(torch)().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()
    y = np.asarray(d["yi"], dtype=np.int64)[sample_rows]
    batch_seconds, batch_lengths = [], []
    for batch_index in range(5):
        rows = np.arange(batch_index * TRAIN_BATCH_SIZE,
                         (batch_index + 1) * TRAIN_BATCH_SIZE)
        semantic, metadata, lengths = cache.batch(rows)
        batch_lengths.append(int(lengths.max()))
        semantic_t = torch.from_numpy(semantic)
        metadata_t = torch.from_numpy(metadata)
        lengths_t = torch.as_tensor(lengths, dtype=torch.long)
        target = torch.as_tensor(y[rows], dtype=torch.long)
        batch_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(semantic_t, metadata_t, lengths_t), target)
        loss.backward()
        optimizer.step()
        elapsed = time.time() - batch_started
        if batch_index:
            batch_seconds.append(elapsed)
    del model, optimizer

    events = int(cache.offsets[-1])
    events_per_run = events / args.benchmark_runs
    full_encode_10k = encode_seconds * 10000 / args.benchmark_runs
    steps_per_fold = EPOCHS * math.ceil((2 * 10000 / 3) / TRAIN_BATCH_SIZE)
    step_seconds = float(np.mean(batch_seconds))
    semantic_fold_seconds = step_seconds * steps_per_fold
    structural_cv_reference = 336.6 + 185.7 + 180.8
    one_fold_seconds = semantic_fold_seconds + structural_cv_reference / 3
    cv_seconds = full_encode_10k + 3 * one_fold_seconds
    max_events = max(batch_lengths)
    batch_working_set_mb = (TRAIN_BATCH_SIZE * max_events *
                            (EMBEDDING_DIM + META_DIM + 16 + 192 +
                             2 * HIDDEN_SIZE) * 4 * 3 / (1024 ** 2))
    peak_rss_mb = _peak_rss_mb()
    full_events_est = events_per_run * 10000
    full_embedding_cache_mb = full_events_est * EMBEDDING_DIM * 2 / (1024 ** 2)
    full_metadata_mb = full_events_est * META_DIM * 4 / (1024 ** 2)
    full_rss_estimate_mb = (None if peak_rss_mb is None else round(
        (peak_rss_mb + full_embedding_cache_mb + full_metadata_mb) * 1.25, 1))
    result = {
        "device": "cpu",
        "torch": torch.__version__,
        "transformers": importlib.import_module("transformers").__version__,
        "safetensors": importlib.import_module("safetensors").__version__,
        "torch_threads": torch.get_num_threads(),
        "benchmark_runs": args.benchmark_runs,
        "benchmark_events": events,
        "events_per_run": round(events_per_run, 2),
        "encoder_seconds": round(encode_seconds, 2),
        "encoder_events_per_second": round(events / encode_seconds, 2),
        "estimated_encoder_10k_seconds": round(full_encode_10k, 2),
        "gru_batches_measured": len(batch_seconds),
        "gru_batch_mean_seconds": round(step_seconds, 2),
        "gru_batch_max_events": max_events,
        "estimated_semantic_fold_train_seconds": round(semantic_fold_seconds, 2),
        "structural_3fold_reference_seconds": structural_cv_reference,
        "estimated_one_fold_seconds": round(one_fold_seconds, 2),
        "estimated_3fold_cv_seconds": round(cv_seconds, 2),
        "estimated_3fold_cv_hours": round(cv_seconds / 3600, 3),
        "estimated_batch_working_set_mb": round(batch_working_set_mb, 1),
        "full_10k_embedding_cache_mb": round(full_embedding_cache_mb, 1),
        "full_10k_event_metadata_mb": round(full_metadata_mb, 1),
        "peak_process_rss_mb": peak_rss_mb,
        "estimated_full_run_peak_rss_mb": full_rss_estimate_mb,
        "structural_runtime_reference": "Exp13 run_production.log fold matrix build times",
        "full_cv_ready": cv_seconds <= 7200,
    }
    (HERE / "cpu_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["full_cv_ready"] else 3


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-dir", default=str(CACHE_DIR / "encoder"))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="allow the heavy CV to run without the target A100")
    parser.add_argument("--benchmark-runs", type=int, default=None,
                        help="measure fixed CPU throughput on 500-1000 runs; do not run CV")
    parser.add_argument("--check-only", action="store_true",
                        help="verify Exp15 baseline and frozen CV contract without fitting")
    args = parser.parse_args(argv)

    if args.check_only:
        helpers = _load_helpers()
        d = helpers.exp15.ENS.load_verified()
        robust = np.asarray(helpers.exp15.ENS.robustness_mask(), dtype=bool)
        _, _, baseline = _validate_baseline(d, robust, helpers)
        print("Exp15 baseline verified: composite=%.10f Macro=%.10f Robustness=%.10f" %
              (baseline["composite"], baseline["macro_f1"],
               baseline["robustness_f1"]))
        print("Exp16 lock verified: folds=3/seed0, turns=all, fusion=50:50, "
              "one encoder, gate=+0.005 composite")
        return 0

    if args.benchmark_runs is not None:
        args.device = "cpu"
        return run_cpu_benchmark(args)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable; no encoder pass or fit started") from exc
    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/A100 unavailable; no encoder pass or fit started")
    if not args.device.startswith("cuda") and not args.allow_cpu:
        raise RuntimeError("CPU run disabled by default; pass --allow-cpu explicitly")
    torch.set_num_threads(8)
    return run_cv(args)


if __name__ == "__main__":
    raise SystemExit(main())
