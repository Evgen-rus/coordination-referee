"""One frozen semantic-sequence experiment against the Exp15 label baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
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
INPUT_FORMAT_VERSION = 1

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
META_NAMES = ("relative_position", "intent_present", "refs_present",
              "log_refs_count", "log_artifact_count_so_far",
              "log_artifacts_this_turn", "log_state_count_so_far",
              "log_state_updates_this_turn")
META_DIM = len(META_NAMES)

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


def _event_turn(event):
    value = event.get("t")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("event timestamp is not an integer: %r" % value) from exc


def prepare_run(record):
    """Build a separate goal string and ordered, per-message encoder inputs.

    Only run-content columns are read. No run_id or target is used as input.
    Artifact/state progress is causal: counts at turn t include records with
    timestamp <= t, never records from the future.
    """
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
    art_times = [_event_turn(event) for event in artifacts
                 if isinstance(event, dict)]
    state_times = [_event_turn(event) for event in states
                   if isinstance(event, dict)]
    art_at = Counter(t for t in art_times if t is not None)
    state_at = Counter(t for t in state_times if t is not None)
    known_art = sorted(t for t in art_times if t is not None)
    known_state = sorted(t for t in state_times if t is not None)

    turn_texts, metadata = [], []
    prior_t = -1
    denom = max(1, len(messages) - 1)
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("message %d is not an object" % index)
        turn = _event_turn(message)
        turn = index if turn is None else turn
        if turn < prior_t:
            raise ValueError("messages are not in chronological order")
        prior_t = turn

        sender = _text(message.get("from"), "unknown")
        receiver = _text(message.get("to"), "unknown")
        has_intent = bool(str(message.get("intent") or "").strip())
        refs = message.get("refs") or []
        if not isinstance(refs, list):
            refs = [refs]
        tool = _text(message.get("tool"), "[none]")
        turn_texts.append(
            "message type: %s; sender id: %s; sender role: %s; "
            "receiver id: %s; receiver role: %s; intent present: %s; "
            "tool: %s; text: %s" % (
                _text(message.get("type"), "unknown"), sender,
                roles.get(sender, "unknown"), receiver,
                roles.get(receiver, "unknown"),
                "yes" if has_intent else "no", tool,
                _text(message.get("text"))))
        metadata.append((
            index / denom,
            float(has_intent),
            float(bool(refs)),
            math.log1p(len(refs)),
            math.log1p(sum(t <= turn for t in known_art)),
            math.log1p(art_at[turn]),
            math.log1p(sum(t <= turn for t in known_state)),
            math.log1p(state_at[turn]),
        ))

    return {
        "goal_text": "goal: %s" % _text(record.get("goal")),
        "turn_texts": turn_texts,
        "turn_meta": np.asarray(metadata, dtype=np.float32).reshape(-1, META_DIM),
    }


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
        lengths = self.offsets[rows + 1] - self.offsets[rows] + 1
        width = int(lengths.max())
        semantic = np.zeros((len(rows), width, EMBEDDING_DIM), dtype=np.float32)
        metadata = np.zeros((len(rows), width, META_DIM), dtype=np.float32)
        for slot, row in enumerate(rows):
            lo, hi = int(self.offsets[row]), int(self.offsets[row + 1])
            semantic[slot, 0] = self.embeddings[row]
            if hi > lo:
                semantic[slot, 1:hi - lo + 1] = self.embeddings[
                    self.n_runs + lo:self.n_runs + hi]
                metadata[slot, 1:hi - lo + 1] = self.meta[lo:hi]
        return semantic, metadata, lengths.astype(np.int64)


def build_embeddings(frame: pd.DataFrame, model_dir: Path, cache_dir: Path,
                     csv_path: Path, device: str, log=print):
    """Cache frozen mean-pooled goal/turn embeddings; preserve all turns."""
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
        offsets[i + 1] = offsets[i] + len(seq["turn_texts"])
    meta = (np.concatenate([s["turn_meta"] for s in sequences], axis=0)
            if offsets[-1] else np.zeros((0, META_DIM), dtype=np.float32))
    texts = [s["goal_text"] for s in sequences]
    texts.extend(text for seq in sequences for text in seq["turn_texts"])

    manifest = {
        "format_version": INPUT_FORMAT_VERSION,
        "encoder_id": ENCODER_ID,
        "encoder_revision": ENCODER_REVISION,
        "encoder_weights_sha256": weight_sha,
        "input_csv_sha256": sha256_file(csv_path),
        "rows": len(frame),
        "turns": int(offsets[-1]),
        "embedding_dim": EMBEDDING_DIM,
        "max_wordpieces_per_goal_or_turn": MAX_WORDPIECES,
    }
    manifest_path = cache_dir / "manifest.json"
    embedding_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "turn_meta.npy"
    offsets_path = cache_dir / "turn_offsets.npy"
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
                           (len(frame) + int(offsets[-1]), EMBEDDING_DIM)
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
        shape=(len(frame) + int(offsets[-1]), EMBEDDING_DIM))
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
                log("encoder: %d/%d goal+turn strings" %
                    (min(start + len(batch), len(texts)), len(texts)))
    embeddings.flush()
    del encoder, tokenizer
    np.save(meta_path, meta, allow_pickle=False)
    np.save(offsets_path, offsets, allow_pickle=False)
    manifest["embedding_cache_sha256"] = sha256_file(embedding_path)
    manifest["encode_seconds"] = round(time.time() - start_time, 2)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    log("encoder pass: %.1fs; turns kept=%d (no turn truncation)" %
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
    log("turn truncation=none; every message kept in original order")
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

        p_semantic = train_semantic_fold(
            cache, y, train_rows, valid, FOLD_MODEL_SEEDS[fold], args.device,
            log=log)
        p_candidate[valid] = ((1.0 - FUSION_WEIGHT) * p_struct +
                              FUSION_WEIGHT * p_semantic)
        seen[valid] += 1
        base_fold = _score_rows(valid, p_base[valid], d, robust,
                                {"success": baseline["success"].to_numpy()}, helpers)
        candidate_fold = _score_rows(valid, p_candidate[valid], d, robust,
                                     {"success": baseline["success"].to_numpy()}, helpers)
        fold_results.append({"fold": fold, "n_train": len(train_rows),
                             "n_valid": len(valid), "baseline": base_fold,
                             "candidate": candidate_fold,
                             "composite_delta": candidate_fold["composite"] -
                                               base_fold["composite"],
                             "seconds": round(time.time() - t_fold, 2)})
        log("fold %d composite %.6f -> %.6f (%+.6f), %.1fs" %
            (fold, base_fold["composite"], candidate_fold["composite"],
             fold_results[-1]["composite_delta"], fold_results[-1]["seconds"]))
        del structural, x_train, x_valid, xagg_train, xagg_valid

    if not np.all(seen == 1) or not np.isfinite(p_candidate).all():
        raise AssertionError("outer OOF coverage is incomplete or non-finite")
    rows = all_rows
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
        "candidate": candidate_metrics,
        "deltas": deltas,
        "per_class_f1_delta": class_deltas,
        "folds_won": fold_wins,
        "folds": fold_results,
        "gate": {"frozen": GATE, "checks": checks, "passed": passed},
        "runtime_seconds": round(time.time() - t_cv, 2),
        "encoder_revision": ENCODER_REVISION,
        "turn_truncation": "none",
        "fusion_weight_semantic": FUSION_WEIGHT,
        "fault_turn_head": "unchanged Exp13 L1 peaks; no semantic localizer trained",
    }
    np.save(HERE / "candidate_label_proba.npy", p_candidate.astype(np.float32),
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-dir", default=str(CACHE_DIR / "encoder"))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="allow the heavy CV to run without the target A100")
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
