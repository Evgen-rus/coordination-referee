# Exp19 — StepFinder-inspired fault-turn localizer

**Status: pre-CV plan. No Exp19 training or CV result has been observed.**
Exp15 remains the production baseline; classification, success, and production
files are frozen.

## Hypothesis and scope

Replace only the Exp13 L1 `fault_turn` output with a learned, class-conditioned
full-sequence step scorer. The scorer sees frozen Exp16 MiniLM event vectors
and all observable event context, while its outputs are restricted to message
turns. The hypothesis is that agent-aware temporal scoring identifies the
primary fault more accurately than the Exp13 per-class window-probability peak.

The label predictions, success predictions, and all production behavior stay
fixed. The public leaderboard is not used. No production ZIP is built unless
the complete CV gate passes.

## Frozen inputs and alignment

- Read only `.cache/exp16_semantic_sequence/`; do not run the encoder. Require
  the pinned Exp16 manifest, input CSV SHA, MiniLM weight SHA, embedding-file
  SHA, semantic-text digest, `event_meta`, and `event_offsets` to match.
- Frozen cache evidence at plan time: 10,000 rows, 590,146 events, embeddings
  `(590146, 384)` float16, metadata `(590146, 12)`, and 10,001 offsets. The
  train CSV SHA is `8e8feb4ee284d2acd918b533921631fa905d47adf8d34881f89359964a1bcfbc`;
  the embedding SHA is
  `1602d13b63480346f3c63932f49978e34ec6edc9cf450a88d6f1222f40e39220`.
- Bind rows to the original CSV and frozen Exp15 OOF by ordered `run_id`; all
  10,000 IDs must match exactly and be unique. The source CSV SHA is the cache
  row-order seal.
- Reuse Exp16 event preparation exactly: goal first, then all observable
  messages, shared-state writes, and artifacts sorted by original numeric `t`,
  fixed tie order `message -> state -> artifact`, then source-list index.
  Untimed events follow timed events in that same type/source order. Never
  rewrite or increment source `t` after inserting context events.
- Reconstruct and retain, for every event, its type, original source index,
  original `t`, and message-turn mapping. Assert the Exp16 text digest,
  metadata, and offsets before fitting. The source schema defines message
  `t` as dense zero-based and equal to message-list position; verify this for
  all runs and map each message's `t` to its event-sequence position. All
  457,203 message events and all context events remain; no sequence truncation.
- All event vectors and metadata are inputs only. `label`, `success`, and
  `fault_turn` are absent from input construction. Training targets are joined
  separately after input/cache validation.

## Frozen architecture

Use the existing frozen 384d Exp16 MiniLM event embeddings, already mean
pooled, L2-normalized, and stored as float16. Convert to float32 for training.
Project content `384 -> 128`; project the 12 Exp16 side-metadata values to 128
and add them to the content projection.

For the 32d agent-aware representation, use a trainable embedding of the event
actor's **role string**, never its raw random agent ID. Message actor is
`from`; artifact actor is `by`/`agent`; state actor is `agent`/`by`; goal and
unresolved roles use an unknown bucket. For message events, actor role is the
sender role; receiver role is already present in the frozen semantic text.
Build each fold's role vocabulary from outer-train input rows only; unseen
validation roles map to unknown.

Condition the shared sequence model on one fault class with a learned `6 x 32`
class embedding projected to 128 and added to every event's 128d input. For
training rows use only that row's true fault class. For validation use only its
frozen Exp15 predicted class. Never pass a validation target class.

Port the StepFinder model components and keep these settings fixed:

- LayerNorm -> two-layer bidirectional LSTM (`hidden=64` each direction) ->
  LayerNorm, giving 128d contextual event states.
- Agent-aware residual attention: 2 heads x 32d; cosine role-embedding bias
  scaled by `alpha=0.1`; masked mean agent gate with the same alpha.
- Step score: MLP `128 -> 64 -> 1` plus `beta=0.9` times the mean normalized
  multi-scale temporal difference at scales `[1, 2]`.
- Set `gamma=0`: no late-turn penalty.
- Auxiliary temporal head predicts the next contextual state. Train with
  masked single-positive cross-entropy over **message events only** plus
  `lambda_temporal=0.9` times the source-compatible temporal consistency MSE
  over every valid event transition.
- Fixed training: 12 epochs, batch size 16, AdamW, learning rate `0.001`,
  weight decay `0.00001`, dropout `0.5`, seed `42`. Retain StepFinder's
  gradient-norm clipping at `1.0`. Use the final epoch; no early stopping,
  checkpoint selection, scheduler, or tuning.
- Dynamic right padding and packed LSTM sequences; no turn/event cap. Keep a
  separate valid-event mask and message-output mask. Artifacts, state writes,
  and goal affect context/auxiliary loss but can never be selected as output.

The source basis is StepFinder `model.py`, `feature_construction.py`,
`collate_fn.py`, and `main.py`. Do not port its test-accuracy checkpoint
selection or its Qwen3 encoder. Preserve the StepFinder MIT copyright and
permission notice with any substantial copied source in
`THIRD_PARTY_NOTICES.md`.

## Frozen CV and scoring

- Before any model fit, reproduce the Exp15 baseline from its frozen OOF
  artifacts: Composite `0.7938624419`, hit@2 `0.6315277778`, plus exact Macro,
  Robustness, Success, predicted labels, success predictions, and run order.
  Use the official evaluator. If any baseline check differs, stop before fit.
- Use the exact existing 3 outer folds:
  `StratifiedKFold(3, shuffle=True, random_state=0)`, and assert them against
  the Exp15 verified folds and frozen OOF row IDs.
- For each fold, fit only on outer-train rows with a true fault class and
  labeled, valid `fault_turn`. One positive target is the message event whose
  unchanged source `t` equals `fault_turn`. No outer-validation row or target
  is included in training, epoch selection, or preprocessing.
- At validation, condition on the frozen Exp15 OOF predicted label. If it is
  `clean`, emit `-1`; otherwise choose the highest score among that run's
  message-event positions only and return the mapped original message `t`.
- Keep Exp15 labels and success predictions byte-for-byte unchanged. Score
  `fault_turn_hit_at_k(..., k=2)` with `evaluation/metrics.py`; classwise hit@2
  uses the same official hit condition restricted to each true fault class.
- Baseline turns are Exp13 `peak_pos[row, frozen_predicted_class]` loaded
  through the existing fail-closed verified Exp15/Exp13 helper. Do not rebuild
  or tune the L1 peaks.

## Frozen promotion gate

Compare the full OOF localizer against the frozen Exp13 L1 turns, holding
labels and success predictions fixed. All conditions must pass:

1. hit@2 improves by at least `+0.05`.
2. Composite improves by at least `+0.005`.
3. hit@2 improves in at least 2/3 folds.
4. No fault class's hit@2 falls by more than `0.05` absolute.
5. Label and success predictions remain exactly unchanged.
6. No leakage and no validation-based model selection.

Because the only changed score component is hit@2 with weight `0.10`, the
first two thresholds are mathematically equivalent; check and report both.
Any failure means **STOP**: do not change settings, relax the gate, or create a
production ZIP.

## Checks and runtime

Before benchmark/CV, implement tests for: exact event-to-message turn mapping;
message-only outputs; clean -> `-1`; target perturbation invariance; outer-fold
disjointness; train-true / validation-predicted class conditioning; every
message turn retained; deterministic inference; and official metric parity.
Run those tests and the frozen baseline-parity check before any fit.

Then benchmark the complete localizer on 256 deterministic outer-fold-0
faulty training sequences, stratified across sequence-length quartiles, for one
epoch. Measure wall time and peak process memory; extrapolate the full three
folds at 12 epochs. If the projected CPU CV exceeds 120 minutes, stop before
full CV and report the estimate. Run no other heavy training concurrently.

If the gate passes, separately measure production feasibility: offline bundle
size, full 10,000-train plus 4,000-test pipeline wall time against 1,800
seconds, peak RAM, and GPU/CPU memory needs. Passing CV alone does not authorize
production if the platform runtime or offline bundle requirements fail.
