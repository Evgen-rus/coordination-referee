# Exp16 — frozen semantic event sequence

**Status: one pre-CV input correction was completed before the first Exp16 CV;
no Exp16 CV result had been observed at that time.** The single fixed CV is now
complete and failed its pre-registered promotion gate. Exp15 remains production
at `fdd588c4412c7cc5cb4f6a1f18bc1d17cbeb264a`.
Its OOF predictions are frozen in `baseline_exp15_oof.csv`; `baseline.json`
records the inputs and hashes. Production files and ZIP are unchanged.

## One hypothesis

Pretrained semantic event representations add information missing from the
current Exp13 312 structural features and improve shifted-vocabulary
classification.

## Frozen experiment configuration

| Part | Choice |
| --- | --- |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, English, Apache-2.0, 22.7M parameters, 384 dimensions |
| Pin | Hub revision `46605decb5369335a3847c9f41bb0b896c07dd1a`; `model.safetensors` SHA-256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| Bundle | About 92 MB: safetensors, model/tokenizer configs, tokenizer/vocabulary, and model card/license notice. No duplicate `.bin`, `.ot`, or `.h5` weights. |
| Encoder | Frozen `eval()` / inference-only; attention-mask mean pooling, then L2 normalize. Every event is encoded separately, max 256 wordpieces per event. |
| Event text | Goal; then message type, sender/receiver roles, raw intent (`[none]` when empty), tool, message text; artifacts encode producer role, subtask, type, status; state writes encode writer role, key, op, value. Raw agent IDs and artifact ids/hashes are excluded from semantic text. |
| Event order | Goal is always first. Remaining events sort by numeric `t`, with fixed equal-time order `message -> state -> artifact`; source order breaks ties within a type. Untimed events follow timestamped events in the same fixed type/source order. |
| Side metadata | Event-type one-hot; causal relative position `i/(i+1)` and fixed timestamp transform `t/(1+abs(t))`; time-known bit; message intent/refs presence and log refs count; log artifact/state prefix counts. Prefix counts include the current event and never read later events. |
| Sequence | Goal plus every observable message, state write, and artifact. No event cap/truncation; ragged batches use packed sequences. Tokenizer truncation is local to each event at 256 wordpieces. |
| Semantic head | One 1-layer bidirectional GRU (hidden 128 per direction), fixed 10 epochs, batch 64, AdamW `1e-3`, weight decay `1e-4`, dropout `0.15`; no early stopping or search. Only this small classifier is trained. |
| Structural channel | Fold-local Exp13 312-feature LightGBM and probabilities, rebuilt with the existing corrected Exp15 fold helper. Structural and semantic probabilities remain separate until fixed 50:50 fusion. |
| CV | Same verified `StratifiedKFold(3, shuffle=True, random_state=0)` folds and per-fold seeds as before. Each outer fold fits on outer-train and predicts outer-validation. Frozen event embeddings use run content only; no target or run id enters the cache. Runs are sequential. |
| Fault turn | Exp13 class-conditioned L1 peak remains unchanged. No semantic fault-turn head is trained in this experiment. |
| Diagnostics | Save structural-only, semantic-only, and fixed 50:50 candidate OOF metrics; save semantic probabilities separately. Promotion gate applies only to the pre-fixed 50:50 candidate. |

The deterministic equal-time policy is fixed before CV: `message -> state ->
artifact`. Relative position and time do not normalize by full-run length, so
adding a later event cannot alter earlier event metadata. All event text is
tokenized independently; chronology is represented by the packed sequence,
not by concatenating a run into one string.

## Gate — fixed before CV

Continue to a separate fault-turn experiment only if all conditions pass on
the fixed 50:50 candidate:

1. Composite improves by at least `+0.005` over Exp15.
2. Macro F1 and Robustness F1 both improve.
3. Composite improves in at least 2 of the 3 folds.
4. No class's aggregate F1 falls by more than `0.02` absolute.

Structural-only and semantic-only results are diagnostics; they cannot replace
the candidate in the gate. Otherwise **STOP**; do not tune the encoder, fusion,
or GRU after seeing the result. No public leaderboard is used for selection.
Exp15 production remains unchanged unless a later CV is accepted.

## Pre-CV CPU benchmark

The fixed 500-run benchmark encoded 30,426 events in 248.96 s (122.21
events/s), then measured four 64-run GRU training batches at 0.28 s/batch
(maximum 140 events in a padded batch). It used PyTorch `2.14.1+cpu`,
Transformers `5.19.0`, and Safetensors `0.8.0` with 8 CPU threads. Peak process
RSS was 629.6 MB; estimated 10k event embedding cache is 445.7 MB, event
metadata 27.9 MB, and conservative full-run RSS estimate 1,378.9 MB.

The extrapolation gave 4,979 s for 10k-run encoding, 530 s per fold including
the Exp13 structural runtime reference, and 6,571 s (1.825 h) for 3-fold CV.
Observed encoding took 4,822.9 s for 590,146 events; the three CV folds took
1,251.7 s total. The embedding cache was built once and reused across folds.
Benchmark details are in `cpu_benchmark.json`.

## Exp16 CV result — fixed candidate

| Metric | Exp15 | 50:50 candidate | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| Composite | 0.793862 | 0.798614 | +0.004751 | FAIL: requires +0.005 |
| Macro F1 | 0.812955 | 0.816192 | +0.003238 | PASS |
| Robustness F1 | 0.759984 | 0.772847 | +0.012863 | PASS |
| Fold wins | — | 2/3 | — | PASS |
| Worst per-class F1 delta | — | `duplicated_work` −0.014324 | — | PASS: drop limit −0.02 |

Fold composite deltas: fold 0 `−0.003089`, fold 1 `+0.002232`, fold 2
`+0.015332`. The semantic-only diagnostic was materially below baseline
(composite `0.638624`, Macro `0.629219`, Robustness `0.571111`); the structural
diagnostic exactly reproduced Exp15. The fixed candidate therefore **fails**
only the composite threshold, by `0.000249`. Decision: **STOP**. Do not tune
the encoder, fusion, or GRU and do not start the semantic fault-turn experiment.

OOF metrics and fold details are in `results.json`; structural, semantic, and
candidate probabilities are saved separately as `*_label_proba.npy`.

## Input sources

- Task and runtime limits: `docs/task.md`, `docs/faq.md`
- Data fields and event schema: `docs/data_schema.md`
- Class definitions: `docs/ontology.md`
- Metrics: `docs/metrics.md`
- Encoder model card and pooling recipe: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Pinned model files: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/46605decb5369335a3847c9f41bb0b896c07dd1a
