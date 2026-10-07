# Exp16 — frozen semantic turn sequence

**Status: configuration frozen; CV not run.** Exp15 remains production at
`fdd588c4412c7cc5cb4f6a1f18bc1d17cbeb264a`. The Exp15 OOF predictions are
captured in `baseline_exp15_oof.csv`; `baseline.json` records their inputs and
hashes. Production files and ZIP are not changed.

## One hypothesis

Frozen per-turn semantic representations add information missing from the
current Exp13 312 structural features, especially under shifted vocabulary.

## Frozen configuration

| Part | Choice |
| --- | --- |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, English, Apache-2.0, 22.7M parameters, 384 dimensions |
| Pin | Hub revision `46605decb5369335a3847c9f41bb0b896c07dd1a`; `model.safetensors` SHA-256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| Bundle | About 92 MB: safetensors, model/tokenizer configs, tokenizer/vocabulary, and model card/license notice; omit duplicate `.bin`, `.ot`, and `.h5` weights. Well under the 5 GB limit. |
| Encoder | Frozen `eval()` / inference-only; attention-mask mean pooling, then L2 normalize. Each goal and message is encoded separately, at most 256 wordpieces each. No job-time network access. |
| Turn text | One formatted record per message: protocol type, sender/receiver IDs and roles, intent-presence bit, tool, and that message's text. Raw intent value is not added as a class-name shortcut. |
| Sequence | `[goal embedding, message 0, …, message N-1]`, in source order. No turn cap or turn truncation; ragged batches use packed sequences. Relative position uses the original message index. |
| Progress | Per-turn side values: intent/ref presence, ref count, artifact count so far / at this turn, shared-state update count so far / at this turn. Progress counts only include records timestamped at or before the message turn. |
| Semantic head | One 1-layer bidirectional GRU (hidden 128 per direction), fixed 10 epochs, batch 64, AdamW `1e-3`, weight decay `1e-4`, dropout `0.15`; no early stopping or search. Only this small classifier is trained. |
| Structural channel | Fold-local Exp13 312-feature LightGBM and its probabilities, rebuilt with the existing corrected Exp15 fold helper. Semantic and structural probabilities are separate until a fixed 50:50 average. No learned blend weights. |
| CV | Same verified `StratifiedKFold(3, shuffle=True, random_state=0)` folds as Exp15. Each outer fold fits both branches on outer-train and predicts outer-validation. Runs are sequential. Frozen embeddings use run content only; no target or run ID enters the cache. |
| Fault turn | No semantic localizer in this experiment. Use the existing verified Exp13 per-class L1 peak positions, selected by the candidate class, for hit@2 scoring. If the label gate passes, the next experiment may train a class-conditioned per-turn score on these same turn states. |

The local demo data has a 156-turn train maximum and a 151-turn test maximum;
all turns are retained. The official schema warns that platform test runs are
longer and differ in domain/wording. Variable-length batching therefore avoids
silently discarding chronology if a platform run is longer.

For offline preparation, run `python experiments/exp16_semantic_sequence/fetch_encoder.py`
on a connected machine. It fetches only the pinned inference files, verifies
the weight hash, and stores them under ignored `.cache/`; the experiment and
eventual submission load local files only.

## Gate — fixed before CV

Continue to a separate fault-turn experiment only if all conditions pass:

1. Composite improves by at least `+0.005` over Exp15.
2. Macro F1 and Robustness F1 both improve.
3. Composite improves in at least 2 of the 3 folds.
4. No class's aggregate F1 falls by more than `0.02` absolute.

Otherwise **STOP**; do not try another encoder. No public leaderboard is used
for selection. Exp15 production is unchanged unless a later CV is accepted.

## Runtime and current blocker

Engineering estimate: roughly 15–25 minutes for an A100 run, with a hard
30-minute ceiling; this estimate is unmeasured. The model is only 22.7M
parameters, but encoding time depends on message count/length. The pinned
~92 MB bundle is cached locally and the weight SHA-256 was verified. This
checkout has no CUDA device, PyTorch, or Transformers, so the full embedding/CV
run cannot be measured here. `run_experiment.py` refuses to start a heavy CPU
fit unless `--allow-cpu` is explicit. A full 10,000-train/4,000-test runtime
measurement on the task A100 is required before this can be considered
production-ready.

## Input sources

- Repository task: `docs/task.md`
- Schema: `docs/data_schema.md`
- Failure ontology: `docs/ontology.md`
- Offline model/runtime rules: `docs/faq.md`
- Official model card and pooling recipe:
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Pinned model files:
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/46605decb5369335a3847c9f41bb0b896c07dd1a
