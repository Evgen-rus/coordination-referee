# Exp21 Fault-First Referee — frozen one-shot CV plan

**Status: preregistered before CV. No Exp21 CV result had been observed when
this plan was committed.** This is one fixed 3-fold experiment on the available
GitHub demo train. It is not a test-set or competition-score estimate.

## Data and baseline

- Source: this repository's checked-in, compressed demo dataset, expanded to
  `data/train.csv`; README labels it a local-development demo. It is not the
  platform's actual training file. Row count is 10,000.
- Train SHA256:
  `8e8feb4ee284d2acd918b533921631fa905d47adf8d34881f89359964a1bcfbc`.
- Frozen Exp15 prediction SHA256:
  `bad28b0528a291e96d7291f004ae72c6d571a93c38c8483d7adf889be7a291c6`.
  Its run_id sequence is checked against the demo train before scoring.
- Official Exp15 demo metrics, verified from the frozen OOF through
  `evaluation/metrics.py`: Macro F1 `0.8129545586738590`, Robustness F1
  `0.7599843692318033`, Success F1 `0.8949086161879896`, hit@2
  `0.6315277777777778`, Composite `0.7938624418508566`.

## Single fixed model

The implementation is `model.py` in this directory: a trainable 32,768-bucket
word EmbeddingBag (64d), 16d reference EmbeddingBag, 8d categorical embeddings,
12 metadata values, 128d event projection, two-layer bidirectional GRU
(hidden 64 per direction; recurrent dropout 0.15), learned attention pooling,
7-class label head, and six class-conditioned message-onset scores. It has
2,289,334 parameters. No external encoder or model download is used. The
architecture and loss are unchanged from the checked-in preflight.

Training settings: 12 epochs, batch 16, AdamW learning rate `0.001`, weight
decay `0.0001`, label cross-entropy plus `0.5 *` faulty-train-row onset
cross-entropy, head dropout `0.2`, model seeds 42/43/44. No early stopping,
thresholds, ensembles, or parameter search.

## Inputs and leakage controls

Each run is represented as a goal event followed by every observable message,
shared-state write, and artifact. Events sort by numeric `t`; ties use the
frozen type order message → state → artifact, then source index. Messages retain
their original turn indices. Every event is retained. Target columns and
`run_id` are excluded by an input-field whitelist; only train-fold labels and
fault turns supervise their corresponding fit. The validation onset channel
uses the candidate's predicted class. Only message events can be returned as
fault turns; clean predictions return `-1`.

Exp15 predicted labels and success outputs remain frozen. The candidate replaces
only the label and fault-turn predictions in the evaluated row; success is
copied unchanged from Exp15. No production solution, ZIP, or leaderboard state
is modified during CV.

## Folds and comparison

Use exactly `StratifiedKFold(n_splits=3, shuffle=True, random_state=0)` on the
demo labels. For each outer fold, train on its two-thirds only and produce one
OOF prediction for each held-out row. Compare these predictions against the
verified frozen Exp15 OOF on the same rows, using the official
`evaluation/metrics.py`. Report aggregate metrics, each fold, and per-class F1;
fold deltas are diagnostics, not a selection rule.

Save each fold's weights and held-out predictions. Reload every saved fold
checkpoint and rerun only its held-out inference; require exact equality of
class probabilities and fault turns before accepting any result. Also require
baseline parity within `1e-12`, exact run_id order, no train/validation overlap,
one OOF assignment per row, unchanged Exp15 success outputs, valid message-turn
outputs, and clean `-1` outputs.

## Fixed gate

The only score gate is **official Composite strictly greater than the frozen
Exp15 Composite** on the full OOF rows. No minimum delta or additional Macro,
Robustness, fold-win, or per-class threshold is imposed. A zero or negative
delta is STOP. Exact checkpoint replay and all comparison/leakage checks above
must pass before a positive delta is treated as confirmed. Metrics by fold and
class remain visible to assess how the aggregate change arose. This one CV is
not repeated or tuned after its results are seen.

## Runtime and packaging boundary

The existing local CPU benchmark (PyTorch 2.14.1+cpu, 6 torch threads, 12
logical CPUs) measured/projection-estimated: Exp15 train+inference `593.4 s`,
candidate event input build for 14,000 rows `53.3 s`, candidate inference for
4,000 rows `6.7 s`. With locally pre-trained candidate weights, the point
estimate for Exp15 plus candidate preparation/inference is `653.4 s`; applying
a 25% margin to the candidate-only part gives `668.4 s`, below 1,800 seconds.
This is a local CPU estimate, not a measurement on a DSWorks instance or A100.
Local 3-fold candidate CV is projected at `2,548.7 s` and is permitted to run
outside the submission job. Official task FAQ permits pre-trained weights in
the ZIP and inference-only use; the ZIP limit is 5 GB and the job is offline.

If and only if this OOF passes the fixed gate, a separate all-demo-train fit and
package smoke check may be prepared with Exp15 success inference unchanged.
That package must still be validated for the competition invocation, full
runtime, and size before any submission. Demo CV is not hidden-test validation.
