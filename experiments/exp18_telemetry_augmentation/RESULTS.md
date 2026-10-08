# Exp18 — fixed telemetry-loss augmentation

**Decision: STOP. Exp15 remains the production baseline.** The pre-registered
candidate did not meet its frozen composite gate; no tuning or production
change followed.

## Design and reproduction

One training-only corrupted view was added for each selected parent run using
the four frozen schema rates. This produced 3,809 additional views from 10,000
original runs. Validation always used original runs. Synthetic views stayed
with their parent in every outer and inner split, and the three inner window
fits ran sequentially. The model, 312-feature path, folds, seeds, Exp15 success
predictions, and Exp13 class-conditioned fault-turn peaks were unchanged.

The 500-run CPU benchmark included 192 selected views and projected 39.9
minutes for full control plus candidate CV, below the 120-minute cap. Actual
wall time was 951 seconds (15.9 minutes), including the repeated benchmark,
input preparation, exact Exp15 control, and candidate CV. The control matched
the frozen baseline metrics and reproduced Exp13 label predictions in all
three folds.

## OOF result

| Metric | Exp15 control | Exp18 candidate | Delta |
|---|---:|---:|---:|
| Composite | 0.7938624419 | 0.7954212064 | +0.0015587645 |
| Macro F1 | 0.8129545587 | 0.8139992312 | +0.0010446725 |
| Robustness F1 | 0.7599843692 | 0.7644078601 | +0.0044234908 |
| Success F1 (pinned) | 0.8949086162 | 0.8949086162 | 0.0000000000 |
| Fault-turn hit@2 (pinned localizer) | 0.6315277778 | 0.6308333333 | -0.0006944444 |

`dropped_handoff` F1 improved by `+0.0100759372`. No class dropped more than
`0.02`; the largest decrease was `goal_drift` at `-0.0080873433`. Composite
fold deltas were `+0.003790`, `-0.002822`, and `+0.003971` (2/3 wins). The
candidate changed 563 labels: 252 baseline errors became correct, 232 correct
labels became wrong, and 79 changed while remaining wrong; accuracy increased
from `0.8272` to `0.8292`.

The gate passed every condition except composite improvement of at least
`+0.008`. The observed `+0.001559` is below that threshold, so the frozen
experiment stops. Do not adjust augmentation rates, seeds, model, or gate from
this result. No public leaderboard was used.

Structural-control and candidate OOF label probabilities are saved separately
in `structural_label_proba.npy` and `candidate_label_proba.npy`. Full metrics,
per-class deltas, fold details, gate checks, and runtime data are in
`results.json`, `cpu_benchmark.json`, and `run.log`. The Exp15 production ZIP
and source were not changed; no new ZIP was built.
