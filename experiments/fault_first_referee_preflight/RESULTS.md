# Exp21 Fault-First Referee — final result

**Decision: STOP.** One preregistered 3-fold OOF completed on the available
repository demo train. No second CV, tuning, full-data refit, production change,
ZIP, or leaderboard run was made.

## Data and baseline

- Data: 10,000-row GitHub repository demo, not the platform training set.
- Train SHA256: `8e8feb4ee284d2acd918b533921631fa905d47adf8d34881f89359964a1bcfbc`.
- Frozen Exp15 OOF SHA256:
  `bad28b0528a291e96d7291f004ae72c6d571a93c38c8483d7adf889be7a291c6`.
- Baseline passed five-metric parity within `1e-12`; frozen run_id order and
  uniqueness matched all 10,000 train rows.

## Official OOF metrics

| Metric | Exp15 | Fault-First | Delta |
|---|---:|---:|---:|
| Composite | 0.7938624419 | 0.7624898400 | -0.0313726018 |
| Macro F1 | 0.8129545587 | 0.7524623929 | -0.0604921658 |
| Robustness F1 | 0.7599843692 | 0.7436449602 | -0.0163394091 |
| Success F1 | 0.8949086162 | 0.8949086162 | 0 |
| fault_turn hit@2 | 0.6315277778 | 0.6611111111 | +0.0295833333 |

Metrics were scored with the official `evaluation/metrics.py`. Success
predictions are exactly the frozen Exp15 predictions.

### Per-class F1

| Class | Exp15 | Fault-First | Delta |
|---|---:|---:|---:|
| clean | 0.8783947414 | 0.8226348364 | -0.0557599050 |
| dropped_handoff | 0.6550094518 | 0.5156626506 | -0.1393468012 |
| duplicated_work | 0.8410819949 | 0.7130820399 | -0.1279999550 |
| deadlock | 0.8336092715 | 0.7340339531 | -0.0995753184 |
| conflict | 0.9033280507 | 0.8502715283 | -0.0530565224 |
| goal_drift | 0.8443186413 | 0.8373729476 | -0.0069456937 |
| runaway_loop | 0.7349397590 | 0.7941787942 | +0.0592390351 |

### Fold Composite

| Fold | Rows | Exp15 | Fault-First | Delta |
|---|---:|---:|---:|---:|
| 0 | 3,334 | 0.7905699305 | 0.7635186116 | -0.0270513189 |
| 1 | 3,333 | 0.7987070993 | 0.7697261053 | -0.0289809940 |
| 2 | 3,333 | 0.7908629737 | 0.7481235062 | -0.0427394675 |

All three fold Composite deltas are negative. The hit@2 gain did not offset the
classification losses, especially for `dropped_handoff`, `duplicated_work`, and
`deadlock`.

## Integrity and saved artifacts

- OOF run_id order is exact and unique; fold counts are 3,334 / 3,333 / 3,333.
- Validation rows are assigned once; train/validation sets do not overlap.
- Inputs exclude targets and run_id; the pre-CV input-invariance checks passed.
- Each checkpoint was reloaded and its validation probabilities and turns
  matched exactly. `saved_weight_replay_exact: true`.
- OOF CSV SHA256:
  `2d432f783b302ab1c25f5da4bb3661d36d88c06890a6bf45e5803b1cf003010e`.
- OOF probabilities SHA256:
  `d29bab6123763db16be6828eb21b32b949c2a15500dcabae7040387cd6531208`.
- OOF fault_turn SHA256:
  `0b22593696b90134632dd8cd2e9752d81e8267086ff01490e571f9c9240492ce`.
- Three fold checkpoints and their SHA256/size are listed in `results.json`.
- Full stdout/stderr is in `diagnostics/exp21_cv.log`.

## Runtime

Local CPU only: PyTorch `2.14.1+cpu`, 6 torch threads, 12 logical CPUs.

| Stage | Actual time |
|---|---:|
| Event input build | 37.3 s |
| Fold training total | 2,522.3 s (folds: 843.3 / 838.9 / 840.1 s) |
| Fold validation inference | 16.8 s |
| Saved-weight replay | 17.4 s |
| Complete CV process | 2,601.6 s |

Contest inference was not run because the candidate failed its gate. Existing
local measurements estimate Exp15 plus candidate input preparation/inference at
653.4 s, or 668.4 s with a 25% margin on candidate-only work. This is a local
CPU estimate, not a DSWorks/A100 measurement. The official FAQ permits locally
pretrained weights in an offline ZIP, but no candidate package was created.

## Gate

The preregistered gate was Composite strictly greater than Exp15. It failed by
`-0.0313726018`, so the experiment is STOP. The result is a demo OOF finding,
not a hidden-test or leaderboard result. Exp15 production remains unchanged.
