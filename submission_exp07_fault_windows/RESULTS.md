# Exp07 production submission

Second public submission. Ships the Exp07 PROMOTE configuration exactly as
validated in `experiments/exp07_fault_windows`; nothing tuned, added or dropped.

## Heads

| head | features | source |
|---|---|---|
| `label` | **300** | 249 Exp06b `full − age` foundation + 51 aggregates of a turn-level window LightGBM |
| `success` | **185** | Exp03 B, unchanged |
| `fault_turn` | — | official rule-based `localize` on the predicted label |

The window model scores every candidate turn (83 structural local features,
fixed ±2 positive zone, 6 fault classes + background, pre-declared weights
background 0.25 / each fault 0.125) and its per-window probabilities are
summarised into 51 run-level aggregates.

## What is deliberately NOT in here

- **L1 window-peak turn and the L2 hybrid.** L2's 0.60 confidence cut was fixed
  after seeing L1, so it is a post-hoc diagnostic. The pre-registered Exp07
  headline used L0, the official localizer. A parity check fails the build if
  `main()` ever references a peak, an argmax or a hybrid cut.
- **Any hyper-parameter search.** Fixed at the validated values.
- **Any new feature.** The foundation is the same 249 columns Submission 1 used,
  proven value-identical.

## Honest stacking

The window model is a second-stage learner, so its features must be out-of-sample
for the runs the label head trains on. With 3 stratified inner folds, every train
run gets its 51 window features from a window model that never saw it; test runs
get theirs from a window model fitted on all train runs. `verify_stacking`
re-derives the fold membership and hard-fails on any run predicted by a model
trained on it — it runs on every invocation and is visible in the log.

## Verification

| check | script | result |
|---|---|---|
| feature parity (249 + 185, name/order/value) | `scripts/check_submission_parity_exp07.py` | 60/60 PASS, max abs diff 0 |
| window feature parity (83) | same | PASS over 5,628 windows, max abs diff 0 |
| target parity (`window_targets`, weights) | same | PASS over 90 cases |
| aggregation parity (51) | same | PASS over 250 runs + degenerate shapes, max abs diff 0 |
| model hyper-parameters | same | PASS, attribute by attribute |
| self-containment (repo hidden) | same | PASS |
| no targets-as-features / no paths / no network | same | PASS, 9 modules |
| end-to-end equivalence | `scripts/verify_submission_equivalence_exp07.py` | PASS |
| smoke run (1200→400) | `solution.py` | PASS, 73.6 s |
| full run (10k→4k) | `solution.py` | PASS, **272.4 s** |
| validator | `scripts/validate_submission.py` | OK, 4000 rows |
| fallback path (`run_id` only) | `solution.py` | OK, valid constant output |

### Equivalence detail

Holding out 1,500 real train runs and fitting the submission on the remaining
8,500 (exactly as the platform would) reproduces the experiment's system B on
**92.3%** of labels, and scores **macro F1 0.7910** on that holdout versus the
experiment's OOF 0.7894. The residual disagreement is expected and correct: the
experiment's OOF came from 3 outer folds, this fit saw a different 85/15 split,
so the two models are genuinely different fits of the same validated pipeline.
`fault_turn` matched the official localizer on **1500 of 1500** rows.

## Runtime

Full 10,000 → 4,000: **272 s (4.6 min)** single run, CPU only. Breakdown:
foundation build ~22 s, window dataset 457k windows ~40 s, 3 inner window models
~125 s, full-train window model ~66 s, two classifier heads ~66 s.

## Environment

CPU-only, offline. `numpy`, `pandas`, `scikit-learn`, `lightgbm`. No absolute
paths, no network, no bundled data. `solution.py` sits at the archive root.
