# Exp19 hybrid localization diagnostic

**Purpose:** read-only retrospective check of the fixed Exp15 + Exp19
class-conditioned fault-turn rule. No model training, MiniLM encoding,
parameter search, production build, or leaderboard use was performed.

## Inputs and integrity

- Loaded all 10,000 rows from `exp19_stepfinder_localizer/oof_predictions.csv`
  and the frozen `exp16_semantic_sequence/baseline_exp15_oof.csv`.
- Both files contain 10,000 unique `run_id` values in the exact verified
  `data/train.csv` order. Predicted `label` and `success` match row for row.
- The Exp15 baseline CSV hash, Exp13 OOF probability hash, train CSV hash, and
  Exp15 production ZIP hash passed the existing Exp16 baseline validator.
- The hybrid choice is based only on the frozen **predicted Exp15 label**:
  `dropped_handoff` / `duplicated_work` use Exp15 L1;
  `deadlock` / `conflict` / `goal_drift` / `runaway_loop` use Exp19;
  `clean` returns `-1`. True labels and target turns are used only to score.
- Rule assignments: 2,082 rows use Exp15, 4,937 use Exp19, and 2,981 predicted
  `clean` rows return `-1`.

## Clean diagnostic correction

The frozen Exp15 OOF has a nonnegative L1 turn on all 2,981 rows whose
**predicted** class is `clean`. For this diagnostic only, those baseline turns
were set to `-1`; no source OOF, NumPy artifact, or Exp15 production file was
edited. Official hit@2 stays unchanged at `0.6315277778`: true-clean rows are
excluded, and a faulty run predicted as clean is already a class mismatch and
counts as a miss regardless of its turn.

## Full OOF metrics

Metrics use `evaluation/metrics.py`. Hit@2 was also recalculated independently
from per-row conditions; it matches the official function exactly.

| Metric | Exp15 L1, clean → -1 | Fixed hybrid | Delta |
|---|---:|---:|---:|
| Macro F1 | 0.8129545587 | 0.8129545587 | 0 |
| Robustness F1 | 0.7599843692 | 0.7599843692 | 0 |
| Success F1 | 0.8949086162 | 0.8949086162 | 0 |
| Fault-turn hit@2 | 0.6315277778 | **0.7102777778** | **+0.0787500000** |
| Composite | 0.7938624419 | **0.8017374419** | **+0.0078750000** |

The composite delta is exactly `0.10 × hit@2 delta`, since the other three
components are unchanged.

### Hit@2 by true fault class

Each class has 1,200 runs overall. A hit requires the predicted class to equal
the true class and the chosen turn to be within ±2 of the target.

| True class | Hits / runs | Hybrid hit@2 |
|---|---:|---:|
| dropped_handoff | 577 / 1,200 | 0.4808333333 |
| duplicated_work | 895 / 1,200 | 0.7458333333 |
| deadlock | 893 / 1,200 | 0.7441666667 |
| conflict | 942 / 1,200 | 0.7850000000 |
| goal_drift | 1,019 / 1,200 | 0.8491666667 |
| runaway_loop | 788 / 1,200 | 0.6566666667 |

### Three folds

Each fold has 2,400 faulty runs; each class has 400 runs per fold.

| Fold | Exp15 hit@2 | Hybrid hits / 2,400 | Hybrid hit@2 | Composite | Delta vs Exp15 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6179166667 | 1,661 / 2,400 | 0.6920833333 | 0.7979865972 | +0.0074166667 |
| 1 | 0.6358333333 | 1,729 / 2,400 | 0.7204166667 | 0.8071654326 | +0.0084583333 |
| 2 | 0.6408333333 | 1,724 / 2,400 | 0.7183333333 | 0.7986129737 | +0.0077500000 |

Class hit@2 by fold (rows are **true** fault class):

| True class | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|
| dropped_handoff | 0.447500 | 0.477500 | 0.517500 |
| duplicated_work | 0.757500 | 0.737500 | 0.742500 |
| deadlock | 0.715000 | 0.765000 | 0.752500 |
| conflict | 0.767500 | 0.797500 | 0.790000 |
| goal_drift | 0.837500 | 0.865000 | 0.845000 |
| runaway_loop | 0.627500 | 0.680000 | 0.662500 |

## Class-conditioned selection diagnostic

For each held-out fold, the two other folds alone determined which localizer
to use for each **predicted** non-clean class. Selection compared row-level
hit rates within truly faulty runs predicted as that class. Every split chose
the same map: Exp19 for `deadlock`, `conflict`, `goal_drift`, and
`runaway_loop`; Exp15 for `dropped_handoff` and `duplicated_work`; `-1` for
`clean`.

| Held-out fold | Train-only Δ hit@2: dropped / duplicate / deadlock / conflict / goal / runaway | Held-out hit@2 | Held-out Composite |
|---:|---|---:|---:|
| 0 | -0.40933573 / -0.08064516 / +0.04084158 / +0.09163803 / +0.26555024 / +0.07714286 | 0.6920833333 | 0.7979865972 |
| 1 | -0.40036232 / -0.08476821 / +0.03585147 / +0.08455468 / +0.25539568 / +0.06966618 | 0.7204166667 | 0.8071654326 |
| 2 | -0.42476190 / -0.07702523 / +0.04654088 / +0.07770270 / +0.26309524 / +0.07725322 | 0.7183333333 | 0.7986129737 |

**Retrospective caveat:** the hybrid idea and class map were proposed after the
Exp15/Exp19 results across all folds had been inspected. The leave-one-fold-out
calculation is therefore a stability diagnostic, not an independent CV result
or a promotion estimate.

## History and protocol guard

Exp19 was absent from `experiments/results.csv`. Its historical standalone STOP
has now been recorded there as `not_promoting`, with the original Exp19
Exp15→Exp19 metrics and results path. The hybrid diagnostic is not recorded as
a new experiment or a promotion.

Ran `.venv/Scripts/python.exe experiments/protocol_guard.py`: exit code 1.
Exp15 remains current best. The stop tail is `exp16, exp18, exp19`; the existing
`ROOT_CAUSE_REVIEW.md` covers `exp09, exp10, exp11, exp12` and is stale. The
guard blocks inventing another experiment until the review is complete for
exactly `[exp16, exp18, exp19]`, with all five required answers (hypotheses,
transfer, headroom, noise, warrant). No next experiment was started.

One guard limitation was found: it reads `results.json` only from the
`promotion_gate` key, while Exp19 stores its result under `gate`. Thus the
Exp19 STOP is classified from the newly recorded CSV verdict and is not
independently cross-checked against `gate.passed` by this guard. The guard still
correctly blocks on the three-STOP tail.

## Production runtime feasibility

Existing evidence is CPU-only:

- Exp16 measured MiniLM encoding at 122.21 events/s and took **4,822.9 s** for
  590,146 train events. Its benchmark estimated **4,979.2 s** for the 10,000
  training runs. The measured encoder pass alone is 2.68× the 1,800 s limit,
  before Exp15 or StepFinder work.
- The existing Exp15 production ZIP measured **593.4 s** for 10,000 train runs
  and 4,000 test runs, under the limit by itself.
- Exp15's nested research CV took about **1,886 s**; this is not an exact
  production single-fit timing, but it also does not demonstrate a sub-1,800 s
  end-to-end path.
- Exp19's three-fold CPU CV took **2,346.93 s** (peak RSS 1,067 MB), with
  745–837 s per fold. These CV timings are not production single-fit timings.
- No GPU/A100 runtime was measured. No Exp16/Exp19 encoder/model weights or
  offline production bundle exist in those experiment directories; the
  unchanged Exp15 ZIP contains neither MiniLM nor StepFinder.

**Assessment:** Exp15 alone fits the limit in its measured package run, but CPU
production for the combined system is infeasible under 1,800 s from the measured
MiniLM encoder pass alone. GPU/offline feasibility is unverified; there is no
evidence to claim the full Exp15 + MiniLM + StepFinder pipeline meets the
limit. No production ZIP was made.

## Verdict

**Promising for an independent verification, not PROMOTE.** The fixed routing
rule raises hit@2 and Composite in all three folds, and the same class map is
selected in each two-fold diagnostic. However, that evidence is retrospective,
and production feasibility is not established. Exp15 production remains
unchanged; the protocol guard requires the current root-cause review before
another experiment.
