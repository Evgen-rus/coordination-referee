# Exp18 — fixed telemetry-loss augmentation

**Status before CV: plan frozen; no Exp18 CV result has been observed.** Exp15
remains the current production baseline. Exp16 semantic fusion is closed, and
Exp17 candidate-event localization was not run because its measured non-GD
oracle ceiling is below the proposed localization gate.

## One hypothesis

Adding one training-only degraded view of runs at the telemetry-loss rates
documented in `docs/data_schema.md` improves classification on the train-side
Robustness slice, especially `dropped_handoff`, without changing the Exp15
production success head or Exp13 L1 localizer.

## Frozen corruption policy

For every original training run, draw four independent Bernoulli flags with
`numpy.random.default_rng(20261008)` in CSV row order. The fixed probabilities
are the schema rates: 0.12 for clearing all message `intent` fields; 0.08 for
clearing `refs` on exactly half of message indices sampled without replacement;
0.10 for clearing all artifact `subtask` fields; and 0.15 for truncating the
text of exactly half of message indices to independent lengths drawn uniformly
from 12 through 30 characters. The latter half-message convention gives a
fixed interpretation to the schema's unspecified word “part”. Flags may
co-occur. Append one transformed view if any flag fires; otherwise append no
copy. Keep every original row. Do not alter event order, message type, agent
roles/identity, artifact IDs/hashes, state entries, label, success, or
`fault_turn`. No transformation reads a target. Validation receives original
runs only.

This adds approximately 38% training views before fold filtering. The rates
are frozen from the official schema and will not be adjusted to match observed
CV outcomes. No hidden generator or run-ID mapping is used.

## Model and CV

- Candidate channel: existing Exp15/Exp13 312-feature label path: 249
  foundation features, 51 corrected Exp12 window aggregates, 12 Exp13 wait
  graph features.
- Candidate head: the current Exp13 multiclass LightGBM (`n_estimators=600`,
  `learning_rate=0.05`, `num_leaves=63`, `random_state=42` and all remaining
  parameters from `ENS.label_params(42)`). No new features or tuning.
- Outer CV: same `StratifiedKFold(3, shuffle=True, random_state=0)` over the
  10,000 original runs. Candidate validation rows are always the original
  views. Exp15 saved OOF success predictions stay fixed. Exp13 L1 peak turns
  stay fixed and are selected using each candidate predicted class.
- Window aggregates: retain the existing corrected Exp12 grouping and Exp13
  window model. In every outer/nested split, stratify the original parent runs
  with the existing 3-fold seed 0; assign each synthetic view to its parent's
  fold. Fit only on views whose parents are in the fit side. This prevents an
  original or corrupted view from appearing on opposite sides of any split.
  Fit the three inner window models sequentially (`workers=1`); the existing
  Exp14 audit verified workers 1 and 3 produce identical predictions. The
  LightGBM per-model thread setting and all model parameters stay unchanged.
- Exact control: run the unaugmented Exp15 corrected 312-feature path on the
  same folds first and require its OOF label predictions to match committed
  Exp13 D. Require the unchanged Exp15 success predictions, L1 peak turns,
  baseline metric values, row order, and official evaluator to match their
  frozen artifacts before candidate scoring.
- Success: use the committed Exp15 OOF success predictions unchanged.
- Candidate success/turn handling: score candidate labels with the same
  saved Exp13 class-conditioned peak-position matrix used by Exp15. No
  localizer is trained or tuned.
- Robustness: use the existing Exp07 train-side hard/Robustness mask. It is a
  proxy for the hidden test shift, not the hidden test labels.
- Runtime: first benchmark the fixed feature/window path on a deterministic
  500-run subset and estimate the full 3-fold CV cost. If the measured
  projection exceeds two hours, stop before full CV and report the limit.

## Frozen gate

Compare only the fixed augmented candidate with the exact Exp15 baseline.

1. Composite delta is at least `+0.008`.
2. Macro F1 and Robustness F1 do not fall.
3. `dropped_handoff` F1 improves (the preselected target class).
4. Composite improves in at least 2/3 folds.
5. No class F1 drops by more than `0.02` absolute.

If any condition fails: **STOP**. Do not tune rates, truncation fraction,
seed, model, or folds. Do not create a ZIP. Public leaderboard values are not
used for selection. Exp15 production stays untouched.
