# Exp20 — Temporal Agent–Task Graph

**Status:** preregistered; no Exp20 CV result has been observed.  This file is to be committed by itself before implementation results or CV.

## Hypothesis and scope

One fixed hypothesis: explicit, time-ordered links among delegating agents, assigned agents, subtasks, artifacts, waits, and shared-state keys contain classification information missing from the current Exp15 312-feature label representation. The primary expected signal is `dropped_handoff`.

The only candidate change is a fixed 24-column graph summary appended to the Exp15 312 label features. Keep the LightGBM architecture and parameters, success head and its saved OOF predictions, Exp13 L1 peak-turn matrix, existing outer folds, seeds, and official metrics unchanged. No embeddings, neural models, feature search, thresholds, per-class routing, leaderboard use, production edits, or ZIP creation.

## Observable graph and frozen event order

Use only `agents`, `messages`, `artifacts`, `shared_state`, and their observable fields. `topology` is deliberately not used: it describes allowed communication, not a task/result event. Nodes are run-local Agent IDs, exact Subtask keys, Artifact IDs, and exact shared-state keys. Raw agent/artifact IDs are used only to join nodes inside one run and never become model categories.

Build events from messages, state writes, and artifacts. Sort by `(numeric t, event-type rank, original list index)`, with the frozen rank **message → state → artifact**. Do not rewrite or renumber source `t`; at equal `t`, all messages precede state writes, which precede artifacts, and source order resolves ties within each type. Events without a valid numeric `t` cannot establish temporal links.

Subtask matching is exact after trimming outer whitespace and case-folding; no fuzzy, lexical, or semantic matching. A concrete assignment is a `handoff` with empty `refs`, nonempty `intent`, and known `from`/`to` Agent nodes. The delegator and assignee are attributes of one Agent→Subtask assignment edge. A matching assignee response is a later `inform`, `status`, `state_update`, or `final` message from `to` with the same exact Subtask key; this is an observable response proxy because the schema has no dedicated acknowledgement field. It does not inspect message prose.

An Artifact→Subtask edge uses a nonempty artifact `subtask`, or a unique, explicit `refs` link from a handoff with a nonempty matching `intent`. A result is a later artifact for that Subtask, or a later handoff with nonempty `refs` and the exact `intent`; when refs resolve uniquely to artifacts, their creator (`by`) is the result actor, otherwise the observable handoff sender is used. A result closes every currently open assignment for the exact Subtask. Missing or conflicting identifiers never create a guessed link. A repeated assignment before that first result remains a separate open assignment.

Other graph edges are: assignee Agent→Subtask response; Agent→Artifact creation; sender Agent→recipient Agent handoff; Agent→Agent wait parsed only through Exp13's frozen `wait_events`; and writer Agent→State-key write. Wait-to-result links require the same explicit Subtask and a later result actor equal to the named awaited agent. Exact state `key`, `op`, and `value` remain edge attributes; only equality/change relations enter numeric features, never raw values as model categories.

## Fixed candidate feature block (24 numeric columns)

Let `A` be concrete assignments, `R` assignments with a later linked result, `S` the subset with a matching assignee response, and `O` the subset whose assignee emits a message with a different nonempty intent before the result (or run end). Rates use their named set as denominator and are zero for an empty denominator. Time span is `max(1, max(event.t)-min(event.t))`; normalized durations divide by this span. Delivery delay is `result.t-assignment.t` in source turns.

| Feature | Fixed definition |
| --- | --- |
| `g_known_assignments` | `|A|` |
| `g_assignments_without_later_result` | `|A \ R|` |
| `g_unresolved_assignment_rate` | `|A \ R| / |A|` |
| `g_executor_response_without_result` | Count in `S` that is not in `R` |
| `g_response_without_result_rate` | Previous count / `|S|` |
| `g_other_task_activity_before_result` | Count in `O` that is not in `R` |
| `g_other_task_activity_rate` | Previous count / number of assignments with an assignee |
| `g_delivery_delay_mean_turns` | Mean delay over `R` |
| `g_delivery_delay_p90_turns` | Nearest-rank 90th percentile delay over `R` |
| `g_delivery_delay_max_rel` | Maximum delay over `R` / time span |
| `g_peak_open_assignments` | Maximum number of open assignment edges at an event boundary |
| `g_open_assignment_area_rel` | Time-weighted open-assignment count / (`|A| × time span`) |
| `g_max_open_age_rel` | Maximum age of any open assignment at result or run end / time span |
| `g_repeat_assignment_before_result` | Assignment edges added while that exact Subtask already has an open edge |
| `g_repeat_before_result_rate` | Previous count / `|A|` |
| `g_other_executor_deliveries` | Resolved assignments whose linked result actor differs from the assigned agent |
| `g_other_executor_delivery_rate` | Previous count / `|R|` |
| `g_wait_task_links` | Wait edges with explicit Subtask and named awaited agent |
| `g_wait_resolved_by_named_agent_rate` | Such waits followed by a same-Subtask result from the named agent / `g_wait_task_links` |
| `g_wait_to_delivery_delay_mean_rel` | Mean wait-to-result delay / time span, over resolved links |
| `g_state_multiwriter_keys` | Number of state keys written by more than one distinct known Agent |
| `g_state_cross_writer_events` | Writes whose known writer differs from the preceding known writer on that key |
| `g_state_cross_writer_overrides` | Cross-writer events whose `op` is exactly `override` |
| `g_state_cross_writer_value_changes` | Cross-writer events whose exact `value` differs from the preceding write on that key |

All 24 columns are output in this order as finite float64 values; no feature selection, threshold, or post-result deletion is allowed. Existing `wg_*` columns stay in Exp15 and are not regenerated or replaced. Exact column equality with any of the 312 incumbent columns is a pre-CV STOP, not a reason to edit the block.

## Inputs, folds, and baseline reproduction

Feature construction must be invariant to any changes in `label`, `success`, and `fault_turn`; it never reads those fields or `run_id`. Use the current Exp13/Exp12 corrected fold-local feature builder to produce the incumbent 312-column matrices, then append the same graph block to train and test. Use the existing three folds, equivalent to `StratifiedKFold(3, shuffle=True, random_state=0)`, with no overlap and one held-out prediction per row.

**Control A:** refit the unchanged Exp15 312-feature label model on each outer-train fold. Before fitting candidate B, require its OOF predicted classes to equal the frozen Exp15/Exp13 label OOF row-for-row, its L1 turns to equal the frozen L1 output, and its official metrics to reproduce the saved Exp15 baseline within `1e-12`: Macro F1 `0.8129545587`, Robustness F1 `0.7599843692`, Success F1 `0.8949086162`, hit@2 `0.6315277778`, Composite `0.7938624419`. Any failure is STOP before candidate B.

**Candidate B:** same corrected 312 columns plus the frozen graph block; same train rows, folds, class labels, LightGBM parameters, and seed. Parameters are exactly `experiments/exp07_fault_windows/common.py::lgb_label`: multiclass 7 classes, 600 estimators, learning rate `0.05`, 63 leaves, min child samples `20`, subsample `0.9`, subsample frequency `1`, column sample `0.8`, L2 `1.0`, seed `42`, all available CPU threads, verbose `-1`.

Use saved Exp15 `success_pred_B.npy` unchanged, after verifying its 10,000-row alignment to `data/train.csv` and the committed run-ID digest. For each predicted class use only the frozen Exp13 L1 peak-position column; `clean` maps to `-1`. Do not use `Exp19`.

## Pre-CV checks and runtime gate

Before the single candidate CV: run the unit tests below; verify target perturbation invariance, deterministic output, consistent agent-ID renaming, exact event tie order, no train/validation overlap, 10,000-row run-ID alignment, full row retention, and exact Control A parity. Compare each graph column against the existing 312 columns and STOP on exact duplicates. Report maximum absolute correlation as a diagnostic only; it cannot select or alter features.

Benchmark graph extraction on 500–1,000 representative runs, plus one representative LightGBM fit/inference batch, without MiniLM or multiple concurrent heavy trainings. Estimate the added cost for 10,000 train + 4,000 test and add it to the measured Exp15 production runtime of 593.4 s. Only proceed when this measurement-based total is no more than 1,800 s and memory is within the existing production environment; otherwise record a preflight STOP and do not run candidate CV. Do not substitute a GPU/A100 estimate. This estimate is separate from research CV runtime.

## Frozen promotion gate

All conditions are mandatory against Exp15:

1. Composite delta ≥ `+0.005`.
2. Macro F1 delta ≥ `0`.
3. Robustness F1 delta ≥ `0`.
4. `dropped_handoff` F1 delta ≥ `+0.01`.
5. Composite improves on at least 2 of 3 outer folds.
6. No class F1 decreases by more than `0.02` absolute.
7. No target leakage, fold overlap, baseline mismatch, or changed success predictions.

No parameter, feature, threshold, or gate changes after CV results are observed. On any failed gate, STOP without a second candidate, ZIP, production change, or leaderboard submission.

## Required checks and outputs

Unit tests cover: target-field perturbation; graph edge linkage and deterministic IDs; assignment without result; response without result; later result; repeated assignment before result; same subtask assigned to different agents; missing/ambiguous intent, refs, subtask, or IDs; cross-agent substitute result; wait linked to later delivery; shared-state writer links; target independence; ID-renaming invariance; event tie ordering; all rows retained; fold disjointness; exact Control A parity; and official metric parity.

After the one CV, save `RESULTS.md`, `results.json`, `run.log`, frozen OOF predictions, runtime measurements, test output, and one `experiments/results.csv` row. If promoted, report the result and stop: production and ZIP remain untouched until separately authorized. Public leaderboard is never used.
