# Exp17 / Exp18 direction selection

## Checkpoint 1 — baseline and diagnostics

- Repository HEAD at audit start: `10f6cdb` (`Record Exp16 semantic sequence CV results`); working tree was clean.
- Exp15 remains the production baseline: local composite `0.7938624419`, Macro `0.8129545587`, Robustness `0.7599843692`, Success `0.8949086162`, hit@2 `0.6315277778`. The public score `0.79475` is the value supplied in the task; it was not used for selection.
- Exp16 fixed 50:50 candidate is STOP: composite delta `+0.0047511700`, short of its frozen `+0.005` gate by `0.0002488300`. Macro and Robustness improved, 2/3 folds won, and the largest class drop was `-0.0143238`; that does not waive the failed composite gate. Exp15 production files and ZIP are unchanged.
- Exp16 was missing from the protocol history. It has now been recorded as one STOP in `experiments/results.csv`. `python experiments/protocol_guard.py` reports Exp15 as current best and 1/3 consecutive STOPs; another experiment is allowed. The guard prints a Python 3.14 launcher warning on this machine but still completes and exits 0.
- The diagnostics use the existing train-side Robustness proxy. The exact hidden test shift flag is unavailable, so length/team/topology comparisons are diagnostic proxies only.

## Three independent analyses

### Luna A — class-conditioned candidate events

Candidates were extracted from observable events without labels or `fault_turn`; targets were attached only after generation. Current L1 correct-class hit@2 is `0.79625`, versus `0.63153` over all faulty runs. The table below reports candidate oracle coverage, not achievable selector quality.

| Class | Candidates/run among true class | Candidate coverage ±2, correct-class runs | Current L1 hit@2, correct-class runs | L1 misses covered by candidate oracle |
|---|---:|---:|---:|---:|
| dropped_handoff | 2.43 | 0.859 | 0.833 | 89 / ~116 |
| duplicated_work | 2.28 | 0.830 | 0.899 | 69 / ~101 |
| deadlock | 2.71 | 0.217 | 0.838 | 57 / ~163 |
| conflict | 0.83 | 0.717 | 0.728 | 221 / ~310 |
| goal_drift | 38.67 | 0.980 | 0.662 | 348 / ~353 |
| runaway_loop | 0.53 | 0.124 | 0.831 | 13 / ~144 |

All candidate oracles together cover 797 L1 misses (11.1 percentage points of the 7,200 faulty runs), but 348 depend on a broad goal-drift detector emitting about 39 candidates/run. Excluding that pool leaves an oracle ceiling of `449 / 7,200 = +0.0624 hit@2`, below Exp17's proposed `+0.07` gate. Coverage is stable by fold, but this does not establish a selector: without goal-drift, covered misses are 154/145/150 across the three folds. CF is the only clear unexplored candidate signal, yet its oracle coverage `0.717` is already below the current L1 `0.728`. Existing realizable rules are worse than L1 for DH (`0.632` vs `0.833`) and RL episode-start (`0.252` vs `0.831`). Secondary ontology-compatible signatures overlap strongly on cross-class errors, so a class-conditioned selector can follow a secondary anomaly.

**Decision: do not run Exp17 CV.** The remaining upper bound without the nonspecific goal-drift pool is below the frozen gate, and there is no measured realizable six-class selector.

### Luna B — classification and Robustness

Exp15 OOF error rates rise with run length: `13.9%` for at most 42 messages, `18.7%` for 43–70, and `30.4%` above 70. The hardest weak class is DH: recall falls from `69.6%` on short runs to `19.8%` above 70. The all-intent-empty group has 1,198 runs (12.0%) and error `25.8%` versus `16.1%` otherwise; DH recall there is `51.4%` versus `58.7%`. The same-turn/ref delivery proxy is weak, and text length alone cannot identify clipping because 96% of runs contain a naturally short message. The exact hidden shift labels are unavailable.

Exp16 provides diagnostics only: its candidate lowers overall error on runs above 70 messages from `30.4%` to `28.5%`, but lowers DH recall on the all-intent-empty group from `51.4%` to `42.6%`. This is not a reason to reuse semantic fusion.

**One supported next hypothesis:** add a fixed, label-preserving, training-only copy with documented intent/refs/artifact-subtask/text loss rates, while keeping validation runs original. The exact policy, seed, fold handling, model, and gate are frozen in `exp18_telemetry_augmentation/PLAN.md` before CV.

### Luna C — Exp16 saved-probability postmortem

The saved OOF arrays show the fixed candidate corrects 250 Exp15 label errors and creates 203 new ones (net 47 more correct). Runaway-loop fixes/breaks are 65/15 (`F1 +0.02536`); deadlock 56/27 (`+0.01314`). It harms DH 19/57 (`F1 -0.00746`) and duplicated work 10/41 (`-0.01432`). Candidate fold composite changes are `-0.003089`, `+0.002232`, `+0.015332`; fold 2 supplies most of the gain and only that fold improves Macro. Robustness gains are concentrated in some blackboard, runaway-loop, and deadlock groups, but robustness fold changes are `-0.00837`, `+0.01889`, `+0.02942`.

**Decision: close the semantic classification branch.** Do not tune fusion or train a semantic fault-turn model from Exp16.

## Checkpoint 2 — direction comparison

| Direction | Current errors | Measured potential | Stability | Runtime | Decision |
|---|---|---|---|---|---|
| Exp17 candidate-event localization | L1 misses remain across all six fault classes | Oracle covers `+0.111` hit@2 only with broad GD candidates; `+0.062` without them, both upper bounds | Covered misses are fold-stable; realizable selector signal is not measured; DH/RL simple rules lose to L1 | No training; event extraction only | STOP before CV; no measured six-class selector supports the `+0.07` gate |
| Exp16 semantic fusion | DH and duplicated-work regressions; new errors span folds | Actual composite `+0.004751`, fails frozen `+0.005`; semantic-only far below baseline | Fold 2 dominates; folds 0/1 lower Macro | 1,251.7 s observed CV | Close branch; no tuning or semantic localizer |
| Exp18 telemetry-loss augmentation | Long runs and all-intent-empty groups are weak, especially DH | Not yet measured; only task-documented loss rates support the mechanism | To be established by the fixed 3-fold CV; hidden-shift comparisons remain proxies | Requires rebuilding all 312-feature views and parent-grouped window aggregates; benchmark before full CV | Run one fixed experiment because the observed telemetry channel and single hypothesis are explicitly supported by the schema |

Exp18 is the selected next experiment. No improvement is promised. A failed gate ends this direction without parameter changes.

## Sources

- `experiments/exp16_semantic_sequence/results.json`, `PLAN.md`, and saved OOF probability arrays
- `experiments/EXP13_ERROR_AUDIT.md`
- `experiments/exp01_fault_turn_ranker/RESULTS.md`
- `experiments/exp08_window_localizer/RESULTS.md`
- `experiments/PROTOCOL.md` and `experiments/protocol_guard.py`
- `docs/ontology.md`, `docs/data_schema.md`, `docs/metrics.md`, `docs/task.md`, and `docs/faq.md`
