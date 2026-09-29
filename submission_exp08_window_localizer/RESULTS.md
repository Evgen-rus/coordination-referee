# Exp08 production submission

Third submission. Ships the Exp08 L1 window-peak localizer, validated in
`experiments/exp08_window_localizer`. Exactly one behavioural change from
Submission 2.

## The change

```python
# WAS (Exp07)
pred_turn = [localize(run, lab) for run, lab in zip(te_runs, pred_label)]

# NOW (Exp08)
te_peak = pt.peak_turns(Pte, Wte["run"], len(te_runs), pred_label)
pred_turn = [int(t) for t in te_peak]
```

For a run predicted as fault class `c`, `fault_turn` is the turn maximising
`P[run, t, c]` — where the window model is most confident that the fault it has
already named actually happens. `clean` -> `-1`; an empty window block -> `-1`.
No threshold, no offset, no class-specific rule, no calibration.

The window model was **already being fitted** in Submission 2: the 51 run-level
aggregates are the entire new signal for the label head. Exp07 computed the peak
positions and discarded them. This reads them back out of the same probability
blocks — no extra model, no extra features, no material extra runtime.

## Heads

| head | features | change |
|---|---|---|
| `label` | **300** (249 Exp06b `full − age` + 51 window aggregates) | **none** |
| `success` | **185** (Exp03 B) | **none** |
| `fault_turn` | window peak for the predicted class | **L0 -> L1** |

## Measured effect (Exp07 honest OOF, official metric)

| | L0 | L1 | delta |
|---|---|---|---|
| hit@2 | 0.4308 | **0.6086** | **+0.178** |
| composite | 0.7517 | **0.7694** | **+0.0178** |

3/3 outer folds improve (+0.173 / +0.180 / +0.180). The composite delta is
entirely the hit@2 term; macro, robustness and success are unchanged floats.

**The trade, stated plainly:** hit@0 falls from 0.3618 to 0.1142. The window
model finds the right neighbourhood far more often and the exact turn far less
often. Only hit@2 is scored, so under this metric the trade is favourable.

**The caveat that must travel with it:** L1 was discovered post-hoc on Exp07's
OOF, on the same 10 000 runs whose labels produced it. Every number above is
measured on data the rule was chosen on. The public leaderboard is the real
external validation.

## Verification

| check | result |
|---|---|
| 8 feature modules byte-identical to Submission 2 | PASS (sha256) |
| `label` byte-identical | **4000 / 4000** |
| `success` byte-identical | **4000 / 4000** |
| `fault_turn` changed | 2504 of 4000 rows |
| every clean run -> `fault_turn = -1` | PASS |
| label / success / window model params | identical attribute by attribute |
| feature counts (300 / 185), `aggregate.AGG_NAMES` | pinned, unchanged |
| peak helper == `grouping.aggregate_block` | PASS, 16/16 |
| validator `scripts/validate_submission.py` | OK, 4000 rows |
| full run 10k -> 4k | **374.9 s (6.2 min)**, far inside the 30-min budget |
| standalone ZIP, extracted outside the repo | PASS |

`peak_turn.py` is proved to be the same function as Exp07's
`grouping.aggregate_block` on live probabilities from a really-fitted window
model, plus adversarial shapes: 0/1/2-window runs, shuffled row order, exact
ties, one-hot rows, an empty middle run, and nothing scored at all.

## Not in here

- **L2 hybrid or any confidence threshold.** Exp07's L2 scored 0.5122 and its
  0.60 cut was fixed *after* seeing L1 — post-hoc, excluded, as in Exp07.
- **Any hyper-parameter search, new feature, or new classifier.**
- **Any calibration or ensemble.**

## Honest stacking

The window model is a second-stage learner, so its features must be out-of-sample
for the runs the label head trains on. With 3 stratified inner folds, every train
run gets its 51 window features from a window model that never saw it; test runs
get theirs from a window model fitted on all train runs. `verify_stacking`
re-derives the fold membership and hard-fails on any run predicted by a model
trained on it — it runs on every invocation.

The test-run peak turns come from the **same** window model that produced the
test features, so a run's `fault_turn` is the turn whose probability also fed the
label head. There is no second, differently-fitted model behind `fault_turn`.

## Environment

CPU-only, offline. `numpy`, `pandas`, `scikit-learn`, `lightgbm`. No absolute
paths, no network, no bundled data. `solution.py` sits at the archive root.
</content>