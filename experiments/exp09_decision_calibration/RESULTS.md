# Exp09 - decision-layer calibration

**Verdict: STOP. Promotion gate FAILED.** No production build, no ZIP, no second
calibration method tried.

One hypothesis, no new model, no new feature, no retrain:

> Can 6 additive class offsets in log-probability space — and nothing else —
> raise the official composite over Exp08?

```
pred = argmax_c ( log(max(P_c, 1e-15)) + delta_c ),   delta_clean = 0
```

---

## 1. Headline

| metric | A = Exp08 (no offsets) | B = cross-fitted | delta |
|---|---|---|---|
| Macro F1 | 0.7894323366835861 | 0.7883748956510434 | **-0.00106** |
| Robustness F1 | 0.7368470046834376 | 0.7376937133573590 | +0.00085 |
| Success F1 | 0.8643757406010988 | 0.8643757406010988 | 0 (pinned) |
| fault_turn hit@2 | 0.6086111111111111 | 0.6097222222222223 | +0.00111 |
| **COMPOSITE** | **0.7694453917139285** | **0.7692394594772485** | **-0.00021** |

Cross-fitted composite is **0.00021 BELOW** the Exp08 baseline. 211 of 10000
OOF labels changed; 208 emitted `fault_turn` values changed as a consequence.
Accuracy 0.8061 -> 0.8046 (diagnostic only, not the objective).

The gate needed **+0.003**. The measured delta is **-0.00021** — the wrong
sign, and a tenth of the bar in magnitude.

## 2. Promotion gate — FAILED

| criterion | threshold | value | |
|---|---|---|---|
| composite delta | >= +0.003 | **-0.000206** | FAIL |
| folds won | >= 2/3 | **1/3** | FAIL |
| robustness drop | <= 0.003 | 0.000847 | PASS |
| worst class F1 drop | <= 0.02 | 0.005058 (`duplicated_work`) | PASS |

Two of four criteria fail, and they fail on the two that matter: the effect is
not there, and it is not consistent. The two that pass are *safety* criteria —
they say the change is not actively harmful, not that it is useful.

## 3. Per-fold

| fold | offsets (log space) | labels changed | delta composite | delta macro | delta hit@2 | |
|---|---|---|---|---|---|---|
| 0 | `[0, +0.400, -0.150, -0.050, 0, +0.100, +0.050]` | 58/3334 | **+0.00061** | +0.00259 | +0.00458 | WIN |
| 1 | `[0, +0.400, -0.375, 0, +0.400, +0.150, -0.100]` | 63/3333 | -0.00015 | -0.00382 | -0.00292 | LOSS |
| 2 | `[0, +0.400, +0.375, -0.3625, -0.100, -0.1125, -0.300]` | 90/3333 | -0.00141 | -0.00186 | +0.00167 | LOSS |

## 4. Why it failed — the offsets are not a real signal

**5 of 6 fault offsets disagree in sign across the three independent
calibration-train fits.** Only `dropped_handoff` is stable, and it is pinned at
the `+0.40` bound in all three — it is the one class the objective wants pushed
as hard as the box allows, which is itself a sign the search is exploiting a
weak signal rather than measuring a correction.

| class | fold 0 | fold 1 | fold 2 | |
|---|---|---|---|---|
| dropped_handoff | +0.400 | +0.400 | +0.400 | consistent (at the bound) |
| duplicated_work | -0.150 | -0.375 | **+0.375** | **sign flip** |
| deadlock | -0.050 | 0.000 | -0.363 | **sign flip** |
| conflict | 0.000 | **+0.400** | -0.100 | **sign flip** |
| goal_drift | +0.100 | +0.150 | -0.113 | **sign flip** |
| runaway_loop | +0.050 | -0.100 | -0.300 | **sign flip** |

A direction that reverses when you change which two thirds of the data you can
see is not a property of the model — it is a property of which rows happened
to be in the sample. Each fit gained **+0.0038 to +0.0059 composite on its own
training rows** (e.g. fold 2: 0.770275 -> 0.776222), and each lost on held-out
rows. That gap is the whole story: the search is fitting fold noise, and the
cross-fit exists precisely to expose that.

The per-fold deltas also show the shape of the failure. Where the fitted
offsets happened to be conservative (fold 0, 58 label changes), the held-out
delta was positive. Where they were aggressive (fold 2, 90 changes), it was the
worst. The gain is not a calibration correction — it is a monotone trade of
macro F1 for robustness F1, and it is roughly a wash.

## 5. The gate caught two things worth recording

**The reproduction gate caught a real bug in this experiment's own code, before
any search ran.** The first draft of the cross-check called
`fault_turn_hit_at_k` with full-length label lists next to faulty-only turn
lists. The metric's `zip` truncates to the shortest input, so it paired row
*i*'s label with row *i*'s turn and reported hit@2 = 0.4867 against a true
0.6086. This is the same class of misalignment that made Exp07 ship hit@2
0.1843 against the true 0.4258, and the pre-registered baseline is what caught
it. Had the baseline not been pinned to 1e-12, this would have been a
confident, wrong, and entirely plausible-looking experiment.

**A test caught a claim I was about to make.** The brief asks for a test that
the candidate hit@2 "uses the peak predicted class, not the true class". The
test I first wrote asserted the two readings give *different scores*, and it
failed. They do not — and the reason is structural, not a bug:

> `fault_turn_hit_at_k` counts a hit **only when `pred == y_true`**. So on every
> row that can contribute a hit, the predicted column *is* the true column, and
> the two readings are numerically identical.

Verified on the real 10000-row OOF: both give 0.6086111111111111 at `delta = 0`
and 0.6116666666666667 at a non-zero delta. The readings do differ in what they
**emit** — `T[pred]` and `T[y_true]` disagree on 1903 of 10000 rows, exactly the
misclassified faulty runs — but they can never differ in the score. So the test
now pins the emitted column (the only observable) and records the invariance
explicitly, rather than asserting a distinction that does not exist.

## 6. Method, and what it cost

* **Reuse only.** 32 provenance checks, strict load in **1.4 s**. No Exp07
  training, no feature recomputation, no LightGBM, no new model.
* **Search.** Deterministic coordinate descent inside a space declared before
  any number was seen: bounds `[-0.40, +0.40]`, steps
  `[0.20, 0.10, 0.05, 0.025, 0.0125]`, `clean` pinned to 0, tie-break
  composite > `sum(delta^2)` > fixed class order. Bounds were never widened and
  the schedule was never changed.
* **Exact determinism.** Every step is an integer multiple of the finest step
  (16/8/4/2/1) and `0.40` is exactly 32 units, so the search runs on an integer
  grid and converts with `units * 0.0125`. No accumulated float drift, and
  reruns are bit-identical (tested).
* **Cost.** 157 + 168 + 216 = **541 candidate evaluations for all three folds,
  ~0.07 s each**. The whole research run is **2.8 s** end to end, including the
  10.5 s robustness-mask build (cached thereafter). Target was seconds to a few
  minutes; the actual figure is 2.8 s.
* **Objective correctness.** The fast objective is asserted equal to
  `evaluation.metrics` on 9 probe candidates spanning the declared range,
  including boundary values — max abs diff **0.0**. Success F1 is pinned and
  never recomputed.

## 7. What this does and does not license

This is **OOF meta-CV** — a cross-fitted decision layer over Exp07's existing
OOF probabilities. It is *not* a fully nested retrain of the Exp07 base model:
those probabilities came from a single Exp07 run, and no offset ever saw a
held-out fold during the fold that validated it, but the base model was not
re-fitted inside a further inner loop. The public leaderboard remains the
external validation.

The public Exp08 score of 0.7696 was **not** used at any point. It is a
reference only; the search never reads it, and no offset was chosen by looking
at it.

Nothing here says a decision layer cannot help. It says that *this* decision
layer, on *these* OOF probabilities, has no signal to find: the best it does is
move 211 labels around and trade macro against robustness at a net loss. Given
5-of-6 sign instability, the honest reading is that the composite surface in
this direction is noise-dominated, and a larger or different search space would
find fold noise more reliably, not less.

**Not done, deliberately:** no second search space, no temperature scaling, no
thresholds, no L2 hybrid, no new features, no new classifier, no retrain, no
change to L1, no change to the success head, no production build, no ZIP, no
upload.

## 8. Reproduce

```
python experiments/exp09_decision_calibration/run_experiment.py    # 2.8 s
python experiments/exp09_decision_calibration/test_calibration.py  # 10/10, 1 skipped
```

The single skip is the production-constants test, which is skipped *because the
gate failed*: it asserts that no offsets were fitted and that no submission
directory exists, so a stray build cannot pass unnoticed.
