# Exp11 — fixed 3-seed probability ensemble of the final label head

**Verdict: STOP. No production build, no ZIP, no submission.**

One hypothesis, no search space, no free parameter. The official metric is
**composite**, and that is what the verdict is based on.

---

## 1. Headline

| metric | current best (seed 42) | fixed 3-seed average | delta |
|---|---|---|---|
| **COMPOSITE** | 0.7694453917 | **0.7696021570** | **+0.000157** |
| Macro F1 | 0.7894323367 | 0.7894887896 | +0.000056 |
| Robustness F1 | 0.7368470047 | 0.7369722712 | +0.000125 |
| fault_turn hit@2 | 0.6086111111 | 0.6095833333 | +0.000972 |
| Success F1 | 0.8643757406 | 0.8643757406 | 0.000000 (pinned) |
| accuracy | 0.8061 | 0.8065 | +0.0004 *(diagnostic only)* |

The composite moved **+0.000157**. The gate required **+0.002** — the observed
delta is **12.7x smaller than the bar**, and it also sits *below* the
pre-declared weak band of [+0.001, +0.002), so this is not "inconclusive", it is
empty.

**201 of 10000 OOF labels changed (2.01%)**, and 196 `fault_turn` values moved
with them.

## 2. Promotion gate (pre-declared before the first fit)

| criterion | required | observed | verdict |
|---|---|---|---|
| composite delta | >= +0.002 | +0.000157 | **FAIL** |
| outer folds won | >= 2 / 3 | 1 / 3 | **FAIL** |
| robustness drop | <= 0.003 | +0.000125 (it *rose*) | PASS |
| worst class F1 drop | <= 0.02 | 0.00238 (`dropped_handoff`) | PASS |

**GATE FAILED** on both criteria that carry the hypothesis. The two safety
criteria passed, which is what "harmless but empty" looks like — the same
signature Exp09 and Exp10 produced.

## 3. Fold-by-fold

| fold | n_val | labels changed | baseline composite | ensemble composite | delta | |
|---|---|---|---|---|---|---|
| 0 | 3334 | 61 | 0.7661703747 | 0.7704013658 | **+0.0042310** | WIN |
| 1 | 3333 | 75 | 0.7747447945 | 0.7718215167 | −0.0029233 | LOSS |
| 2 | 3333 | 65 | 0.7664411916 | 0.7651781529 | −0.0012630 | LOSS |

Fold 0's macro and robustness gains are the largest in the table (+0.0025 macro,
+0.0107 robustness) and fold 0 is still worth only +0.0042 composite, because
macro carries 0.50 of the objective and the total effect across all 10000 rows
cancels to +0.000157. **A gain that lives in one fold out of three is a fold
effect, not a model effect.**

Note the hit@2 column of the per-fold table: fold 0 moved it (+0.0029) and
folds 1 and 2 did not move it at all (0.0). Not one of the 140 labels changed
on folds 1–2 converted into a localisation hit. The headline hit@2 gain of
+0.00097 is therefore a **single-fold** gain, and hit@2 is 0.10 of the objective
— so the entire hit@2 contribution to the +0.000157 composite delta is
approximately 0.0001, and everything else is rounding.

## 4. Individual seeds — diagnostic only, no seed was selected

| seed | macro | robustness | hit@2 | composite | vs baseline | labels differing from seed 42 |
|---|---|---|---|---|---|---|
| 42 | 0.789432 | 0.736847 | 0.608611 | 0.7694453917 | +0.000000 | 0 |
| 137 | 0.788454 | 0.733015 | 0.608611 | 0.7679980257 | **−0.001447** | 328 |
| 2026 | 0.789181 | 0.735529 | 0.610139 | 0.7691431612 | **−0.000302** | 359 |

**This table is the real finding.** Two things are visible:

1. The three seeds disagree on **328 and 359 labels** out of 10000 (3.3% / 3.6%),
   and `max|dP|` reaches 0.477. They are genuinely different models, so this is
   not a case of the ensemble being fed three copies of one model.

2. **The spread between individual seeds (0.00145, from 42 down to 137) is about
   9x larger than the gain the ensemble buys (+0.000157).** Seed-to-seed variance
   dominates any ensembling effect by an order of magnitude. No arithmetic on
   three seeds can recover a signal that is smaller than the disagreement
   between them.

The average does beat **all three** of its own members, which is the textbook
variance-reduction effect and the only positive sign in this experiment. It is
also smaller than the noise it is averaging over, so it does not survive the
gate. **No best seed was chosen and none may be**: the seed set was fixed
before any number was seen, and `test_ensemble.py` fails if the recorded
diagnostic ever implies a selection other than the incumbent.

## 5. Per-class F1 (official `f1_per_class`)

| class | baseline | ensemble | delta |
|---|---|---|---|
| clean | 0.865003 | 0.867408 | **+0.002405** |
| dropped_handoff | 0.635637 | 0.633255 | **−0.002382** (worst) |
| duplicated_work | 0.829642 | 0.830221 | +0.000578 |
| deadlock | 0.721739 | 0.720246 | −0.001493 |
| conflict | 0.903175 | 0.902238 | −0.000937 |
| goal_drift | 0.846898 | 0.845627 | −0.001271 |
| runaway_loop | 0.723932 | 0.727427 | **+0.003495** |

**Worst per-class delta: −0.002382 (`dropped_handoff`)**, against a gate
allowance of 0.02 — comfortable, but the sign pattern matters more than the
size: the ensemble **helps `clean` and `runaway_loop` and hurts four of the six
fault classes.** A variance-reduction argument predicts gains spread roughly
evenly; a gain concentrated in the two easiest classes and paid for by the hard
ones is a shift in the clean/fault operating point, not a better model.

Note that the two biggest movers are exactly the classes with the largest
existing F1 (`clean` 0.865, `runaway_loop` 0.724) and the two losers are among
the hardest (`dropped_handoff` 0.634). This is consistent with the ensemble
nudging ambiguous runs toward whichever class its near-tied probabilities favour,
which buys easy classes and costs hard ones.

## 6. Confusion transitions

Of the **201** changed labels, the split into three disjoint parts is:

| outcome | rows |
|---|---|
| wrong -> **right** | 86 |
| right -> wrong | 82 |
| wrong -> **different** wrong | 33 |
| net accuracy change | **+4 runs** (+0.04pp) |

The 201 rows are therefore **86 vs 82 on the rows where correctness could
change at all**, and accuracy moves from 0.8061 to 0.8065. A 201-row re-roll
that lands 86–82 is indistinguishable from a fair coin. This is the same
fold-noise signature Exp09 found with class offsets and Exp10 found with the
A/B blend: the change is real, it is stable in sign, and it is worth nothing.

(Reported in `results.json` as `changed_label_stats`, which records the
`wrong_to_wrong` count explicitly so the three parts sum to the 201 changed
rows instead of leaving a tail unaccounted for.)

The transitions are near-symmetric in the dominant pairs (`dropped_handoff->clean`
14 against `clean->dropped_handoff` 14; `deadlock->clean` 13 against
`clean->deadlock` 11), i.e. the ensemble is not systematically re-routing runs
between classes — it is re-rolling near-ties.

## 7. Reproduction and parity — what was actually proven

| check | result |
|---|---|
| strict loader (`reuse.load` + `load_exp07`) | 32 provenance checks, pass, fail-closed |
| outer fold index hashes vs sealed manifest | equal, all 3 folds |
| **baseline reproduction, 5 numbers to 1e-12** | **exact** (macro / robustness / success / hit@2 / composite) |
| baseline labels vs committed Exp07 OOF CSV | identical on all 10000 rows |
| L1 turn vs a direct re-read of the sealed peaks | identical; clean -> −1, every fault gets a real turn |
| 3 parameter dicts differ in | `random_state` **only**, mechanically asserted |
| feature count / order | 300 = 249 + 51, one column order, identical for all seeds |
| **seed 42 vs the sealed Exp07 OOF probabilities** | **BIT-IDENTICAL** (sha256 `edbaf8d7…`) |
| fast path vs `evaluation.metrics` | max abs diff **0.0** on 4 real candidates |

**The seed-42 bit-parity is the load-bearing result.** It proves the rebuilt
300-feature matrix and the refitted seed-42 head are the *same objects* current
best was scored on, which turns every delta in this report from a comparison of
two pipelines into a comparison of two **seeds** on one pipeline. It is an
identity (float32 buffer hash equality), not a tolerance.

It also means the one-model case is exact by construction: `P_ensemble` is the
mean of the same float32 matrices the incumbent is stored as, so seed 42 alone
reproduces 0.7694453917139285 to the last bit and no part of the delta can come
from float drift or a different fold assignment.

## 8. What was NOT done

No other seeds. No 2 or 4 models. No weights between seeds. No best-seed
selection. No A/B blend. No offsets, calibration or thresholds. No window-model
ensemble. No new features, no new classifier, no Optuna, no leaderboard tuning.
No second attempt after the gate failed, no Exp12. The gate was applied once,
with the thresholds written into `run_experiment.py` before the first fit, and
`test_ensemble.py` re-derives each criterion value from the reported columns to
prove none of them moved.

Public Exp08 (0.7696) is recorded in `results.json` as a **reference only** and
is read by nothing.

## 9. Separate finding — an Exp07 defect, reported and NOT fixed here

While transcribing `runner.run()` I found a real bug in Exp07.

`experiments/exp07_fault_windows/runner.py:229` calls

```python
blocks = grp.predict_runs(P, rw[np.isin(rw, hold)], len(tr_runs))
```

`grouping.predict_runs` iterates `for r in range(n_runs)` while `rw` carries
**global** run indices up to 9999, and `len(tr_runs)` is about 6666. Every
outer-train run whose global index is `>= len(tr_runs)` therefore never receives
a probability block, so its **51 window aggregate features stay exactly zero**.

Measured on fold 0: **2182 of 6666 outer-train runs (32.7%)**; per inner fold,
749 / 720 / 713 of 2222 rows (33.7% / 32.4% / 32.1%).

The asymmetry matters: the **outer-validation** call on the next lines uses
`predict_runs(Pva, ..., n)` with `n = 10000`, so validation rows get their real
aggregates. The label head is therefore **trained** on a mixture where ~1/3 of
rows carry a "no window evidence" signature of 51 zeros, and **evaluated and
shipped** on rows where 0% do. The 51 window features are not distributed
identically between train and inference, and all-zeros is not a neutral value —
it is a value LightGBM can split on.

**This experiment reproduces the defect verbatim and does not fix it.** Fixing
it would change the 300-feature matrix, which is the exact thing the bit-parity
gate exists to hold constant; a corrected matrix would be a different feature
set testing a different hypothesis, and it would invalidate the reproduction
gate this report rests on. It is recorded here as a candidate for a separate
experiment with its own pre-registered reproduction targets.

How much it is worth is **not** estimated here, deliberately: fixing it moves
current best's own numbers, so it cannot be measured inside an experiment whose
baseline is current best.

## 10. Runtime

| stage | time |
|---|---|
| strict verified load (32 checks) | 0.15 s |
| Exp07 cache load (`build_all`, 457 203 windows) | 3.3 s (cache HIT) |
| 3 outer folds x seed 42 / 137 / 2026 | 463 / 503 / 464 s |
| scoring, diagnostics, gate, report | ~35 s |
| **total research runtime** | **1497.6 s (24.96 min)** |

Two bugs in this experiment's own code were found and fixed before any number
was reported, both caught by the parameter-discipline assert rather than by a
metric: the seed-vs-reference comparison raised on an empty diff for the
reference seed itself, and then had its condition inverted. Neither reached a
fit. A third was a **test** bug, and it is worth naming because it would have
been easy to paper over: the reproducibility test asserted
`became_correct + became_wrong == n_changed`, which is simply false on real
data — 33 of the 201 changed rows were wrong -> wrong, moving between two
different incorrect classes. The assertion was wrong, not the report; it was
corrected to a proper three-way partition rather than loosened to a tolerance.
The fit is deterministic and was not re-run to fix it: the single added integer
was recounted from the same per-seed matrices the report was computed from, and
all five headline numbers are unchanged to the last bit.

No production build, therefore **no production runtime** and **no ZIP**. Both
are forbidden after a failed gate and `test_ensemble.py` asserts that neither
`submission_exp11_label_seed_ensemble/` nor its `.zip` exists.

## 11. Conclusion

Averaging three LightGBM label heads that differ only in `random_state` **does
not** improve the official composite over the single incumbent.

The mechanism is variance reduction, and it is real but tiny: the average beats
all three of its members. It is nonetheless an order of magnitude below both
the promotion bar (+0.000157 vs +0.002) and the disagreement between the seeds
it averages (0.00145). Three seeds is not enough averaging to matter when a
single seed choice is worth 9x the ensemble gain — and the 86-vs-82 split of
the 201 changed labels says the remaining effect is a coin flip.

This is the **third consecutive STOP** (Exp09 offsets, Exp10 A/B blend, Exp11
seed ensemble), so `experiments/ROOT_CAUSE_REVIEW.md` is now mandatory before
any Exp12 is designed. The common shape across all three is worth stating
plainly in that review: each bought a small in-sample gain, each lost it on
held-out rows, and none of the three touched the thing that actually limits
this system — the Exp07 window-feature defect in section 9, which is a
correctness bug rather than a tuning knob.

## 12. Files

| file | contents |
|---|---|
| `run_experiment.py` | gate, honest OOF over 3 folds x 3 seeds, diagnostics |
| `ensemble.py` | the 3-seed fit, the mean, the Exp10 scoring reuse, parity helpers |
| `test_ensemble.py` | 13 tests, including the no-build-after-FAIL assertion |
| `seed_42_oof.npy`, `seed_137_oof.npy`, `seed_2026_oof.npy` | per-seed honest OOF probabilities (float32) |
| `results.json` | all metrics, per-class, per-fold, gate, parity hashes |
| `run.log` | full runtime log |
| `add_wrong_to_wrong.py` | one-off: adds the `wrong_to_wrong` count to `results.json` by recounting the 201 changed rows from the saved per-seed matrices. The fit was **not** re-run; `run_experiment.py` now emits this field directly, so a fresh run makes it redundant. It is kept because the committed `results.json` came from the rerun, and deleting it would hide how that field was added |
| `update_results_csv.py` | one-off: appends the exp11 row to `experiments/results.csv` and asserts the existing rows are byte-identical afterwards |
