# Exp10 - probability blend of the Exp07 A and B label heads

**Verdict: STOP. Promotion gate FAILED.** No production build, no ZIP, no
second blend tried, no submission uploaded.

One hypothesis, no new feature, no new model, no retrain, nothing else changed:

> Can a plain probability-level ensemble of the two label heads Exp07 already
> fitted — **A** = 249-feature foundation, **B** = the shipped 300-feature
> `B_plus_window` — improve the official composite over B alone?

```
P_blend = (1 - alpha) * P_A + alpha * P_B
pred    = argmax(P_blend)
```

`alpha = 0` is pure A, `alpha = 1` is exactly Exp08's B. **One scalar.** No
class-specific alpha, no offsets, no temperature, no geometric/log blend, no
thresholds, no stacking model.

---

## 1. Headline

| metric | A = Exp08 (alpha = 1) | B = cross-fitted blend | delta |
|---|---|---|---|
| Macro F1 | 0.7894323366835861 | 0.7889459789785602 | **-0.00049** |
| Robustness F1 | 0.7368470046834376 | 0.7346640733932716 | **-0.00218** |
| Success F1 | 0.8643757406010988 | 0.8643757406010988 | 0 (pinned) |
| fault_turn hit@2 | 0.6086111111111111 | 0.6075000000000000 | **-0.00111** |
| **COMPOSITE** | **0.7694453917139285** | **0.7685453689277627** | **-0.00090** |

The cross-fitted composite is **0.00090 BELOW** the Exp08 baseline. 77 of
10000 OOF labels changed; 77 `fault_turn` values changed as a consequence.
Accuracy 0.8061 → 0.8060 (diagnostic only, not the objective).

The gate needed **+0.002**. The measured delta is **-0.00090** — the wrong
sign, and less than half the bar in magnitude.

## 2. Promotion gate — FAILED

| criterion | threshold | value | |
|---|---|---|---|
| composite delta | >= +0.002 | **-0.000900** | FAIL |
| folds won | >= 2/3 | **0/3** | FAIL |
| robustness drop | <= 0.003 | 0.002183 | PASS |
| worst class F1 drop | <= 0.02 | 0.003361 (`dropped_handoff`) | PASS |

Two of four criteria fail, and they fail on the two that matter: the effect is
not there, and it is not consistent. The two that pass are *safety* criteria —
they say the change is not actively harmful, not that it is useful.

## 3. Per-fold — the selection never transferred

| fold | n_train | n_val | alpha selected | labels changed | delta composite | |
|---|---|---|---|---|---|---|
| 0 | 6666 | 3334 | **1.00** | 0/3334 | +0.000000000000 | LOSS (tie) |
| 1 | 6667 | 3333 | **0.65** | 68/3333 | **-0.001644** | LOSS |
| 2 | 6667 | 3333 | **0.95** | 9/3333 | **-0.000929** | LOSS |

**0/3 folds win.** Fold 0 is a tie by construction: it selected `alpha = 1.00`,
which *is* the baseline, so its 0 changed labels and 0.000000 delta are the
identity operation, not a win. Folds 1 and 2 both selected an alpha away from
1, and both **lost** on their held-out rows. This is the same pattern Exp09
found with class offsets: an in-sample gain that evaporates out of sample.

## 4. Why it failed — the selection is fitting fold noise

The per-fold training curves show each selection bought a tiny in-sample
improvement, and none of it carried:

| fold | train argmax alpha | train composite there | train composite at alpha=1 | in-sample gain |
|---|---|---|---|---|
| 0 | 1.00 | 0.7710339116666942 | 0.7710339116666942 | +0.000000 |
| 1 | 0.65 | 0.7678700354137142 | 0.7668612888197289 | +0.001009 |
| 2 | 0.95 | 0.7712354910822170 | 0.7702745140636910 | +0.000961 |

Folds 1 and 2 each gained roughly +0.001 composite on their own selection rows
and then lost -0.0016 / -0.0009 on their held-out rows. The cross-fit exists
precisely to expose that gap, and here it is.

The stability warning is real too. The three folds selected **1.00, 0.65, 0.95**
— a spread of 0.35, three distinct values, which the code classifies as
`NOTABLE INSTABILITY`. Two of the three folds, left to themselves, moved off the
shipped baseline; the third said "don't blend" by choosing the identity. That a
fifth of a blend weight is enough to move a fold's selection, and that the
resulting movement is always harmful on unseen rows, is the substantive finding:
there is no stable, transferable optimum inside this grid.

## 5. Per-class F1 and confusion

| class | A | B | delta |
|---|---|---|---|
| clean | 0.865003 | 0.865870 | +0.000867 |
| dropped_handoff | 0.635637 | 0.632276 | **-0.003361** (worst) |
| duplicated_work | 0.829642 | 0.830350 | +0.000707 |
| deadlock | 0.721739 | 0.719443 | -0.002296 |
| conflict | 0.903175 | 0.903529 | +0.000354 |
| goal_drift | 0.846898 | 0.846898 | 0.000000 |
| runaway_loop | 0.723932 | 0.724255 | +0.000324 |

Of the 77 changed labels, **35 became correct and 36 became wrong — net -1**.
The blend is not systematically better or worse than B; it is noise that happens
to land marginally worse. `dropped_handoff`, the class the project cares most
about, is the worst-hit and moved the wrong way.

## 6. Reproduction gate — PASSED, exact

The Exp08 baseline was re-derived from the verified artifacts through the
official `evaluation.metrics` and matched to 1e-12 on all five numbers
(macro, robustness, success, hit@2, composite) before any alpha was evaluated.
If it had not, the run would have stopped. Full details in `run.log` section 2.

Every probability input is fail-closed: `reuse.load` runs 32 provenance checks
per system (manifest identity, data/code hashes, run order, the exact outer-fold
index vectors, per-array sha256, shape/dtype/finiteness/rowsum), and
`load_exp07.py` additionally verifies the sealed `window_peak_pos` /
`window_peak_prob` (hash, array-buffer hash, dtype, shape, `[0,1]`, full run
coverage). **`ArtifactRejected` is fatal; there is no lenient path.** The A and
B peak seals are additionally checked to be identical, because blending two
probability matrices that did not come from the same Exp07 run would be
meaningless.

`alpha = 1` reproduces Exp08's labels from **three** independent sources (the
verified B labels, the committed `oof_B_plus_window.csv`, and `argmax(P_B)`),
and its `fault_turn` vector is checked against a from-scratch re-read of
`window_peak_pos` plus the official hit@2 and a brute-force recount.

## 7. Method, and what it cost

* **Reuse only.** No Exp07 retrain, no feature recomputation, no LightGBM, no new
  model, no new feature. Strict verified load ~1.5 s; the whole research run is
  **3.0 s** including the (cached) robustness-mask build. Target was seconds to a
  few minutes; the actual figure is 3.0 s.
* **Search.** The 21-point grid `0.00, 0.05, …, 1.00`, fixed before any number
  was seen. 21 evaluations per fold is the entire search. No second grid, no
  finer grid, no Optuna, no random search.
* **Determinism.** Blending is float64 on arrays converted once; the endpoints
  are exact by construction (`1.0*P_A + 0.0*P_B == P_B` bit-for-bit). Reruns
  are bit-identical (tested).
* **Objective correctness.** The fast path is asserted equal to
  `evaluation.metrics` on 9 probe alphas spanning the grid, including both
  endpoints and the selected region — max abs diff 0.0. Every candidate is
  scored on the **full composite**, not macro alone. Success is pinned and never
  recomputed.
* **Tie-break** (pre-declared): higher composite; if within 1e-12, the alpha
  closer to 1.0; still tied, the larger alpha. A tie therefore resolves to
  "keep the Exp08 baseline" — "no evidence" means "do not ship".

## 8. Honest meta-CV — and its limit

For each of Exp07's 3 existing outer folds, alpha was chosen on the **other two
folds only**, frozen, and applied to the held-out fold; the three held-out
prediction vectors were concatenated into one cross-fitted vector over all
10000 rows and scored **once**. That vector is the headline above. Choosing alpha
on all 10000 and then reporting the gain on those same 10000 rows was not done.

A test proves the discipline holds: the three selected alphas are not all equal
(if the held-out fold had leaked into its own selection they would be), and
corrupting the held-out rows' probability matrices leaves the selected alpha
bit-identical.

As in Exp09, and as must be stated:

> **OOF meta-CV is not a fully nested retrain of Exp07; the public leaderboard
> remains the external validation.**

The base probabilities were produced once by a single Exp07 run; no alpha saw a
held-out fold during the fold that validated it, but the base models were not
re-fitted inside a further inner loop. The public Exp08 score of **0.7696** is
recorded as a **reference only** and was never read by the search — the test
suite asserts it sits in `results.json` only as that reference.

The full-OOF curve is printed in `run.log` as a **diagnostic** (its in-sample
argmax is `alpha = 0.95` at composite 0.76982, i.e. +0.00038 over Exp08). It is
explicitly *not* a validation result: it is the surface the selection searches,
on the same rows the score would be read from.

## 9. What this does and does not license

This is **OOF meta-CV** over Exp07's existing OOF probabilities, and it
measures the **blend layer's** contribution only. The cross-fitted composite is
0.00090 below the baseline. The full 21-point curve, evaluated honestly
out-of-sample, never produces a transferable gain; the two folds that moved off
`alpha = 1` both lost on unseen rows, and the full-OOF in-sample "winner"
(0.95) is precisely the kind of small, unstable gain the cross-fit is built to
reject.

Nothing here says a probability ensemble of A and B can never help. It says that
**this** simple blend, on **these** OOF probabilities, at **this** grid
resolution, has no stable signal: the best it does is move 77 labels around and
trade macro/robustness/hit@2 at a net loss, with a fold-selected alpha that
disagrees across folds. Given 0/3 fold wins and a negative composite delta, the
honest reading is that the composite surface in this direction is noise-dominated
around `alpha = 1`, and a finer grid would find fold noise more reliably, not
less — which is exactly what the experiment forbids trying.

**Not done, deliberately:** no class-specific blend, no second or finer grid, no
calibration offsets, no temperature, no Optuna/random search, no new features,
no new models, no public-score tuning, no sequence model, no production build,
no ZIP, no upload.

## 10. Files

| file | contents |
|---|---|
| `blend.py` | verified dual-system loader, frozen grid, corpus, objective (fast == official), tie-break, selection |
| `run_experiment.py` | reproduction gate, blend curve, honest 3-fold meta-CV, comparison, alpha stability, promotion gate |
| `test_blend.py` | 14/14 tests: endpoint parity (labels + fault_turn, 3 sources each), grid/shape/finiteness/alpha-range, determinism, no-leakage, blended-L1, fast==official, success pinned, tie-break, gate-as-declared, no-production-build |
| `results.json` | all metrics, per-fold curves, confusion transitions, gate |
| `run.log` | execution log |

No `submission_exp10_ab_blend/` and no `.zip` — the gate failed, and a test
asserts their absence so a stray build cannot pass unnoticed.

## 11. Reproduce

```
python experiments/exp10_ab_probability_blend/run_experiment.py    # 3.0 s
python experiments/exp10_ab_probability_blend/test_blend.py       # 14/14
```
