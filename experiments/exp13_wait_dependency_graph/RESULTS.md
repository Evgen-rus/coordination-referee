# Exp13 — wait-dependency graph for `deadlock` — **PROMOTE**

**Status:** research result PASS. Baseline reproduced bit-for-bit, candidate beat it on
every pre-declared criterion. No production build and no ZIP have been made yet.

---

## 1. Hypothesis and scope

**One hypothesis:** an explicit representation of *who waits for whom* adds relational
information the current 300-feature representation does not contain, and improves the
official composite.

**Frozen and untouched:** the 12 `wg_*` features, the 6 parser templates, LightGBM and its
parameters, class weights, seeds, folds, decision rule (argmax, no threshold), the success
head, the window model, the 51 aggregates, the Exp08 L1 localiser and the pinned success F1.
`goal_drift` was not modelled and no feature targets it.

`fault_turn_hit_at_k` is the organiser-confirmed implementation (a wrong class is a miss) and
was **not** modified.

---

## 2. Root cause of the earlier refit mismatch — CONFIRMED

The first attempt built one global `(10000, 300)` matrix. That is invalid, and
`diag_fold_specific.py` proves it without fitting anything:

| Quantity | Value |
|---|---|
| Runs in >=2 outer-train folds | 10000 |
| **Train aggregates DIFFER between folds** | **6667 (66.67%)** |
| Aggregates identical | 3333 |
| Runs whose stored fold is not the only fold they belong to | 6667 |

**Why:** the 51 window aggregates of an outer-train row are produced by *that fold's* honest
inner window models. A run is outer-train in 2 of 3 folds, so it receives genuinely different
train columns depending on which fold is being built. A single global matrix can only keep one
of them, so it presented the wrong honest representation to one of the two label fits that
must see it. The previous `max|dP| = 0.913` was this, not noise.

The earlier numbers (composite `+0.016190`, deadlock F1 `+0.1287`) are therefore marked
**INVALID / PRELIMINARY DUE TO NON-COMPARABLE BASELINE** and kept, not deleted, in
`results_preliminary_INVALID.json`.

---

## 3. The corrected scheme

Per outer fold, and only per outer fold, the exact Exp07 path is rebuilt:

* `tr_runs = sorted(set(tr))`; 3 inner `StratifiedKFold(3, seed=0)`; `verify_stacking`
* one set of inner window fits -> `Xagg_tr_f` via Exp07's grouping **verbatim**, including the
  known global-index quirk (reproduced on purpose — the baseline *is* Exp08 as shipped)
* one window model on all outer-train -> `Xagg_va_f`
* `X_A_tr_f = [found[tr], Xagg_tr_f]`, `X_A_va_f = [found[va], Xagg_va_f]`

The window fits run **once per fold** and both arms consume the output:

```
X_B_tr_f = [X_A_tr_f, Xwg[tr]]
X_B_va_f = [X_A_va_f, Xwg[va]]
```

A and B therefore differ in **exactly the 12 appended columns** and in nothing else — same rows,
same folds, same 300 columns, same window probabilities, same aggregates, same params, same seed.

---

## 4. CONTROL A sealed parity — PASS

| Check | Result |
|---|---|
| sha256 | `edbaf8d766f97f28bce12e72fbf9fee6…` — **identical to sealed** |
| argmax labels differing | **0 / 10000** |
| max abs probability diff | **0.000e+00** (bit-for-bit) |
| macro / robustness / success / hit@2 / composite | all 5 exact, diff `0.00e+00` |

This is the same sha Exp11 and Exp12 reproduced, so A is provably current best.

---

## 5. Corrected paired results

| Metric | A (sealed Exp08) | B (paired) | delta |
|---|---|---|---|
| macro_f1 | 0.789432 | 0.815868 | **+0.026436** |
| robustness_f1 | 0.736847 | 0.767013 | **+0.030166** |
| success_f1 | 0.864376 | 0.864376 | +0.000000 (pinned) |
| fault_turn_hit2 | 0.608611 | 0.632778 | **+0.024167** |
| **composite** | **0.769445** | **0.792622** | **+0.023176** |

**Per-class F1 — no class fell:**

| Class | A | B | delta |
|---|---|---|---|
| clean | 0.8650 | 0.8784 | +0.0134 |
| dropped_handoff | 0.6356 | 0.6648 | +0.0291 |
| duplicated_work | 0.8296 | 0.8416 | +0.0120 |
| **deadlock** | **0.7217** | **0.8367** | **+0.1150** |
| conflict | 0.9032 | 0.9059 | +0.0027 |
| goal_drift | 0.8469 | 0.8482 | +0.0013 |
| runaway_loop | 0.7239 | 0.7355 | +0.0116 |

**Deadlock detail:** precision 0.7545 -> 0.8262, recall 0.6917 -> 0.8475;
tp 830->1017, fp 270->**214**, fn 370->**183**. `deadlock -> clean` fell **123 -> 43**.

**Fold deltas:** f0 `+0.024856`, f1 `+0.019565`, f2 `+0.024141` — **3/3**.
Deadlock per fold: `+0.1009`, `+0.1168`, `+0.1273` — **3/3**.

**Diagnostics:** 708/10000 labels changed; wrong->right **426**, right->wrong **191**;
of 1200 true deadlock, **212 fixed / 25 broken**. `wg_*` gain share 0.0994, led by
`wg_max_pair_span_rel` (0.0381) and `wg_n_same_subtask_recip_pairs` (0.0285).

---

## 6. Gate — unchanged thresholds, PASS 6/6

| Criterion | Value | |
|---|---|---|
| composite delta >= +0.002 | +0.02318 | PASS |
| folds won >= 2 | 3 | PASS |
| deadlock F1 delta >= +0.015 | +0.11495 | PASS |
| deadlock folds won >= 2 | 3 | PASS |
| robustness >= -0.003 | +0.03017 | PASS |
| no class worse than -0.02 | 0.00000 | PASS |

---

## 7. Preliminary vs corrected (diagnostic only)

| | Preliminary (invalid) | Corrected | Change |
|---|---|---|---|
| composite delta | +0.016190 | **+0.023176** | grew |
| deadlock F1 delta | +0.1287 | **+0.1150** | shrank |
| changed labels | 717 | 708 | -9 |
| wrong->right / right->wrong | 447 / 180 | 426 / 191 | -21 / +11 |

The preliminary gain was real but **understated**: with a valid baseline the composite delta
grew by ~43%. Deadlock F1 was overstated (an illegal baseline flattered it) but stays far above
the +0.015 bar either way.

---

## 8. Leakage

Target-perturbation probe: overwriting `label`, `success` and `fault_turn` on 300 runs left the
feature matrix **identical** (max|diff| 0). The parser reads only `type`, `from`, `text`, `t`.
Unit tests 55/55 PASS, including all 6 template families.

---

## 9. Not done, deliberately

No second feature variant, no re-search, no parser or gate change. No production submission and
no ZIP yet — that is the next step and is gated on this research result being accepted.