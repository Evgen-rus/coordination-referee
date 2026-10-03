# Exp15 - success stacking from Exp13 label probabilities (`exp15_success_label_stacking`)

**Status: PROMOTE (gate 5/5).**
One frozen hypothesis, one appended block of 7 raw Exp13 label probabilities on
the SUCCESS head, one honest nested OOF. Label predictions, Macro, Robustness,
hit@2 and `fault_turn` are exactly unchanged; only `success_f1` moves. Production
is **not** built here — by protocol a PASS waits for approval.

---

## 1. The single hypothesis

> The current success head loses useful information because it is still the
> 185-feature Exp03 model and does not observe the current strong Exp13
> coordination-label probabilities. Honest out-of-fold label probabilities
> should improve success prediction.

## 2. What is and is not touched

**Touched:** one block of 7 columns on the SUCCESS head — the raw
`predict_proba` of the current Exp13 label model, in `LABELS` order
(`lp_clean` … `lp_runaway_loop`). No entropy, margin, max-prob, hard-label
one-hot, threshold or class-weight column.

**Not touched:** the label head (matrix, params, seed, folds); `fault_turn` / L1;
Exp14's 16 `ep_*` columns; the success LightGBM hyper-parameters and
`random_state=42`; the outer folds; thresholds; feature selection; seed search.

**CONTROL A** = 185 Exp03 features, production `make_success_model()`.
**CANDIDATE B** = the SAME 185 + 7 honest label probabilities = 192.
A and B share the first 185 columns (`np.array_equal` per fold) and differ in
exactly 7 appended columns.

## 3. Honest stacking

For each outer fold (seed-0 `StratifiedKFold(3)`, sealed Exp07/Exp03 folds):

- **validation rows** — Exp13 corrected 312-feature label model fitted on all
  outer-train (the Exp13 D fold path, rebuilt here);
- **training rows** — nested 3-fold OOF strictly inside outer-train
  (`StratifiedKFold(3, shuffle=True, random_state=1)`). Each nested fold rebuilds
  the full corrected 312-feature representation (3 honest Exp07 window fits →
  51 aggregates via Exp12's corrected mapping → outer-train window model for
  nested-validation aggregates → label fit) and predicts only its held-out rows.

Exclusion is proven structurally (fit set never contains the held-out row;
every outer-train row held out exactly once). OOF vs in-sample refit:
`max|dP| ≈ 0.997`, ~1200/6666 argmax differ — the training probabilities are
not in-sample.

## 4. Deviation from the written design (recorded, not hidden)

A first execution had two defects, both surfaced by this script's own gates:

1. nested splitter was built with `N_META` (7) instead of `N_INNER` (3);
2. GATE 2 compared the argmax of the training-row meta array (which also carries
   nested probabilities) against Exp13 D, and failed on 625 rows.

That run is archived under `_first_run_7fold/`. It was **not** re-scored after
seeing its delta; the accepted run uses the designed 3 nested folds and compares
parity on `p_meta_outer` (outer-validation probabilities only). Neither defect
touched the success head's 185 base columns.

## 5. Baseline reproduction — PASS, exact

```
   success_f1 expected 0.8643757406010988  got 0.8643757406010988  diff 0.00e+00  PASS
   predictions vs committed Exp03 B csv: 0 / 10000 differ -> IDENTICAL
```

Carried Exp13 D metrics (frozen inputs, never refitted):

| metric | value |
| --- | --- |
| macro_f1 | 0.8129545586738590 |
| robustness_f1 | 0.7599843692318033 |
| fault_turn_hit2 | 0.6315277777777778 |
| composite (with A success) | 0.7892825105128229 |

Committed Exp13 D OOF sha256:
`f4e70a4280c825613e54cde7a843915323a89643be78138ad5c48f2ced131853`.

## 6. Label parity — PASS

Outer label path matches Exp13 D argmax **10000 / 10000**;
`max|dP| = 2.980e-08` (float32 rounding). Macro / Robustness / hit@2 recomputed
from that vector match the carried values at `0.00e+00`. The `fault_turn` L1
vector is byte-identical to Exp13 D's (7019 non-`-1`).

## 7. A vs B (success only)

| metric | A (185) | B (185+7) | delta |
| --- | --- | --- | --- |
| **success F1** | 0.8643757406 | **0.8949086162** | **+0.030533** |
| precision | 0.857815 | 0.896860 | +0.039045 |
| recall | 0.871038 | 0.892966 | +0.021928 |
| FP | 665 | 473 | -192 |
| FN | 594 | 493 | -101 |
| TP | 4012 | 4113 | +101 |
| TN | 4729 | 4921 | +192 |
| **composite** | 0.7892825105 | **0.7938624419** | **+0.004580** |

`composite_B - composite_A == 0.15 * (success_B - success_A)` holds exactly
(Macro / Robustness / hit@2 frozen).

### Fold stability (success)

| fold | n | F1 A → B | delta | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | 3334 | 0.867056 → 0.893519 | **+0.026463** | 227→153 | 183→169 |
| 1 | 3333 | 0.864286 → 0.897478 | **+0.033192** | 218→152 | 200→161 |
| 2 | 3333 | 0.861815 → 0.893740 | **+0.031925** | 220→168 | 211→163 |

Folds won **3/3**. Worst fold delta **+0.026463**.

### Changed predictions

Changed 741 / 10000 (7.41%): **wrong→right 517**, right→wrong 224, wrong→wrong 0.

### Class-conditional success F1

| true class | n | succ.prev | F1 A | F1 B | delta |
| --- | --- | --- | --- | --- | --- |
| clean | 2800 | 0.9464 | 0.9195 | 0.9444 | +0.0248 |
| dropped_handoff | 1200 | 0.1742 | 0.6935 | 0.7160 | +0.0225 |
| duplicated_work | 1200 | 0.6208 | 0.8857 | 0.9015 | +0.0158 |
| deadlock | 1200 | 0.0417 | 0.3944 | 0.5816 | **+0.1872** |
| conflict | 1200 | 0.3842 | 0.8651 | 0.8856 | +0.0205 |
| goal_drift | 1200 | 0.1433 | 0.7200 | 0.7614 | +0.0414 |
| runaway_loop | 1200 | 0.2658 | 0.7674 | 0.7779 | +0.0105 |

Every class improves. The largest absolute lift is on `deadlock` (rare success
positives; the label probabilities are highly informative there).

### Gain share of the 7 probabilities

Total gain share of the 7 meta columns: **31.215%**. Top: `lp_clean` (24.5%),
then `lp_deadlock` (2.1%). 5 of 7 land in the top-20 features.

## 8. Production design (not built)

A PASS implies a production build that:

1. keeps `submission_exp13_wait_dependency_graph` as the base;
2. leaves the label head (312 features), window model, aggregates, L1 and every
   threshold byte-identical;
3. widens **only** the success head: 185 → 192, appending the 7 raw label
   probabilities in `LABELS` order;
4. at train time, produces those 7 columns by the same honest nested OOF recipe
   used here (outer folds = sealed seed-0; nested seed = 1);
5. at test time, fits the label model on all train rows and predicts the 7
   probabilities for every test row once.

**Not done here.** No `submission_exp15_*`, no ZIP, no leaderboard upload.
Production waits for approval.

## 9. Gate — PASS 5/5

| criterion | required | measured | |
| --- | --- | --- | --- |
| success_f1 delta | >= +0.010 | **+0.030533** | PASS |
| folds won | >= 2/3 | **3/3** | PASS |
| composite delta | >= +0.0015 | **+0.004580** | PASS |
| no fold worse than | -0.005 | **+0.026463** | PASS |
| Macro / Robustness / hit@2 / L1 | exactly unchanged | yes | PASS |

**VERDICT: PROMOTE.** Production created: **NO**.

## 10. Honest limitations

1. **OOF on the same 10000 train runs.** The public leaderboard remains the
   real external validation.
2. **Single nested seed (`NESTED_SEED=1`).** No seed search; the fold check is
   consistency, not significance.
3. **Nested training probabilities come from smaller fits** (~4444 rows) than
   the outer validation probabilities (~6666). That is the price of honesty;
   production will use full-train probabilities at test time.
4. **`fault_turn` inherits Exp08's post-hoc caveat.** hit@2 is unchanged here
   and is still an upper bound, not an estimate.
5. **First-run deviation** (7 nested folds, Gate-2 mis-comparison) is archived,
   not discarded silently.

## 11. Reproduction

```bash
.venv/Scripts/python.exe experiments/protocol_guard.py
.venv/Scripts/python.exe experiments/exp15_success_label_stacking/run_experiment.py
.venv/Scripts/python.exe -m pytest experiments/exp15_success_label_stacking/test_stacking.py -q
```

Runtime of the accepted run: ~31.4 min (1886 s).
