# Exp07 - fault-centred window model

**Verdict: PROMOTE.**

One hypothesis tested, no new foundation, no hyper-parameter search, no post-hoc
subsetting. The official metric is **composite**, and that is what the selection
is based on.

---

## 1. Headline

| metric | A: foundation (249) | B: + window (300) | delta | criterion |
|---|---|---|---|---|
| **COMPOSITE** | 0.7456 | **0.7517** | **+0.0061** | required >= +0.005 -> **MET** |
| Macro F1 | 0.7827 | 0.7894 | +0.0067 | desirable >= +0.005 -> **MET** |
| Robustness F1 | 0.7279 | 0.7368 | +0.0090 | not worse than -0.003 -> **MET** |
| fault_turn hit@2 | 0.4258 | 0.4308 | +0.0050 | reported only |
| Success F1 | 0.8644 | 0.8644 | +0.0000 | pinned, not changed |

Fold-by-fold Macro F1 - **3/3 folds improve**, and the per-fold spread is tight:

| fold | A | B | delta |
|---|---|---|---|
| 0 | 0.7731 | 0.7796 | +0.0065 |
| 1 | 0.7898 | 0.7976 | +0.0079 |
| 2 | 0.7848 | 0.7909 | +0.0061 |

**All five pre-declared success criteria are met.** This clears the primary
threshold (composite +0.005) without relying on the weak 0.10-weight hit@2 term.

A useful consistency check: system A reproduces Exp06b's validated `full - age`
Macro of **0.7827** on this split exactly, so the gain is measured against a
foundation that is known to be correct.

## 2. Per-class F1

| class | A | B | delta |
|---|---|---|---|
| clean | 0.8579 | 0.8650 | +0.0071 |
| dropped_handoff | 0.6106 | 0.6356 | **+0.0250** |
| deadlock | 0.7049 | 0.7217 | +0.0168 |
| goal_drift | 0.8432 | 0.8469 | +0.0037 |
| duplicated_work | 0.8274 | 0.8296 | +0.0023 |
| conflict | 0.9043 | 0.9032 | -0.0012 |
| runaway_loop | 0.7305 | 0.7239 | -0.0066 |

No class loses more than 0.02 (criterion met). The two classes that improve most,
`dropped_handoff` and `deadlock`, are exactly the ones whose failures are
*spatially* identifiable - an assignment that is never resolved, or a cycle of
status messages.

## 3. Confusion pairs (counts out of 1200 each)

| pair | A | B | delta |
|---|---|---|---|
| dropped_handoff -> clean | 203 | 171 | **-32** |
| dropped_handoff -> deadlock | 99 | 92 | -7 |
| deadlock -> dropped_handoff | 59 | 59 | 0 |
| duplicated_work -> clean | 89 | 76 | -13 |
| runaway_loop -> clean | 114 | 110 | -4 |
| goal_drift -> clean | 40 | 41 | +1 |

The dominant failure mode of the project - a dropped handoff being dismissed as a
clean run - falls by 16%. That is a real, interpretable improvement, not a
redistribution.

## 4. Slices

| slice | n | A | B | delta | 95% CI | verdict |
|---|---|---|---|---|---|---|
| topo_mesh | 1481 | 0.7842 | 0.8013 | **+0.0172** | [+0.0035, +0.0303] | **CI excludes zero** |
| hard/robustness | 1290 | 0.7279 | 0.7368 | +0.0091 | [-0.0065, +0.0248] | spans zero |
| long(>p66) | 3210 | 0.7133 | 0.7176 | +0.0043 | [-0.0059, +0.0144] | spans zero |
| no_intent | 1198 | 0.6779 | 0.6792 | +0.0013 | [-0.0162, +0.0189] | spans zero |
| topo_blackboard | 1010 | 0.8018 | 0.7967 | -0.0050 | [-0.0198, +0.0097] | spans zero |

`topo_mesh` is the one slice whose confidence interval excludes zero. Given
Exp06b's finding that Exp06's `topo_mesh` gain was a seed-0 artifact, this is
worth noting as the *first* topo_mesh effect in the project with a CI that clears
zero - but it rests on one OOF run and should be treated as provisional.

## 5. Significance

Macro F1, run-level paired bootstrap (20 000 resamples, macro F1 **recomputed on
each resample**):

- delta **+0.0067**, 95% CI **[+0.0015, +0.0119]**, **p = 0.0096**

The composite is not a per-run estimand - it is a fixed weighted sum of three
whole-set statistics - so it is not bootstrapped over runs. The honest treatment
at n=3 paired OOF runs is a fold-level check: per-fold deltas
+0.0065 / +0.0079 / +0.0061, **3/3 wins**. With n=3 that is a consistency check,
not a significance level, and it is reported as such.

Two estimator bugs were found and fixed while producing these numbers, and both
are now gated:

- a hand-rolled macro F1 returned **1.3058** for a quantity whose true value is
  0.7827 (it did not mask to `y_true == c` before counting false positives), which
  inflated the reported delta to +0.0162. `significance.py` now asserts its
  estimator reproduces `results.json` to 1e-9 before bootstrapping anything.
- a composite bootstrap over runs double-counted the robustness slice, reporting
  +0.0033 against the true +0.0061. Replaced by the fold-level check above.

## 6. The hit@2 discrepancy, explained

`results.json` reports 0.4258 -> 0.4308. An early version of `diagnostics.py`
reported 0.1843 -> 0.1927 for what appeared to be the same thing. Those were
**not** two definitions - the second number was a bug.

`diagnostics.py` built the turn array by skipping `clean` runs, then zipped it
positionally against the unfiltered `y_true` / `fault_turn` arrays. Every
element after the first `clean` row was therefore misaligned. Both figures are
now on the single official definition (`metrics.fault_turn_hit_at_k`):

- denominator = **every truly-faulty run** (7200), not only runs the system
  called faulty;
- a run whose *label* was misclassified counts as a **miss**;
- arrays are index-aligned with the full run list.

`verify_hit_at_k.py` recomputes the official number independently and fails if the
two ever disagree; it currently matches to 6 decimals (0.425833 / 0.430833).
**If you see a second hit@2 number anywhere, it is a bug, not a different metric.**

## 7. Localisation: L0 / L1 / L2

All three evaluated on the official definition, for system B:

| turn source | hit@0 | hit@1 | hit@2 | note |
|---|---|---|---|---|
| **L0** official baseline localiser | 0.3618 | 0.3981 | **0.4308** | used in the headline |
| **L1** window peak for predicted label | 0.1142 | 0.3400 | **0.6086** | +0.178 over L0 |
| L2 hybrid (window if conf >= 0.60) | 0.2518 | 0.3790 | 0.5122 | **post-hoc, diagnostic only** |

L2's 0.60 threshold was fixed *after* seeing L1, so it is a post-hoc diagnostic
and **is excluded from the headline result** - the reported Exp07 configuration
uses L0, the official localiser, exactly as pre-registered.

L1 is dramatically better at hit@2 (+0.178) but much worse at hit@0 (0.114 vs
0.362). The reading: the window model places its peak *near* the right region but
not on the exact turn, whereas the rule-based localiser nails the exact turn when
it fires. The metric rewards ±2, so L1 wins by a lot.

Per-class L1 hit@2 (system B):

| class | hit@0 | hit@1 | hit@2 | n |
|---|---|---|---|---|
| duplicated_work | 0.113 | 0.377 | 0.728 | 1200 |
| conflict | 0.111 | 0.386 | 0.700 | 1200 |
| runaway_loop | 0.142 | 0.363 | 0.592 | 1200 |
| deadlock | 0.116 | 0.320 | 0.588 | 1200 |
| goal_drift | 0.103 | 0.329 | 0.577 | 1200 |
| dropped_handoff | 0.100 | 0.265 | 0.468 | 1200 |

Peak offset vs true `fault_turn`: median **+0**, exact hit 0.1271, within ±2 0.6724,
within ±5 0.7514, later than +2 in only 0.1540 of cases. The distribution is
tightly centred, which is what makes L1 useful.

**This corrects the working assumption stated when the experiment was launched.**
The expectation was that the window model would add fault-type signal but not
improve localisation. It does improve localisation substantially at the metric's
own tolerance. Whether that survives a genuine competition run is untested.

## 8. Window model behaviour

OOF over 457 203 validation windows, background = 92.6% of all windows:

- overall accuracy 0.9356; background precision 0.9427, recall 0.9918
- per-fault window recall: duplicated_work 0.3688, conflict 0.2257, deadlock 0.2077,
  runaway_loop 0.2020, goal_drift 0.1984, **dropped_handoff 0.1708**

Window-level fault recall is low, which is expected and not a problem: each fault
class has only 5 positive windows out of ~45, and the model must localise, not
merely type. Precision is high (0.61-0.74), so when it does fire on a window it is
usually right about the type.

**Does it find the root-cause region, or just recognise the run type?** Both, and
the split is measurable. The `duplicated_work` and `conflict` window recalls are
the highest (0.3688 / 0.2257) and their L1 hit@2 is also highest (0.728 / 0.700):
for these, the model genuinely discriminates a local region. For
`dropped_handoff`, window recall is lowest (0.1708) and L1 hit@2 lowest (0.468) -
here the run-level context dominates and the window adds type evidence rather
than position evidence. That is consistent with the class-level result:
`dropped_handoff` F1 still improved the most (+0.0250), but via the aggregated
`area` features, not via localisation.

## 9. Which window features carried the signal

Window block share of total gain in the B model: **13.43%** of all splits. All 51
window features rank inside the global top 100.

Top window features by gain share:

| feature | gain share |
|---|---|
| p_goal_drift_area | 2.07% |
| g_bg_mean | 1.26% |
| p_duplicated_work_area | 0.84% |
| p_goal_drift_amax | 0.72% |
| p_dropped_handoff_atop3 | 0.70% |
| p_dropped_handoff_area | 0.69% |
| p_deadlock_atop3 | 0.66% |
| g_probs_entropy | 0.57% |
| p_deadlock_area | 0.49% |
| p_dropped_handoff_amax | 0.44% |

The dominant family is **`_area` and `_atop3` / `_amax`** - soft aggregates of
window probability mass over the whole run. The hard positional features
(`_pos_max`, `_pos_first_hi`, `_n_hi`) are not what the classifier leans on. So the
label head is using the window model mainly as a **distributional evidence
channel** ("how much probability mass of type X is anywhere in this run"), with
position as a weaker secondary cue. That is a cleaner story than "it found the
fault turn", and it is why the label gain does not depend on localisation being
perfect.

## 10. Design decisions, fixed in advance

| decision | value | note |
|---|---|---|
| positive zone | +-2 around true `fault_turn` | fixed, not tuned |
| wider context | +-5 | for local-change features only |
| window features | 83 | design band 50-120 |
| aggregation | 51 | design band 30-60 |
| window model | LightGBM, 300 trees, lr 0.08, 63 leaves | no search |
| class weighting | background 0.25, each fault 0.125 | pre-declared vector, identical in every fold |
| outer / inner folds | 3 / 3, seed 0 | same folds as Exp05/06/06b |
| success head | Exp03 B, 185 feats, F1 pinned 0.8644 | unchanged |
| fault_turn | official baseline localiser | unchanged |

No post-hoc subset search was performed, and none should be: only A vs B was
compared, as pre-registered.

## 11. Leakage audit

`verify_no_leakage.py` - all checks pass:

- **targets not in features** - `window_features.py`, `aggregate.py`,
  `grouping.py` AST-scanned; zero target references. The builder takes a parsed
  run, which carries run content only, so leakage is impossible by construction.
- **causality** - features are bit-identical (max abs diff 0.0) when
  `label` / `success` / `fault_turn` are altered in the input row.
- **true `fault_turn` used only for target construction** - it appears solely in
  `window_dataset.window_targets`.
- **clean runs get no synthetic fault target** - verified: 0 fault windows across
  61 clean runs; all clean runs carry `fault_turn == -1`.
- **radius respected** - every faulty run gets 3-5 fault windows, never more than
  the fixed +-2 zone.
- **outer-validation never trains the window model** - the outer-valid features
  come from a model fitted on outer-train only.
- **no in-sample window predictions for the final classifier** - inner 3-fold
  cross-fitting, verified at runtime by `verify_stacking`, which re-derives fold
  membership and fails on empty folds, overlap, out-of-range assignment, wrong
  length, or non-partitioning. Unit-tested against 6 deliberately broken
  assignments.
- **no absolute local paths, no network** - AST-scanned across all modules.
- **baseline untouched** - `git diff --stat -- baseline/ evaluation/` empty;
  `features.py` sha256[:12]=72b1373b8c03, `localize.py` =b9831906f783,
  `solution.py` =76b7e692c605.

The in-sample check matters in practice: fitted on 300 runs the window model hits
**100% in-sample** accuracy, because features like `w_n_msgs` identify the run.
Cross-fitting is what makes the reported number honest.

## 12. Runtime

| stage | time |
|---|---|
| foundation build | 22 s |
| window dataset (457 203 windows, 83 features) | ~35 s |
| main A/B run, 3 folds x (1 outer window model + 3 inner + 2 classifiers) | 509 s |
| diagnostics | ~420 s |
| significance (20k bootstraps, 6 slices) | ~545 s |
| leakage audit | 8 s |

Total ~25 min single-run, dominated by bootstrapping. A production submission
would add ~2-3 min of window-model fitting, well within a 30-min job limit.

## 13. Verdict: PROMOTE

Reasons, in order of weight:

1. **Composite +0.0061 clears the pre-declared +0.005 bar**, with Macro +0.0067
   (p=0.0096), Robustness +0.0090, and 3/3 folds improving. This is the first
   experiment since Exp03/Exp05 to clear its own threshold on the official metric.
2. The gain is concentrated in a **mechanistically explicable** way:
   `dropped_handoff` F1 +0.0250 and `dropped_handoff -> clean` errors -32.
3. `topo_mesh` finally moves with a **CI that excludes zero** (+0.0172), which is
   the slice every previous experiment has failed to move credibly.
4. It adds **51 features, not a new architecture** - the foundation is untouched,
   so the risk of the gain being a foundation artefact is low.

Caveats that must travel with the promotion:

- **Single seed, 3 folds.** The fold-level check is consistency evidence, not
  significance. The run-level bootstrap on Macro is significant (p=0.0096), but
  `full_minus_age` was itself chosen post-hoc, so some optimism is expected -
  Exp06b's 5-seed study is the right template for confirming this on the
  competition train before treating the number as final.
- **`topo_mesh` is one slice on one split.** It is the most interesting finding
  here and also the most fragile.
- **L1 localisation is not in the promoted configuration.** The +0.178 hit@2
  finding is real and large, but wiring it in would be a separate, post-hoc change
  to a metric worth only 0.10, on a threshold that was itself chosen after the
  fact. It belongs in a follow-up experiment, not in this promotion.
- Success F1 is pinned, so Exp07 says nothing about the success head.

**Recommended next step:** ship this as the second public submission now, since
it has cleared the pre-registered bar, and confirm with a 5-seed run on the
competition train in the same way Exp06b confirmed `full - age`. If the
competition data lands first, run the seeds before uploading.

## 14. Files

| file | contents |
|---|---|
| `window_features.py` | 83 structural window features (no targets) |
| `window_dataset.py` | target construction, fixed weights |
| `aggregate.py` | 51 run-level aggregates, fixed 0.5 threshold |
| `grouping.py` | window-row to per-run block mapping |
| `common.py` | foundation build, models, metrics, slices |
| `run_experiment.py` | 3-fold A/B with inner cross-fitting + stacking assert |
| `diagnostics.py` | window model behaviour, L0/L1/L2, peak offsets |
| `significance.py` | gated run-level + fold-level bootstrap |
| `verify_no_leakage.py` | 9-group leakage audit |
| `verify_hit_at_k.py` | pins diagnostics hit@k to the official metric |
| `results.json`, `significance.json`, `diagnostics.json` | metrics |
| `oof_A_foundation.csv`, `oof_B_plus_window.csv` | OOF predictions |
| `window_*.npy` | peak positions, peak probabilities, window OOF labels |
| `run.log`, `diag.log`, `significance.log` | runtime logs |
