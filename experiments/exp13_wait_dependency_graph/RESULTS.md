# Exp13 - explicit wait-dependency graph (`exp13_wait_dependency_graph`)

**Status: PROMOTED -> `submission_exp13_wait_dependency_graph`.**
One new feature block, one frozen hypothesis, two independent measurements
(historical OOF semantics and production-corrected semantics), both gates passed.

---

## 1. The single hypothesis

The ontology defines `deadlock` relationally: *A waits for B's subtask, B waits
for A's, a status exchange happens, and no progress follows.* The existing
representation can only approximate that through aggregate proxies
(`share_status`, r ~ 0.72), because the 300 features count message edges of
**every** type together and never parse *who is being waited for*.

`wait_graph` parses the six official waiting-message templates, builds the
directed relation `sender -> awaited_agent`, and summarises it.

## 2. The frozen block

12 features, fixed **before** any model was fitted. The regex list is a literal
transcription of the six template families named in the audit and is asserted to
be exactly six entries; no regex was added, removed or loosened after any CV
number was seen.

| feature | meaning |
| --- | --- |
| `wg_n_wait_edges` | total parsed wait events |
| `wg_n_unique_wait_edges` | distinct `sender -> awaited` pairs |
| `wg_n_recip_pairs` | reciprocal wait pairs (A waits B and B waits A) |
| `wg_has_recip_pair` | indicator |
| `wg_recip_wait_share` | share of wait events inside reciprocal pairs |
| `wg_n_same_subtask_recip_pairs` | reciprocal pairs **on the same subtask** |
| `wg_has_same_subtask_recip_pair` | indicator - the ontology's core condition |
| `wg_earliest_recip_pos_rel` | relative position of the first reciprocal pair |
| `wg_max_pair_wait_events` | most waits inside any single pair |
| `wg_max_pair_repeat_after_onset` | repeat pressure after that pair forms |
| `wg_max_pair_span_rel` | relative span of the pair |
| `wg_n_agents_in_recip_pairs` | agents participating in reciprocal waits |

**Leakage.** Only `messages` entries and their `t` / `from` / `text` fields are
read. `test_features.py` (55 tests) proves it by target perturbation: the matrix
is invariant under arbitrary rewriting of `label`, `success` and `fault_turn`.

## 3. Measurement 1 - research OOF (historical Exp08 semantics)

Comparison against the **sealed** Exp08 OOF. Control A reproduced the sealed
artifact bit for bit (`max|dP| = 0.0`, sha `edbaf8d766f97f28bce12e72fbf9fee6`),
so the delta below is a feature effect on one pipeline, not a difference
between two pipelines.

| metric | Exp08 (A) | + wait graph (B) | delta |
| --- | --- | --- | --- |
| composite | 0.7694453917 | **0.7926215140** | **+0.023176** |
| macro F1 | 0.7894323367 | 0.8158680494 | +0.026436 |
| robustness F1 | 0.7368470047 | 0.7670134020 | +0.030166 |
| hit@2 | 0.6086111111 | 0.6327777778 | +0.024167 |
| success F1 | 0.8643757406 | 0.8643757406 | 0 (pinned) |
| **deadlock F1** | 0.7217391304 | **0.8366927190** | **+0.114954** |

3/3 folds on composite **and** on deadlock F1. No class fell at all.

## 4. Measurement 2 - PRODUCTION SEMANTICS (corrected mapping)

**Why this second measurement exists.** Exp12 proved that the shipped
production Exp08 uses *corrected* window-aggregation semantics, while the
historical OOF carries the global/local indexing defect (~33% of outer-train rows
had 51 zeroed aggregates). Historical Exp08 is therefore not a faithful model of
production Exp08, and measurement 1 alone does not justify a production change.

**This is not a new hypothesis.** Same 12 features, same parser, same LightGBM
parameters, same seeds, same folds, no tuning. Only the aggregation semantics of
the shared base change, and they change for **both** arms at once:

```
CONTROL C   = Exp12 corrected representation, no wait graph  (= Exp12 arm B)
CANDIDATE D = the SAME corrected fold-local matrices + the frozen 12 wg_* columns
```

C and D consume **one** set of inner window fits and **one** pair of 300-column
fold-local matrices per fold, so they differ in exactly the 12 appended columns.

### 4.1 Control C reproduction gate - PASS

CONTROL C must reproduce Exp12's committed corrected numbers before D is scored.
It did, **bit for bit**:

| metric | Exp12 committed | Control C | diff |
| --- | --- | --- | --- |
| macro F1 | 0.7873785254579829 | 0.7873785254579829 | 0.00e+00 |
| robustness F1 | 0.7368422139406114 | 0.7368422139406114 | 0.00e+00 |
| success F1 | 0.8643757406010988 | 0.8643757406010988 | 0.00e+00 |
| hit@2 | 0.6093055555555555 | 0.6093055555555555 | 0.00e+00 |
| composite | 0.7684867328598647 | 0.7684867328598647 | 0.00e+00 |

OOF sha256 `b7e92bd4e14fa59c8c340e65c70317d4...` == Exp12's committed arm-B sha.
5/5 PASS.

### 4.2 C vs D under production semantics

| metric | C (corrected) | D (+ wait graph) | delta |
| --- | --- | --- | --- |
| **composite** | 0.7684867329 | **0.7892835288** | **+0.020796** |
| macro F1 | 0.7873785255 | 0.8129544747 | +0.025576 |
| robustness F1 | 0.7368422139 | 0.7599841122 | +0.023142 |
| hit@2 | 0.6093055556 | 0.6315277778 | +0.022222 |
| success F1 | 0.8643757406 | 0.8643757406 | 0 (pinned) |
| **deadlock F1** | 0.7143476376 | **0.8336054376** | **+0.119258** |

Per-class F1 - **no class falls**:

| class | C | D | delta |
| --- | --- | --- | --- |
| clean | 0.8669 | 0.8784 | +0.0115 |
| dropped_handoff | 0.6294 | 0.6550 | +0.0257 |
| duplicated_work | 0.8363 | 0.8411 | +0.0047 |
| **deadlock** | 0.7143 | **0.8336** | **+0.1193** |
| conflict | 0.9010 | 0.9033 | +0.0023 |
| goal_drift | 0.8408 | 0.8443 | +0.0035 |
| runaway_loop | 0.7229 | 0.7349 | +0.0121 |

### 4.3 The main target, `deadlock`

| | C | D |
| --- | --- | --- |
| precision | 0.7444 | **0.8281** (+0.0838) |
| recall | 0.6867 | **0.8392** (+0.1525) |
| tp / fp / fn | 824 / 283 / 376 | 1007 / 209 / 193 |

The gain is not a threshold trade - both precision and recall rise together,
because the block supplies information the aggregates could not express.

Error transitions, all in the right direction:

| transition | C | D |
| --- | --- | --- |
| `deadlock -> clean` | 111 | **51** (-60) |
| `clean -> deadlock` | 59 | **24** (-35) |
| `dropped_handoff -> deadlock` | 95 | **66** (-29) |
| `runaway_loop -> deadlock` | 72 | **59** (-13) |

### 4.4 Fold stability (3 outer folds, no seed search)

| fold | composite Δ | macro Δ | robustness Δ | hit@2 Δ | deadlock F1 |
| --- | --- | --- | --- | --- | --- |
| 0 | +0.020559 | +0.025672 | +0.022391 | +0.021250 | 0.7221 -> 0.8319 (+0.1098) |
| 1 | +0.024510 | +0.026530 | +0.036979 | +0.020000 | 0.7186 -> 0.8394 (+0.1208) |
| 2 | +0.016185 | +0.024880 | +0.004816 | +0.025417 | 0.7018 -> 0.8294 (+0.1276) |

**3/3 folds won on composite, macro, robustness, hit@2 and deadlock F1.**

## 5. Gates

### 5.1 Original strict Exp13 gate - PASS 6/6 (unchanged, not retuned)

| criterion | required | measured | |
| --- | --- | --- | --- |
| composite delta | >= +0.002 | **+0.020796** | PASS |
| folds won | >= 2/3 | **3/3** | PASS |
| deadlock F1 delta | >= +0.015 | **+0.119258** | PASS |
| deadlock folds won | >= 2/3 | **3/3** | PASS |
| robustness | >= -0.003 | **+0.023142** | PASS |
| no class worse than | -0.02 | **none fell** | PASS |

### 5.2 Production confirmation gate - PASS 5/5

| criterion | measured | |
| --- | --- | --- |
| composite delta > 0 | +0.020796 | PASS |
| >= 2/3 folds composite > 0 | 3/3 | PASS |
| deadlock F1 delta > 0 | +0.119258 | PASS |
| >= 2/3 folds deadlock F1 > 0 | 3/3 | PASS |
| no class F1 drop > 0.02 | 0.0 (none fell) | PASS |

## 6. Historical gain vs corrected-semantics gain

| semantics | composite Δ | deadlock F1 Δ |
| --- | --- | --- |
| historical Exp08 (defective) -> +wg | +0.023176 | +0.114954 |
| **corrected Exp12 -> +wg** | **+0.020796** | **+0.119258** |

**The wait-graph finding survives removal of the mapping defect.** The composite
gain shrinks by ~10% (0.0232 -> 0.0208) and the deadlock gain slightly *grows*
(0.1150 -> 0.1193). Both are far above their gates under either semantics, so
the production decision does not depend on which OOF is consulted. The residual
difference between the two columns is a property of the *baseline*, not of the
new features.

## 7. Production build

`submission_exp13_wait_dependency_graph` = `submission_exp08_window_localizer`
plus **exactly one** change: the 12 `wg_*` columns appended to the label matrix
(300 -> 312).

- 10 of the 11 inherited modules are **sha256 byte-identical** to sub08
  (`aggregate`, `features`, `lifecycle`, `localize`, `new_features`, `peak_turn`,
  `temporal_features`, `window_dataset`, `window_features`, plus sub08's own test).
- `solution.py` differs only in the docstring, one import, one constant, and the
  label-matrix assembly. The 300 production columns are the same 249 foundation
  + the same 51 aggregates in the same order.
- `make_label_model` / `make_success_model` / `make_window_model` return
  **identical `get_params()`**. Window model, 51 aggregates, honest inner
  cross-fit, success head (185 feats), L1 localiser, seeds: unchanged.
- `wait_graph.py` is the research module verbatim; numerically identical to it on
  real runs; all 6 template families fire on real data.

### 7.1 Parity checks - all PASS

Module hashes, model params, column layout, block-equals-research equality,
6/6 templates firing, target-perturbation invariance, no non-finite values,
fallback on an incomplete schema, and the prediction delta below.

### 7.2 End-to-end on the real 10k -> 4k test

| | sub08 | sub13 |
| --- | --- | --- |
| runtime | 600.8 s | **375.0 s** |
| success predictions | - | **byte-identical, 4000/4000** |
| label changed | - | 322 / 4000 (8.05%) |
| deadlock predictions | 427 | **496 (+69)** |
| fault_turn changed | - | 317 |
| `fault_turn` moved **without** a label change | - | **0** |

Largest transitions, all moving *into* `deadlock` or *out of* it:

```
clean           -> deadlock        57      deadlock -> clean             17
dropped_handoff -> deadlock        27      deadlock -> dropped_handoff  22
runaway_loop    -> deadlock        24      deadlock -> runaway_loop     18
```

Every one of the 317 changed `fault_turn` values coincides with a changed
predicted label, i.e. they are the L1 peak being re-read for the new class, not a
change in L1 behaviour. `clean` rows carry `-1` in both submissions.

**Validator:** OK, 4000 rows, format valid, run_id order matches
`sample_submission.csv`.

## 8. Honest limitations

1. **Both measurements are OOF on the same 10000 train runs.** The public
   leaderboard remains the real external validation.
2. **`fault_turn` inherits Exp08's post-hoc caveat.** Exp08's L1 rule was
   discovered on Exp07's OOF, so hit@2 0.6086 is an upper bound on that gain,
   not an estimate of it. Exp13 does not make that number more or less honest;
   it only moves the label the peak is read for.
3. **`success` F1 is pinned, not measured.** No `wg_*` feature targets
   `success`, and the head is unchanged, so the experiment pins it at the
   validated 0.8644 by construction.
4. **Single seed, 3 folds.** The fold check here is consistency, not
   significance. No seed search was performed, by design.
5. **The two measurements are not independent samples.** C vs D shares folds,
   features and seeds with the historical run; they are two *baselines*, not two
   datasets. The 10% composite shrinkage is the informative part, and it is
   small.
6. **The test-set prediction delta is not scored.** `test.csv` has no labels
   here; the 322 changed labels are reported, not evaluated.

## 9. Reproduce

```bash
python -m pytest experiments/exp13_wait_dependency_graph/test_features.py -q
python experiments/exp13_wait_dependency_graph/paired_fold_local.py
python experiments/exp13_wait_dependency_graph/production_confirmation.py
python submission_exp13_wait_dependency_graph/check_parity.py
```
