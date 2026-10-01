# Exp13 - submission: explicit wait-dependency graph

`submission_exp13_wait_dependency_graph` is `submission_exp08_window_localizer`
with **exactly one** behavioural change:

```
pred  = LGB(300 production features)                 # submission_exp08
pred  = LGB(300 production features + 12 wg_* wait)  # THIS submission
```

Everything else - the turn-level window LightGBM, its 83 structural features, the
51 run-level aggregates and their honest inner cross-fit, the 185-feature success
head, the L1 window-peak localiser, the seeds and every decision threshold - is
carried over untouched. This is a feature addition, not a configuration change.

---

## Why a wait-dependency graph

The ontology defines `deadlock` **relationally**: *A waits for B's subtask, B
waits for A's, a status exchange occurs, and no progress follows.* The 300
production features can only approximate that condition through aggregate
proxies (`share_status`, r ~ 0.72 with the new counts), because they count
message edges of every type together and never parse **who is being waited
for**.

`wait_graph.py` parses the six official waiting-message templates, builds the
directed relation `sender -> awaited_agent`, and summarises it into 12 features:

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

The block was frozen **before** any model was fitted. The regex list is a literal
transcription of the six template families named in the audit and is asserted to
be exactly six entries; no regex was added, removed or loosened after any
cross-validation number was seen.

**Leakage.** Only `messages` entries and their `t` / `from` / `text` fields are
read. `label`, `success` and `fault_turn` are never touched, which
`check_parity.py` proves by perturbation rather than by inspection: overwriting
those three fields on real runs moves not one value.

---

## Promotion basis: measured twice, both gates cleared

Exp12 established that the shipped Exp08 uses **corrected** window-aggregation
semantics while the historical OOF carries a global/local indexing defect. Since
production Exp08 is therefore not measured by the historical OOF, Exp13 was
re-validated against corrected semantics before this build.

### Against the sealed Exp08 OOF (research)

Control A reproduced the sealed artifact bit for bit (`max|dP| = 0.0`), so the
delta is a feature effect on one pipeline.

| metric | Exp08 | + wait graph | delta |
| --- | --- | --- | --- |
| composite | 0.769445 | **0.792622** | **+0.023176** |
| macro F1 | 0.789432 | 0.815868 | +0.026436 |
| robustness F1 | 0.736847 | 0.767013 | +0.030166 |
| hit@2 | 0.608611 | 0.632778 | +0.024167 |
| **deadlock F1** | 0.721739 | **0.836693** | **+0.114954** |

3/3 folds. No class fell.

### Against corrected (production) semantics

Control C reproduced Exp12's committed corrected numbers to 1e-12 **and** with
an identical sha256, so the candidate delta is again a feature effect on one
pipeline.

| metric | C (corrected) | D (+ wait graph) | delta |
| --- | --- | --- | --- |
| **composite** | 0.768487 | **0.789284** | **+0.020796** |
| macro F1 | 0.787379 | 0.812954 | +0.025576 |
| robustness F1 | 0.736842 | 0.759984 | +0.023142 |
| hit@2 | 0.609306 | 0.631528 | +0.022222 |
| **deadlock F1** | 0.714348 | **0.833605** | **+0.119258** |

3/3 folds on composite, macro, robustness, hit@2 **and** deadlock F1.

### The finding survives the mapping fix

| semantics | composite Δ | deadlock F1 Δ |
| --- | --- | --- |
| historical (defective) | +0.023176 | +0.114954 |
| **corrected (production)** | **+0.020796** | **+0.119258** |

The composite gain sheds ~10% and the deadlock gain slightly grows. Both clear
their gates under either semantics, so the production decision does not depend
on which OOF is consulted.

**Production confirmation gate: PASS 5/5.** Composite delta > 0; 3/3 folds;
deadlock F1 delta > 0; 3/3 deadlock folds; no class F1 drop > 0.02 (none fell).
**Original strict Exp13 gate: also PASS 6/6**, unchanged and not retuned.

### The gain is information, not a threshold trade

Under corrected semantics, `deadlock` precision **0.7444 -> 0.8281** and recall
**0.6867 -> 0.8392** both rise; tp 824 -> 1007, fp 283 -> 209, fn 376 -> 193.
The error transitions move the right way too:

| transition | C | D |
| --- | --- | --- |
| `deadlock -> clean` | 111 | **51** |
| `clean -> deadlock` | 59 | **24** |
| `dropped_handoff -> deadlock` | 95 | **66** |
| `runaway_loop -> deadlock` | 72 | **59** |

---

## Verification

| check | result |
| --- | --- |
| 10 inherited modules sha256 byte-identical to sub08 | PASS |
| `make_label_model` / `make_success_model` / `make_window_model` `get_params()` identical | PASS |
| the 300 production columns preserved, 12 appended after them | PASS |
| block numerically identical to the research module on real runs | PASS |
| all 6 template families fire on real data | PASS |
| block invariant under target perturbation | PASS |
| no non-finite values | PASS |
| success head still 185 features (not widened) | PASS |
| fallback on an incomplete schema | PASS |
| validator, 4000 rows, run_id order matches `sample_submission.csv` | PASS |

### End-to-end 10k train -> 4k test, each submission a separate process

| | sub08 | sub13 |
| --- | --- | --- |
| runtime | 600.8 s | **375.0 s** |
| success predictions | - | **byte-identical, 4000/4000** |
| label changed | - | 322 / 4000 (8.05%) |
| deadlock predictions | 427 | **496 (+69)** |
| `fault_turn` changed | - | 317 |
| `fault_turn` moved **without** a label change | - | **0** |

Largest transitions, all moving into or out of `deadlock`:

```
clean           -> deadlock        57      deadlock -> clean             17
dropped_handoff -> deadlock        27      deadlock -> dropped_handoff  22
runaway_loop    -> deadlock        24      deadlock -> runaway_loop     18
```

Every one of the 317 changed `fault_turn` values coincides with a changed
predicted label - they are the L1 peak being re-read for the new class, not a
change in L1 behaviour. `clean` rows carry `-1` in both submissions.

**Runtime** 375.0 s (6.2 min) on the full 10k -> 4k test, CPU-only, inside the
30-minute job limit. The 12 deterministic features cost no measurable runtime.

---

## Caveats that travel with this submission

1. **The public leaderboard is the real external check.** Both validation
   measurements are out-of-fold on the same 10000 train runs.
2. **`fault_turn` inherits Exp08's post-hoc caveat.** Exp08's L1 rule was
   discovered on Exp07's OOF, so hit@2 0.6086 is an upper bound on that gain and
   not an estimate of it. Exp13 does not make that number more or less honest; it
   only changes which class the peak is read for.
3. **Success F1 is pinned, not measured.** No `wg_*` feature targets `success`
   and the head is unchanged, so it stays at the validated 0.8644 by
   construction.
4. **The 322 changed test labels are reported, not scored** - `test.csv` carries
   no labels here.

---

## Reproduce

```bash
python check_parity.py                       # module/param/schema/leakage checks
python solution.py --train ../data/train.csv \
                   --test  ../data/test.csv  --output predictions.csv
python ../scripts/validate_submission.py --pred predictions.csv --test ../data/test.csv
```
</content>