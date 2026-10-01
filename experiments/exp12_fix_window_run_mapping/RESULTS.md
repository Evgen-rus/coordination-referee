# Exp12 - fix the global/local run-index mapping in the honest inner cross-fit

**Verdict: STOP. Promotion gate FAILED.** No production build, no ZIP, no second
mapping fix, no tuning, no Exp13.

One hypothesis, zero free parameters, no search space:

> Fixing the global/local run-index mapping in Exp07's honest inner cross-fit
> removes the train/inference mismatch and improves the official composite.

---

## 1. Headline

| metric | A = control, mapping WITH bug | B = mapping FIXED | delta |
|---|---|---|---|
| Macro F1 | 0.7894323366835861 | 0.7873785254579829 | **-0.00205** |
| Robustness F1 | 0.7368470046834376 | 0.7368422139406114 | -0.0000048 |
| Success F1 | 0.8643757406010988 | 0.8643757406010988 | 0 (pinned) |
| fault_turn hit@2 | 0.6086111111111111 | 0.6093055555555555 | **+0.00069** |
| **COMPOSITE** | **0.7694453917139285** | **0.7684867328598647** | **-0.00096** |

The gate needed **+0.002**. The measured delta is **-0.00096** — the wrong sign.
546 of 10000 OOF labels changed.

**The fix did what it said. It made the score worse.** Both halves of that
sentence are established below; the second one is the actual finding.

## 2. The production nuance, answered before the experiment

The brief asked whether `submission_exp08_window_localizer` shares the defect,
and said not to take the expected answer on faith. It was decided by executing
production's own code, not by reading it.

**Production does NOT contain the defect. CASE A.**

| probe | result |
|---|---|
| real window rows, global ids `[9997, 9998, 9999]`, 10000-run space | **3/3 kept, 0 lost** |
| synthetic sparse ids `[1, 7, 9]`, row for id 7 | equals a direct `aggregate()` call |
| same inputs through Exp07's grouping, `n_runs=len(want)` | **0/3 kept** |

The reason is structural. Production builds its window dataset with `run`
indexing the runs it was *handed* (`build_window_matrix`: `rid.append(np.full(F.shape[0], i))`),
and aligns with `bounds = np.searchsorted(sr, want)` iterating `want` itself. Its
id space and its iteration space are the same space, so a sparse or offset id
set cannot lose rows. Exp07's `predict_runs` instead iterates
`range(n_runs)` — a *positional* range — over a searchsorted built from
*global* ids. That mismatch is the whole defect.

**Consequence, and it is the important part of this run:** the shipped model was
never affected. The bug was in the *measurement apparatus*. Exp07's OOF
training matrix, and therefore every local score computed from it — Exp08's
0.7694, and the numbers Exp09/Exp10/Exp11 were measured against — came from a
trainer that was fed a third zeroed-out columns, while the thing being scored at
inference time was always fed complete ones.

## 3. Baseline policy and the reproduction gate

The candidate cannot reproduce Exp08, and is not supposed to. The gate therefore
runs against **control A** — the same pipeline with the bug present:

| metric | expected | A got | diff |
|---|---|---|---|
| Macro F1 | 0.7894323366835861 | 0.7894323366835861 | **0.0** |
| Robustness F1 | 0.7368470046834376 | 0.7368470046834376 | **0.0** |
| Success F1 | 0.8643757406010988 | 0.8643757406010988 | **0.0** |
| fault_turn hit@2 | 0.6086111111111111 | 0.6086111111111111 | **0.0** |
| Composite | 0.7694453917139285 | 0.7694453917139285 | **0.0** |

5/5 within 1e-12, and not merely within tolerance: control A's probabilities are
**byte-identical** to the sealed Exp07 B OOF artifact
(`edbaf8d766f97f28...` both sides). Candidate B hashes to
`b7e92bd4e14fa59c...`. Control A *is* the incumbent, so the paired comparison
below is a genuine A/B on one pipeline, not a comparison of two pipelines.

## 4. Coverage — measured structurally, not by "!= 0"

Losses are counted by **ownership**: a run is lost when it owns at least one
selected window row and no block is ever produced for it. A zero numeric vector
is not evidence — a run with no strong window legitimately aggregates to zeros —
so no assertion here is based on aggregate values.

| fold | outer-train runs | OLD lost | FIXED lost | per-inner OLD losses |
|---|---|---|---|---|
| 0 | 6666 | **2182** | **0** | 749 / 720 / 713 |
| 1 | 6667 | **2223** | **0** | 743 / 757 / 723 |
| 2 | 6667 | **2261** | **0** | 766 / 775 / 720 |

The OLD column matches the pre-registered expectation from the root-cause review
exactly. FIXED is 0/0/0.

**The fix is surgical.** The number of outer-*train* rows whose 51 aggregates
actually changed is `2182 / 2223 / 2261` — exactly the lost counts, not a
superset. The fix repairs precisely the rows the bug corrupted and touches
nothing else. Foundation columns, validation aggregates, window targets, folds,
seeds, label parameters, success head and the sealed L1 peak matrix are all
asserted identical before scoring.

## 5. Honest stacking, after the fix

The fix touches the honesty path, so it was re-proved rather than assumed:

* every outer-train run held out **exactly once** across the 3 inner folds;
* each inner fold's hold-out set **disjoint** from the set its window model was
  fitted on — no run scores itself;
* global -> local map **bijective** on `tr_runs` (asserted in construction);
* every held-out window row resolves to the local position of the run that owns
  it, or the run raises.

Each inner window model was fitted **once**; its single probability matrix was
grouped two ways. No second set of models was trained, so A and B differ only at
the grouping step.

## 6. Promotion gate — FAILED

| criterion | threshold | value | |
|---|---|---|---|
| composite delta | >= +0.002 | **-0.000959** | FAIL |
| folds won | >= 2/3 | **1/3** | FAIL |
| robustness drop | <= 0.003 | 0.0000048 | PASS |
| worst class F1 drop | <= 0.02 | 0.007391 (`deadlock`) | PASS |

| fold | A | B | delta | |
|---|---|---|---|---|
| 0 | 0.766170 | 0.765639 | -0.000531 | LOSS |
| 1 | 0.774745 | 0.769232 | **-0.005513** | LOSS |
| 2 | 0.766441 | 0.770273 | **+0.003832** | WIN |

## 7. Why it lost — the incumbent was calibrated to the broken trainer

Per-class F1 deltas show the shape, and it is not a uniform degradation:

| class | A | B | delta |
|---|---|---|---|
| clean | 0.865003 | 0.866885 | **+0.001881** |
| duplicated_work | 0.829642 | 0.836348 | **+0.006706** |
| runaway_loop | 0.723932 | 0.722861 | -0.001071 |
| conflict | 0.903175 | 0.901021 | -0.002154 |
| goal_drift | 0.846898 | 0.840836 | -0.006062 |
| dropped_handoff | 0.635637 | 0.629351 | **-0.006286** |
| deadlock | 0.721739 | 0.714348 | **-0.007391** |

Two classes gained (`clean`, `duplicated_work`), four lost. The two biggest
losses are the two most confusable fault classes, and `dropped_handoff` — the
class the project cares most about — lost in every configuration tested across
Exp09, Exp10, Exp11 and now Exp12.

The reading this supports: **the zeroed window columns were not neutral damage
to be removed for free — they were part of the training distribution the label
head was fitted against.** A third of its training rows carried an all-zero
51-column block; the model learned to treat "all window aggregates zero" as a
recognisable regime rather than as missing data, and spent capacity accordingly.
Remove the regime and the input distribution it was fitted to changes, and so
does every weight. That is why the correction is not free: it is not adding
information to a fixed model, it is **retraining the label head on a different
distribution and measuring whether the new model is better**. On this data, on
these folds, it is not.

This is also why the fold pattern is unstable rather than uniformly negative —
`+0.003832` on fold 2 against `-0.005513` on fold 1. A retraining-induced shift
of this size is the same order as the noise documented in the root-cause review,
and 1/3 wins says there is no reliable direction in it.

## 8. What this licenses, and what it does not

**It does not** say the bug was harmless, that the current 0.7694 is an honest
estimate of the shipped system, or that the fix should be reverted in the OOF
pipeline. Two facts stand on their own:

1. **Production was never broken** (§2). The shipped model's behaviour is
   unchanged by anything found here, and no new ZIP is warranted.
2. **The OOF estimate and the production model were measuring different things.**
   The local number 0.7694453917 was produced by a trainer that saw zeroed
   window features for a third of its rows, while the artefact being scored was
   built by a correct one. The number is still the right *reproduction target*
   for every experiment measured against it — it is simply not, on its own, an
   unbiased estimate of the shipped system.

**It does not** license fixing the OOF pipeline and re-baselining on the fixed
number. That would replace a 0.7694 that every experiment has been pinned to
with a different number, without evidence that the new one is better — and the
gate just failed on exactly that question. Re-baselining is a separate decision
for a separate run, and the evidence here argues *against* taking it.

## 9. Not done, deliberately

No second mapping fix, no variant mapping, no threshold or blend, no retuning,
no re-baselining, no Exp13, no production build, no ZIP, no upload. Public
Exp08 `0.7696` was read as a reference only and never entered the gate, a
parameter, or a choice between implementations — the hypothesis has no search
space to search.

## 10. Files and reproduce

| file | contents |
|---|---|
| `mapping_fix.py` | the fix, coverage accounting, production probe, honest-stacking verifier, paired run |
| `run_experiment.py` | production check, paired run, reproduction gate, official scoring, gate |
| `test_mapping_fix.py` | 21/21, incl. the defect reproduced on synthetic ids `[1, 7, 9]` |
| `results.json` | every number above, machine-readable |
| `run.log` | execution log |

```
python experiments/exp12_fix_window_run_mapping/run_experiment.py    # 618.8 s
python experiments/exp12_fix_window_run_mapping/test_mapping_fix.py  # 21/21
```

**One implementation note worth recording.** The first working version of the fix
fed local ids into `aggregate_block` while its `slot` was still keyed by global
ids, so blocks whose local id was not also a global id were dropped through
`if r not in slot: continue` — recovering only ~33% of runs while *looking*
complete. It was caught because the outer-train row count that changed
(`4484`) exceeded the number of rows the bug had actually corrupted (`2182`).
The fixed branch now builds `slot` in local coordinates
(`{k: k for k in range(n_tr_runs)}`), after which the changed-row count equals
the lost-run count exactly. This is the same *shape* of bug as the original
defect, one layer down: mixing index spaces and having the mismatch absorb rows
silently. The count check that caught it is now a standing assertion.
