status: complete
covers: [exp16, exp18, exp19]
hypotheses: Exp16 tested frozen MiniLM event-sequence probabilities fused 50:50 with Exp15 structure; Exp18 tested one schema-rate training-only telemetry-loss augmentation; Exp19 tested a fixed StepFinder-inspired sequence localizer replacing only Exp13 L1.
transfer: Exp16 gained +0.004751 composite but missed +0.005 and concentrated gains in fold 2; Exp18 gained +0.001559 against +0.008 with a slight hit@2 loss; Exp19 gained +0.001944 composite and +0.019444 hit@2 but failed both thresholds and sharply harmed two classes.
headroom: The strongest remaining measured signal is relational: Exp13's direct wait graph added +0.020796 composite and +0.119258 deadlock F1 under corrected semantics; the current 312-feature baseline includes those 12 wait summaries but not a unified temporal agent-task graph.
noise: Reblending semantic/structural probabilities, changing augmentation rates, tuning the localizer, or selecting per-class routes from these same OOF folds is not supported; the hybrid's +0.007875 composite is retrospective and not a promotion result.
warrant: Warrant one pre-registered sparse, typed, time-ordered agent-task interaction graph experiment beyond the existing wait summaries; first require a measured CPU train-plus-test estimate below 1800 seconds, and treat the existing three-fold OOF as internal evidence only.

---

# Root-Cause Review — Exp16 / Exp18 / Exp19

**Current production baseline:** Exp15, Composite `0.7938624419`. The Exp15 ZIP and production code remain unchanged. This review uses only committed reports, machine-readable results, and saved OOF artifacts. No model, embedding, or parameter search was run.

## Hypotheses and observed transfer

| Run | Fixed hypothesis | Result against Exp15 | Root-cause reading |
|---|---|---|---|
| Exp16 | Frozen MiniLM event sequence adds shifted-vocabulary information; fixed 50:50 fusion with Exp13 structure. | Composite `0.7986136119` (`+0.0047511700`, gate `+0.005`); Macro `+0.003238`, Robustness `+0.012863`, hit@2 `-0.000833`; 2/3 fold wins. | Composite missed by `0.00024883`; fold deltas were `-0.003089`, `+0.002232`, `+0.015332`, so fold 2 supplied most of the lift. Semantic-only Composite was `0.638624`. This is a weak, fold-sensitive semantic channel, not evidence that MiniLM has no signal. |
| Exp18 | One training-only view corrupted at official telemetry-loss rates improves Robustness, especially `dropped_handoff`. | Composite `0.7954212064` (`+0.0015587645`, gate `+0.008`); Macro `+0.001045`, Robustness `+0.004423`, hit@2 `-0.000694`; 2/3 fold wins. | The augmentation changed 563 labels but transferred only a small net composite gain. It improved `dropped_handoff` F1 by `+0.010076`, while `goal_drift` fell `0.008087`. Simulating missing telemetry alone did not close the target gap. |
| Exp19 | A fixed full-sequence StepFinder-inspired scorer improves the primary fault turn over Exp13 L1. | Composite `0.7958068863` (`+0.0019444444`, gate `+0.005`); hit@2 `0.6509722222` (`+0.0194444444`, gate `+0.05`); 3/3 folds improved. | Localizer signal is real in each fold but too small overall; class hit@2 fell `0.280000` on `dropped_handoff` and `0.075833` on `duplicated_work`, violating the `0.05` cap. A single scorer does not transfer uniformly across fault types. |

The three STOPs are not evidence of one implementation defect: baseline reproduction and frozen gates passed, and each run measured its declared candidate. They show that (a) fixed semantic fusion did not clear its narrow gate, (b) one synthetic corruption policy had limited transfer, and (c) one shared localizer had substantial class-specific trade-offs.

## Hybrid result — diagnostic only

The fixed rule uses only the frozen Exp15 **predicted** class: Exp15 L1 for `dropped_handoff` / `duplicated_work`, Exp19 for `deadlock` / `conflict` / `goal_drift` / `runaway_loop`, and `-1` for `clean`. Official and independent row-wise hit@2 both equal `0.7102777778`; Composite is `0.8017374419`, a `+0.0078750000` delta. Macro, Robustness, and Success are unchanged.

A leave-one-fold-out routing diagnostic selected the same four Exp19 classes on all three splits and evaluated positively on each held-out fold. However, the hybrid idea and class map were proposed after the full OOF results had been seen. The leave-one-fold-out calculation cannot undo that hindsight. Treat this as a retrospective stability diagnostic, **not** PROMOTE and not independent validation. The only clean-specific correction was in the diagnostic vector: all 2,981 predicted-clean Exp15 turns were set to `-1`; source OOF and production artifacts were not changed. The official hit@2 is unchanged because clean targets are excluded and faulty runs predicted clean already count as class errors.

## Does the evidence warrant a new representation?

It does **not prove** that a graph is necessary. Exp16–Exp19 reject their specific fixed candidates, not every semantic or relational representation. But further blending, corruption-rate changes, or per-class tuning on the same OOF outputs has no valid warrant: the hybrid itself is post hoc.

There is separate positive evidence for relational representation. Under corrected production semantics, the existing Exp13 directed wait-dependency block improved Composite from `0.7684867329` to `0.7892835288` (`+0.020796`), deadlock F1 by `+0.119258`, and Robustness by `+0.023142`; no class F1 fell. That 12-column wait graph is already part of Exp15's 312-feature label channel. Therefore repeating the wait graph is not a new experiment. The ontology and schema also expose task assignment/handoff, waits, artifact delivery, and shared-state writes that a broader graph could link explicitly.

**One warranted next direction:** a sparse, typed, time-ordered graph of agent-role ↔ subtask interactions, with observable handoff/wait/result edges and artifact/state-key links. It must add relations beyond the existing `wg_*` summaries and preserve event time. This is a representation-level hypothesis, not a claim of expected score gain. Do not combine it with a hybrid retune or a new sequence encoder.

## Runtime and validation constraints

- The Exp15 production ZIP measured `593.4 s` for 10,000 training and 4,000 test runs on CPU, leaving `1,206.6 s` of the 1,800 s budget in that measured environment.
- Exp16's MiniLM encoder alone took `4,822.9 s` for 590,146 training events on CPU. This rules out the measured CPU path for Exp15 + MiniLM + StepFinder. No GPU/A100 end-to-end timing exists; do not infer one.
- Exp19's three-fold CV took `2,346.93 s` and peak RSS `1,067 MB`; those research-CV numbers are not a production single-fit estimate. No Exp19/Exp16 offline production bundle was built.
- A sparse graph avoids the measured MiniLM encoding cost, but its runtime is unmeasured. Before any CV, benchmark the full intended training/inference path against the 1,800 s limit; if it fails, stop. No A100 speed claim can substitute for a measurement.
- The saved 3-fold OOFs are internal validation on the same 10,000 training runs. The hidden test shift labels are unavailable; Robustness F1 is a train-side proxy. There is no independent external confirmation of the hybrid or a graph candidate.

## Protocol and remaining blockers

This review covers the exact current STOP tail. No Exp20 is created here. Before a future graph experiment, freeze its input representation, folds, baseline reproduction checks, runtime gate, and promotion gate; run only after the protocol guard accepts this review. Production remains blocked until an accepted result and an offline runtime fit the platform limit. External generalization remains unverified without independent evaluation data.

After updating this review, `.venv/Scripts/python.exe experiments/protocol_guard.py` returned exit code `0`: current best `exp15`, STOP tail `exp16, exp18, exp19`, review present/current, and the guard says a new experiment may be started. This is protocol permission only; it is not authorization to create or run Exp20 here.

The protocol guard currently reads `results.json` only from `promotion_gate`; Exp16, Exp18, and Exp19 store their gate under `gate`. Their historical CSV verdicts are STOP, so this does not change the current tail, but the guard does not independently cross-check those JSON gates.

## Archived prior review

The following Exp09–Exp12 review is retained for historical context only; the machine-readable header at the top of this file covers Exp16, Exp18, and Exp19.

# Root-Cause Review — Exp09 / Exp10 / Exp11 / Exp12

**Why this file exists.** `experiments/protocol_guard.py` reports an unbroken tail of
four STOPS and refuses to let a fifth experiment be invented until the stall is
reviewed on paper. Current best is still `exp08` (`B_plus_window`, composite
`0.7694453917139285`); exp09, exp10, exp11 and exp12 all failed their own
pre-declared promotion gates and none produced a build.

This revision supersedes the previous one, which covered `[exp09, exp10, exp11]`
and was correctly rejected by the guard as stale. Every number below is quoted
from the run's own `results.json`. No model was fitted and no metric recomputed
to produce this document.

## 0. The previous review's prediction is now confirmed

The previous revision ended with a conditional:

> If this single check also fails to promote, the correct conclusion is that the
> remaining gap is not reachable from this representation at this evaluation
> resolution, and that further hillclimbing on decision-layer parameters should
> stop rather than continue in a new form.

**Exp12 did not promote. That prediction is confirmed.** The stall is no longer
an open question about whether to keep hillclimbing — the hillclimb is over. The
remaining question, addressed in §5 and §6, is whether anything else is warranted
at all, and the answer is narrower than "try the next thing": only a change of
representation qualifies.

## Summary of the research finding

Four experiments, and they fall into **two structurally different groups**. Keeping
them apart is the whole point of this revision.

| run | what varied | free params | held-out composite delta | folds won | gate |
|---|---|---|---|---|---|
| exp09 | class offsets in log-space | 6 | **-0.000206** | 1/3 | +0.003 needed — FAIL |
| exp10 | A/B blend weight `alpha` | 1 | **-0.000900** | 0/3 | +0.002 needed — FAIL |
| exp11 | seed set, fixed a priori | 0 | **+0.000157** | 1/3 | +0.002 needed — FAIL |
| exp12 | global/local run-index mapping | 0 | **-0.000959** | 1/3 | +0.002 needed — FAIL |

**Group A (exp09–11)** varied how the existing representation becomes a label.
None added information about a run. All three found real in-sample gains and lost
them on held-out rows.

**Group B (exp12)** was a correctness change to the training pipeline itself. It
was not another post-processing experiment and is not treated as one. Its defect
was real, its fix was surgical and verified, and it still lost on the objective.
That is a genuinely different finding, and §3 treats it separately.

## 1. `hypotheses` — what was actually tested

### Group A — decision/variance changes over an existing representation

**Exp09 — decision-layer class calibration.** Six additive offsets in
log-probability space over the sealed Exp07 OOF probabilities, `delta_clean`
pinned to 0, found by deterministic coordinate descent on an integer grid
(`bound +/-0.40`, steps `0.20/0.10/0.05/0.025/0.0125`), scored by the official
`evaluation.metrics.composite()`. 211 of 10000 labels changed. *Can a pure
decision shift, with no new model and no new feature, raise the composite?*

**Exp10 — probability blend of the Exp07 A and B label heads.**
`P_blend = (1 - alpha) * P_A + alpha * P_B` over a frozen 21-point grid
`0.00 ... 1.00`, one scalar, no class-specific alpha. `alpha = 1` is exactly
Exp08. 77 labels changed. *Does the 249-feature foundation head carry
complementary probability information to the shipped 300-feature head?*

**Exp11 — fixed 3-seed ensemble of the final label head.** Unweighted arithmetic
mean of `predict_proba` from three LightGBM label heads differing only in
`random_state` (42, 137, 2026), fitted on identical outer-train rows against an
identical 300-feature matrix. Zero free parameters, nothing searched, so no
selection bias. 201 labels changed. *Is the incumbent score a function of the
representation, or of one arbitrary seed?*

**The common structure.** Every one of the three varied *how the existing
representation is turned into a label*. None varied the representation itself,
and none introduced any new observation of a run. That is the fact that explains
the outcome in §2.

### Group B — representation/pipeline correctness

**Exp12 — fix the global/local run-index mapping in Exp07's honest inner
cross-fit.** `runner.py:229` calls `predict_runs(P, rw[...], len(tr_runs))`;
`predict_runs` iterates `for r in range(n_runs)` — a *positional* range — over a
searchsorted built from *global* run ids spanning `0..9999`. With
`n_runs ~= 6666`, every outer-train run whose global id is `>= len(tr_runs)`
got no block and kept all 51 window aggregates at their `np.zeros`
initialisation. Exp12 changed **only the mapping**: global ids were mapped to
local outer-train positions before grouping, with a local-coordinate `slot` so
`aggregate_block` resolves rows in the same space. Probabilities, window model,
targets, aggregate formulas, folds, seeds, label parameters, success head and the
sealed L1 peak matrix were all untouched. Zero free parameters, no search space,
no tuning.

This is **not** a fifth post-processing experiment and must not be read as one.
It is a test of whether a genuine train/inference inconsistency in the pipeline
was costing score.

## 2. `transfer` — Group A: why the decision-layer changes did not transfer

### Exp09 — offsets that reverse when the sample changes

Five of the six fault offsets change sign depending on which two thirds of the
data the fit was allowed to see (`results.json` records `inconsistent_classes` as
exactly those five):

| class | fold 0 | fold 1 | fold 2 | |
|---|---|---|---|---|
| dropped_handoff | +0.400 | +0.400 | +0.400 | consistent (but pinned at the `+0.40` bound) |
| duplicated_work | -0.150 | -0.375 | **+0.375** | sign flip |
| deadlock | -0.050 | 0.000 | -0.363 | sign flip |
| conflict | 0.000 | **+0.400** | -0.100 | sign flip |
| goal_drift | +0.100 | +0.150 | -0.113 | sign flip |
| runaway_loop | +0.050 | -0.100 | -0.300 | sign flip |

The single stable offset is stable at the *boundary of the search box* in all
three folds, which is what an optimiser does when it is exploiting rather than
measuring. Each search gained `+0.0038` to `+0.0059` composite on the rows it
was fitted on and lost on the rows it had not seen. Held-out delta **-0.000206**.

### Exp10 — a blend weight that will not sit still

The three folds selected `alpha = 1.00, 0.65, 0.95`: three distinct values, spread
`0.35`, classified `NOTABLE INSTABILITY`. The two folds that moved off
`alpha = 1` bought `+0.001009` and `+0.000961` on their own selection rows and
then lost `-0.001644` and `-0.000929` held-out. **0/3 folds won.** Of 77 changed
labels, **35 became correct and 36 became wrong — net -1**. Held-out delta
**-0.000900**.

### Exp11 — an effect one order of magnitude below its own noise floor

| seed | composite | vs incumbent |
|---|---|---|
| 42 (incumbent) | 0.7694453917139285 | — |
| 137 | 0.7679980257218403 | **-0.001447** |
| 2026 | 0.7691431612496809 | -0.000302 |
| 3-seed mean | 0.7696021570323565 | **+0.000157** |

Seed-to-seed spread is `0.0014473659920882`; the ensemble's effect is
`0.00015676531842800934` — **0.11x the spread it is trying to resolve**. Fold
deltas `+0.004231`, `-0.002923`, `-0.001263`, **1/3 won**. At row level 201
labels changed: 86 wrong→right, 82 right→wrong, net +4.

### The Group A conclusion

Across exp09–11 the pattern is identical, and it is a statement about the
information available, not about the three methods:

1. A handful of parameters is fitted on ~6666 rows against a composite whose own
   fold-to-fold variation is of the same order as any effect they could produce.
2. Each fit finds a real in-sample optimum, because one always exists.
3. Each fit's optimum disagrees with the other folds', so it is not a property of
   the data-generating process.
4. Applied out of sample, each therefore loses, or wins by less than its own noise.

**The composite surface in this direction is flat to within the resolution of this
evaluation.** Refining any of these three search spaces would find fold noise more
reliably, not less — which is what each run forbade itself from doing, correctly.

## 3. `transfer` — Group B: Exp12 is a different kind of result

Exp12 must **not** be filed under "another noise-dominated adjustment". Its
finding is more specific and more useful than that.

**What was established, and it is all solid:**

* **The defect was real.** Not a suspected bug, not a stylistic complaint — 2182,
  2223 and 2261 outer-train run-instances per fold (32.7% / 33.3% / 33.9%) owned
  window rows and received no block, measured by *structural ownership* rather
  than by testing aggregates for non-zero values.
* **The fix drove mapping losses to zero** on all three folds, for every run that
  owns window rows.
* **The fix was surgical.** The number of outer-train rows whose 51 aggregates
  actually changed is `2182 / 2223 / 2261` — exactly the lost counts, not a
  superset. Foundation columns, validation aggregates, window targets, folds,
  seeds and label parameters were asserted identical before scoring.
* **Control A reproduced the incumbent exactly**: 5/5 metrics within 1e-12 with
  max diff `0.0`, and byte-identical probabilities
  (`sha256 edbaf8d766f97f28...`). The paired comparison is therefore one pipeline
  with a single changed line, not two pipelines.
* **Outer-validation features were untouched**, as they had to be — validation
  coverage was already complete, and any difference there would have meant the fix
  had reached beyond its scope.
* **Honest stacking survived the fix**: every outer-train run held out exactly
  once, each hold-out disjoint from its own fit, mapping bijective.

**And then the objective moved the wrong way:**

| | A (buggy) | B (fixed) | delta |
|---|---|---|---|
| Composite | 0.7694453917139285 | 0.7684867328598647 | **-0.000959** |
| Macro F1 | 0.7894323366835861 | 0.7873785254579829 | **-0.002054** |
| Robustness F1 | 0.7368470046834376 | 0.7368422139406114 | -0.0000048 |
| fault_turn hit@2 | 0.6086111111111111 | 0.6093055555555555 | **+0.000694** |

Fold deltas `-0.000531` / `-0.005513` / `+0.003832` — **1/3 won**, gate failed on
both delta and folds won.

**The correct reading, stated precisely.** The consistency defect was eliminated
and the composite still fell. It would be wrong to file this as "the fix failed
because of noise" — that repeats the exp09–11 error and loses the actual result.
The supported conclusion is narrower and firmer:

> **Technical correctness and competitive score are not the same quantity.** The
> zeroed window columns were not neutral damage that could be removed for free;
> they were part of the training distribution the label head was fitted against.
> Removing them is not adding information to a fixed model — it is retraining the
> label head on a different input distribution. The retrained head is, on these
> folds, the worse one.

The per-class pattern is consistent with that and not with a random perturbation:
`clean +0.001881` and `duplicated_work +0.006706` gained, while the two most
confusable fault classes lost — `dropped_handoff -0.006286` and
`deadlock -0.007391` — with `goal_drift -0.006062`. The unstable fold pattern
(`+0.0038` against `-0.0055`) is the same order of magnitude as the noise
documented in §2, and 1/3 wins means there is no reliable direction in it.

This finding is durable regardless of the score outcome: the pipeline was
inconsistent, the inconsistency is now measured and closed, and the honest
conclusion is that removing it did not buy accuracy.

## 4. The production finding

**Production Exp08 does not contain the defect. This was measured, not assumed.**

Exp12 probed production's own code before running anything. Production's
`build_window_matrix` assigns `run` ids indexing the runs it was *handed*, and
`aggregate_runs` aligns with `bounds = np.searchsorted(sr, want)` iterating
`want` itself. Its id space and its iteration space are the same space, so a
sparse or offset id set cannot lose rows.

| probe | production | Exp07's grouping, same inputs |
|---|---|---|
| real global ids `[9997, 9998, 9999]`, 10000-run space | **3/3 kept, 0 lost** | **0/3 kept** |
| synthetic sparse ids `[1, 7, 9]`, row for id 7 | equals a direct `aggregate()` | dropped |

Consequences, all of which follow from that one measurement:

* **The shipped Exp08 requires no bugfix.** Nothing in production is broken.
* **No new ZIP is warranted by Exp12**, and none was built. The production
  behaviour already corresponds to the corrected mapping.
* **Exp12 repaired the OOF / training-measurement path, not production
  inference.** These are different code paths and only the former was wrong.
* **Public Exp08 `0.7696` remains the factual external result of the existing
  production system.** It was never read by the search, the gate, or any
  parameter choice in exp12 — it is a reference only, as in exp09–11.

**What this does NOT mean.** It does not make the local OOF apparatus worthless,
and this review does not claim it does. What it establishes is a specific and
narrow mismatch:

> The historical OOF trainer and the production trainer had different window
> aggregation semantics. The local number `0.7694453917139285` was produced by a
> trainer that saw zeroed window features for a third of its training rows, while
> the artefact being scored was built by a correct one.

That number remains the correct **reproduction target** for every experiment
pinned to it — it is a well-defined, verified anchor, and exp12's control branch
reproduced it byte-for-byte. It is simply not, on its own, an unbiased estimate
of the shipped system's behaviour. Those are different jobs, and the number was
only ever asked to do the first one.

## 5. `headroom` — the previous headroom is exhausted

The previous revision of this review named exactly one concrete item of headroom:
the Exp07 global/local indexing defect, and the mapping fix as the one warranted
next experiment. **That item is now spent.**

Exp12 tested it. The defect was real, the fix was correct, the coverage question
is closed at 100%, and the promotion gate failed. This review therefore removes
that claim and does not replace it with "try another mapping fix" — the coverage
measurement is complete and there is nothing left to vary.

What the four results jointly establish about the current 300-feature
representation is that **no small adjustment to it has been found to work**:

* calibration — no (exp09)
* blending two heads — no (exp10)
* averaging across seeds — no (exp11)
* repairing the training pipeline for consistency — score did not improve (exp12)

The remaining measurable headroom is not in the decision layer at all. It is in
the recognition errors themselves:

* Macro F1 sits near `0.789`, and the error is concentrated, not spread. In the
  incumbent, `conflict` is at `0.903` and `clean` at `0.865`, while
  `dropped_handoff` is at `0.636` and `deadlock` at `0.722`.
* `dropped_handoff` is the weakest class and the one the project cares most
  about, and it is the class that moved the wrong way in exp09, exp10, exp11 and
  exp12 alike.
* The one time this project moved the composite substantially was **Exp07**,
  which went from `0.7455549124136548` (249 features, A) to
  `0.7516676139361507` (300 features, B) — **+0.006113** composite and
  `+0.005` hit@2 — and it did so by **adding a window-level evidence channel**,
  not by re-cutting an existing one.

That contrast is the load-bearing observation of this review. Every Group A
experiment and Exp12 itself operated on probabilities derived from a
representation that was already fixed, and none of them moved the objective by
even a third of what adding that one channel moved it. The evidence therefore
points at a single conclusion about where headroom can still be:

> **Further headroom, if it exists, must come from new information about the run —
> a new representation — not from another reworking of the same probabilities.**

This review deliberately stops there. It does not name a feature family, a model
class, an architecture, or an experiment number. Choosing among those is the work
of the next run, and a root-cause review that picks the model is a root-cause
review that has started designing Exp13.

## 6. `noise` — what is closed, and what is not

**Noise-dominated in this pipeline, on this data, on this representation.** Four
consecutive honest held-out checks found no transferable effect:

1. **Additive class offsets in log-probability space** (exp09). 5/6 offsets flip
   sign across folds; in-sample `+0.0038..+0.0059` becomes held-out `-0.000206`.
2. **Probability blending of the Exp07 A and B heads** (exp10). 0/3 folds won;
   selected `alpha` unstable across folds; 35 fixes against 36 breaks.
3. **Seed averaging and, a fortiori, seed selection** (exp11). Effect 0.11x the
   seed-to-seed spread; 86 right-ward against 82 left-ward moves.
4. **Further post-hoc recalibration of these same probability matrices** —
   temperature scaling, thresholding, per-class alphas, stacking on top of the
   same OOF predictions, finer grids, smaller or larger offset bounds — by the
   same argument and at greater cost. Each re-reads the same ~6666 rows with the
   same few degrees of freedom and would move the same small numbers.

The honest scope of this finding: **four consecutive held-out checks in the
current pipeline on the current data did not produce a transferable effect.**
That is not a claim that such methods are incapable of helping. It is a claim
that nothing is currently known about them that justifies a fifth experiment, and
that a fifth such experiment would be indistinguishable from sampling fold noise —
the specific failure mode the cross-fitted protocol exists to expose, and which it
has now exposed four times.

**Exp12's mapping fix is explicitly NOT in this class**, and must not be added to
it. It was a correctness hypothesis, it was honestly tested, and it was honestly
refuted **as a source of score improvement**. That is a different outcome from
"this direction is noise": the defect was real, is fixed, and the fix does not pay.
The instruction this carries forward is narrow and concrete:

* Do not repeat the mapping fix.
* Do not look for "another mapping fix" or a variant of it. Coverage after the
  correction is 100% on all three folds; that question is **closed**.

## 7. `warrant` — is another experiment justified?

**Yes — but the only thing that qualifies is a change of representation.**

A further experiment is warranted in principle, on one condition: it must change
the representation or add new semantic or structural information about a run. The
justification is the Exp07 contrast in §5 — the only substantial composite gain
in the project's history came from adding an evidence channel, and four
subsequent attempts to extract more from the representation that channel feeds
have all failed.

**There is no warrant for:**

* any further post-processing of the existing probabilities;
* calibration, in any form, on these probabilities;
* probability blending of A/B or of any other pair of fitted heads;
* seed averaging, seed selection, or additional seeds;
* another mapping fix or any variant of the one Exp12 closed;
* a small hyperparameter hillclimb on the same 300-feature representation;
* re-baselining the project onto the corrected OOF number. Exp12's gate just
  failed on precisely the question "is the corrected number better?", and the
  evidence argues against taking it.

The honest position is that this is a narrower warrant than the one the previous
revision gave, and that narrowing is the finding. The previous review authorised
exactly one more experiment on a specific, concrete, measured defect. That defect
is now closed, the experiment failed, and the authorisation it granted is spent.

**Choosing the representation hypothesis is not this review's job.** No
architecture, model class, feature family or experiment number is proposed here.
The precondition for the next run is a single independently-motivated
representation hypothesis, stated and justified on its own evidence — not a
continuation of the offset/alpha/seed family, and not a re-run of Exp12 with
different parameters.

If that hypothesis also fails to promote, the correct conclusion is that the
remaining gap is not reachable from run-level aggregates of this kind, and that
the search for it should stop rather than continue in another form.

## 8. Scope of this document

This review changed nothing else. It did not modify exp09, exp10, exp11 or exp12;
did not touch `PROTOCOL.md` or `protocol_guard.py`; did not change
`experiments/results.csv`; did not build or modify any submission; did not fit or
refit any model; and did not run any feature experiment. Every number is quoted
from the four runs' own `results.json`.
