status: complete
covers: [exp09, exp10, exp11]
hypotheses: exp09 = 6 additive class offsets in log-probability space over Exp07 OOF probabilities; exp10 = single-scalar probability blend of the two Exp07 label heads A and B; exp11 = fixed unweighted 3-seed probability ensemble (42, 137, 2026) of the final B_plus_window label head - all three are decision-layer changes over one already-fitted representation, and none adds information about a run.
transfer: all three moved in-sample gains onto held-out rows with the wrong sign or no sign at all - exp09 -0.000206 (1/3 folds), exp10 -0.000900 (0/3), exp11 +0.000157 (1/3) - because each fitted 1-7 free numbers on ~6666 rows and then applied them to ~3334 unseen rows, so what it fit was fold noise, not a transferable correction.
headroom: one concrete, measured item - the incumbent Exp07 honest stacking trains system B with 51 window aggregates that are identically ZERO for ~33% of outer-train runs due to a global-vs-local run-index mismatch, so this is a train/inference feature-distribution mismatch, i.e. a correctness defect worth exactly one controlled experiment, with no claim that fixing it must raise the composite.
noise: additive class offsets, A/B probability blending, seed averaging and seed selection, and further fine post-hoc recalibration of the same already-fitted probability matrices - three consecutive honest held-out checks in this pipeline on this data showed no transferable effect; that is a statement about these three tests, not a claim that such methods can never work.
warrant: YES - a next experiment is warranted, but only as a single separate controlled check of the Exp07 window-aggregation correctness fix, on one hypothesis, and not as another offset/alpha/seed search.

---

# Root-Cause Review — Exp09 / Exp10 / Exp11

**Why this file exists.** `experiments/protocol_guard.py` reports an unbroken tail of
three STOPS and refuses to let a fourth experiment be invented until the stall is
reviewed on paper. Current best is still `exp08`
(`B_plus_window`, composite `0.7694453917139285`); exp09, exp10 and exp11 all failed
their own pre-declared promotion gates and all produced no build.

Every number below is quoted from the run's own `results.json`. Nothing here is
recomputed, and no new model was fitted to produce this document.

## Summary of the research finding

All three experiments were **the same kind of experiment three times**. Each took the
probabilities that Exp07 had already produced and re-cut the decision: move class
scores by a learned amount, mix two probability matrices by a learned weight, average
three models that differ only by seed. None of them looked at a run and learned
anything the existing representation did not already contain.

All three failed in the same way, and the failure is the informative part: each found a
real in-sample gain, and each lost that gain on held-out rows.

| run | what varied | free parameters | held-out composite delta | folds won | gate |
|---|---|---|---|---|---|
| exp09 | class offsets in log-space | 6 (of 7) | **-0.000206** | 1/3 | +0.003 needed — FAIL |
| exp10 | A/B blend weight `alpha` | 1 | **-0.000900** | 0/3 | +0.002 needed — FAIL |
| exp11 | seed set, fixed a priori | 0 | **+0.000157** | 1/3 | +0.002 needed — FAIL |

The three deltas span `0.001057` — about 5x the largest promotion gate in the set.
Two are negative. The conclusion this forces is not "the right offset/alpha/seed has
not been found yet"; it is that **this class of change is noise-dominated around the
incumbent**, and the size of the observed effects is smaller than the size of the
noise being moved.

## 1. `hypotheses` — what was actually tested

**Exp09 — decision-layer class calibration.** Six additive offsets in log-probability
space over the sealed Exp07 OOF probabilities, `delta_clean` pinned to 0, found by
deterministic coordinate descent on an integer grid (`bound +/-0.40`, steps
`0.20/0.10/0.05/0.025/0.0125`), scored by the official `evaluation.metrics.composite()`.
211 of 10000 labels changed. The question was: *can a pure decision shift, with no new
model and no new feature, raise the composite?*

**Exp10 — probability blend of the Exp07 A and B label heads.**
`P_blend = (1 - alpha) * P_A + alpha * P_B` over a frozen 21-point grid
`0.00 ... 1.00`, one scalar, no class-specific alpha. `alpha = 1` is exactly Exp08. 77
labels changed. The question was: *does the 249-feature foundation head carry
complementary probability information to the shipped 300-feature head?*

**Exp11 — fixed 3-seed ensemble of the final label head.** Unweighted arithmetic mean
of `predict_proba` from three LightGBM label heads differing only in `random_state`
(42, 137, 2026), fitted on identical outer-train rows against an identical 300-feature
matrix. Zero free parameters — nothing was searched, so no selection bias. 201 labels
changed. The question was: *is the incumbent score a function of the representation, or
of one arbitrary seed?*

**The common structure.** Every one of the three varied *how the existing
representation is turned into a label*. None varied the representation itself, and none
introduced any new observation of a run. That is what the three have in common, and it
is the single fact that explains the outcome in §2.

## 2. `transfer` — why none of it reached held-out rows

### Exp09 — offsets that reverse when the sample changes

The fitted offsets do not agree with each other across the three independent
calibration-train fits. Five of the six fault offsets change sign depending on which
two thirds of the data the fit was allowed to see:

| class | fold 0 | fold 1 | fold 2 | |
|---|---|---|---|---|
| dropped_handoff | +0.400 | +0.400 | +0.400 | consistent (but pinned at the `+0.40` bound) |
| duplicated_work | -0.150 | -0.375 | **+0.375** | sign flip |
| deadlock | -0.050 | 0.000 | -0.363 | sign flip |
| conflict | 0.000 | **+0.400** | -0.100 | sign flip |
| goal_drift | +0.100 | +0.150 | -0.113 | sign flip |
| runaway_loop | +0.050 | -0.100 | -0.300 | sign flip |

`results.json` records `inconsistent_classes` as exactly those five. The single stable
offset is stable at the *boundary of the search box* in all three folds, which is what
an optimiser does when it is exploiting rather than measuring.

Each search gained substantially on the rows it was fitted on — fold 0
`0.771034 -> 0.774835`, fold 1 `0.766861 -> 0.770772`, fold 2
`0.770275 -> 0.776222`, i.e. `+0.0038` to `+0.0059` — and each lost on the rows it had
not seen (`+0.00061`, `-0.00015`, `-0.00141`). Held-out delta -0.000206 overall. The
search worked exactly as advertised; the advertised target was the problem.

### Exp10 — a blend weight that will not sit still

The three folds selected `alpha = 1.00, 0.65, 0.95`: three distinct values, spread
`0.35`, which the run classifies as `NOTABLE INSTABILITY`. Fold 0 chose the identity
(a tie, not a win). The two folds that moved off `alpha = 1` bought
`+0.001009` and `+0.000961` composite **on their own selection rows** and then lost
`-0.001644` and `-0.000929` on their held-out rows. **0/3 folds won.**

The label-level accounting is the cleanest statement of the whole stall: of 77 changed
labels, **35 became correct and 36 became wrong — net -1**. The blend is not a worse
model; it is a coin flip that happened to land one row down. `dropped_handoff`, the
class the project cares most about, is the worst-hit and moved the wrong way
(`-0.003361`).

### Exp11 — an effect one order of magnitude below its own noise floor

The ensemble is the strongest of the three, and it still fails. Its individual-seed
diagnostic is the key number:

| seed | composite | vs incumbent |
|---|---|---|
| 42 (incumbent) | 0.7694453917139285 | — |
| 137 | 0.7679980257218403 | **-0.001447** |
| 2026 | 0.7691431612496809 | -0.000302 |
| 3-seed mean | 0.7696021570323565 | **+0.000157** |

Seed-to-seed spread is `0.0014473659920882`. The ensemble's effect is
`0.00015676531842800934`. **The effect is 0.11x the spread it is trying to resolve** —
roughly an order of magnitude too small to be read off this data. Folds agreed with
each other on nothing: `+0.004231`, `-0.002923`, `-0.001263`, **1/3 won**.

At row level, 201 labels changed: **86 wrong -> right, 82 right -> wrong, net +4**, plus
33 wrong -> a *different* wrong class. The ensemble is doing exactly what variance
averaging is supposed to do — nudging borderline rows — and the nudges are, in
aggregate, indistinguishable from random.

### The common finding

Across the three runs the pattern is identical, and it is a statement about the
*information available*, not about the three methods:

1. A handful of parameters is fitted on ~6666 rows against a composite whose own
   fold-to-fold variation is of the same order as any effect they could produce.
2. Each fit finds a real in-sample optimum, because one always exists.
3. Each fit's optimum disagrees with the other folds', so it is not a property of the
   data-generating process.
4. Applied out of sample, each therefore loses, or wins by less than its own noise.

The composite surface in this direction is flat to within the resolution of this
evaluation. Widening or refining any of these three search spaces would not add
evidence; it would find fold noise more reliably, which is exactly what each run
forbade itself from doing and was right to forbid.

## 3. `headroom` — the one measured, actionable item

Exp11 surfaced something the other two could not have: while rebuilding the 300-feature
matrix to obtain per-seed OOF probabilities, it had to reproduce Exp07's honest
stacking pipeline exactly — and that pipeline contains an indexing defect.

### The defect, in the code

`experiments/exp07_fault_windows/grouping.py`:

```python
def predict_runs(P: np.ndarray, run_rows: np.ndarray, n_runs: int):
    """Split a flat (n_rows, 7) probability matrix into per-run blocks."""
    order = np.argsort(run_rows, kind="stable")
    sr = run_rows[order]
    bounds = np.searchsorted(sr, np.arange(n_runs + 1))
    for r in range(n_runs):          # <-- iterates LOCAL ids 0 .. n_runs-1
        a, b = int(bounds[r]), int(bounds[r + 1])
        ...
```

The bounds are built over the **global** run-id range, and the caller passes the
**global** ids in `run_rows` (which span `0..9999`). But `runner.py` calls it with two
different `n_runs` for the two different populations:

```226:246:experiments/exp07_fault_windows/runner.py
        Xagg_tr = np.zeros((len(tr), ag.N_AGG))
        slot = {int(r): k for k, r in enumerate(tr_runs)}
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            P = probs[(f, j)]
            blocks = grp.predict_runs(P, rw[np.isin(rw, hold)], len(tr_runs))   # L229
            sub, _, _ = grp.aggregate_block(blocks, ag, slot, len(tr_runs))
            Xagg_tr[assign == j] = sub[assign == j]
        ...
        blocks = grp.predict_runs(Pva, rw[np.isin(rw, va)], n)                  # L244
        slot_va = {int(r): k for k, r in enumerate(va)}
        Xagg_va, pk, pp = grp.aggregate_block(blocks, ag, slot_va, len(va))
```

- **Line 229 (outer-train):** global ids up to `9999`, but the loop only covers
  `0 .. len(tr_runs)-1` (about `6666`). Every outer-train run whose *global* id is
  `>= len(tr_runs)` produces no block, `aggregate_block` leaves its row at the
  `np.zeros` initialisation, and those rows enter system B's feature matrix with
  **all 51 window aggregate features identically zero**.
- **Line 244 (outer-validation):** called with `n`, the full `10000`, so coverage is
  complete.

The 249 foundation features are untouched. The corruption is confined to the 51 window
aggregate columns of system `B_plus_window` — which is the incumbent system.

### The magnitude, measured

Counted directly from the fold definitions and the cached window dataset (no model
fitted):

| fold | outer-train runs | runs with global id >= len(tr_runs) | share |
|---|---|---|---|
| 0 | 6666 | **2182** | **32.7%** |
| 1 | 6667 | 2223 | 33.3% |
| 2 | 6667 | 2261 | 33.9% |
| **overall** | **20000** | **6666** | **33.33%** |

Within each fold the inner cross-fit partitions outer-train into three hold-out sets,
and each set shows the same rate (fold 0: 33.7%, 32.4%, 32.1%). Outer-validation rows
are reachable in 100% of cases in every fold.

So the incumbent was trained such that roughly **one in three of its training runs
looked, in its 51 window columns, like a run with no window evidence at all**, while at
inference time every run it is asked about carries real window aggregates. That is a
train/inference feature-distribution mismatch on a third of the training set, and the
model was fitted across it.

### Why this is headroom, stated carefully

It qualifies as headroom on two independent grounds, and neither requires assuming a
gain:

1. **It is a correctness defect, not a modelling choice.** No defensible protocol
   requires a model to be trained on silently truncated inputs while serving complete
   ones. Fixing it is justified on its own terms, regardless of the score.
2. **It is measurable and unexploited.** The size is known exactly (33.33% of
   outer-train rows, 51 features, 3 folds), it is currently *absorbed* into the
   incumbent rather than tested, and it has never been subject to a held-out check.

**What is explicitly not claimed:** that fixing it will improve the composite. The
effect could be positive, negative, or nil. A model that has learned to work around a
third of its input being zeroed may be well calibrated to that regime; removing the
mismatch can plausibly *cost* score. The honest statement is that the fix changes the
model in a way no held-out data has yet constrained, which is precisely what makes it
worth one controlled experiment rather than an assumption.

One methodological consequence must be declared by whoever runs it, and is flagged
here so it is not discovered late: **a fix is not a meta-CV layer.** Exp09–Exp11 all
reproduced the incumbent exactly and then searched over its artifacts, so the
reproduction gate applied cleanly. A change to the inner cross-fit necessarily
changes the fitted model, so it cannot both reproduce the incumbent to `1e-12` and be
the experiment. Which baseline policy applies is a decision the next run has to make
and state up front — it cannot inherit exp08's reproduction targets by default.

## 4. `noise` — directions already measured as noise-dominated

On this pipeline, on this data, these three consecutive honest held-out checks found no
transferable effect:

1. **Additive class offsets in log-probability space** (exp09). 5/6 offsets flip sign
   across folds; in-sample `+0.0038..+0.0059` becomes held-out `-0.000206`.
2. **Probability blending of the Exp07 A and B heads** (exp10). 0/3 folds won;
   selected `alpha` unstable across folds; 35 fixes against 36 breaks.
3. **Seed averaging and, a fortiori, seed selection** (exp11). Effect 0.11x the
   seed-to-seed spread; 86 right-ward against 82 left-ward moves.
4. **Further fine post-hoc recalibration of these same probability matrices** —
   temperature scaling, thresholding, finer grids, per-class alphas, stacking on top of
   the same OOF predictions — by the same argument and at greater cost. Each of these
   re-reads the same ~6666 rows with the same few degrees of freedom and would move the
   same small numbers.

The honest scope of this finding: **three consecutive held-out checks in the current
pipeline on the current data did not produce a transferable effect.** That is not a
claim that these methods are incapable of helping. It is a claim that nothing is
currently known about them that justifies a fourth experiment, and that a fourth such
experiment would be indistinguishable from sampling fold noise — which is the specific
failure mode the cross-fitted protocol exists to expose, and which it has now exposed
three times.

## 5. `warrant` — is a next experiment justified?

**Yes — but only one, and it is not another one of these three.**

The stall has a real cause and it was located: the incumbent pipeline is
train/inference inconsistent on ~33% of its training rows. That is a correctness
question, it has never been tested, and it is the one item in the stalled tail that is
not a re-parameterisation of an already-fitted decision layer. Everything else in
exp09–exp11 is now measured as noise, and none of it should be revisited.

The next hypothesis is exactly one:

> **Fixing the global/local run-index mapping in Exp07's honest inner cross-fit removes
> the train/inference feature-distribution mismatch and improves the official
> composite.**

That is the whole claim. No expected gain is estimated here, no implementation is
chosen, no search space is specified, and the baseline policy required by §3 is left
open — because choosing any of those is the experiment, and an experiment invented by a
root-cause review is precisely what `PROTOCOL.md` §6 forbids.

If this single check also fails to promote, the correct conclusion is that the
remaining gap is not reachable from this representation at this evaluation resolution,
and that further hillclimbing on decision-layer parameters should stop rather than
continue in a new form.

## 6. Scope of this document

This review changed nothing. It did not modify exp09, exp10 or exp11; did not touch
`PROTOCOL.md` or `protocol_guard.py`; did not fix the Exp07 defect; did not create
`exp12`; did not build a submission or a ZIP; and did not train any model. The only
measurement performed for it was a direct count of run indices over the cached window
dataset, reported in §3.