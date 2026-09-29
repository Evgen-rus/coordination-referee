# Exp08 - L1 window-peak localizer

**Verdict: PROMOTE** (with the post-hoc caveat in section 9, which is not optional).

One question, one rule, no new model:

> Can the official rule-based L0 localizer be replaced by the turn where the
> Exp07 window model is most confident that the **already-predicted** fault
> class occurs?

Nothing else changed. No retrain, no new feature, no new head, no threshold.

---

## 1. Headline

| metric | L0 (Exp07, shipped) | L1 (Exp08) | delta |
|---|---|---|---|
| **fault_turn hit@2** | 0.43083333333333335 | **0.6086111111111111** | **+0.1778** |
| **COMPOSITE** | 0.7516676139361507 | **0.7694453917139285** | **+0.0178** |
| Macro F1 | 0.7894323366835861 | 0.7894323366835861 | 0.0000 (pinned) |
| Robustness F1 | 0.7368470046834376 | 0.7368470046834376 | 0.0000 (pinned) |
| Success F1 | 0.8643757406010988 | 0.8643757406010988 | 0.0000 (pinned) |

The composite delta is exactly `0.10 x 0.1778`. The label, robustness and
success terms are **identical floats**, not re-estimates: Exp08 changes
`fault_turn` and nothing else, so re-deriving them would be noise dressed up as
a result.

The delta is **2.9x larger than Exp07's entire label-model gain** (+0.0061)
came from adding 51 features and 457k windows.

## 2. The rule

```
predicted label == clean      ->  fault_turn = -1
empty window block            ->  fault_turn = -1
otherwise                     ->  fault_turn = argmax_t P[run, t, class(predicted label)]
```

`P[run, t, c]` is the Exp07 window model's probability that turn `t` of this run
belongs to fault class `c`. The class index comes from `window_dataset.W2I`,
whose index 0 is `background` while the label head's index 0 is `clean` -
reading the peak with a *label* index would shift every class by one and still
produce plausible-looking numbers, which is why `check_parity.py` pins the two
orderings explicitly.

No confidence threshold. No offset. No class-specific rule. No calibration. No
ensemble. No second model.

## 3. Reproduction gate - PASSED, exact

Exp07's L1 diagnostic was re-derived from the verified artifacts through the
**official** `evaluation.metrics.fault_turn_hit_at_k` and had to match to 1e-12:

| check | expected | got | |
|---|---|---|---|
| L0 hit@2 | 0.43083333333333335 | 0.43083333333333335 | exact |
| L1 hit@2 | 0.6086111111111111 | 0.6086111111111111 | exact |
| dropped_handoff | 0.4675 | 0.4675 | exact |
| duplicated_work | 0.7275 | 0.7275 | exact |
| deadlock | 0.5883333333333334 | 0.5883333333333334 | exact |
| conflict | 0.7000 | 0.7000 | exact |
| goal_drift | 0.5766666666666667 | 0.5766666666666667 | exact |
| runaway_loop | 0.5916666666666667 | 0.5916666666666667 | exact |

Every hit@k in this experiment is computed twice - once by the official metric,
once by a deliberately naive brute-force recount - and the two are asserted
equal to 1e-12 before any number is reported. This is not ceremony: Exp07
shipped a hit@2 of 0.1843 that disagreed with the official 0.4258 because a
positional `zip` misaligned every row after the first `clean` run, and it took a
separate gate (`verify_hit_at_k.py`) to catch it. A second implementation is
the cheapest insurance against repeating that class of bug.

## 4. hit@0 / hit@1 / hit@2 - the trade being made

| turn source | hit@0 | hit@1 | hit@2 |
|---|---|---|---|
| **L0** official rule-based | **0.3618** | **0.3981** | 0.4308 |
| **L1** window peak | 0.1142 | 0.3400 | **0.6086** |

L1 wins by +0.178 at hit@2 and **loses by -0.248 at hit@0**. This is the whole
story and it is worth stating plainly: the window model finds the right
*neighbourhood* far more often, and the exact turn far less often. The official
composite scores only hit@2, so within this metric the trade is strongly
favourable. A scorer that rewarded the exact turn would reverse it.

Peak offset vs the true `fault_turn` (Exp07 diagnostic, unchanged): median **+0**,
exact 0.1271, within +/-2 0.6724, within +/-5 0.7514. The distribution is
tightly centred, which is why a +/-2 window catches so much of it.

## 5. Per-class L1 hit@2

| class | L1 hit@0 | L1 hit@1 | L1 hit@2 | n |
|---|---|---|---|---|
| duplicated_work | 0.1125 | 0.3767 | **0.7275** | 1200 |
| conflict | 0.1108 | 0.3858 | **0.7000** | 1200 |
| runaway_loop | 0.1425 | 0.3633 | 0.5917 | 1200 |
| deadlock | 0.1158 | 0.3200 | 0.5883 | 1200 |
| goal_drift | 0.1033 | 0.3292 | 0.5767 | 1200 |
| dropped_handoff | 0.1000 | 0.2650 | 0.4675 | 1200 |

L1 beats L0 on **all six classes**. The ordering tracks Exp07's window-level
diagnostics: `duplicated_work` and `conflict` have the highest window recall
(0.369 / 0.226) and the highest L1 hit@2; `dropped_handoff` has the lowest
window recall (0.171) and the lowest L1 hit@2. Where the window model can
actually discriminate a local region, the peak is good; where the signal is
run-level, the peak is a rough guess that still lands within +/-2 more often than
the rule-based localizer does.

## 6. Cross-fold consistency - 3/3

| fold | n_val | faulty | L0 h@0 | L0 h@1 | L0 h@2 | L1 h@0 | L1 h@1 | L1 h@2 | delta h@2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3334 | 2400 | 0.3533 | 0.3929 | 0.4238 | 0.1042 | 0.3179 | 0.5971 | **+0.1733** |
| 1 | 3333 | 2400 | 0.3667 | 0.4021 | 0.4379 | 0.1196 | 0.3525 | 0.6179 | **+0.1800** |
| 2 | 3333 | 2400 | 0.3654 | 0.3992 | 0.4308 | 0.1188 | 0.3496 | 0.6108 | **+0.1800** |

**L1 wins 3/3 folds**, with a tight spread (+0.173 to +0.180). The folds were
re-derived from `StratifiedKFold(3, shuffle=True, random_state=0)` and their
index-vector SHA-256s are asserted equal to the ones sealed in Exp07's manifest
- same folds, not merely same fold sizes.

**This is NOT independent validation.** L1 was discovered post-hoc on exactly
these OOF predictions by Exp07's diagnostics. A 3/3 result on data the rule was
chosen on is a *consistency* check: it shows the gain is not carried by one
unlucky fold. It is not evidence of generalisation, and it must not be reported
as if it were. The genuine external test is the public leaderboard.

## 7. Composite arithmetic

Computed through the official `evaluation.metrics.composite`, and cross-checked
against a hand expansion of the weights:

```
0.50 * 0.7894323366835861   (macro)
+ 0.25 * 0.7368470046834376   (robustness)
+ 0.15 * 0.8643757406010988   (success)
+ 0.10 * 0.6086111111111111   (fault_turn hit@2)
= 0.7694453917139285
```

| | L0 | L1 | delta |
|---|---|---|---|
| hit@2 | 0.43083333333333335 | 0.6086111111111111 | +0.1777777777777778 |
| composite | 0.7516676139361507 | 0.7694453917139285 | +0.0177777777777778 |

`composite()` on the L0 inputs reproduces Exp07's recorded 0.7516676139361507
exactly, which confirms the weight vector and the unchanged heads.

## 8. Provenance

Exp07's manifest did not seal `window_peak_pos.npy` / `window_peak_prob.npy`,
because until now nothing outside a diagnostic read them. Exp08 makes them a
consumer-facing artifact, so they were sealed - **as a provenance upgrade, not a
retrain**. Before any hash was written, `seal_peaks.py` proved:

1. both peak arrays are byte-identical to the git commits that introduced them
   (`06f92aa3`), i.e. untouched since the CV run;
2. the two on-disk copies (`exp07_fault_windows/` and `runs/cv/`) agree;
3. `runs/cv/` is clean against git;
4. the **base manifest already passes `reuse.load`** - the peaks are sealed on
   top of a proven-good base, not in isolation;
5. every `peak_pos[i, c]` is a valid turn index of run `i`'s own window block
   (`0 <= pos < n_windows(run)`), every run was validated, no empty block is
   silently recorded as turn 0, and probabilities are finite inside `[0, 1]`;
6. the L1 hit@2 recomputed from these arrays equals Exp07's committed
   `diagnostics.json` exactly, overall and per class.

`load_exp07.py` then verifies them fail-closed on every use:
`window_peak_sha256` must be present and complete, each file's hash must match,
shapes and dtypes must be exact, and coverage must be total. **`ArtifactRejected`
is fatal** - there is no lenient mode.

`test_peak_provenance.py` proves the failure direction: **12/12** cases, each
staging the real artifacts, tampering with exactly one thing, and asserting
rejection - including a manifest stripped of the peak hashes entirely, a single
flipped peak position, a dtype rewrite, and a `peak_pos` marking an unvalidated
run (caught by *content*, with the hashes made to agree, because that is the
only way it could ever slip through).

## 9. Production build and parity

`submission_exp08_window_localizer/` is `submission_exp07_fault_windows` with
**one executable statement changed**:

```python
# WAS
pred_turn = [localize(run, lab) for run, lab in zip(te_runs, pred_label)]
# NOW
te_peak = pt.peak_turns(Pte, Wte["run"], len(te_runs), pred_label)
pred_turn = [int(t) for t in te_peak]
```

The window model was **already being fitted** in production - the 51 aggregates
are the entire new signal for the label head. Exp07 computed the peak positions
and discarded them; Exp08 reads them back out of the same probability blocks.
No extra model, no extra features, no material extra runtime.

| check | result |
|---|---|
| 8 feature modules byte-identical to Exp07 | PASS (sha256) |
| `label` byte-identical | **4000 / 4000** |
| `success` byte-identical | **4000 / 4000** |
| `fault_turn` changed | 2504 of 4000 rows |
| every clean run -> `fault_turn = -1` | PASS (1246/1246) |
| no predicted-fault run left without a turn | PASS (0 of 2754 at -1) |
| label / success / window model params | identical attribute by attribute |
| feature counts 249 + 51 = 300, 185 success | pinned, unchanged |
| peak-helper parity vs `grouping.aggregate_block` | PASS, 16/16 |
| validator (`scripts/validate_submission.py`) | OK, 4000 rows |
| runtime, full 10k -> 4k | see section 10 |

`peak_turn.py` is proved to be the same function as Exp07's
`grouping.aggregate_block` on live probabilities from a really-fitted window
model, plus adversarial shapes: 0/1/2-window runs, shuffled row order,
duplicate indices, exact ties, one-hot rows, an empty middle run between two
populated ones, and nothing scored at all. Two of those tests failed on first
run and both failures were *test* bugs - a tie fixture built on the wrong column
index, and a comparison that included the `background` column the L1 rule never
reads. Both were corrected in the test, not worked around in the helper.

**STOP condition honoured:** if a single `label` or `success` row had differed,
the build would have been rejected. None did.

One check was rewritten before the final run: "no faulty run gets -1" had been
written as `(... >= 0).all() | True`, which OR-s in a constant and can never
fail - a test that cannot fail is not a test. It now asserts the count is zero,
and it is zero: all 2754 predicted-fault runs received a real turn.

## 10. Runtime

Full 10 000 -> 4 000, CPU only, measured by `check_production_parity.py` running
each submission as a separate process on the same inputs:

| submission | wall clock |
|---|---|
| Exp07 (L0, rule-based localizer) | 379.7 s (6.3 min) |
| **Exp08 (L1, window peak)** | **374.9 s (6.2 min)** |
| overhead of L1 localization | **-4.8 s** |

The L1 rule is *free*: `peak_turns` is one stable `argsort` plus a
`searchsorted` over probability blocks the pipeline has already computed, so the
difference is run-to-run variance in LightGBM timing rather than added work. Both
are far inside the 30-minute job limit.

An earlier session measured the same pair at 360.2 s / 350.0 s, so machine
load moves these by ~5%. The comparison that matters is between the two rows
above, which ran back to back on the same machine in the same session. The
standalone ZIP run was 308.6 s.

## 11. Verdict: PROMOTE

1. **Reproduction is exact** - 8/8 checks to 1e-12, through the official metric,
   with an independent recount agreeing.
2. **+0.178 hit@2 is far outside noise**, and 3/3 folds move by +0.173..+0.180.
3. **Composite +0.0178**, all of it from the hit@2 term, arithmetic verified
   through the official `composite()`.
4. **The production change is one line** and provably cannot touch `label` or
   `success`: 4000/4000 byte-identical, feature modules sha256-identical, model
   params identical, runtime well inside the limit.
5. **The window model already existed in production**, so this is a
   configuration change, not new machinery.

### Caveats that must travel with the promotion

- **L1 was discovered POST-HOC on Exp07's OOF.** Exp07's own diagnostics found
  it while looking for something else, on the same 10 000 runs whose labels
  produced it. Every number in sections 1-7 is measured on data the rule was
  chosen on. This is the single most important caveat in this report: the
  +0.178 is an upper bound on the gain, not an estimate of it. **The public
  leaderboard is the real external validation**, and nothing in this repository
  can substitute for it.
- **hit@0 collapses from 0.3618 to 0.1142.** Under the official metric this is
  irrelevant. If the true grader ever weighted exactness, or a human reviewed
  turns, L1 would be a regression. The composite is the contract, and it scores
  hit@2 - but the collapse is real and should not be forgotten.
- **hit@2 carries only 0.10 of the composite.** The +0.178 is a large move in a
  small term. It is still the single largest available gain in the project, but
  it does not make the label head better, and it should not be described as
  though it did.
- **`dropped_handoff` is the weakest class** (0.4675) - the project has
  historically cared most about that class, and it is where L1 helps least.
- **Single seed, 3 folds.** As in Exp06/Exp06b, the fold check is consistency
  evidence, not significance. No seed study was run and none was pre-registered.

### Not done, deliberately

No L2 hybrid, no 0.60 threshold, no threshold search, no class-specific rules,
no offsets, no calibration, no ensemble, no retraining for the OOF, no new
features, no new label classifier, no leaderboard tuning. Exp07's L2 hybrid
scored hit@2 0.5122 - between L0 and L1 - and its 0.60 cut was fixed *after*
seeing L1, so it remains a post-hoc diagnostic and is excluded from the
headline, exactly as Exp07 excluded it.

## 12. Files

| file | contents |
|---|---|
| `seal_peaks.py` | provenance upgrade: proves the peak arrays are Exp07's, seals their sha256 |
| `load_exp07.py` | fail-closed loader for OOF + peaks; `ArtifactRejected` is fatal |
| `test_peak_provenance.py` | 12 fail-closed tamper tests |
| `run_experiment.py` | the experiment: official metric, reproduction gate, folds, composite |
| `check_production_parity.py` | Exp07 vs Exp08 end-to-end, 10k -> 4k |
| `results.json` | all metrics |
| `production_parity.json` | parity + runtime summary |
| `run.log`, `parity.log` | execution logs |
| `RESULTS.md` | this report |

Production: `submission_exp08_window_localizer/` (+ `.zip`), with
`peak_turn.py` (the helper) and `check_parity.py` (its parity proof).
