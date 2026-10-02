# Exp14 - explicit bounded episode representation (`exp14_trajectory_episode`)

**Status: STOP (gate 4/6).**
One frozen hypothesis, one 16-column block, one paired CV. The baseline reproduced
the committed Exp13 OOF **bit for bit** (sha `f4e70a42…`, 5/5 metrics at 0.00e+00),
so the deltas below are a feature effect on one pipeline, not a difference
between two pipelines. The episode block did not improve `runaway_loop`
classification; it slightly hurt it while lifting robustness.

---

## 1. The single hypothesis

> Explicit bounded repeat-episode representation improves classification of
> `runaway_loop`, because the current 312-feature Exp13 representation stores
> only order-free repeat counts and adjacency streaks, but loses episode
> identity, chronology, interleaving, agent-ring structure and progress inside
> the episode.

Motivated by `experiments/EXP13_ERROR_AUDIT.md`, which graded this direction
**VERDICT B** (observable but genuinely lost) and measured episode identity at
\(r = 0.6158\) against `norm_dup_max` — partially reconstructible, not present.

## 2. The frozen block

Copied verbatim from the audited detector in `experiments/exp13_error_audit.py`:
signature = production `features._norm(text)`, `EP_MIN_LEN = 3`, `EP_GAP_MAX = 6`,
`EP_CYCLE_MIN = 2`. Maximal ordered run of one signature whose consecutive turns
are within `EP_GAP_MAX`; sender is recorded per occurrence, never folded into the
key, because the ontology's runaway case is a ring. Progress = new artifact, new
`shared_state` write, or delivered message inside the episode window.

| feature | meaning |
| --- | --- |
| `ep_n_episodes` | episodes found in the run |
| `ep_has_episode` | indicator |
| `ep_longest_len` | length of the longest episode |
| `ep_longest_span_rel` | its span, relative to run length |
| `ep_repeat_event_share` | repeat events / total messages |
| `ep_max_distinct_agents` | most senders inside any one episode |
| `ep_has_multi_agent_cycle` | indicator: any episode touching >= 2 senders |
| `ep_n_cycle_episodes` | count of such episodes |
| `ep_n_interleaved_episodes` | count with a gap > 1 |
| `ep_interleaved_share` | interleaved / total episodes |
| `ep_n_no_progress_episodes` | episodes with nothing new inside |
| `ep_no_progress_share` | that count / total episodes |
| `ep_longest_no_progress_span_rel` | longest no-progress span, relative |
| `ep_episodes_with_progress_share` | share with any progress event (broad) |
| `ep_earliest_start_rel` | earliest onset, relative |
| `ep_latest_end_rel` | latest end, relative |

Three definitions were resolved **before** any fit and are recorded in
`features.py`:

1. **`ep_episodes_with_progress_share` is the broad reading** (artifact, state
   write, or delivery). The narrow complement would have made it a
   mathematically identical duplicate of `1 - ep_no_progress_share`, which the
   protocol forbids keeping.
2. **`ep_earliest_start_rel` is the minimum over all episodes.** The audit's
   `longest_start_rel` keyed on the longest episode; the name says earliest.
3. **A run with no episode is all zeros**, with the two onset columns at `-1.0`
   as an explicit "no onset exists" sentinel.

**Leakage.** Only `messages`, `artifacts` and `shared_state` are read, and only
their observable fields. `label`, `success` and `fault_turn` are never touched:
target perturbation over 400 runs leaves the matrix **bit-identical**
(`max|d| = 0.0e+00`).

## 3. Baseline reproduction - PASS, bit for bit

```
   macro_f1           expected 0.8129545586738590 got 0.8129545586738590 diff 0.00e+00 PASS
   robustness_f1      expected 0.7599843692318033 got 0.7599843692318033 diff 0.00e+00 PASS
   success_f1         expected 0.8643757406010988 got 0.8643757406010988 diff 0.00e+00 PASS
   fault_turn_hit2    expected 0.6315277777777778 got 0.6315277777777778 diff 0.00e+00 PASS
   composite          expected 0.7892825105128229 got 0.7892825105128229 diff 0.00e+00 PASS
   sha256 A         : f4e70a4280c825613e54cde7a843915323a89643be78138ad5c48f2ced131853
   sha256 exp13 D   : f4e70a4280c825613e54cde7a843915323a89643be78138ad5c48f2ced131853
```

CONTROL A is Exp13 Candidate D verbatim: the 300 corrected columns plus the
frozen 12 `wg_*` columns. Both arms were produced from **one** shared corrected
fold matrix per fold, and A/B differ in exactly the 16 appended columns —
asserted per fold with `np.array_equal` on the base block.

## 4. A vs B

| metric | A (Exp13) | B (+episodes) | delta |
| --- | --- | --- | --- |
| composite | 0.7892825105 | **0.7916180754** | **+0.002335** |
| macro F1 | 0.8129545587 | 0.8113011435 | -0.001654 |
| robustness F1 | 0.7599843692 | 0.7732986719 | +0.013314 |
| hit@2 | 0.6315277778 | 0.6298611111 | -0.001667 |
| success F1 | 0.8643757406 | 0.8643757406 | 0 (pinned) |
| **runaway_loop F1** | **0.7349** | **0.7317** | **-0.0033** |

Per-class F1:

| class | A | B | delta |
| --- | --- | --- | --- |
| clean | 0.8784 | 0.8780 | -0.0004 |
| dropped_handoff | 0.6550 | 0.6550 | +0.0000 |
| duplicated_work | 0.8411 | 0.8355 | -0.0056 |
| deadlock | 0.8336 | 0.8345 | +0.0009 |
| conflict | 0.9033 | 0.9024 | -0.0009 |
| goal_drift | 0.8443 | 0.8420 | -0.0023 |
| **runaway_loop** | 0.7349 | 0.7317 | **-0.0033** |

`runaway_loop` precision 0.7598 → 0.7585 (-0.0013), recall 0.7117 → 0.7067
(-0.0050). The episode block moved **recall down**, which is the opposite of
what an episode signal would have to do to help.

## 5. Confusions

| pair | A | B | delta |
| --- | --- | --- | --- |
| runaway_loop -> clean | 104 | 109 | +5 |
| runaway_loop -> dropped_handoff | 44 | 42 | -2 |
| runaway_loop -> deadlock | 59 | 59 | 0 |
| runaway_loop -> goal_drift | 65 | 65 | 0 |
| clean -> runaway_loop | 80 | 79 | -1 |
| dropped_handoff -> runaway_loop | 88 | 89 | +1 |
| deadlock -> runaway_loop | 32 | 29 | -3 |

Largest improvement: `deadlock -> runaway_loop` -3. Largest regression:
`runaway_loop -> clean` +5 — the exact confusion the hypothesis targeted.

## 6. Changed labels

290 of 10000 rows changed (2.90%): 113 wrong → right, 127 right → wrong,
1615 wrong → wrong. Over the 1200 true `runaway_loop` rows: **20 fixed,
26 broken, 326 still wrong**. The block fixed fewer than it broke, and the
residual pool is still large.

## 7. Fold stability

| fold | n | composite | macro | robustness | hit@2 | RL F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3334 | +0.001366 | +0.000115 | +0.006066 | -0.002083 | 0.6934 → 0.6962 (+0.0028) |
| 1 | 3333 | +0.000218 | -0.002879 | +0.007461 | -0.002083 | 0.7635 → 0.7592 (-0.0043) |
| 2 | 3333 | +0.006530 | -0.002374 | +0.031199 | -0.000833 | 0.7477 → 0.7389 (-0.0088) |

Composite wins 3/3. `runaway_loop` F1 wins 1/3, and macro F1 is negative on
2/3 folds.

## 8. Feature importance — the block was effectively ignored

Episode block total gain share: **0.0031** (0.31%). Episode features in the
top-20: **0 of 16**. In the top-50: **0 of 16**. The highest single member
contributed 0.00063.

| feature | gain |
| --- | --- |
| `ep_longest_span_rel` | 0.00063 |
| `ep_latest_end_rel` | 0.00061 |
| `ep_earliest_start_rel` | 0.00053 |
| `ep_repeat_event_share` | 0.00051 |
| `ep_longest_no_progress_span_rel` | 0.00034 |
| `ep_max_distinct_agents` | 0.00027 |

## 9. Why it failed — the actual finding

**The block is not redundant, it is unhelpful.** The audit's verdict rested on
`max |r|` against the 261 fold-independent columns, and the measurement
confirms the block is genuinely new information: no column is an exact
duplicate (0 of 16), and the highest correlation is 0.6158. Yet LightGBM gave
the whole block 0.31% of the split gain.

The reason is **base rate, not redundancy**. An episode exists in only
**2398 of 10000 runs (24.0%)**. Within the 1200 true `runaway_loop` rows the
signal is not even rare, but the block's discriminative power is diluted by the
7602 rows that have no episode at all and are identical on every column. The
gain share is the honest measurement of that dilution.

**The composite gain is not the target class.** Composite rose +0.0023, driven
almost entirely by robustness +0.0133 while macro fell -0.0017. The episode
block appears to make the model marginally more conservative on the robustness
subset without helping `runaway_loop` at all — and `runaway_loop` was the one
class the hypothesis was about.

**Why the audit predicted B and the experiment says otherwise.** The audit
measured *representational* absence (r = 0.6158) and *separation* on current
errors (6/6 diagnostics, 3/3 fold-stable). It never measured whether a model
would *use* the information once given. Those are different claims. Direction B
was observable and lost — that was accurate. Whether it is *exploitable* was the
untested premise, and the answer here is no.

## 10. Gate — FAIL 4/6

### 10.1 Predeclared gate (unchanged, not retuned)

| criterion | required | measured | |
| --- | --- | --- | --- |
| composite delta | >= +0.002 | +0.002335 | PASS |
| composite folds won | >= 2/3 | **3/3** | PASS |
| runaway_loop F1 delta | >= +0.015 | **-0.003270** | FAIL |
| runaway_loop folds won | >= 2/3 | **1/3** | FAIL |
| robustness | >= -0.003 | +0.013314 | PASS |
| no class worse than | -0.02 | none (worst -0.00555) | PASS |

Two criteria are about the declared target class, and both failed. The gate is
conjunctive: 4/6 is a STOP.

## 11. Honest limitations

1. **The block fired on only 24.0% of runs.** That is a property of the
   detector under this data, not a tuning choice, but it does mean the block is
   mostly zeros and its gain share should be read against that base rate.
2. **`success` F1 is pinned, not measured.** No `ep_*` feature targets
   `success` and the head is unchanged, so it stays at 0.8644 by construction.
3. **Single seed, 3 folds.** The fold check is consistency, not significance.
   No seed search was performed, by design.
4. **The detector does not cap episodes at the ontology's 14 repeats.** It
   reports what it sees; a few real episodes span 4 agents, above the ontology's
   "2-3" ring. Recorded in `test_features.py` 12h rather than silently clipped.
5. **`fault_turn` inherits Exp08's post-hoc caveat.** hit@2 fell -0.0017; Exp14
   does not make that number more or less honest.
6. **This is OOF on the same 10000 train runs** as every prior experiment. The
   public leaderboard remains the real external validation.

## 12. Reproduction note — read before re-running

`protocol_guard.py` was run first (current best `exp13`, 0/3 STOPs, verdict
"a new experiment MAY be started"). The committed Exp13 OOF reproduced bit for
bit only when this script was the **only** Python process on the machine.

That is not a hypothesis, it is a measurement. Three earlier attempts drifted
away from the committed artifact (composite delta 7.48e-04, 3.55e-05, and
8.01e-04) with the fold affected moving between runs (first fold 0, then fold
2), which is the signature of floating-point reduction-order noise rather than a
code difference. `exp07/parallel.py` documents exactly this hazard: the window
model is pinned to `n_jobs=-1` because argmax-derived aggregates make 49 of
10000 predictions sensitive to last-bit changes.

Component-level determinism was verified and is **not** the cause: the same
inner window fit repeated 4x in-process is bit-identical, `workers=1` and
`workers=3` (loky) agree exactly, and the 600-tree label fit is unaffected by an
intervening wider fit. The run must simply be alone on the box.

```bash
python experiments/protocol_guard.py
python experiments/exp14_trajectory_episode/test_features.py
python experiments/exp14_trajectory_episode/run_experiment.py
```

`test_features.py` is a flat script, not pytest-collectable (same as Exp13's):
run it directly. It reports `ALL TESTS PASSED (54 checks)` and exits non-zero on
any failure.