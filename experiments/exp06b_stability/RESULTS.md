# Exp06b — Stability validation of the Exp06 candidates

**Type:** validation-only. No new features, no hyper-parameter changes, no selection.
**Question:** is the `full − age` advantage a real effect, or selection noise from
having picked it out of 12 variants on one 3-fold OOF run?
**Answer:** the Macro F1 advantage is real and stable. The Robustness claim is not —
it was substantially a seed-0 artifact.

---

## 1. What was fixed in advance

Four systems, defined by reference to the Exp06 blocks, never re-derived from these
results. Hyper-parameters are byte-identical to Exp05/Exp06; `random_state=42` inside
LightGBM is held constant, so the only thing varying is the **CV split**.

| system | features | composition |
|---|---|---|
| `exp05_B` | 226 | Exp05 B foundation (reference) |
| `exp06_ALL` | 255 | foundation + all 29 Exp06 temporal features |
| `full_minus_age` | 249 | foundation + backlog + receiver + reassign + tail |
| `full_minus_age_minus_reassign` | 245 | foundation + backlog + receiver + tail |

Feature counts are asserted in code (`common.py`, `N_FEATURES`) and matched at
runtime, so a silent drift in the upstream feature builders would abort the run
rather than quietly change what is being compared.

Seeds `0, 7, 21, 42, 99`, `StratifiedKFold(3, shuffle=True, random_state=seed)`,
15 fits per system, 60 in total. Success F1 pinned to 0.8644 as in Exp03/Exp05/Exp06.

## 2. The headline correction: `long-run dh` is a RECALL, not an F1

Implemented as an assertion, not a comment — `run_validation.py` aborts if the
long-slice precision is ever not exactly 1.0, because at that point the printed
number would silently stop meaning "recall":

```python
assert long["precision"] == 1.0, \
    "long slice precision %.4f != 1.0; not a recall proxy" % long["precision"]
```

It held on all 20 OOF runs. Exp05 §8 has been amended with the same note; every
sub-slice in that table is positives-only and therefore a recall proxy, not an F1.

## 3. Per-seed results

Full table in `SEED_TABLE.md`. Macro F1, the metric that drove the Exp06 pick:

| system | feats | s0 | s7 | s21 | s42 | s99 | **mean** | std | wins vs Exp05 B |
|---|---|---|---|---|---|---|---|---|---|
| `exp05_B` | 226 | 0.7758 | 0.7767 | 0.7775 | 0.7793 | 0.7790 | 0.7777 | 0.0015 | — |
| `exp06_ALL` | 255 | 0.7795 | 0.7806 | 0.7870 | 0.7816 | 0.7818 | 0.7821 | 0.0029 | 5/5 |
| `full_minus_age` | 249 | 0.7827 | 0.7815 | 0.7829 | 0.7835 | 0.7825 | **0.7826** | **0.0007** | **5/5** |
| `full_minus_age_minus_reassign` | 245 | 0.7795 | 0.7803 | 0.7836 | 0.7826 | 0.7806 | 0.7813 | 0.0017 | 5/5 |

Deltas vs Exp05 B on Macro F1:

| system | s0 | s7 | s21 | s42 | s99 | mean | median | std |
|---|---|---|---|---|---|---|---|---|
| `exp06_ALL` | +0.0037 | +0.0039 | +0.0094 | +0.0023 | +0.0028 | +0.0044 | +0.0037 | 0.0029 |
| `full_minus_age` | +0.0069 | +0.0048 | +0.0054 | +0.0042 | +0.0036 | **+0.0050** | +0.0048 | **0.0013** |
| `full_minus_age_minus_reassign` | +0.0037 | +0.0035 | +0.0061 | +0.0033 | +0.0016 | +0.0036 | +0.0035 | 0.0016 |

## 4. Acceptance criteria for `full − age`

| # | criterion | result | met? |
|---|---|---|---|
| 1 | mean Macro F1 > Exp05 B | +0.0050 | **YES** |
| 2 | mean Robustness F1 > Exp05 B | +0.0018 | **YES, but noise** |
| 3 | win on >= 4/5 seeds (Macro) | 5/5 | **YES** |
| 4 | no persistent drop in dh overall | +0.0121, 5/5 seeds | **YES** |
| 5 | `topo_mesh` not back to Exp05 regression | +0.0012, 3/5 | **WEAK** |

**All five criteria are met, but criterion 2 and 5 are met only barely, and the study
shows why they are weak: see §6.**

## 5. Significance (`STATS.md`, full table there)

| test | `full_minus_age` Macro F1 |
|---|---|
| paired t-test over 5 seeds | **p = 0.0009** |
| Wilcoxon over 5 seeds | p = 0.0625 (floored by n=5) |
| pooled paired bootstrap, 20k resamples | **95% CI [+0.0023, +0.0076]**, p = 0.0004 |

The t-test p-value at n=5 is a consistency check, not a real significance level, and
the bootstrap CI is optimistic because all 5 seeds resample the same 10 000 runs. The
honest statement is narrow: **the Macro F1 gain is consistent in sign and magnitude
across all 5 independent splits**, and the smallest of the 5 deltas (+0.0036) is still
positive.

## 6. Two claims from Exp06 that do NOT survive

This is the most useful output of the study.

**The Robustness gain was largely a seed-0 artifact.** Exp06 reported +0.0154
Robustness and called it the best result of the project. Across 5 seeds the mean is
**+0.0018**, the wins are **3/5**, the seed-level std is 0.0073, and the pooled
bootstrap CI is `[-0.0059, +0.0096]` — it **includes zero**. Seed 42 gives
`−0.0057` and seed 7 `+0.0009`; only seed 0 (+0.0135) and seed 21 (+0.0026) carry it.
`exp06_ALL`'s Robustness mean is +0.0003 with a CI of `[-0.0078, +0.0082]`, i.e.
**flat**. A +0.0154 result measured on a single split is not a property of the model;
it is a property of that split. The number should be withdrawn, not restated.

**`topo_mesh` recovery is mostly gone too.** Exp06 reported +0.0092; here the mean is
**+0.0012** with 3/5 wins and CI `[-0.0056, +0.0081]`. It is no longer a clear
reversal of the Exp05 regression — it is directionally right but indistinguishable from
zero. Note `full_minus_age_minus_reassign` does better here (+0.0046, 4/5, CI lower
bound −0.0015), which is *not* a reason to switch to it — it is just another
observation that was not part of the pre-registered comparison.

## 7. The primary target is still not met

`long-run dh` was Exp06's pre-registered headline. Across 5 seeds:

| system | mean Δ recall | median | wins | pooled-boot CI95 |
|---|---|---|---|---|
| `exp06_ALL` | +0.0171 | +0.0236 | 4/5 | [-0.0041, +0.0389] |
| `full_minus_age` | +0.0094 | +0.0147 | 4/5 | **[-0.0112, +0.0307]** |
| `full_minus_age_minus_reassign` | +0.0124 | +0.0265 | 3/5 | [-0.0077, +0.0330] |

Not one of the three reaches the required +0.03, and **every CI includes zero**. The
original Exp06 verdict (`not_promising`, primary target missed) stands.

## 8. Threats to validity

- **This is not an independent confirmation.** Repeated CV on one demo dataset
  measures *stability*, not generalisation. `full − age` was chosen after seeing
  Exp06's single 3-fold OOF, and no amount of re-splitting the same 10 000 runs
  removes that selection step. The expected sign of the effect here is upward-biased.
- **The three significance levels are not independent** and disagree by
  construction; they are reported side by side because collapsing them into one
  number would be dishonest.
- **5 seeds is a small sample.** Wilcoxon cannot go below p = 0.0625 with n = 5, so
  the rank test is structurally incapable of reaching 0.05 here and should not be read
  as a near-miss.
- **Success F1 is pinned**, not re-fitted, matching Exp06's convention. If the new
  features ever affected `success`, this study would not see it.
- **Only the split varies.** LightGBM's `random_state=42` is fixed, so model-seed
  variance is excluded by construction. The real competition run has both.

## 9. Recommendation

**Adopt `full − age` (249 features) as the working candidate for the competition
train, and stop experimenting on demo data.**

Rationale, in order of weight:

1. It is the only system with a Macro F1 gain that is positive on **all 5** seeds,
   with the **smallest** seed-to-seed spread (std 0.0007 vs 0.0029 for `exp06_ALL`)
   — the most stable configuration measured in this project.
2. `dropped_handoff` overall F1 improves +0.0121 on 5/5 seeds (t-test p = 0.0080),
   the most consistent effect measured anywhere in Exp05/Exp06.
3. Robustness and `topo_mesh` should be **demoted to tie-breakers only**. Their
   Exp06 values are not reproducible, and shipping a model chosen partly on them
   would be choosing on noise.

`full_minus_age_minus_reassign` is 4 columns smaller and beats `full_minus_age` on
Robustness and `topo_mesh`, but loses on Macro F1 and composite. It is **not**
selected — selecting it *now*, on the strength of these same five seeds, would be
precisely the post-hoc selection this study was built to avoid. It stays a documented
alternative.

**Explicitly: this is a stability result on demo data, not a confirmation.** The real
test is the competition train arriving tomorrow. The plan there is unchanged and now
has a concrete ordering: confirm Exp03 → Exp05 → Exp06b's `full − age` in that
sequence, each against the same Exp05 B baseline, and treat the Robustness and
`topo_mesh` numbers from this report as **noise bands, not expected gains**.

## 10. Files

| file | contents |
|---|---|
| `common.py` | fixed systems, feature-count assertions, split generator |
| `run_validation.py` | 4 systems × 5 seeds × 3-fold OOF driver |
| `stability_stats.py` | seed-level tests + pooled paired bootstrap |
| `results.json` | raw per-seed metrics |
| `SEED_TABLE.md` | 4 × 5 seed table + all deltas |
| `STATS.md` | significance table |
| `oof_*.csv` | 20 OOF prediction files (4 systems × 5 seeds) |
