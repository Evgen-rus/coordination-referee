# Exp06 - Temporal dynamics of unresolved handoffs

Foundation: Exp05 B (122 baseline + 63 Exp03 + 41 Exp05 lifecycle = 226).
Added: 29 features in 5 blocks. Same folds, same hyperparameters, no new deps.

Verdict: PARTIAL SUCCESS. The pre-registered target was missed and one block is
harmful. The 29-feature bundle is NOT recommended as-is.

Best supported config is `full - age` (23 features): Macro 0.7827 (+0.0069,
McNemar p=0.0022, 3/3 folds). The pre-registered full model (6_B_ALL) reaches
0.7795 (+0.0037, p=0.097 - not significant).

## 1. Pre-registered criteria scorecard

| # | criterion | target | actual (6_B_ALL) | met? |
|---|---|---|---|---|
| 1 | long-run dh F1 | >= +0.03 | **+0.0138** | NO |
| 2 | overall dh F1 not worse | >= 0 | +0.0100 | YES |
| 3 | Macro F1 not worse | >= 0.7758 | 0.7795 | YES |
| 4 | Robustness | >= 0.7114 | 0.7298 | YES |
| 5 | improvement on >= 2/3 folds | 2/3 | 2/3 (fold 0 -0.0008) | MARGINAL |
| 6 | very good: long-run dh | >= +0.05 | +0.0138 | NO |
| 7 | very good: overall dh | >= +0.02 | +0.0100 | NO |
| 8 | very good: composite | >= +0.005 | **+0.0061** | YES |

**3 of 5 required criteria met. The primary target (long-run dropped_handoff) was
missed by a wide margin: +0.0138 against a required +0.03.**

For `7_full_minus_age` the primary target is still missed (+0.0172), but criteria 2-5
all pass and criterion 5 passes decisively (3/3 folds).

## 2. Headline - 12 variants, 3-fold OOF

| variant | feats | Macro | Robust | comp | dMacro | d_dh | d_dh_long |
|---|---|---|---|---|---|---|---|
| **0** Exp05 B | 226 | 0.7758 | 0.7144 | 0.7381 | - | - | - |
| 1 +age | 232 | 0.7739 | 0.7178 | 0.7378 | -0.0019 | -0.0082 | +0.0069 |
| 2 +backlog | 235 | 0.7773 | 0.7205 | 0.7406 | +0.0015 | +0.0063 | +0.0138 |
| 3 +receiver | 234 | 0.7786 | 0.7218 | 0.7417 | +0.0028 | +0.0072 | +0.0138 |
| 4 +reassign | 230 | 0.7787 | 0.7194 | 0.7409 | +0.0029 | -0.0023 | **-0.0319** |
| 5 +tail | 228 | 0.7767 | 0.7233 | 0.7410 | +0.0009 | +0.0044 | +0.0138 |
| **6 = ALL (pre-reg)** | 255 | 0.7795 | 0.7298 | 0.7442 | +0.0037 | +0.0100 | +0.0138 |
| **7 full - age** | 249 | **0.7827** | 0.7279 | **0.7456** | **+0.0069** | +0.0127 | +0.0172 |
| 8 full - backlog | 246 | 0.7817 | 0.7289 | 0.7452 | +0.0059 | +0.0181 | **+0.0405** |
| 9 full - receiver | 247 | 0.7797 | 0.7277 | 0.7439 | +0.0039 | -0.0013 | +0.0035 |
| 10 full - reassign | 251 | 0.7779 | 0.7246 | 0.7420 | +0.0021 | +0.0145 | +0.0103 |
| 11 full - tail | 253 | 0.7813 | 0.7300 | 0.7453 | +0.0055 | +0.0101 | +0.0103 |

**Four of the five drop-one variants beat the full model.** The full 29-feature bundle
is over-parameterised: it dilutes 249 useful columns with features that cost more than
they return.

## 3. Significance

| variant | d Macro | 95% CI | P(>0) | McNemar net | p | folds |
|---|---|---|---|---|---|---|
| 6 ALL | +0.0037 | [-0.0009, +0.0082] | 0.946 | +36 | 0.097 | 2/3 |
| **7 full - age** | **+0.0069** | **[+0.0023, +0.0115]** | **0.998** | **+66** | **0.0022** | **3/3** |
| 8 full - backlog | +0.0059 | [+0.0015, +0.0104] | 0.997 | +55 | 0.0086 | 3/3 |
| 11 full - tail | +0.0055 | [+0.0010, +0.0100] | 0.992 | +55 | 0.0091 | 3/3 |
| 9 full - receiver | +0.0039 | [-0.0004, +0.0083] | 0.965 | +43 | 0.0331 | 2/3 |
| 4 +reassign | +0.0029 | [-0.0010, +0.0068] | 0.928 | +32 | 0.089 | 3/3 |
| 3 +receiver | +0.0028 | [-0.0013, +0.0072] | 0.911 | +27 | 0.175 | 2/3 |
| 2 +backlog | +0.0015 | [-0.0022, +0.0056] | 0.792 | +16 | 0.399 | 2/3 |
| 5 +tail | +0.0009 | [-0.0031, +0.0044] | 0.671 | +9 | 0.658 | 2/3 |
| 1 +age | -0.0019 | [-0.0054, +0.0019] | 0.155 | -14 | 0.442 | 1/3 |

The pre-registered full model is **not** statistically significant (p=0.097).
`full - age` is (p=0.0022, CI excludes zero, 3/3 folds).

**Is the age block harmful?** `full - age` beats `full` by +0.0032 Macro, but the CI
[-0.0005, +0.0070] includes zero and McNemar p=0.084. So the age block is *probably*
mildly harmful, not certainly. It is certainly not helpful: adding it to the
foundation gives -0.0019 (1/3 folds), the only block with a negative sign.

## 4. dropped_handoff by run length

| sub-slice | n | B | +age | +backlog | +receiver | +reassign | +tail | ALL | full-age |
|---|---|---|---|---|---|---|---|---|---|
| all | 1200 | 0.6770 | 0.6762 | 0.6849 | 0.6878 | 0.6733 | 0.6857 | 0.6864 | 0.6853 |
| short (<=p33) | 489 | 0.7896 | 0.7866 | 0.8015 | 0.8059 | 0.8000 | 0.7985 | **0.8102** | 0.8046 |
| medium (p33-p66) | 372 | 0.6809 | 0.6785 | 0.6785 | 0.6809 | 0.6738 | 0.6855 | 0.6691 | 0.6764 |
| **long (>p66)** | 339 | 0.4661 | 0.4730 | 0.4798 | 0.4798 | 0.4342 | 0.4798 | 0.4798 | 0.4832 |
| longest 20% | 222 | 0.4086 | 0.3971 | 0.4086 | 0.3855 | 0.3676 | 0.4143 | 0.3971 | 0.4126 |
| no_intent | 148 | 0.5769 | 0.5837 | 0.5769 | 0.6296 | 0.5905 | 0.6103 | 0.6233 | 0.6229 |
| mesh | 192 | 0.7243 | 0.7114 | 0.7327 | 0.7114 | 0.7071 | 0.7327 | 0.7157 | 0.7271 |
| blackboard | 128 | 0.7071 | 0.7200 | 0.7327 | 0.7389 | 0.7005 | 0.7264 | 0.7200 | 0.7188 |

**A measurement problem that must be stated before the numbers are read.** The long
slice contains only *true* `dropped_handoff` runs, so precision there is 1.0 **by
construction** and its F1 collapses to 2R/(1+R). Long-slice "F1" is therefore a
**recall diagnostic**, not a real F1. Two consequences:

- `+reassign` moving it **-0.0319** means recall fell 0.3038 -> 0.2773, about 9
  additional missed long runs. Real, but on 339 runs.
- Bootstrapped as recall, the deltas are: ALL +0.0124 [-0.0206, +0.0472];
  full-age +0.0151 [-0.0177, +0.0472]; full-backlog +0.0351 [+0.0000, +0.0708].
  **Only `full - backlog` has a CI that excludes zero** - and it barely touches it.

`longest 20%` (n=222) never improves in any variant; the single best is `full - age`
0.4126 (+0.0040). The very longest runs remain the hardest cell in the whole project.

Overall `dropped_handoff` F1 improves: 0.5979 -> 0.6079 (ALL) / 0.6106 (full-age) /
0.6160 (full-backlog), with precision and recall both rising (P 0.7190->0.7328,
R 0.5117->0.5233 for full-age).

## 5. Per-class and slices (foundation vs full)

| class | 0_B | 6_ALL | d | class | 0_B | 6_ALL | d |
|---|---|---|---|---|---|---|---|
| clean | 0.8513 | 0.8570 | +0.0056 | conflict | 0.9028 | 0.9004 | -0.0024 |
| dropped_handoff | 0.5979 | 0.6079 | +0.0100 | goal_drift | 0.8383 | 0.8378 | -0.0006 |
| duplicated_work | 0.8136 | 0.8202 | +0.0066 | runaway_loop | 0.7256 | 0.7273 | +0.0018 |
| deadlock | 0.7011 | 0.7057 | **+0.0047** | | | | |

| slice | 0_B | 6_ALL | d |
|---|---|---|---|
| no_intent | 0.6631 | 0.6744 | +0.0113 |
| long (>p66) | 0.7012 | 0.7050 | +0.0038 |
| longest 20% | 0.6735 | 0.6743 | +0.0009 |
| **hard/robustness** | 0.7144 | 0.7298 | **+0.0154** |
| topo_mesh | 0.7762 | 0.7855 | +0.0092 |
| topo_blackboard | 0.7873 | 0.7996 | +0.0122 |

Two things stand out. **Robustness +0.0154 is the largest gain of any Exp06 variant and
the only metric that clearly exceeds its "very good" bar.** And **`topo_mesh`, the slice
that regressed in Exp05 (-0.0082), now improves +0.0092** - the receiver-activity
block appears to have fixed it.

`deadlock` finally edges up (+0.0047) for the first time, though still marginal.

## 6. Confusion

| true -> predicted | B | ALL | d |
|---|---|---|---|
| dropped_handoff -> clean | 210 | 196 | -14 |
| dropped_handoff -> deadlock | 99 | 97 | -2 |
| dropped_handoff -> runaway_loop | 96 | 99 | +3 |
| deadlock -> clean | 150 | 136 | -14 |
| duplicated_work -> clean | 103 | 91 | -12 |

All three target pairs improve or hold, and the feared leak - `dh -> runaway_loop`
rising to justify a falling `dh -> clean` - did not materialise (+3 only).

## 7. Ablation

**Incremental (primary - each block added to the clean 226-column foundation):**

| block | d Macro | d dh | d dh long | d Rob | d comp | folds |
|---|---|---|---|---|---|---|
| age | -0.0019 | -0.0082 | +0.0069 | +0.0034 | -0.0003 | 1/3 |
| backlog | +0.0015 | +0.0063 | +0.0138 | +0.0061 | +0.0025 | 2/3 |
| receiver | +0.0028 | +0.0072 | +0.0138 | +0.0074 | +0.0036 | 2/3 |
| reassign | +0.0029 | -0.0023 | **-0.0319** | +0.0050 | +0.0028 | 3/3 |
| tail | +0.0009 | +0.0044 | +0.0138 | +0.0089 | +0.0030 | 2/3 |

**Drop-one from the full model** (interpret with care - `bd_slope`, `bd_grows_late`,
`bd_tail_growth` and `ua_oldest_share_gt*` are mutually substitutable, so drop-one
understates a block):

| dropped | Macro | d vs foundation | d dh long |
|---|---|---|---|
| age | 0.7827 | +0.0069 | +0.0172 |
| backlog | 0.7817 | +0.0059 | +0.0405 |
| tail | 0.7813 | +0.0055 | +0.0103 |
| receiver | 0.7797 | +0.0039 | +0.0035 |
| reassign | 0.7779 | +0.0021 | +0.0103 |

The two tables disagree in an informative way. Incrementally, `reassign` looks like the
*best* block (+0.0029, 3/3 folds). As a drop-one it looks like the *least* important
(+0.0021). The resolution: the blocks are strongly interdependent - `reassign` looks
good only on top of the other four, and carries `bd_*` and `ra_*` in their place when
they are removed. Neither table should be read in isolation, which is exactly why the
pre-registered full model is the honest primary and it came out *not significant*.

**Importance (full model):** 29 new features take 7.97% of total gain; all 29 receive
non-zero gain; 1 in top-20, 9 in top-50. By block: age 2.55%, backlog 1.85%, receiver
1.92%, reassign 1.19%, tail 0.46%. Top features: `ua_age_p90_rel` (1.10%),
`ua_age_max_rel` (0.69%), `rt_reassign_lat_mean_rel` (0.63%), `bd_max_rel` (0.56%),
`ua_age_mean_rel` (0.46%).

Note the tension: the age block holds the **largest** gain share yet is the only block
that hurts OOF Macro F1. This is precisely why `gain share` was removed as a stopping
criterion in review - importance and OOF effect disagree here, and the metric is what
counts.

## 8. Verification

`verify_no_leakage.py` - 10/10 PASS: no target column read (AST scan); no lexicon in
executable code; 29 declared = 29 produced, finite, no constants; **zero name overlap
with Exp05's 54**; features bit-identical after flipping label / success / fault_turn
(max diff 0); deterministic; `bd_grows_late` = 0 on an empty 0->0 backlog (review fix 1);
`bd_slope` scale-free - identical at 1x and 4x run length (review fix 2); baseline files
unchanged (`features.py` sha256 `72b1373b8c03`).

`test_backlog.py` - 25/25 PASS on synthetic runs with hand-computed expectations,
including the review fix 4 case (a handoff at 95% of the run is excluded from the 80%
cut).

**Two real bugs were caught by these tests during implementation:**

1. `rt_old_owner_result_after` was initially implemented from lifecycle attribution,
   where it could **never** fire: Exp05's greedy LIFO matching books a late delivery
   against the *most recent* assignment, so when the original owner finishes after the
   task moved on, the result lands on the new owner's lifecycle. Rewritten as a direct
   message/artifact-log test. My own unit test caught it - the feature was silently 0
   on a run that obviously should fire.
2. The first long-slice recall bootstrap reported a **zero-width confidence interval**,
   which is impossible. Cause: `np.where` was given a bootstrap index array where a
   boolean mask was expected, silently shrinking the denominator. Fixed with an explicit
   `recall_at` helper; the corrected CI is [-0.0206, +0.0472].

## 9. Threats to validity

- **The headline target failed.** +0.0138 on long-run dh against a required +0.03. Only
  `full - backlog` reaches the vicinity (+0.0405) and it is a drop-one variant, i.e.
  discovered by looking at validation.
- **12 variants were scored on the same OOF run.** `full - age` and `full - backlog` are
  post-hoc selections; their CIs do not account for that search. Treat them as
  candidates for confirmation, not as results.
- **The long-slice "F1" is a recall proxy** (precision is 1.0 by construction). n=339
  implies about +-0.035 on recall. Every long-slice delta except `full - backlog` is
  inside that band.
- `longest 20%` (n=222) never improves; about +-0.065 noise.
- Small dh slices: `no_intent` 148, `mesh` 192, `blackboard` 128, so +-0.05 to 0.07.
- Single seed (`random_state=42`); fold std about 0.009.
- Success F1 held fixed at 0.8644 by design (no Exp06 feature targets success).

## 10. Conclusion

**Направление подтверждено частично, в урезанном виде.**

What worked: `receiver` and `backlog` blocks each lift Robustness (+0.0074, +0.0061)
and both lift long-run dh (+0.0138); the full bundle gives the best **Robustness**
result of the project (+0.0154) and finally moves `topo_mesh` in the right direction
(+0.0092). `deadlock` edges up for the first time (+0.0047). Composite +0.0061 clears
the "very good" bar.

What did not: the primary target is missed (+0.0138 vs +0.03 required), the
pre-registered full model is not statistically significant (p=0.097), and the `age`
block is at best neutral and probably mildly harmful despite holding the largest
gain share. `reassign` is the only block that damages the target it was supposed to
help (-0.0319 on long-run dh).

**Recommended configuration: `full - age` (23 features, 249 columns)** - Macro
0.7827, Robustness 0.7279, composite 0.7456, dMacro +0.0069 with McNemar p=0.0022 and
3/3 folds. This is the only Exp06 configuration that is both statistically supported
and free of a block with a negative sign.

Two things must not be claimed: that the long-run target was achieved, and that
`full - backlog` (+0.0405) is a real result - it is the best of 12 on the same OOF and
needs independent confirmation before it counts.

Suggested next step, not started: re-run `full - age` and `full - age - reassign` on a
**different seed** to separate the confirmed effect from selection noise, and only
then decide whether to fold this block set into the Exp05 foundation.

## Files

- `features.py` - 29 temporal-dynamics features in 5 blocks
- `test_backlog.py` - 25 synthetic-run definition tests
- `verify_no_leakage.py` - 10-check leakage / causality audit
- `common.py`, `run_experiment.py`, `report.py` - 12-variant 3-fold driver
- `stat_common.py`, `significance.py` - bootstrap, McNemar, long-slice recall
- `results.json`, `oof_*.csv` (12), `run.log`, `significance.log`
