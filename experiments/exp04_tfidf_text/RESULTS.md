# Exp04 — Light TF-IDF text signal blended with the Exp03 B tabular model

**Hypothesis.** The baseline uses text almost exclusively as keyword counters. A
light TF-IDF text classifier should add signal that the structural features do not
have — especially where `intent` telemetry is missing — and a blend of the text
model with Exp03 B should beat either model alone.

**Verdict: REJECT as a source of additional score, but KEEP the Robustness gain.
Text-only is far too weak (Macro F1 0.64 vs 0.76). The blend at α=0.50 adds only
+0.0015 Macro F1 (not significant, p = 0.17), but it *does* lift Robustness F1 by
+0.0089 — the first change since the baseline to move that metric. The headline
hypothesis (text rescues `no_intent` and `dropped_handoff`) is **refuted.****

---

## 1. Text representation

`build_docs.py` builds one document per run (mean 4 906 chars, p95 8 350):

```
[GOAL] <goal>
[HANDOFF] [SUB] <intent> [ROLE] <role> <text> [TOOL] <tool> [DELIVERED] ...
[ART] <type> [ARTSTATUS] <status> [STATEKEY] <key> [STATEOP] <op>
[NOKEYWORD] [UNANSWERED_MANY] [REASSIGN] [TIE] [DUP] [DEAD] [CONFLICT] [DRIFT] [LOOP]
```

Design decisions worth stating:

- **Agent IDs are never emitted.** The baseline's `_norm` already turns them into
  `a#`, but I emit `[ROLE]` (orchestrator / analyst / …) instead, since roles are
  meaningful and stable while IDs are arbitrary.
- **Protocol type and intent are injected as special tokens** (`[HANDOFF]`,
  `[SUB] collect_usage_data`), so the linear model can condition on protocol
  context rather than on bare words.
- **Structural events are exposed as presence tokens** (`[NOKEYWORD]`,
  `[CONFLICT]`, `[LOOP]`, …), each computed from protocol structure alone. This
  is deliberate: it tests whether *wording* adds anything beyond the structure,
  and the token list below (§7) shows the model leans on these markers heavily.
- **`label`, `success`, `fault_turn` are never read.** No domain column exists in
  the data, so the shift domain is approximated by a `compliance_audit` goal
  regex (2 293 train runs) — leakage-free, diagnostic only.

## 2. Leakage and control

Same 3 folds as every prior experiment: `StratifiedKFold(3, shuffle=True, random_state=0)`.
No baseline file was modified. `success` was held at the Exp03 B OOF value
(0.8644) as instructed; `fault_turn` uses the official rule-based localiser on each
system's own predicted label, so composite stays honest.

**α selection is honest.** Both the text variant *and* α are picked on an **inner
split of the training fold** (`test_size=0.4`, seed 7, 9 combinations scored). The
validation fold is never used to choose them. The inner pick was unstable
(`word`/0.50, `char`/0.50, `wc`/0.75 across the three folds), which is itself a
finding — see §9. Fixed α = 0.25 / 0.50 / 0.75 are also reported.

## 3. Headline — 3-fold OOF

| system | Macro F1 | Robustness F1 | composite | ΔMacro | ΔRobust | ΔComp | no_intent | longest20% |
|---|---|---|---|---|---|---|---|---|
| baseline_flat (122) | 0.7570 | 0.7178 | 0.7284 | −0.0074 | +0.0024 | −0.0035 | 0.6567 | 0.6635 |
| **exp03_B (185)** | **0.7644** | 0.7154 | 0.7318 | — | — | — | 0.6634 | 0.6677 |
| text_word | 0.6242 | 0.5673 | 0.6150 | −0.1401 | −0.1481 | −0.1168 | 0.5476 | 0.5328 |
| text_char | 0.6173 | 0.5739 | 0.6123 | −0.1471 | −0.1415 | −0.1195 | 0.5323 | 0.5366 |
| text_wc (word+char) | 0.6429 | 0.5896 | 0.6306 | −0.1215 | −0.1258 | −0.1013 | 0.5591 | 0.5535 |
| blend α=0.25 | 0.7262 | 0.6807 | 0.7005 | −0.0382 | −0.0347 | −0.0313 | 0.6384 | 0.6451 |
| **blend α=0.50** | **0.7659** | **0.7243** | **0.7345** | +0.0015 | **+0.0089** | **+0.0027** | 0.6663 | 0.6720 |
| blend α=0.75 | 0.7665 | 0.7206 | 0.7341 | +0.0021 | +0.0052 | +0.0023 | 0.6580 | 0.6700 |
| **blend_honest** | 0.7659 | 0.7224 | 0.7341 | +0.0015 | +0.0070 | +0.0023 | 0.6631 | 0.6719 |

`blend_honest` = per-fold inner-selected (variant, α), never tuned on validation.
It lands within 0.0001 of fixed α=0.50, so the α choice is not doing the work.

**Text-only is decisively weak: −0.12 to −0.15 Macro F1.** No configuration of a
linear model on TF-IDF comes close to the 185-feature tabular model.

## 4. Fold-by-fold (the "2 of 3 folds" criterion)

| system | fold 0 | fold 1 | fold 2 | Δ vs exp03_B | folds won |
|---|---|---|---|---|---|
| exp03_B | 0.7516 | 0.7754 | 0.7661 | — | — |
| text_wc | 0.6361 | 0.6502 | 0.6417 | −0.1154 / −0.1252 / −0.1244 | 0/3 |
| blend α=0.50 | 0.7558 | 0.7716 | 0.7703 | +0.0042 / **−0.0038** / +0.0042 | **2/3** |
| blend_honest | 0.7558 | 0.7716 | 0.7701 | +0.0042 / **−0.0038** / +0.0040 | **2/3** |

The blend wins 2 of 3 folds (fold 1 is slightly negative). Consistent with a real
but very small effect.

## 5. Significance — the effect is NOT significant

Paired bootstrap (2 000 resamples) and exact McNemar, blend_honest vs exp03_B:

| test | result |
|---|---|
| Δ Macro F1 | **+0.0015**, 95% CI **[−0.0019, +0.0051]** |
| P(Δ > 0) | 0.793 |
| McNemar | fixed 154, broke 130, net **+24**, two-sided **p = 0.1722** |
| McNemar for fixed α=0.50 | fixed 207, broke 179, net +28, p = 0.1693 |

The confidence interval **includes zero** and McNemar is far from significant.
Compare with Exp03 B vs baseline, which gave p = 0.0068 and a CI excluding zero.
**By the same evidence standard, the text blend does not earn its place on Macro F1.**

## 6. The hypothesis is refuted on exactly the slices it targeted

| slice | n | exp03_B | text_word | text_char | text_wc | blend |
|---|---|---|---|---|---|---|
| **no_intent** | 1 198 | 0.6634 | 0.5476 | 0.5323 | 0.5591 | 0.6631 (**−0.0003**) |
| has_intent | 8 802 | 0.7762 | 0.6334 | 0.6267 | 0.6527 | 0.7778 |
| hard/robustness | 1 290 | 0.7154 | 0.5673 | 0.5739 | 0.5896 | **0.7224 (+0.0070)** |
| compliance_audit | 2 293 | 0.7533 | 0.6148 | 0.6314 | 0.6386 | 0.7625 |
| main_domain | 7 707 | 0.7678 | 0.6269 | 0.6127 | 0.6440 | 0.7670 |

**`no_intent` is where the text model is weakest, not strongest** (0.55 vs 0.66),
and the blend changes it by −0.0003 — literally nothing. Text *cannot* rescue
missing-intent runs, which makes sense: the reason those runs are hard is that
the *structural* field is missing, and text cannot reconstruct a missing
`intent` field from prose.

**`dropped_handoff` recall tells the same story:**

| system | recall (n = 1 200) |
|---|---|
| baseline_flat | 0.5042 |
| exp03_B | 0.4992 |
| text_word | 0.3383 |
| text_wc | 0.3783 |
| blend_honest | 0.4942 |

The text model is 12–17 points *worse* at the class the hypothesis was supposed to
fix, and the blend does not move it. `dropped_handoff` is defined by an assignment
that was never answered — an absence. Text models are poor at absence, which is
precisely what the 122 structural features already encode.

**Two sub-goals, two refutations.** Per the task's own rule — "если текст не помогает
no_intent, одна из главных гипотез считается опровергнутой" — the main hypothesis
is refuted.

## 7. Why: the text model is not really reading text

Class-specific features from the fold-0 word model:

| class | most distinctive features |
|---|---|
| clean | `[DRIFT] [REASSIGN]`, `@ [HANDOFF]`, `final [ART]` |
| dropped_handoff | `[NOKEYWORD]`, `[NOKEYWORD] [REASSIGN]`, `[TIE] [UNANSWERED_MANY]`, `understood` |
| duplicated_work | `[DUP]`, `[DUP] [NOKEYWORD]`, `[DELIVERED] [HANDOFF]` |
| deadlock | `[STATUS]`, `[STATUS] [SUB]`, `[DELIVERED] [STATUS]`, `for a#` |
| conflict | `override [STATEKEY]`, `override [STATEOP]`, `that contradicts`, `disagree` |
| goal_drift | `[SUB] deliver`, `the_datacenter_migration`, `the_office_relocation` |
| runaway_loop | `db_query`, `[TOOL] db_query`, `query returned`, `another pass` |

The top discriminative features are overwhelmingly **my own structural marker
tokens and protocol type tokens** — `[DUP]`, `[NOKEYWORD]`, `[STATUS]`,
`override`, `db_query`. Almost none are ordinary English words describing *meaning*.

**The text model is rediscovering the structural features through a lossy text
channel.** `override` and `[DUP]` are exactly what `state_overrides` and
`dup_hash_pairs` already encode, and the Exp03 B tabular model already has them
cleanly. That is the whole explanation for why text-only is 12 points worse and
why blending adds almost nothing: it is not new information.

Coefficient-mass concentration (share of total |coef| mass):

| model | top-20 | top-100 | top-1000 | top-5000 |
|---|---|---|---|---|
| word | 1.8% | 5.8% | 28.7% | 79.3% |
| char | 1.6% | 6.2% | 34.2% | 88.7% |
| word+char | 1.0% | 3.3% | 18.4% | 56.9% |

The mass is **spread over thousands of features, not concentrated in a few template
words.** That is the opposite of the overfitting-to-templates worry, and it means
the model is not memorising a handful of class-specific phrases — it is learning a
diffuse, heavily redundant representation. `char` is the most diffuse of the three
(top-1000 = 34.2% in 6 950 features), i.e. the most over-parameterised.

## 8. word vs char — word is the more robust choice

| | word (1-2) | char_wb (3-5) | word+char |
|---|---|---|---|
| Macro F1 | 0.6242 | 0.6173 | **0.6429** |
| Robustness F1 | 0.5673 | **0.5739** | 0.5896 |
| no_intent | **0.5476** | 0.5323 | 0.5591 |
| vocabulary | 8 569 | 6 950 | 15 519 |
| fit / fold | **7.6 s** | 36.1 s | 47.2 s |
| infer / fold | **1.6 s** | 12.5 s | 14.0 s |

The scenario the task warned about — "word scores high on CV but transfers worse
than char" — **did not occur here**, because word never scored high in the first
place. word beat char on ordinary Macro F1 (+0.0069) and on `no_intent`
(+0.0153), while char edged it on Robustness (+0.0066). The gap is small and
inconsistent in direction.

**word is preferable on cost: 5.5× faster to fit, 8× faster at inference, at
equal quality.** char's only advantage is Robustness, and it is a 0.0066 edge
inside a model that is 0.14 behind — not worth the runtime or the diffuse
feature space. word+char is the best text-only model but costs the most and still
trails tabular by 0.12.

## 9. Complementarity: real, but unexploitable at this blend

The text model is right where the tabular model is wrong on **615–649 of 2 184
tabular misses (28–30%)** — genuine complementarity. An oracle that falls back to
text only on tabular errors would reach Macro F1 **0.8339** (vs 0.7644). Mean
agreement with Exp03 B is 70.7%.

So a large amount of complementary signal demonstrably exists — the failure is
that a *fixed convex blend* cannot exploit it. Probability averaging pulls the
tabular model toward a much weaker classifier instead of deferring to it only
where it is uncertain. A gated/hierarchical combination (text consulted only on
low-confidence tabular runs) is the obvious follow-up, and it is precisely the
family Exp02 ruled out — worth revisiting only with fresh evidence.

The inner-fold α selection being unstable (`word`/0.50, `char`/0.50, `wc`/0.75)
reinforces this: there is no stable optimum to tune toward.

## 10. Success criteria scorecard

| criterion | target | actual | met? |
|---|---|---|---|
| Macro F1 rises vs Exp03 B | ≥ +0.005 | **+0.0015** (CI includes 0) | ❌ |
| Robustness F1 not worse | ≥ −0.003 | **+0.0089** | ✅ |
| no_intent slice rises noticeably | — | **−0.0003** | ❌ |
| effect on ≥ 2 of 3 folds | 2/3 | 2/3 | ✅ |
| *very good:* Macro | ≥ +0.01 | +0.0015 | ❌ |
| *very good:* Robustness | ≥ +0.01 | +0.0089 | ❌ |
| *very good:* dropped_handoff | ≥ +0.02 | **−0.0010** | ❌ |

1 of 4 required criteria met, 0 of 3 bonus thresholds met.

## 11. Runtime and memory — the practical verdict

| model | vocabulary | nnz/run | sparse bytes/run | fit/fold | infer/fold |
|---|---|---|---|---|---|
| text_word | 8 569 | 484 | 3.9 kB | **7.6 s** | **1.6 s** |
| text_char | 6 950 | 2 101 | 16.8 kB | 36.1 s | 12.5 s |
| text_wc | 15 519 | 2 585 | 20.7 kB | 47.2 s | 14.0 s |

- **Full driver wall clock: 753 s (12.5 min)** — within the 30-minute limit.
- **Peak process RSS: 1 293 MB (1.26 GB)** — far below the 32 GB limit.
- Vocabulary sizes are small (7–16 k) because `min_df=5` prunes hard; the
  `max_features=60 000` cap never binds. No million-feature dictionary was created.
- Fitting is dominated by `LogisticRegression` with lbfgs on 17 M non-zeros.
  Converting to `SGDClassifier(loss="log_loss")` or `liblinear` would cut this
  sharply, but the whole text branch is being rejected on accuracy grounds, so
  optimising it is pointless.

Resource cost is **not** the reason to reject text. Accuracy is.

## 12. Threats to validity

- **The text branch is a single configuration per variant.** `C=4.0`, lbfgs,
  `min_df=5` were fixed a priori and never tuned, per instructions. A tuned text
  model might close part of the 0.12 gap — but the §7 diagnostic suggests the gap
  is structural (the model has no access to information the tabular model lacks),
  not a hyperparameter artefact.
- **My document representation is one reasonable choice.** A different one might
  expose more signal. The bracket tokens were designed to *help* text, not to
  handicap it.
- **`no_intent` has n = 1 198**, so slice estimates carry ≈ ±0.03 noise. The
  −0.0003 blend delta is a true null, but ±0.01 movements there are not meaningful.
- **Single seed** (`random_state=42`); fold std ≈ 0.009, which is 6× the blend
  effect.
- **Blend effect sizes (+0.0015 Macro) are below the resolution of this CV setup.**
  Even if real, they are not measurable here.

## 13. Conclusion

**Отказаться от текста как источника score; оставить blend α=0.50 только как
кандидат на Robustness — и то с оговорками.**

Reasoning:

1. **The headline hypothesis is refuted on its own terms.** Text does not help
   `no_intent` (−0.0003) and does not help `dropped_handoff` (−0.0010, recall
   0.50 → 0.49). Both target classes got worse or stayed flat.
2. **The blend's Macro gain is not significant** (+0.0015, CI [−0.0019, +0.0051],
   p = 0.17) — an order of magnitude weaker than Exp03 B's p = 0.0068.
3. **The mechanism is understood, and it is bad news.** The text model's top
   features are its own structural markers, and 28–30% of its correct answers land
   on tabular misses, so it is a lossy re-encoding of signals the tabular model
   already holds cleanly.
4. **The one real gain is Robustness F1 +0.0089**, the first upward move on that
   metric since the baseline. It is below the +0.01 target and not statistically
   established, but it is the *only* reason to keep the branch, and it is
   directionally consistent across the shift slices.
5. **Ablation discipline says keep it:** α=0.25 (text weight 0.75) collapses to
   0.7262, so a badly-weighted blend is actively harmful. α=0.50–0.75 is a narrow
   safe band.

**Recommendation:** do **not** adopt text for Macro F1. If the contest's
distribution shift is as severe as `docs/data_schema.md` describes (33 % of test
runs out-of-distribution, "другой набор формулировок"), retain blend α=0.50 as a
*candidate* for the final submission and validate it once more on a second seed
before committing. Otherwise drop the branch and spend the remaining effort on the
classes that are actually weak and that text provably cannot reach:
`dropped_handoff` (0.587) and `no_intent` (0.663).

**Not pursued, and why:** no LightGBM-on-sparse, no TF-IDF tuning, no manual token
pruning after inspecting validation errors (that would be CV tuning), no
embeddings, no LLM, no gate-based combination (Exp02 territory).

## Files

- `build_docs.py` — target-free document construction
- `run_experiment.py` — 3-fold driver for all 9 systems, honest inner-fold α
- `diagnostics.py` — per-fold table, bootstrap + McNemar, complementarity, no_intent dive
- `mem.py` — ctypes/psapi peak-RSS probe (no `psutil` in this venv, no pip)
- `results.json`, `oof_*.csv` (9 files), `run.log`, `diagnostics.log`
