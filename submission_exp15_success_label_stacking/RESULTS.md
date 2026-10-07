# Exp15 production submission

Built from the packaged Exp13 submission. The label head remains 312 features
with the Exp13 wait-graph block; `fault_turn` keeps the Exp13 L1 window peak.

The success head uses the existing 185 Exp03 features plus the seven raw Exp13
label probabilities in `LABELS` order, for 192 features. Its LightGBM model and
parameters are unchanged. Training probabilities are honest 3-fold OOF
(`StratifiedKFold`, seed 1). Each fold rebuilds the Exp13 312-feature label
input with its own 3-fold honest window cross-fit; the held-out label
probabilities are produced by a label model that excludes those rows. Test
probabilities come from the final Exp13 label model trained on all training
rows.

Research acceptance was recorded in
`experiments/exp15_success_label_stacking/RESULTS.md`: success F1 improved from
0.8643757406 to 0.8949086162, composite from 0.7892825105 to 0.7938624419,
and all three folds improved. Label, fault-turn, macro, robustness, and hit@2
were invariant in the research comparison.

Packaging verification is recorded with the produced
`submission_exp15_success_label_stacking.zip`. The archive contains only the
standalone runtime modules and this report. It is not uploaded automatically.

## Verification

- Unit contract tests: 2 passed; Exp13 static parity: passed.
- Smoke standalone run (1,200 train → 400 test): 122.8 s; validator passed.
- Full standalone run (10,000 train → 4,000 test): 593.4 s, under the 1,800 s
  limit; validator passed. `label` and `fault_turn` match Exp13 on all 4,000
  rows.
- ZIP size and SHA-256 are recorded in `experiments/results.csv`.
- Archive checks passed: flat runtime-only layout, no datasets, absolute paths,
  or network references; extracted standalone execution and fallback passed.
