# Exp20 Control A fold-0 diagnostic

**Status: cause not proven; Candidate B remains blocked.** This is a diagnostic-only follow-up. The original Exp20 `results.json`, `RESULTS.md`, `run.log`, frozen Exp13 OOF and production files were not changed.

## Finding

The saved original Control A output has 47 changed labels, all in fold 0. These are not float precision flips: casting Control A probabilities to `float32` leaves all 47 changes. Median frozen top-two margin is 0.081549; 15/47 frozen margins are at most 0.05. The median per-row maximum absolute probability change is 0.091594, and 42/47 exceed 0.05. Folds 1 and 2 have zero label changes and max probability difference about `3e-8`, consistent with the saved `float32` vs Control A `float64` representation.

The 47 IDs and per-row class/probability details are in `original_control_a_mismatches.csv`. There are 24 distinct directed class transitions.

## Stage review

| Stage | Evidence | Result |
|---|---|---|
| A. Data/order/folds | Verified Exp08 manifest data SHA, Exp13/Exp20 run IDs and all three seed-0 fold validation hashes | Exact match |
| B. Foundation, 249 columns | No historical foundation matrix is saved. Current verified content-addressed cache SHA: `9d2ec053fd688f448f6643d779fcbf0404279151c5dc1e4fe2df65d6daeb87c0`; identical in both replays | Cannot compare historically |
| C. Window inputs/predictions | No historical inner/outer window probability matrices are saved. Both replays produced bit-identical inner probability arrays and outer-validation window probabilities | No replay instability observed; cannot compare to old Control A run |
| D. Window aggregates, 51 columns | Historical matrices absent. All four captured aggregate output hashes matched exactly between replays | No replay instability observed; cannot compare historically |
| E. Exp13 `wg_*`, 12 columns | No historical matrix saved separately. Current frozen source rebuilt deterministically with SHA `8009571285716226972e8d358e019ff486cadcb8d14f9bddaa4b6e486081a05c` | Cannot compare historically |
| F. Final 312-feature classifier | Original saved OOF differs in 47 fold-0 rows; two isolated replays produced the frozen classes on all 3,334 fold-0 rows | First observable historical divergence is final OOF; internal first divergence remains unknown |

Each isolated replay used Exp13's existing `corrected_fold_matrices` generator and stopped after its first yield, so only fold 0 was built. The generated 300- and 312-column fold-0 matrix hashes were identical between repeats. The final probability arrays were also bit-identical between repeats.

## Isolated replay and baseline check

Both fold-0 replays had **0/3,334** label mismatches vs frozen Exp13 and max absolute probability difference `2.97988279607e-08`. The two replay probability matrices were bit-identical. Combining one replay's fold 0 with the already saved Control A folds 1 and 2 yields 0/10,000 class mismatches and 0 L1-turn mismatches; official metric recomputation matches Exp15 within `1e-12` for all five metrics. This is a diagnostic reconstruction, not a fresh three-fold Control A execution.

| Metric | Reconstructed | Frozen Exp15 | Absolute delta |
|---|---:|---:|---:|
| Macro F1 | 0.812954558674 | 0.812954558674 | 0 |
| Robustness F1 | 0.759984369232 | 0.759984369232 | 0 |
| Success F1 | 0.894908616188 | 0.894908616188 | 1.11e-16 |
| hit@2 | 0.631527777778 | 0.631527777778 | 0 |
| Composite | 0.793862441851 | 0.793862441851 | 0 |

## Determinism assessment

Current runtime: Python 3.11.16, LightGBM 4.7.0, NumPy 2.4.6, scikit-learn 1.9.1, joblib 1.6.0. `n_jobs=-1`; inner fits use the existing three joblib workers; `OMP_*`, MKL, OpenBLAS and related thread environment variables were unset. The content-addressed Exp07 caches were hits and their code/data fingerprint matches the verified manifest. Control A stores float64 probabilities while the frozen artifact stores float32; precision conversion does not explain the original 47 flips.

Exp20 currently materializes all three fold matrices before fitting the final Control A classifiers, while the Exp13 confirmation caller fits its heads after each yielded fold. This is a code-order difference to retain in the next audit, not proof of cause. Exp14 already documents fold-sensitive reproduction drift under other Python-process load and a prior rounded composite drift of `7.48e-04`, close to this Exp20 drift (`0.000748348469`). The original Exp20 run did not save process-load or per-stage hashes, so contention cannot be established retroactively. The two clean replays plus Exp14 make execution-context-sensitive window-model behavior the leading explanation, but it remains an inference.

## Verdict

1. **Cause proven? No.** Evidence strongly points to execution-context-sensitive nondeterminism, but the original run lacks workload and intermediate-stage records.
2. **First differing stage?** First *observable* difference is final fold-0 OOF probability/class output. Historical B–E matrices are absent, so the first internal stage is unknown.
3. **Nature of 47 changes?** Substantial probability shifts, not rounding-only near ties; see the CSV for all rows.
4. **Isolated fold 0?** Two identical replays recovered frozen labels exactly and matched each other bit for bit.
5. **Next step / safe continuation?** Do not change model parameters or features. Before Candidate B, require a full Control A-only 3-fold replay under recorded, uncontended conditions with per-fold input, window-probability, aggregate and final-matrix hashes, exact 10,000-row labels/L1 and five-metric parity. No Candidate B was trained here.
