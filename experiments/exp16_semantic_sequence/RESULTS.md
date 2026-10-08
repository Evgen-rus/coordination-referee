# Exp16 — frozen semantic event sequence

**Decision: STOP. Exp15 remains the current production baseline.**

The fixed 50:50 candidate missed its predeclared composite gate. Its composite
was `0.7986136119` versus Exp15 `0.7938624419`, a gain of `+0.0047511700`
against the required `+0.005` (short by `0.0002488300`). Macro F1 improved by
`+0.0032377324`, Robustness F1 by `+0.0128625486`, and 2/3 folds improved.
The largest per-class drop was `duplicated_work` at `-0.0143238474`, within the
registered `-0.02` limit. The candidate therefore failed only the composite
threshold; the gate is not relaxed.

The structural-only diagnostic exactly reproduced Exp15. Semantic-only
performance was substantially below baseline. Semantic probabilities and all
fold diagnostics are preserved in `results.json` and the three OOF probability
arrays. No semantic fault-turn model was trained. No production files or ZIP
were changed.

The frozen plan, input correction record, and runtime benchmark are in
`PLAN.md` and `cpu_benchmark.json`.
