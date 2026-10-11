# Exp20 — preflight STOP

Control A did not reproduce frozen Exp15 baseline

Control A was fitted on the fixed outer folds. Candidate B was not trained.

| Metric | Control A | Frozen Exp15 | Absolute difference |
|---|---:|---:|---:|
| composite | 0.794610790320 | 0.793862441851 | 0.00074834846905 |
| fault_turn_hit2 | 0.631388888889 | 0.631527777778 | 0.000138888888889 |
| macro_f1 | 0.813755843572 | 0.812954558674 | 0.000801284898591 |
| robustness_f1 | 0.761430748866 | 0.759984369232 | 0.00144637963457 |
| success_f1 | 0.894908616188 | 0.894908616188 | 1.11022302463e-16 |

- Label prediction mismatches: 47 / 10000.
- Mismatch counts by outer fold: {'0': 47, '1': 0, '2': 0}.
- L1 fault-turn outputs identical: False.

No candidate CV was run. Exp15 production remains unchanged.

## Control A by fold

| Fold | Rows | Frozen composite | Control A composite | Delta | Frozen Macro | Control A Macro | Frozen Robustness | Control A Robustness | Frozen hit@2 | Control A hit@2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3334 | 0.7905699305 | 0.7928372676 | +0.0022673371 | 0.8013657954 | 0.8037982873 | 0.7762703535 | 0.7806413848 | 0.6179166667 | 0.6175000000 |
| 1 | 3333 | 0.7987070993 | 0.7987070993 | +0.0000000000 | 0.8197832754 | 0.8197832754 | 0.7624417787 | 0.7624417787 | 0.6358333333 | 0.6358333333 |
| 2 | 3333 | 0.7908629737 | 0.7908629737 | +0.0000000000 | 0.8176619496 | 0.8176619496 | 0.7355506816 | 0.7355506816 | 0.6408333333 | 0.6408333333 |

## Runtime

- Corrected incumbent matrix build: 419.3 s.
- Control A fits: 67.3 s total.
- CPU production estimate: 602.8 s / 1800 s; benchmark peak RSS 590.6 MB.
- This research preflight timing is not a full candidate CV or production run.
