# Exp19 StepFinder-inspired fault-turn localizer

**Decision: STOP.** The frozen candidate improves full-OOF `fault_turn_hit@2`
by `+0.019444`, below the required `+0.05`, and Composite by `+0.001944`, below
`+0.005`. It also drops class hit@2 by `0.280000` for `dropped_handoff` and
`0.075833` for `duplicated_work`, exceeding the `0.05` per-class limit.
Settings were not changed after seeing results. Exp15 production is unchanged;
no production ZIP was created and nothing was submitted to a leaderboard.

## Frozen run

- Plan and gate were committed before CV (`a909c8a`); implementation and tests
  were committed before CV (`7fb8a42`).
- Reused the verified Exp16 frozen MiniLM 384d event cache; encoder recomputed:
  **no**. It contains 590,146 ordered events, including 457,203 message turns.
- Baseline parity passed: Composite `0.7938624418508566`, hit@2
  `0.6315277777777778`, Macro `0.8129545586738590`, Robustness
  `0.7599843692318033`, Success `0.8949086161879896`. Run order, labels, and
  success predictions matched the frozen Exp15 OOF file exactly.
- The only scored change was the fault-turn localizer. Fold-local models used
  12 fixed epochs; each validation class came from frozen Exp15 OOF predictions.
  There was no validation-based checkpoint selection or tuning.
- The adaptation uses frozen event content projected `384 -> 128`, 32d role
  and class embeddings, two-layer BiLSTM, agent-aware 2 x 32 attention,
  multi-scale differences at `[1, 2]`, message-only step scores, and temporal
  consistency loss. It uses `gamma=0`; artifacts/state events add context but
  cannot be selected as fault turns.

## Full OOF metrics

| Metric | Exp15 L1 baseline | Exp19 candidate | Delta |
|---|---:|---:|---:|
| Macro F1 | 0.812955 | 0.812955 | 0 |
| Robustness F1 | 0.759984 | 0.759984 | 0 |
| Success F1 | 0.894909 | 0.894909 | 0 |
| Fault-turn hit@2 | 0.631528 | 0.650972 | +0.019444 |
| Composite | 0.793862 | 0.795807 | +0.001944 |

The candidate improves hit@2 in all three folds:

| Fold | Baseline hit@2 | Candidate hit@2 | Delta | Composite delta |
|---:|---:|---:|---:|---:|
| 0 | 0.617917 | 0.634167 | +0.016250 | +0.001625 |
| 1 | 0.635833 | 0.661250 | +0.025417 | +0.002542 |
| 2 | 0.640833 | 0.657500 | +0.016667 | +0.001667 |

## Hit@2 by true fault class

| Class | Exp15 L1 | Exp19 | Delta |
|---|---:|---:|---:|
| dropped_handoff | 0.480833 | 0.200833 | -0.280000 |
| duplicated_work | 0.745833 | 0.670000 | -0.075833 |
| deadlock | 0.703333 | 0.744167 | +0.040833 |
| conflict | 0.691667 | 0.785000 | +0.093333 |
| goal_drift | 0.575833 | 0.849167 | +0.273333 |
| runaway_loop | 0.591667 | 0.656667 | +0.065000 |

The gate passed only the fold-win condition (3/3). It failed both aggregate
improvement thresholds and the per-class deterioration condition. Labels and
success predictions were unchanged; fold overlap and validation-selection
checks passed.

## Runtime and verification

- CPU throughput benchmark: 256 representative faulty runs, one epoch in
  `7.47 s`; projected 3-fold CV `71.45 min`, below the `120 min` start limit.
- Actual CV: `2346.93 s` (`39.12 min`); peak process RSS `1067 MB`.
- Cache audit confirmed train CSV and embedding SHA-256, event order, offsets,
  metadata, and all fault-turn-to-message mappings. No MiniLM pass was run.
- Unit tests: **8 passed**. The tests cover event alignment/order, target
  perturbation invariance, future-event isolation, 200+ messages, fold overlap,
  class conditioning, message-only output, clean -> `-1`, deterministic
  inference, official metric behavior, and Exp15 baseline parity.
- Production packaging/runtime feasibility was not evaluated because the CV
  gate failed.

Machine-readable metrics and OOF turn outputs are in `results.json`,
`oof_predictions.csv`, `baseline_fault_turn.npy`, and
`candidate_fault_turn.npy`.
