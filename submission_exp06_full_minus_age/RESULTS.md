# Submission — Exp06b `full − age` (249 features)

Production build for DSWorks. **Not an experiment.** Nothing was tuned, added or
dropped; this reproduces the configuration validated in `experiments/exp06b_stability`.

## Heads and their feature sets

| head | feature set | count | why |
|---|---|---|---|
| `label` | 122 baseline + 63 Exp03 norm + 41 Exp05 lifecycle + 23 Exp06 temporal | **249** | Exp06b `full − age`; the 6 `ua_` age features are dropped as the one block with a negative incremental effect |
| `success` | 122 baseline + 63 Exp03 norm | **185** | **Exp03 B, deliberately not 249** — see below |
| `fault_turn` | — | — | official baseline `localize` on the *predicted* label, unchanged |

### Why success stays at 185 and is not widened to 249

Every Exp05/Exp06/Exp06b comparison **pins** success F1 to the Exp03 B value of
0.8644, because no Exp05 or Exp06 feature targets `success` — refitting it would add
noise without informing the comparison. Widening the success head to 249 features here
would ship an unvalidated change to a head carrying 0.15 of the composite score, and
would break the correspondence between the shipped model and every number in
`experiments/results.csv`. Kept at 185 on purpose.

## Verification

| check | result |
|---|---|
| label matrix == experiment, 249 cols, same order | **max abs diff 0.0** |
| success matrix == experiment, 185 cols, same order | **max abs diff 0.0** |
| no `ua_` (age) feature in the label matrix | pass |
| imports resolve from a temp dir outside the repo | pass |
| no absolute local paths, no network, no data files | pass (all 6 modules, AST-scanned) |
| full run on demo 10k/4k | 74 s, exit 0 |
| `scripts/validate_submission.py` | `OK: 4000 rows, format valid` |
| smoke run, test with `run_id` only | valid constant fallback, exit 0 |
| run from an **extracted ZIP** outside the repo | exit 0, predictions identical |
| baseline / evaluation / data modified | **no** (`git diff` empty) |

Reproduce with:

```bash
python scripts/check_submission_parity.py        # feature + import + leakage parity
python scripts/compare_submission_to_oof.py     # prediction sanity vs validated OOF
```

## Vendored modules and provenance

| file | copied from | change |
|---|---|---|
| `features.py` | `baseline/features.py` | none (identical) |
| `localize.py` | `baseline/localize.py` | none (identical) |
| `new_features.py` | `experiments/exp03_length_normalization/new_features.py` | import lines only |
| `lifecycle.py` | `experiments/exp05_handoff_lifecycle/lifecycle.py` | import lines only |
| `temporal_features.py` | `experiments/exp06_temporal_dynamics/features.py` | import lines only |
| `solution.py` | new — assembly of the three validated matrices | — |

The three modified files differ from their sources **only** in `sys.path` bootstrap
lines (repo-relative → local directory). No feature logic was edited; this is
confirmed by exact-value parity against the experiment matrices.

## Archive

`submission_exp06_full_minus_age.zip` — 24 795 bytes, 6 entries, **flat, `solution.py`
at the archive root** (no wrapping folder).

## Known behaviours, not bugs

- 3 of 4000 rows are `conflict` with `fault_turn = -1`. The official `localize` looks
  for an `op == "override"` state entry and falls through to `-1` when a key was only
  ever `set`. Verified on those exact runs; this is baseline behaviour, unchanged.
- Label distribution vs the validated seed-0 OOF: total variation distance 0.0132.

## Not yet done

The ZIP is **built but not uploaded**. DSWorks accepts 5 submissions/day, 30 min/job,
no network — so the upload is a deliberate separate step.
