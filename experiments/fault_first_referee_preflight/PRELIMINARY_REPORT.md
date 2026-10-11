# Fault-First Referee — pre-CV feasibility review

## Superseding one-shot plan — 2026-10-11

The earlier STOP below assumed the candidate had to train inside the contest
job. The user has now authorized local pretraining and inference-only packaged
weights; the official task FAQ confirms this is permitted. The old measured
benchmark is unchanged. `PLAN.md` supersedes the old gate and runtime decision:
the fixed candidate package estimate is 653.4 seconds, or 668.4 seconds with a
25% margin on candidate input/inference. No Exp21 CV result had been observed
when that plan was committed.

**Status: STOP before CV.** The root-cause review is current and
`protocol_guard.py` permits a next experiment, but this prototype fails the
measured CPU runtime gate. No OOF training, model selection, production edit,
ZIP, or leaderboard activity occurred. `preflight.json` contains the raw
benchmark measurements.

## Task and data findings

- The task predicts one ontology-selected root-cause class. The official task
  page states the fixed priority when several symptoms occur and defines
  `fault_turn` as the first message of the root failure. Success is a separate
  target. The platform run is offline, has a 1,800-second limit, and may offer
  a GPU profile for neural solutions; no particular GPU or timing is promised.
  [Official task page](https://dsworks.ru/en/champ/aij26-referee)
- The checked-in `data/train.csv` is a 10,000-row demo, not the platform's real
  train. It has 2,800 clean rows, 1,200 per fault class, 7,200 valid faulty
  `fault_turn` annotations, and no turn outside its message sequence. It has
  590,146 total goal/message/artifact/state events; median 56 and maximum 189
  events per run. Local observed topology values are `star`, `pipeline`,
  `hierarchical`, `mesh`, and `blackboard`. The official page describes a
  different four-name topology set, so the actual job input must be inspected
  rather than assuming that the demo vocabulary transfers.
- Observable ontology cues exist for all six faults: unreturned assignments,
  duplicated subtask delivery, reciprocal waits with later progress or no
  progress, conflicting shared-state writes, sustained departure from the
  goal, and repeated actions without new artifacts/state. The ambiguity is
  real: healthy short iteration overlaps `runaway_loop`; a one-way missing
  delivery overlaps `deadlock`; about a quarter of faulty examples may have a
  secondary anomaly; and telemetry loss obscures intents, refs, and artifact
  subtasks. Those cues support a hypothesis, not perfect identification.

## What the earlier work establishes

- Exp01's per-message argmax ranker reduced hit@2 from `0.4074` to `0.1964`.
  It did not encode the rule “choose the first qualifying event” and used
  pointwise causal inputs.
- Exp16 already used a bidirectional GRU over complete events for label
  classification. Its frozen MiniLM 50:50 candidate gained `+0.004751`
  Composite, below its `+0.005` gate; semantic-only Composite was `0.638624`
  and the fold deltas were uneven. Its CPU encoder pass took `4,822.9 s`.
  Therefore future context alone is not a new hypothesis, and MiniLM cannot be
  repeated under the measured CPU budget.
- Exp19 changed only localization. It raised hit@2 by `+0.019444` and
  Composite by `+0.001944`, below both gates, and caused large classwise hit@2
  losses for `dropped_handoff` and `duplicated_work`.
- Exp13's saved error audit identifies 507 `dropped_handoff` errors and 346
  `runaway_loop` errors, and found that episode identity/onset was absent from
  aggregate features. It also found that a simple episode-start rule localized
  worse than L1. Exp14's aggregate episode block did not improve
  `runaway_loop`. Thus this prototype tests a joint learned sequence objective,
  not another onset rule or aggregate episode block.
- Exp20 is still a technical STOP before Candidate B: historical Control A
  differed on 47 fold-0 labels. Two later isolated fold-0 replays matched
  frozen labels but did not explain the historical divergence. No graph score
  is interpretable from Exp20.

## Fixed prototype under preflight

The prototype is deliberately different from Exp16/19 in its supervision and
input channel: it learns from the raw typed event stream without MiniLM and
shares one sequence encoder between root-class and onset losses.

1. Input is a separate goal event followed by every message, state write, and
   artifact, sorted by numeric `t`, then fixed type order `message → state →
   artifact`, then source order. Message source `t` is checked against its
   original turn. No event-count cap is used.
2. Each event gets deterministic hashed word-token IDs (32,768 buckets) from
   its typed text. Frozen event text preserves message type, sender/receiver
   roles, raw intent, tool and text; artifact producer role/subtask/type/status;
   and state writer role/key/op/value. No raw agent or artifact IDs become
   language tokens. Run-local categorical slots keep agent, task, and artifact
   identity links; refs and Exp13's fixed six-template wait parser are aligned
   to the event. Twelve time/type/intent/ref/progress fields are carried as
   metadata. All input construction whitelists observable fields and excludes
   `label`, `success`, `fault_turn`, and `run_id`.
3. A trainable 64-d hashed-token EmbeddingBag and categorical embeddings feed a
   128-d event projection, a 2-layer bidirectional GRU (hidden 64 each way),
   and learned attention pooling. The pooled vector predicts 7 classes.
   Message events also receive six class-conditioned onset scores; non-message
   events can never be returned as the fault turn.
4. Training uses `CE(label) + 0.5 * CE(onset)` for faulty training rows only.
   The onset head is trained with the training example's true class. At
   validation/inference, the predicted class selects its onset channel; clean
   returns `-1`. Exp15 success OOF predictions remain fixed.
5. Frozen settings: 12 epochs, batch 16, AdamW `1e-3`, weight decay `1e-4`,
   encoder dropout `0.15`, head dropout `0.2`, fold seed 0 with the existing
   three StratifiedKFold splits, model seeds 42/43/44. No early stopping or
   parameter search.

If a later exact-profile benchmark clears runtime, the proposed gate is
Composite `+0.025` or better, positive Macro and Robustness deltas, Composite
wins on at least 2/3 folds, and no class F1 drop greater than `0.02`; success
must remain exactly Exp15. Transfer slices are diagnostics only: runs with at
least 70 messages, teams over 6 agents, mesh/blackboard topology, all intents
missing, refs missing on at least half of messages, and missing artifact
subtasks. The aspirational `0.90` Composite is not a forecast. The official
page ranks Macro F1 as the primary metric; Composite is a separate local
objective. This is not yet a committed preregistration: CV requires a separate
PLAN commit after the runtime gate passes.

Target/agent-renaming, chronology tie-order, future-event metadata, full-turn
retention (205 messages), 513-run batch alignment, and deterministic CPU
inference checks passed. Frozen Exp15 OOF artifacts also reproduce all five
official metrics within `1e-12` on the demo data. The local Exp15 OOF cannot be
used for another train file: it is tied to the demo train hash and row order.

## Runtime gate — fail

Measured on the current local CPU environment: 12 logical CPUs, PyTorch
`2.14.1+cpu`, 6 torch threads. Benchmark: 513 runs / 30,385 events / 1,055,773
tokens, plus eight actual training batches of 16 runs and a 513-run inference
pass.

| Component | Seconds |
| --- | ---: |
| Candidate fit, projected to 10,000 runs | 1,273.3 |
| Candidate inference, projected to 4,000 runs | 6.7 |
| Input build, projected to 14,000 runs | 53.3 |
| Existing Exp15 training/inference runtime | 593.4 |
| Projected Exp15 + candidate total | **1,926.8** |
| Same estimate with 25% margin on candidate | **2,260.2** |
| Candidate three-fold CV, projected | **2,548.7** |

The measured benchmark-sample process peak was 565.9 MB; projected flat input
arrays for 14,000 runs are about 183 MB. Full production peak memory was not
measured. The point runtime estimate already exceeds the job limit by 126.8 s.
The Exp15 honest nested OOF benchmark was 1,886.31 s on this host, before
adding candidate CV, so the complete real-train OOF comparison does not fit a
1,800-second CPU job on measured evidence.

The official GPU image may make neural training faster, but this prototype was
not run on that image or on a DSWorks instance. Its documented torch version
also differs from the local benchmark version. Without that measurement,
neither production fit nor full OOF feasibility is established. The candidate
CV runner fails closed unless a preregistered `PLAN.md` and a passing runtime
preflight are present. Neither is present; no CV was started.

## Real-train OOF procedure and current blocker

The correct audit procedure inside a train-only job is: freeze and hash the
real train row order; use the same three stratified folds; fit the candidate
only on each fold's two-thirds train rows; collect one prediction per held-out
row; independently produce honest Exp15 OOF predictions (including the
nested label-probability path required by its success head); then score Macro,
Robustness, per-class F1, Success, hit@2, and Composite with
`evaluation/metrics.py`. Do not read the job's unlabeled test targets. This
procedure is specified but not feasible within the measured CPU time limit:
Exp15 alone took 1,886.31 s and candidate CV is estimated at 2,548.7 s. An
actual DSWorks resource measurement is required before claiming otherwise.

## Decision

**STOP before CV.** The event representation has plausible coverage of the
ontology and the prototype checks pass, but the frozen single-model design
does not meet the measured 1,800-second production budget when Exp15 success
is preserved. Do not run CV or proceed to production based on a presumed GPU
speedup. The one unresolved technical path is a measured run on the exact
available DSWorks instance type; until then, no full-train result or score
claim is available.

No `experiments/results.csv` row was added: this was an implementation and
runtime preflight STOP, not an OOF experiment result. Exp15 production,
historical Exp20 results, ZIPs, and leaderboard state are untouched.
