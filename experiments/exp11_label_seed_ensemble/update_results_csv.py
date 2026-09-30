"""Append the Exp11 row to experiments/results.csv.

Follows the convention of the two most recent STOP rows (exp09, exp10): the
11 columns are id, name, baseline macro, candidate macro, macro delta,
baseline robustness, candidate robustness, robustness delta, verdict, summary,
RESULTS.md path.  Appended, not rewritten; existing rows are not touched.
"""
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CSV = os.path.join(ROOT, "experiments", "results.csv")
res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))

b = res["comparison"]["baseline_seed42"]
e = res["comparison"]["ensemble_3seed"]
folds = res["folds"]
seeds = res["individual_seeds_diagnostic"]["seeds"]

summary = (
    "REJECTED - no production build, no ZIP. Single hypothesis: plain "
    "probability-level ensemble of THREE label heads fitted on identical rows "
    "and identical 300 features, differing ONLY in random_state; "
    "P_ensemble = (P_seed42 + P_seed137 + P_seed2026)/3, pred = argmax, equal "
    "weights, NO free parameter and no search space. The seed set was fixed "
    "BEFORE any number was seen; no other seeds, no 2 or 4 models, no weights, "
    "no best-seed selection, no A/B blend, no offsets/calibration/thresholds, "
    "no window-model ensemble, no new features or classifiers, no Optuna, no "
    "leaderboard tuning. Window model, its random_state=42, the 51 aggregates, "
    "the sealed window peaks, the Exp08 L1 localiser and the pinned success F1 "
    "are all UNCHANGED; L1 is re-read for the ENSEMBLE PREDICTED label, so "
    "hit@2 moves with the label rather than being held fixed. STRICT REUSE: 32 "
    "provenance checks through reuse.load plus the fail-closed window-peak "
    "seal, ArtifactRejected fatal; outer fold index hashes re-derived from "
    "seed-0 StratifiedKFold(3) and asserted equal to the sealed manifest. "
    "REPRODUCTION GATE PASSED EXACT, all 5 numbers to 1e-12 (macro "
    "0.7894323366835861, robustness 0.7368470046834376, success "
    "0.8643757406010988, hit@2 0.6086111111111111, composite "
    "0.7694453917139285) before any fit, plus baseline labels identical to the "
    "committed Exp07 OOF CSV on all 10000 rows. THE LOAD-BEARING PARITY: the "
    "300-feature matrix was rebuilt once by the standard Exp07 mechanism and "
    "seed 42 reproduces the sealed Exp07 OOF probabilities BIT for BIT "
    "(sha256 buffer equality, max abs diff 0.0), so every delta below is a "
    "seed effect on one pipeline and not a difference between two pipelines; "
    "the one-model case is an exact identity rather than a tolerance. 3 "
    "parameter dicts differ in random_state and nothing else, mechanically "
    "asserted; 300 columns in one order for all seeds; fast path == "
    "evaluation.metrics with max abs diff 0.0 on 4 real candidates. HEADLINE "
    "(honest 3-fold OOF, every seed fitted on outer-train only and predicting "
    "outer-val only, concatenated into one 10000-row vector and scored once): "
    "composite 0.7694453917139285 -> 0.7696021570323565, delta +0.000157 - 12.7x "
    "below the pre-declared +0.002 bar and also BELOW the pre-declared weak "
    "band [+0.001,+0.002), so this is empty, not inconclusive. macro +0.000056, "
    "robustness +0.000125, hit@2 +0.000972, success pinned at 0.8644; 201/10000 "
    "OOF labels changed (2.01%) and 196 fault_turn values moved with them. "
    "GATE FAILED on the two criteria that carry the hypothesis - composite "
    "delta and 1/3 folds won (fold 0 +0.0042310, fold 1 -0.0029233, fold 2 "
    "-0.0012630), i.e. the gain lives in ONE fold and cancels out over all "
    "10000 rows; the two safety criteria passed (robustness ROSE 0.000125, "
    "worst class F1 drop 0.00238 dropped_handoff), i.e. harmless but empty. "
    "WHY IT FAILED, the actual finding: the three seeds genuinely disagree - "
    "328 and 359 label differences out of 10000, max|dP| 0.477 - so this is "
    "not three copies of one model, but the seed-to-seed spread (0.00145 from "
    "42 down to 137) is about 9x LARGER than the gain the average buys "
    "(+0.000157), so there is nothing for three seeds to average out. The "
    "average does beat all three of its own members (137 = -0.001447, 2026 = "
    "-0.000302), the textbook variance-reduction effect and the only positive "
    "sign here, and it is still an order of magnitude under the bar. The 201 "
    "changed labels split 86 wrong->right vs 82 right->wrong (33 wrong->wrong), "
    "a coin flip worth +4 runs. Per-class deltas are +0.0024 clean and +0.0035 "
    "runaway_loop against -0.0024 dropped_handoff, -0.0015 deadlock, -0.0013 "
    "goal_drift, -0.0009 conflict: the ensemble helps the two EASIEST classes "
    "and pays for it in the hard ones, which is a shift in the clean/fault "
    "operating point rather than a better model. NO SEED WAS SELECTED and the "
    "headline is the fixed average; test_ensemble.py fails if the recorded "
    "individual-seed diagnostic ever implies a selection other than the "
    "incumbent. SEPARATE FINDING, REPORTED NOT FIXED: a real Exp07 defect - "
    "runner.py:229 calls predict_runs(P, rw[...], len(tr_runs)) with GLOBAL run "
    "indices up to 9999 while predict_runs iterates range(len(tr_runs)) ~ 6666, "
    "so 2182 of 6666 outer-train rows on fold 0 (32.7%; 33.7/32.4/32.1% per "
    "inner fold) get NO block and their 51 window aggregates stay exactly ZERO "
    "in the training matrix, while the outer-VALIDATION call uses n=10000 and "
    "every validation row gets real aggregates - i.e. the 51 window features "
    "are not distributed identically between train and inference and all-zeros "
    "is a splittable value, not a neutral one. This experiment reproduces the "
    "defect verbatim because fixing it would change the 300-feature matrix, "
    "which is precisely what the bit-parity gate holds constant; its worth is "
    "deliberately NOT estimated here, since fixing it moves current best's own "
    "numbers and cannot be measured inside an experiment whose baseline IS "
    "current best. Three bugs were found and fixed before reporting: two in "
    "the parameter-discipline assert (an empty diff for the reference seed, "
    "then an inverted condition) and one TEST bug - the reproducibility check "
    "asserted became_correct+became_wrong==n_changed, which is false on real "
    "data because 33 of 201 rows were wrong->wrong; the assertion was wrong, "
    "not the report, and was corrected to a three-way partition rather than "
    "loosened. 13/13 tests pass. Cost: strict load 0.15s, 3 seeds x 3 folds "
    "1497.6s then 1383.9s on a deterministic rerun, total research runtime "
    "24.96 min. NOT DONE, DELIBERATELY: no second seed set, no weights, no "
    "production submission, no ZIP, no upload, no Exp12. The public leaderboard "
    "remains the external validation and public Exp08 0.7696 is a reference "
    "only, never read by anything. With Exp09 and Exp10 this is the THIRD "
    "consecutive STOP, so experiments/ROOT_CAUSE_REVIEW.md is now mandatory "
    "before any Exp12 is designed."
)

row = ["exp11", "label_seed_ensemble",
       "%.4f" % b["macro_f1"], "%.4f" % e["macro_f1"],
       "%+.4f" % (e["macro_f1"] - b["macro_f1"]),
       "%.4f" % b["robustness_f1"], "%.4f" % e["robustness_f1"],
       "%+.4f" % (e["robustness_f1"] - b["robustness_f1"]),
       "not_promoting", summary,
       "experiments/exp11_label_seed_ensemble/RESULTS.md"]
assert len(row) == 11, len(row)
assert len(row[8]) == len("not_promoting")

with open(CSV, "r", encoding="utf-8", newline="") as fh:
    existing = list(csv.reader(fh))
assert not any(r[0] == "exp11" for r in existing), "exp11 row already present"

buf = io.StringIO()
csv.writer(buf, lineterminator="\n").writerow(row)
with open(CSV, "a", encoding="utf-8", newline="") as fh:
    fh.write(buf.getvalue())

after = list(csv.reader(open(CSV, "r", encoding="utf-8", newline="")))
assert len(after) == len(existing) + 1
assert after[:-1] == existing, "an existing row was modified"
print("appended exp11; rows %d -> %d" % (len(existing), len(after)))
print("row: %s | %s | macro %s->%s | rob %s->%s | %s"
      % (row[0], row[1], row[2], row[3], row[5], row[6], row[8]))
