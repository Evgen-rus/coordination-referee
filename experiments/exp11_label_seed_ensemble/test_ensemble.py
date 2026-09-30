"""Tests for Exp11 - the fixed 3-seed label-head probability ensemble.

A test that passes on a broken implementation is worse than no test, so each
case below is written to fail on the specific mistake it guards against.

RUN THIS AFTER ``run_experiment.py``.  The expensive fits are NOT repeated: the
recorded ``results.json`` is re-checked against the per-seed matrices the run
saved, and the load-bearing bit-parity claim is re-verified against the sealed
artifacts rather than re-derived by a fresh refit.

  * the frozen seed set is exactly {42, 137, 2026} - not searched, not 2 or 4
    models, no weights, no best-seed selection;
  * the three parameter dicts differ in ``random_state`` and NOTHING else;
  * seed 42 alone is BIT-IDENTICAL to the sealed baseline, so the one-model
    case is an identity rather than a tolerance;
  * the ensemble is a plain arithmetic mean, and demonstrably NOT a geometric
    mean, a median, a hard vote or a per-row argmax;
  * the fast path equals ``evaluation.metrics`` on the real candidates;
  * success F1 is a pinned constant no label rule can move;
  * hit@2 reads the peak of the ENSEMBLE PREDICTED label, not the true one;
  * nothing in the decision path reads ``y_true``;
  * no fold ever saw its own validation rows;
  * the gate is the pre-declared one, applied exactly;
  * the gate FAILED, so no submission and no ZIP may exist.

Run:  python experiments/exp11_label_seed_ensemble/test_ensemble.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(ROOT, "experiments", "exp10_ab_probability_blend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ensemble as K                                          # noqa: E402
from ensemble import (BASE_SEED, GATE, N_RUNS, SEEDS, SUCCESS_F1_PINNED,  # noqa: E402
                      TOL, WEAK_BAND, LABELS, CLEAN)

RESULTS = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
SUBMISSION = os.path.join(ROOT, "submission_exp11_label_seed_ensemble")
SUBMISSION_ZIP = os.path.join(ROOT, "submission_exp11_label_seed_ensemble.zip")

_cache = {}
results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print("  PASS  %s" % name)
    except AssertionError as e:
        results.append((name, False, str(e)))
        print("  FAIL  %s -> %s" % (name, e))
    except Exception as e:  # noqa: BLE001
        results.append((name, False, "%s: %s" % (type(e).__name__, e)))
        print("  ERROR %s -> %s: %s" % (name, type(e).__name__, e))


def ctx():
    """Verified corpus + the three per-seed matrices, loaded once.

    The matrices are re-read from the artifacts the run wrote, so the tests
    exercise exactly the arrays the reported numbers came from.
    """
    if "c_base" not in _cache:
        d = K.load_verified()
        rob = K.robustness_mask()
        P = {int(s): np.load(os.path.join(HERE, "seed_%d_oof.npy" % s))
             for s in SEEDS}
        _cache.update(d=d, rob=rob, P=P,
                      c_base=K.corpus_for(d["proba_B"], d, rob,
                                         labels=d["labels_B"]),
                      c_ens=K.corpus_for(K.average([P[s] for s in SEEDS]), d, rob))
    return _cache


# ---------------------------------------------------------------------------
# 1. the frozen search space - this is ONE hypothesis, not a search
# ---------------------------------------------------------------------------

def test_seed_set_is_frozen_and_is_exactly_three():
    """{42, 137, 2026}, equal weights, no search space, no free parameter."""
    assert tuple(SEEDS) == (42, 137, 2026), SEEDS
    assert len(set(SEEDS)) == 3
    assert RESULTS["seeds"] == [42, 137, 2026]
    assert RESULTS["weights"] == [1.0 / 3.0] * 3
    ss = RESULTS["search_space"]
    assert ss["free_parameters"] == 0
    assert ss["seed_set"] == [42, 137, 2026]
    # a seed outside the frozen set must be refused at the constructor
    for bad in (0, 7, 43, 2025, 3141):
        try:
            K.label_params(bad)
        except ValueError:
            continue
        raise AssertionError("label_params accepted seed %r outside the frozen set"
                             % bad)
    # a model count other than 3 must be refused by the average
    P = ctx()["P"]
    for wrong in ([42, 137], [42, 137, 2026, 7]):
        try:
            K.average([P[42]] * len(wrong))
        except ValueError:
            continue
        raise AssertionError("average accepted %d models" % len(wrong))


def test_three_models_differ_only_in_random_state():
    """Same rows, same 300 columns, same hyper-parameters; only the seed moves."""
    out = K.assert_params_differ_only_in_random_state()   # raises if not
    base = out[BASE_SEED]
    for s in SEEDS:
        assert out[s]["random_state"] == s
        diff = {k for k in set(base) | set(out[s])
                if base.get(k) != out[s].get(k)}
        assert diff == (set() if s == BASE_SEED else {"random_state"}), (s, diff)
    assert (base["objective"] == "multiclass" and base["num_class"] == 7
            and base["n_estimators"] == 600 and base["learning_rate"] == 0.05
            and base["num_leaves"] == 63 and base["min_child_samples"] == 20
            and base["subsample"] == 0.9 and base["subsample_freq"] == 1
            and base["colsample_bytree"] == 0.8 and base["reg_lambda"] == 1.0)
    fm = RESULTS["feature_matrix"]
    assert fm["n_features"] == 300 and fm["n_agg"] == 51
    assert fm["same_columns_all_seeds"] is True
    assert len(fm["column_order"]) == 300 == len(set(fm["column_order"]))


def test_seed42_alone_is_bit_identical_to_the_sealed_baseline():
    """The one-model case is an IDENTITY, not a tolerance.

    This is the load-bearing claim: it proves the rebuilt 300-feature matrix
    and the refitted seed-42 head are the same objects current best was scored
    on, so every reported delta is attributable to the seed change alone.
    """
    c = ctx()
    d, P = c["d"], c["P"]
    assert K.sha32(P[BASE_SEED]) == K.sha32(d["proba_B"]), \
        "seed 42 does not reproduce the sealed Exp07 OOF probabilities bit for bit"
    assert RESULTS["feature_matrix"]["seed42_bit_identical_to_sealed"] is True
    assert np.array_equal(P[BASE_SEED].argmax(1), d["labels_B"])
    for s in SEEDS:
        if s == BASE_SEED:
            continue
        assert K.sha32(P[s]) != K.sha32(d["proba_B"]), \
            "seed %d is bit-identical to seed 42 - it is not a different model" % s
        assert RESULTS["per_seed_bit_parity"][str(s)]["labels_differ_vs_seed42"] > 100
    c42 = K.corpus_for(P[BASE_SEED], d, c["rob"], labels=d["labels_B"])
    m42, _ = K.full_official(c42, c42.predict(1.0))
    for k, v in (("macro_f1", 0.7894323366835861),
                 ("robustness_f1", 0.7368470046834376),
                 ("success_f1", 0.8643757406010988),
                 ("fault_turn_hit2", 0.6086111111111111),
                 ("composite", 0.7694453917139285)):
        assert abs(m42[k] - v) < 1e-12, (k, m42[k], v)
    assert abs(m42["composite"] - RESULTS["baseline"]["composite"]) < 1e-12


# ---------------------------------------------------------------------------
# 2. the ensemble arithmetic - a plain mean, and demonstrably only that
# ---------------------------------------------------------------------------

def test_ensemble_is_a_plain_arithmetic_mean():
    """Equal-weight arithmetic mean in float64 over the float32 per-seed casts.

    Pins the arithmetic, not just the output: a geometric mean, a median, a
    hard vote or a per-row argmax would all "work" and all score similarly,
    but they are DIFFERENT functions and this experiment tested only this one.
    """
    c = ctx()
    P = c["P"]
    ref = (P[42].astype(np.float64) + P[137].astype(np.float64)
           + P[2026].astype(np.float64)) / 3.0
    got = K.average([P[s] for s in SEEDS])
    assert np.array_equal(got, ref), "average() is not the declared plain mean"
    assert got.shape == (N_RUNS, 7)
    assert np.isfinite(got).all() and (got >= 0).all() and (got <= 1).all()
    assert np.allclose(got.sum(axis=1), 1.0, atol=1e-9)

    geo = np.exp((np.log(np.maximum(P[42].astype(np.float64), 1e-15))
                  + np.log(np.maximum(P[137].astype(np.float64), 1e-15))
                  + np.log(np.maximum(P[2026].astype(np.float64), 1e-15))) / 3.0)
    assert not np.allclose(got, geo, atol=1e-6), \
        "the ensemble equals a geometric mean - the arithmetic is not the declared one"
    med = np.median(np.stack([P[s].astype(np.float64) for s in SEEDS]), axis=0)
    assert not np.allclose(got, med, atol=1e-6), "the ensemble is a median"
    votes = np.stack([P[s].argmax(1) for s in SEEDS])
    vote_pred = np.array([np.bincount(v, minlength=7).argmax() for v in votes.T])
    assert not np.array_equal(got.argmax(1), vote_pred), \
        "the ensemble label equals a hard vote - that is a different rule"
    assert not np.array_equal(got.argmax(1), P[42].argmax(1)), \
        "the ensemble label equals seed 42's label - nothing was ensembled"


def test_success_is_pinned_and_immovable():
    """Success F1 is a constant; no label rule may recompute it."""
    c = ctx()
    assert SUCCESS_F1_PINNED == 0.8643757406010988
    for key in ("baseline_seed42", "ensemble_3seed"):
        assert RESULTS["comparison"][key]["success_f1"] == SUCCESS_F1_PINNED
    assert RESULTS["comparison"]["delta"]["success_f1"] == 0.0
    from metrics import WEIGHTS
    assert WEIGHTS["success_f1"] == 0.15
    for seed in SEEDS:
        cs = K.corpus_for(c["P"][seed], c["d"], c["rob"])
        m, _ = K.full_official(cs, cs.predict(1.0))
        assert m["success_f1"] == SUCCESS_F1_PINNED, seed
    ind = RESULTS["individual_seeds_diagnostic"]
    assert set(ind["seeds"]) == {"42", "137", "2026"}
    assert "no seed is selected" in ind["note"]
    best = max(ind["seeds"], key=lambda k: ind["seeds"][k]["composite"])
    if best != "42":
        raise AssertionError(
            "the individual-seed diagnostic implies a best seed %r other than "
            "the incumbent - this experiment must never select a seed" % best)


# ---------------------------------------------------------------------------
# 3. honesty of the measurement
# ---------------------------------------------------------------------------

def test_fast_path_equals_official_metrics_on_real_candidates():
    """The reporting path must equal ``evaluation.metrics`` itself."""
    c = ctx()
    for tag, cc in (("baseline", c["c_base"]), ("ensemble", c["c_ens"]),
                    ("seed137", K.corpus_for(c["P"][137], c["d"], c["rob"])),
                    ("seed2026", K.corpus_for(c["P"][2026], c["d"], c["rob"]))):
        o = K.Objective(cc, np.arange(N_RUNS))
        fast, off = o.metrics(1.0), o.official(1.0)
        for key in ("macro_f1", "robustness_f1", "fault_turn_hit2", "composite"):
            assert abs(fast[key] - off[key]) < 1e-12, (tag, key, fast[key], off[key])


def test_reported_numbers_are_reproducible_from_the_saved_artifacts():
    """Recompute the headline from the saved per-seed matrices and compare.

    An end-to-end check of results.json against the arrays: a number typed in by
    hand, or computed from a different matrix, fails here.
    """
    c = ctx()
    pe = c["c_ens"].predict(1.0)
    pb = c["c_base"].predict(1.0)
    me, _ = K.full_official(c["c_ens"], pe)
    mb, _ = K.full_official(c["c_base"], pb)
    for key in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
                "composite"):
        assert abs(me[key] - RESULTS["comparison"]["ensemble_3seed"][key]) < 1e-12, key
        assert abs(mb[key] - RESULTS["comparison"]["baseline_seed42"][key]) < 1e-12, key
    n_chg = int((pe != pb).sum())
    assert n_chg == RESULTS["comparison"]["n_labels_changed"], n_chg
    assert abs((me["composite"] - mb["composite"])
               - RESULTS["comparison"]["delta"]["composite"]) < 1e-12
    from metrics import f1_per_class
    yt = c["c_base"].name(c["c_base"].yi)
    pc_b = f1_per_class(yt, c["c_base"].name(pb), LABELS)
    pc_e = f1_per_class(yt, c["c_ens"].name(pe), LABELS)
    for cls in LABELS:
        assert abs(pc_e[cls]
                   - RESULTS["comparison"]["per_class_f1"]["ensemble"][cls]) < 1e-12
        assert abs((pc_e[cls] - pc_b[cls])
                   - RESULTS["comparison"]["per_class_f1"]["delta"][cls]) < 1e-12
    tr = RESULTS["comparison"]["confusion_transitions"]
    assert sum(tr.values()) == n_chg, (sum(tr.values()), n_chg)
    st = RESULTS["comparison"]["changed_label_stats"]
    # A changed label is not necessarily right->wrong or wrong->right: a run can
    # move from one WRONG class to a different WRONG class and stay wrong.  So
    # the invariant is a partition into three disjoint parts, not a sum of two:
    #   wrong -> right, right -> wrong, wrong -> wrong.
    # An earlier version of this test asserted became_correct + became_wrong ==
    # n_changed, which is simply false on real data - 33 of the 201 changed rows
    # here were wrong->wrong - so it would have rejected a correct report.
    assert st["changed"] == n_chg
    assert st["became_correct"] + st["became_wrong"] <= n_chg
    assert st["net"] == st["became_correct"] - st["became_wrong"]
    n_wrong_to_wrong = n_chg - st["became_correct"] - st["became_wrong"]
    assert n_wrong_to_wrong >= 0
    # fault_turn moves with the label, and a clean prediction is never given a turn
    n_ft = int((c["c_ens"].turns_for(pe) != c["c_base"].turns_for(pb)).sum())
    assert n_ft == RESULTS["comparison"]["n_fault_turn_changed"], (
        "fault_turn changed count: recomputed %d, recorded %d"
        % (n_ft, RESULTS["comparison"]["n_fault_turn_changed"]))
    turns_e = c["c_ens"].turns_for(pe)
    assert bool((turns_e[pe == CLEAN] == -1).all()), \
        "an ensemble clean prediction did not receive fault_turn -1"
    assert int((turns_e >= 0).sum()) == int((pe != CLEAN).sum()), \
        "a non-clean prediction was left without a turn, or a clean one got one"


def test_hit2_and_composite_move_with_the_predicted_label():
    """hit@2 reads the peak of the ENSEMBLE PREDICTED class, L1 rule unchanged.

    The Exp08 L1 rule is re-applied to whatever label vector it is handed; it
    is NOT held fixed while the label moves, and it does NOT silently switch to
    reading the true class's peak - which would score identically and emit a
    different column, the trap Exp09 established.
    """
    c = ctx()
    d = c["d"]
    pred = c["c_ens"].predict(1.0)
    turns = c["c_ens"].turns_for(pred)
    ref = np.full(N_RUNS, -1, dtype=np.int64)
    for i in range(N_RUNS):
        lab = LABELS[pred[i]]
        if lab != "clean":
            ref[i] = int(d["peak_pos"][i, K.W2I[lab]])
    assert np.array_equal(turns, ref), \
        "the vectorised L1 turn differs from a direct re-read of the sealed peaks"
    assert bool((turns[pred == CLEAN] == -1).all()), \
        "an ensemble clean prediction did not receive fault_turn -1"
    assert bool((turns[pred != CLEAN] >= 0).all())
    from metrics import fault_turn_hit_at_k
    yi = c["c_ens"].yi
    f = np.where(yi != CLEAN)[0]
    h2 = fault_turn_hit_at_k([LABELS[yi[i]] for i in f], [LABELS[pred[i]] for i in f],
                             [int(c["c_ens"].fturn[i]) for i in f],
                             [int(turns[i]) for i in f], k=2)
    assert abs(h2 - RESULTS["comparison"]["ensemble_3seed"]["fault_turn_hit2"]) < 1e-12
    hits = sum(1 for i in f if pred[i] == yi[i] and turns[i] >= 0
               and abs(int(turns[i]) - int(c["c_ens"].fturn[i])) <= 2)
    assert abs(hits / len(f) - h2) < 1e-12
    rows = np.arange(N_RUNS)
    e_pred, e_true = c["c_ens"].T[rows, pred], c["c_ens"].T[rows, yi]
    d_diff = e_pred != e_true
    assert int(d_diff.sum()) > 100, "the real OOF does not separate the columns"
    assert int((d_diff & (pred == yi)).sum()) == 0, \
        "the two columns differ where pred == y_true, which is impossible"


def test_no_true_label_leakage_in_the_decision_path():
    """Nothing that EMITS a prediction may read ``y_true``.

    The emitted label vector and the emitted turn vector must be bit-identical
    when every true label is replaced by noise.  The scores necessarily move -
    they are computed against y_true - but the decision cannot.
    """
    c = ctx()
    d, P = c["d"], c["P"]
    ens = K.average([P[s] for s in SEEDS])
    ref = K.corpus_for(ens, d, c["rob"])
    rp = ref.predict(1.0)
    rt = ref.turns_for(rp)
    fake = dict(d)
    fake["yi"] = np.random.default_rng(0).integers(0, 7, N_RUNS)
    c2 = K.corpus_for(ens, fake, c["rob"])
    assert np.array_equal(c2.predict(1.0), rp), \
        "the predicted label moved when y_true was replaced"
    assert np.array_equal(c2.turns_for(rp), rt), \
        "the emitted turn moved when y_true was replaced"
    for s in SEEDS:
        a = K.corpus_for(P[s], d, c["rob"]).predict(1.0)
        b = K.corpus_for(P[s], fake, c["rob"]).predict(1.0)
        assert np.array_equal(a, b), s


def test_no_fold_ever_saw_its_own_validation_rows():
    """Each outer fold's held-out rows are absent from its own training set.

    The seed set is fixed a priori, so there is no per-fold selection to leak;
    what must hold is the structural claim that every fit saw outer-train only.
    The recorded fold index hashes are compared to the sealed manifest, the
    train/val sets are proven disjoint, and each fold's recorded score is
    recomputed on the held-out rows alone.
    """
    c = ctx()
    d = c["d"]
    man = list(d["manifest"]["fold_val_sha256"])
    assert d["fold_sha"] == man, "outer folds differ from the sealed manifest"
    assert len(d["folds"]) == 3
    cover = np.zeros(N_RUNS, dtype=int)
    for f, va in enumerate(d["folds"]):
        va = np.asarray(va, dtype=np.int64)
        cover[va] += 1
        tr = np.setdiff1d(np.arange(N_RUNS), va)
        assert not np.intersect1d(va, tr).size
        rec = RESULTS["folds"][f]
        assert rec["val_sha256"] == man[f]
        assert rec["n_val"] == len(va) and rec["n_train"] == N_RUNS - len(va)
        ob = K.Objective(c["c_base"], va)
        oe = K.Objective(c["c_ens"], va)
        assert abs(ob.official(1.0)["composite"]
                   - rec["baseline"]["composite"]) < 1e-12
        assert abs(oe.official(1.0)["composite"]
                   - rec["ensemble"]["composite"]) < 1e-12
        assert rec["wins"] == (rec["delta_composite"] > 0)
    assert cover.min() == 1 and cover.max() == 1, \
        "some rows are never validated - this would not be honest OOF"
    wins = sum(1 for r in RESULTS["folds"] if r["wins"])
    assert wins == RESULTS["promotion_gate"]["folds_won"]
    assert wins == int(RESULTS["fold_wins"].split("/")[0])
    assert len(RESULTS["folds"]) == 3


# ---------------------------------------------------------------------------
# 4. the gate - pre-declared, applied exactly, and FAILED
# ---------------------------------------------------------------------------

def test_gate_is_exactly_as_predeclared():
    """The four criteria are the ones written down before the first fit."""
    g = RESULTS["promotion_gate"]
    assert GATE == {"composite_delta_min": 0.002, "folds_won_min": 2,
                     "robustness_drop_max": 0.003, "class_f1_drop_max": 0.02}
    assert RESULTS["gate"] == GATE
    assert WEAK_BAND == (0.001, 0.002)
    assert g["criteria"][0]["name"].startswith("composite delta")
    d_comp = (RESULTS["comparison"]["ensemble_3seed"]["composite"]
              - RESULTS["comparison"]["baseline_seed42"]["composite"])
    assert abs(g["criteria"][0]["value"] - d_comp) < 1e-12
    d_rob = (RESULTS["comparison"]["ensemble_3seed"]["robustness_f1"]
             - RESULTS["comparison"]["baseline_seed42"]["robustness_f1"])
    assert abs(g["criteria"][2]["value"] + d_rob) < 1e-12
    pcd = RESULTS["comparison"]["per_class_f1"]["delta"]
    worst = min(pcd, key=lambda k: pcd[k])
    assert g["worst_class_f1_drop"]["class"] == worst
    assert abs(g["worst_class_f1_drop"]["drop"] + pcd[worst]) < 1e-12
    assert g["passed"] == all(x["passed"] for x in g["criteria"])
    assert g["unexpected_failures"] == []


def test_weak_band_is_declared_and_the_verdict_is_stop():
    """A delta in [+0.001, +0.002) is weak/inconclusive and resolves to STOP.

    The observed delta is below the weak band entirely, so the verdict is STOP
    on the plain criterion.  This pins that the band was declared before the
    run and that no seed, weight or variant was substituted afterwards.
    """
    d_comp = RESULTS["comparison"]["delta"]["composite"]
    g = RESULTS["promotion_gate"]
    if WEAK_BAND[0] <= d_comp < WEAK_BAND[1]:
        assert g["weak_inconclusive_band"] is True
    else:
        assert d_comp < WEAK_BAND[0], \
            "delta is above the weak band but the run did not flag it"
    assert RESULTS["promoted"] is False
    assert g["passed"] is False
    assert RESULTS["final_ensemble"]["fitted"] is False
    assert d_comp < GATE["composite_delta_min"], d_comp
    assert "fixed a priori" in RESULTS["caveat"]


def test_gate_failed_so_no_production_build_exists():
    """The gate FAILED, so no ensemble may be shipped and no build may exist.

    Not a skip: a stray production build after a failed gate is exactly the
    outcome this experiment is forbidden from producing, and it would be easy
    to miss in a directory listing.
    """
    assert RESULTS.get("promoted") is False, \
        "results.json claims promotion despite a failed gate"
    assert RESULTS["promotion_gate"]["passed"] is False
    fe = RESULTS.get("final_ensemble", {})
    assert fe.get("fitted") is False, "a production ensemble was fitted after a FAIL"
    assert not RESULTS.get("production"), \
        "a production block was written despite a failed gate"
    assert not os.path.exists(SUBMISSION), \
        "submission_exp11_label_seed_ensemble exists although the gate FAILED"
    assert not os.path.exists(SUBMISSION_ZIP), \
        "submission_exp11_label_seed_ensemble.zip exists although the gate FAILED"
    assert RESULTS["public_exp08_reference_only"] == 0.7696
    assert RESULTS["baseline"]["composite"] != RESULTS["public_exp08_reference_only"]


def main():
    print("=" * 78)
    print("Exp11 fixed 3-seed label-head ensemble - tests")
    print("=" * 78)
    print("  promotion gate: %s (expected FAIL) -> the no-build test runs"
          % ("PASS" if RESULTS.get("promoted") else "FAIL"))
    tests = [
        ("seed set is frozen and is exactly three models",
         test_seed_set_is_frozen_and_is_exactly_three),
        ("3 models differ only in random_state; 300 cols locked",
         test_three_models_differ_only_in_random_state),
        ("seed 42 alone is BIT-IDENTICAL to the sealed baseline",
         test_seed42_alone_is_bit_identical_to_the_sealed_baseline),
        ("ensemble is a plain arithmetic mean (not geo/median/vote)",
         test_ensemble_is_a_plain_arithmetic_mean),
        ("success is pinned and immovable; no seed is selected",
         test_success_is_pinned_and_immovable),
        ("fast path == official evaluation.metrics on real candidates",
         test_fast_path_equals_official_metrics_on_real_candidates),
        ("reported numbers are reproducible from the saved artifacts",
         test_reported_numbers_are_reproducible_from_the_saved_artifacts),
        ("hit@2 and composite move with the ENSEMBLE PREDICTED label",
         test_hit2_and_composite_move_with_the_predicted_label),
        ("no true-label leakage in the decision path",
         test_no_true_label_leakage_in_the_decision_path),
        ("no fold ever saw its own validation rows",
         test_no_fold_ever_saw_its_own_validation_rows),
        ("promotion gate is exactly as pre-declared",
         test_gate_is_exactly_as_predeclared),
        ("weak band declared; delta is below it; verdict is STOP",
         test_weak_band_is_declared_and_the_verdict_is_stop),
        ("gate FAILED, so no submission and no ZIP exist",
         test_gate_failed_so_no_production_build_exists),
    ]
    for name, fn in tests:
        check(name, fn)
    npass = sum(1 for _, o, _ in results if o)
    print("\n%d/%d passed" % (npass, len(results)))
    for name, o, err in results:
        if not o:
            print("  FAILED: %s -> %s" % (name, err))
    return 1 if npass != len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
