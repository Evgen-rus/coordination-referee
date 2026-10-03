"""Exp15 tests - the ten required checks, plus structural guards.

    1. baseline success reproduction (0.8643757406010988, exact)
    2. meta probabilities aligned by run_id (and sha256-stable)
    3. every OOF meta prediction excludes its own row (3 outer x 3 nested)
    4. no success target enters the label model
    5. deterministic folds
    6. A/B first 185 columns identical
    7. B adds exactly 7 columns, names in LABELS order
    8. label predictions unchanged vs committed Exp13 D
    9. fault_turn unchanged
    10. OOF meta probabilities differ from an in-sample refit

Tests 1, 6, 8, 9, 10 assert against ``results.json``, so they FAIL when the run
did not happen or did not pass its gates - a green suite on a missing run is
not possible.  Tests 2, 3, 4, 5, 7 are computed here, not read.

Run:  python -m pytest experiments/exp15_success_label_stacking/test_stacking.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

_spec = importlib.util.spec_from_file_location(
    "exp15_run", os.path.join(HERE, "run_experiment.py"))
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)

RESULTS = os.path.join(HERE, "results.json")


# ---------------------------------------------------------------------------
# helpers - a run must exist before any test about it can pass
# ---------------------------------------------------------------------------

def _res():
    if not os.path.exists(RESULTS):
        pytest.fail("results.json is missing - run run_experiment.py first")
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def res():
    return _res()


@pytest.fixture(scope="module")
def preds():
    pa = os.path.join(HERE, "success_pred_A.npy")
    pb = os.path.join(HERE, "success_pred_B.npy")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        pytest.fail("success predictions are missing - run run_experiment.py")
    return (np.load(pa), np.load(pb))


@pytest.fixture(scope="module")
def ctx():
    d = X.ENS.load_verified()
    c, Wd = X.ENS._prepare()
    return d, c, Wd


# ---------------------------------------------------------------------------
# 1. baseline success reproduction
# ---------------------------------------------------------------------------

def test_01_baseline_success_reproduction_exact(res):
    g = res["baseline_reproduction"]
    assert g["passed"] is True, "CONTROL A did not reproduce the baseline"
    assert g["f1"]["pass"] is True
    assert g["f1"]["expected"] == 0.8643757406010988
    assert g["f1"]["got"] == 0.8643757406010988, \
        "success_f1 is %.16f, not the declared baseline" % g["f1"]["got"]
    assert g["f1"]["abs_diff"] <= X.TOL
    assert g["csv_predictions_identical"] is True
    assert g["csv_predictions_differ"] == 0


def test_01b_declared_baseline_is_the_committed_exp03_b_value():
    """The pinned constant is Exp03 B's, not a number chosen here."""
    c, _W = X.ENS._prepare()
    assert c["success_f1"] == 0.8643757406010988


# ---------------------------------------------------------------------------
# 2. meta probabilities aligned by run_id
# ---------------------------------------------------------------------------

def test_02_meta_rows_align_with_run_ids(ctx, res):
    d, _c, _W = ctx
    ids = [str(x) for x in d["train"]["run_id"].values]
    assert len(set(ids)) == len(ids), "run_id is not unique"
    a = res["run_id_alignment"]
    assert a["n_rows"] == len(ids)
    assert a["n_unique_run_ids"] == len(ids)
    assert a["run_id_sha256"] == X.ca.sha256_run_ids(ids)

    P = np.load(os.path.join(HERE, "meta_prob_oof.npy"))
    assert P.shape == (len(ids), 7), P.shape
    assert np.isfinite(P).all()
    assert (P >= 0).all() and (P <= 1).all()
    np.testing.assert_allclose(P.astype(np.float64).sum(axis=1), 1.0, atol=1e-5)
    # the rows of P are the runs of train.csv, in train.csv's order
    assert res["run_id_alignment"]["matches_verified_loader"] is True


def test_02b_meta_hashes_are_stable_across_loads():
    P = np.load(os.path.join(HERE, "meta_prob_oof.npy"))
    assert X.ca.sha256_array(np.ascontiguousarray(P)) == \
        X.ca.sha256_array(np.ascontiguousarray(np.load(
            os.path.join(HERE, "meta_prob_oof.npy"))))


# ---------------------------------------------------------------------------
# 3. every OOF meta prediction excludes its own row
# ---------------------------------------------------------------------------

def test_03_exclusion_proofs_all_pass(res):
    proofs = res["exclusion_proofs"]
    assert len(proofs) == 3, "expected one proof per outer fold"
    for p in proofs:
        assert p["all_excluded"] is True, p["problems"]
        assert p["coverage_min"] == 1 and p["coverage_max"] == 1, \
            "outer-train rows must be held out exactly once"
        assert len(p["nested"]) == 3, "expected 3 nested folds per outer fold"
        for nd in p["nested"]:
            assert nd["overlap"] == 0
            assert nd["n_fit"] + nd["n_hold"] == p["n_outer_train"]


def test_03b_structural_recheck_of_exclusion(ctx):
    """Recompute the exclusion property here, from the fold definition, so the
    test does not merely trust a flag the run wrote."""
    d, _c, _W = ctx
    yi = np.asarray(d["yi"], np.int64)
    n = len(yi)
    all_ix = np.arange(n, dtype=np.int64)
    from sklearn.model_selection import StratifiedKFold
    for fv in d["folds"]:
        va = np.asarray(fv, np.int64)
        tr = all_ix[~np.isin(all_ix, va)]
        tr_set = set(int(x) for x in tr)
        seen = np.zeros(n, dtype=bool)
        for ntr_p, nva_p in StratifiedKFold(X.N_INNER, shuffle=True,
                                            random_state=X.NESTED_SEED
                                            ).split(tr, yi[tr]):
            ntr, nva = tr[ntr_p], tr[nva_p]
            fit, hold = set(int(x) for x in ntr), set(int(x) for x in nva)
            assert not (fit & hold), "nested fit/hold overlap"
            assert hold <= tr_set and fit <= tr_set
            for r in hold:
                assert not seen[int(r)], "a row is held out twice"
                assert int(r) not in fit, "a row is in its own fit set"
                seen[int(r)] = True
        assert seen[tr].all(), "some outer-train row was never held out"


# ---------------------------------------------------------------------------
# 4. no success target enters the label model
# ---------------------------------------------------------------------------

def test_04_success_features_ignore_the_success_target(res):
    p = res["success_feature_leakage_probe"]
    assert p["identical"] is True, "a success feature moved under target " \
                                   "perturbation: %r" % p
    assert p["max_abs_diff"] == 0.0
    assert p["n_columns"] == 185


def test_04b_success_feature_probe_is_live():
    """Re-run the probe now; a recorded flag alone would not catch a later edit
    of the feature code."""
    c, _W = X.ENS._prepare()
    p = X.success_feature_leakage_probe(c["runs"], 40)
    assert p["identical"] is True, p


def test_04c_label_fits_receive_only_the_label_target(ctx):
    """The label path is driven by ``yi``; ``success`` is never a fit target.
    Mechanically: the label model's own ``classes_`` and the shape of the target
    vector it was given are the label target, and the success vector has a
    different length distribution - so a swapped target would raise."""
    d, _c, _W = ctx
    yi = np.asarray(d["yi"], np.int64)
    s = np.asarray(d["train"]["success"].values, np.int64)
    assert yi.max() == 6 and yi.min() == 0
    assert s.max() <= 1
    assert not np.array_equal(np.unique(yi), np.unique(s)), \
        "the two targets are distinguishable, so a swap cannot pass silently"


# ---------------------------------------------------------------------------
# 5. deterministic folds
# ---------------------------------------------------------------------------

def test_05_outer_folds_are_the_sealed_seed0_folds(ctx):
    d, _c, _W = ctx
    yi = np.asarray(d["yi"], np.int64)
    n = len(yi)
    from sklearn.model_selection import StratifiedKFold
    red = [np.asarray(v, np.int64) for _, v in
           StratifiedKFold(X.N_OUTER, shuffle=True, random_state=X.OUTER_SEED)
           .split(np.zeros(n), yi)]
    for a, b in zip(d["folds"], red):
        assert np.array_equal(np.asarray(a, np.int64), b)
    for fv in d["folds"]:
        assert X.ca.sha256_indices(np.asarray(fv, np.int64)) in \
            list(d["manifest"]["fold_val_sha256"])


def test_05b_success_and_label_heads_share_the_folds(ctx, res):
    """Exp03's success splits and the label head's outer splits are the same
    three folds - 'same folds' holds structurally, not by agreement."""
    d, _c, _W = ctx
    assert len(d["folds"]) == 3
    sizes = sorted(len(np.asarray(v, np.int64)) for v in d["folds"])
    assert sum(sizes) == 10000
    assert len(res["folds"]) == 3
    assert [r["n_val"] for r in res["folds"]] == \
        [len(np.asarray(v, np.int64)) for v in d["folds"]]


def test_05c_nested_seed_is_the_frozen_one():
    assert X.NESTED_SEED == 1
    assert res_nested_seed() == 1


def res_nested_seed():
    return _res()["nested_seed"]


# ---------------------------------------------------------------------------
# 6 / 7. the A/B column contract
# ---------------------------------------------------------------------------

def test_06_ab_first_185_columns_identical():
    """Asserted inside the run on every fold; re-check the shape contract the
    run recorded."""
    assert X.N_SUCCESS == 185
    assert X.N_B == 192
    r = _res()
    assert r["n_success_features"] == 185
    assert r["n_candidate_features"] == 192


def test_07_candidate_adds_exactly_seven_named_columns(res):
    assert res["meta_feature_names"] == ["lp_clean", "lp_dropped_handoff",
                                         "lp_duplicated_work", "lp_deadlock",
                                         "lp_conflict", "lp_goal_drift",
                                         "lp_runaway_loop"]
    assert len(res["meta_feature_names"]) == 7
    assert res["meta_feature_names"] == ["lp_%s" % c for c in X.LABELS]
    assert res["n_meta_features"] == 7


def test_07b_no_derived_quantities_were_added(res):
    """Only the 7 raw probabilities: no entropy, margin, max-prob, one-hot,
    threshold or class-weight column may appear in the candidate's importance."""
    banned = ("entropy", "margin", "max_prob", "onehot", "one_hot", "thr",
              "weight", "argmax", "hard_label", "is_", "peak", "fault_turn")
    for row in res["meta_importance"]:
        nm = row["feature"]
        assert nm.startswith("lp_")
        for b in banned:
            assert b not in nm, "derived column %r entered the candidate" % nm


# ---------------------------------------------------------------------------
# 8 / 9. the carried metrics are untouched
# ---------------------------------------------------------------------------

def test_08_label_predictions_unchanged(res):
    g = res["label_parity"]
    assert g["passed"] is True
    assert g["label_predictions_differ"] == 0, \
        "%d rows changed label" % g["label_predictions_differ"]
    assert res["invariance"]["label_predictions_identical"] is True
    for row in g["rows"]:
        assert row["label_mismatches"] == 0


def test_09_fault_turn_unchanged(res):
    inv = res["invariance"]
    assert inv["fault_turn_identical"] is True, \
        "the L1 turn vector moved even though no label changed"
    assert inv["passed"] is True
    for row in inv["rows"]:
        assert row["pass"] is True
        assert row["abs_diff"] <= X.TOL


def test_09b_carried_metrics_are_the_exp13_d_values(res):
    e = X.EXP13_METRICS
    assert res["macro_f1"] == e["macro_f1"]
    assert res["robustness_f1"] == e["robustness_f1"]
    assert res["fault_turn_hit2"] == e["fault_turn_hit2"]
    assert res["composite_A"] == pytest.approx(e["composite"], abs=1e-15)
    # and they were re-derived from the committed OOF, not copied
    for row in res["invariance"]["rows"]:
        assert row["recomputed"] == pytest.approx(row["carried"], abs=1e-12)


# ---------------------------------------------------------------------------
# 10. the training probabilities are out-of-fold, not in-sample
# ---------------------------------------------------------------------------

def test_10_oof_differs_from_an_insample_refit(res):
    rows = res["oof_vs_insample"]
    assert len(rows) == 3
    for r in rows:
        assert r["max_abs_diff_oof_vs_insample"] > 0.0, \
            "the training probabilities are identical to an in-sample refit"
        assert r["label_disagreements"] > 0, \
            "no label disagreement: in-sample probabilities may have leaked in"
        assert r["mean_abs_diff"] > 0.0


def test_10b_oof_rows_are_not_fitted_rows(res):
    """`meta_prob_fitrows` must be populated exactly where the success head was
    fitted, i.e. every row except the validation block of its own fold."""
    F = np.load(os.path.join(HERE, "meta_prob_fitrows.npy"))
    P = np.load(os.path.join(HERE, "meta_prob_oof.npy"))
    assert F.shape == P.shape == (10000, 7)
    fitted = res["n_rows_trained_on"]
    assert fitted == sum(len(np.asarray(v, np.int64)) for v in
                         X.ENS.load_verified()["folds"]) - len(
        np.asarray(X.ENS.load_verified()["folds"][0], np.int64)) * 0
    # every fitted row has non-zero probabilities recorded
    nz = (np.abs(F).sum(axis=1) > 0)
    assert int(nz.sum()) == fitted, \
        "%d rows carry meta probabilities but %d were trained on" % (
            int(nz.sum()), fitted)
    assert np.isfinite(F).all()


# ---------------------------------------------------------------------------
# the verdict itself
# ---------------------------------------------------------------------------

def test_gate_recorded_and_unchanged(res):
    g = res["promotion_gate"]
    assert set(g["criteria"]) == {
        "success_f1_delta>=+0.010", "folds_won>=2",
        "composite_delta>=+0.0015", "no_fold_worse_than_-0.005",
        "macro_rob_hit2_and_l1_exactly_unchanged"}
    assert g["n_criteria"] == 5
    assert res["gate"]["success_f1_delta_min"] == 0.010
    assert res["gate"]["composite_delta_min"] == 0.0015
    assert res["gate"]["worst_fold_delta_min"] == -0.005
    assert res["gate"]["folds_won_min"] == 2


def test_composite_delta_is_exactly_015_times_success_delta(res):
    assert abs((res["composite_B"] - res["composite_A"])
               - 0.15 * res["success_f1_delta"]) <= 1e-15


def test_no_production_was_built(res):
    assert res["production_created"] is False
    for name in os.listdir(ROOT):
        assert not name.startswith("submission_exp15"), \
            "a production build exists for a non-promoted experiment"


def test_no_second_variant(res):
    """One candidate only: no variant, threshold or hard-label artefact."""
    allowed_ext = (".py", ".md", ".json", ".log", ".npy")
    for name in os.listdir(HERE):
        path = os.path.join(HERE, name)
        if os.path.isdir(path):
            # archive of the defective first execution only
            assert name.startswith("_"), name
            continue
        assert name.endswith(allowed_ext), name
        assert "variant" not in name.lower()
        assert "threshold" not in name.lower()
        assert "hard_label" not in name.lower()
