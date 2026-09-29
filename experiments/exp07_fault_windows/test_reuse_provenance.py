"""Provenance tests for reuse.load - fast, no LightGBM, no training.

Every case builds a small synthetic artifact set in a temp directory, writes a
matching manifest, then tampers with exactly ONE thing and asserts the artifact
is REJECTED.  A test that passes because the artifact was accepted is a failed
test: the point of this file is that reuse.load fails closed.

The synthetic set uses the REAL 10000-row train.csv, so shapes, class counts
and fold sizes match production, but every array is cheap random data and the
probabilities are synthetic - nothing here reproduces an Exp07 prediction.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cache as ca  # noqa: E402
import modes as MD  # noqa: E402
import reuse  # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
SYSTEM = "B_plus_window"

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


def build_fixture(tmp, n=10000, seed=0):
    """A complete, fully-verified artifact set.  Returns its context."""
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                        usecols=["run_id", "label"])
    ys = train["label"].values.astype(object)
    yi = np.array([L2I[l] for l in ys], dtype=int)
    run_ids = [str(x) for x in train["run_id"]]

    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(7), size=n).astype(np.float32)
    labels = P.argmax(1).astype(np.int64)

    d = tmp
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "oof_proba_%s.npy" % SYSTEM), P)
    np.save(os.path.join(d, "oof_%s.npy" % SYSTEM), labels)
    with open(os.path.join(d, "oof_runid.json"), "w", encoding="utf-8") as fh:
        json.dump(run_ids, fh)

    folds = reuse.fold_assignments(yi)
    man = {
        "schema": reuse.SCHEMA, "verified": True, "mode": MD.CV,
        "seed": reuse.SEED, "n_outer": reuse.N_OUTER, "n_inner": reuse.N_INNER,
        "n_runs": n, "n_classes": 7, "full_coverage": True,
        "data_sha256": ca.data_sha256(), "code_sha256": ca.code_sha256(),
        "code_files_sha256": ca.code_files_sha256(),
        "runner_model_path_sha256": reuse.runner_model_path_sha256(),
        "run_id_sha256": ca.sha256_run_ids(run_ids),
        "y_true_sha256": ca.sha256_run_ids(ys),
        "proba_sha256": {SYSTEM: ca.sha256_array(P)},
        "label_sha256": {SYSTEM: ca.sha256_array(labels)},
        "fold_val_sha256": [ca.sha256_indices(v) for v in folds],
        "folds": [{"fold": i, "n_val": int(len(v))}
                  for i, v in enumerate(folds)],
    }
    return {"dir": d, "yi": yi, "run_ids": run_ids, "man": man,
            "P": P, "labels": labels, "folds": folds, "tmp": tmp,
            "manifest_path": os.path.join(d, "honest_manifest.json")}


def write_man(ctx, man=None):
    with open(ctx["manifest_path"], "w", encoding="utf-8") as fh:
        json.dump(man if man is not None else ctx["man"], fh, indent=2)


def do_load(ctx, **kw):
    return reuse.load(system=SYSTEM, mode=MD.CV, yi=ctx["yi"],
                      train_run_ids=ctx["run_ids"],
                      artifact_dir=ctx["dir"], **kw)


def expect_reject(ctx, must_mention=None, **kw):
    try:
        do_load(ctx, **kw)
    except reuse.ArtifactRejected as e:
        if must_mention and must_mention.lower() not in str(e).lower():
            raise AssertionError("rejected, but message %r does not mention "
                                 "%r" % (str(e)[:300], must_mention))
        return str(e)
    raise AssertionError("artifact was ACCEPTED but should have been rejected")


def expect_accept(ctx, **kw):
    try:
        r = do_load(ctx, **kw)
    except reuse.ArtifactRejected as e:
        raise AssertionError("artifact was REJECTED but is valid: %s" % e)
    if not r["ok"]:
        raise AssertionError("ok is not True")
    return r


def main():
    tmp = tempfile.mkdtemp(prefix="exp07_reuse_test_")
    try:
        # ---- 1. a healthy artifact is accepted ----------------------------
        def t1():
            ctx = build_fixture(os.path.join(tmp, "ok"))
            write_man(ctx)
            r = expect_accept(ctx)
            assert r["proba"].shape == (10000, 7), r["proba"].shape
            assert r["labels"].shape == (10000,), r["labels"].shape
            assert r["n_runs"] == 10000, r["n_runs"]
            assert len(r["run_ids"]) == 10000
            np.testing.assert_array_equal(r["labels"], ctx["labels"])
            for k in ("data_sha256", "code_sha256", "fold_val_sha256",
                      "proba_sha256", "run_id_sha256"):
                assert k in r["manifest"], k
        check("healthy artifact is ACCEPTED with full shape (10000, 7)", t1)

        # ---- 2. one changed probability value -> reject -------------------
        def t2():
            ctx = build_fixture(os.path.join(tmp, "proba"))
            write_man(ctx)
            P = ctx["P"].copy()
            P[4321, 3] += np.float32(1e-3)      # single value, single row
            np.save(os.path.join(ctx["dir"], "oof_proba_%s.npy" % SYSTEM), P)
            expect_reject(ctx, "proba sha256")
        check("one changed probability value -> REJECTED", t2)

        # ---- 3. changed seed -> reject ------------------------------------
        def t3():
            ctx = build_fixture(os.path.join(tmp, "seed"))
            m = dict(ctx["man"])
            m["seed"] = 1
            write_man(ctx, m)
            expect_reject(ctx, "seed")
        check("changed seed -> REJECTED", t3)

        # ---- 3b. seed change must also change the derived folds -----------
        def t3b():
            ctx = build_fixture(os.path.join(tmp, "seedfold"))
            write_man(ctx)
            yi2 = ctx["yi"].copy()
            # a different seed yields different folds; the consumer re-derives
            # with SEED=0 and must catch it
            from sklearn.model_selection import StratifiedKFold
            other = list(StratifiedKFold(3, shuffle=True, random_state=7)
                         .split(np.zeros(len(yi2)), yi2))
            got = [ca.sha256_indices(np.asarray(v, dtype=np.int64))
                   for _, v in other]
            assert got != ctx["man"]["fold_val_sha256"], \
                "test bug: seed 7 produced identical folds"
        check("a different seed really does change the fold hashes", t3b)

        # ---- 4. changed fold assignment -> reject -------------------------
        def t4():
            ctx = build_fixture(os.path.join(tmp, "fold"))
            m = dict(ctx["man"])
            f = [dict(x) for x in m["folds"]]
            # swap two rows between folds 0 and 1: SAME SIZES, different rows
            v0 = np.setdiff1d(*[np.asarray(x) for x in
                                [np.arange(10000)]] +
                               [np.sort(ctx["folds"][0])])
            a, b = ctx["folds"][0][0], ctx["folds"][1][0]
            new0 = np.sort(np.concatenate([np.setdiff1d(ctx["folds"][0], [a]),
                                           [b]]))
            new1 = np.sort(np.concatenate([np.setdiff1d(ctx["folds"][1], [b]),
                                           [a]]))
            m["fold_val_sha256"] = [ca.sha256_indices(new0),
                                   ca.sha256_indices(new1),
                                   ctx["man"]["fold_val_sha256"][2]]
            m["folds"] = f
            write_man(ctx, m)
            # sizes are unchanged - this is exactly the case a size-only
            # check cannot see
            assert [len(x) for x in ctx["folds"]] == \
                   [len(new0), len(new1), len(ctx["folds"][2])]
            expect_reject(ctx, "indices")
        check("changed fold assignment (same sizes) -> REJECTED", t4)

        # ---- 5. changed run_id order -> reject ----------------------------
        def t5():
            ctx = build_fixture(os.path.join(tmp, "runid"))
            write_man(ctx)
            swapped = list(ctx["run_ids"])
            swapped[0], swapped[1] = swapped[1], swapped[0]
            with open(os.path.join(ctx["dir"], "oof_runid.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(swapped, fh)
            expect_reject(ctx, "order")
        check("reordered run_id -> REJECTED", t5)

        # ---- 5b. train.csv order differs from stored ----------------------
        def t5b():
            ctx = build_fixture(os.path.join(tmp, "runid_train"))
            write_man(ctx)
            ctx["run_ids"] = list(ctx["run_ids"])
            ctx["run_ids"][0], ctx["run_ids"][5] = \
                ctx["run_ids"][5], ctx["run_ids"][0]
            expect_reject(ctx, "order")
        check("current train order != stored order -> REJECTED", t5b)

        # ---- 6. each missing mandatory field -> reject --------------------
        def t_missing(field):
            def t():
                ctx = build_fixture(os.path.join(tmp, "miss_" + field))
                m = dict(ctx["man"])
                m.pop(field, None)
                write_man(ctx, m)
                expect_reject(ctx, None)
            check("missing required manifest field %r -> REJECTED" % field,
                  t)

        for field in ("verified", "seed", "n_outer", "n_inner", "schema",
                      "data_sha256", "code_sha256", "run_id_sha256",
                      "proba_sha256", "label_sha256", "fold_val_sha256",
                      "full_coverage", "n_runs", "n_classes",
                      "runner_model_path_sha256"):
            t_missing(field)

        # ---- 7. wrong shapes -> reject ------------------------------------
        def t7a():
            ctx = build_fixture(os.path.join(tmp, "shape_cols"))
            P = ctx["P"][:, :5].copy()           # (10000, 5)
            np.save(os.path.join(ctx["dir"], "oof_proba_%s.npy" % SYSTEM), P)
            expect_reject(ctx, "shape")
        check("wrong proba shape (10000, 5) -> REJECTED", t7a)

        def t7b():
            ctx = build_fixture(os.path.join(tmp, "shape_rows"))
            P = ctx["P"][:9999].copy()           # 9999 rows
            np.save(os.path.join(ctx["dir"], "oof_proba_%s.npy" % SYSTEM), P)
            expect_reject(ctx, "row")
        check("wrong row count (9999) -> REJECTED", t7b)

        def t7c():
            ctx = build_fixture(os.path.join(tmp, "shape_lab"))
            L = ctx["labels"][:500].copy()
            m = dict(ctx["man"])
            m["label_sha256"] = {SYSTEM: ca.sha256_array(L)}  # hash updated
            write_man(ctx, m)
            np.save(os.path.join(ctx["dir"], "oof_%s.npy" % SYSTEM), L)
            expect_reject(ctx, "label")
        check("wrong label array shape -> REJECTED", t7c)

        # ---- extra: non-finite, argmax mismatch, tampered labels ---------
        def t_nan():
            ctx = build_fixture(os.path.join(tmp, "nan"))
            P = ctx["P"].copy()
            P[10, 0] = np.nan
            m = dict(ctx["man"])
            m["proba_sha256"] = {SYSTEM: ca.sha256_array(P)}  # hash updated
            write_man(ctx, m)
            np.save(os.path.join(ctx["dir"], "oof_proba_%s.npy" % SYSTEM), P)
            expect_reject(ctx, "finite")
        check("NaN in probabilities -> REJECTED", t_nan)

        def t_argmax():
            ctx = build_fixture(os.path.join(tmp, "argmax"))
            L = ctx["labels"].copy()
            L[7] = (L[7] + 1) % 7                 # label no longer = argmax
            m = dict(ctx["man"])
            m["label_sha256"] = {SYSTEM: ca.sha256_array(L)}
            write_man(ctx, m)
            np.save(os.path.join(ctx["dir"], "oof_%s.npy" % SYSTEM), L)
            expect_reject(ctx, "argmax")
        check("labels != argmax(proba) -> REJECTED", t_argmax)

        def t_partial():
            ctx = build_fixture(os.path.join(tmp, "partial"))
            m = dict(ctx["man"])
            m["full_coverage"] = False
            m["early_stopped"] = True
            write_man(ctx, m)
            expect_reject(ctx, "complete")
        check("early-stopped (partial) run -> REJECTED", t_partial)

        def t_nomanifest():
            ctx = build_fixture(os.path.join(tmp, "nomanifest"))
            write_man(ctx)
            os.remove(ctx["manifest_path"])
            # yi/run_ids are mandatory, so pass them - otherwise the mandatory
            # check would fire first and this test would prove nothing about
            # the missing manifest
            try:
                reuse.load(system=SYSTEM, mode=MD.CV, yi=ctx["yi"],
                           train_run_ids=ctx["run_ids"],
                           artifact_dir=ctx["dir"])
            except reuse.ArtifactRejected as e:
                assert "no manifest" in str(e).lower(), e
                return
            raise AssertionError("missing manifest was ACCEPTED")
        check("absent manifest -> REJECTED", t_nomanifest)

        def t_unverified():
            ctx = build_fixture(os.path.join(tmp, "unverified"))
            m = dict(ctx["man"])
            m["verified"] = False
            write_man(ctx, m)
            expect_reject(ctx, "verified")
        check("verified=false -> REJECTED", t_unverified)

        def t_data():
            ctx = build_fixture(os.path.join(tmp, "data"))
            m = dict(ctx["man"])
            m["data_sha256"] = "0" * 64
            write_man(ctx, m)
            expect_reject(ctx, "train.csv sha256")
        check("changed train.csv hash -> REJECTED", t_data)

        def t_code():
            ctx = build_fixture(os.path.join(tmp, "code"))
            m = dict(ctx["man"])
            m["code_sha256"] = "0" * 64
            write_man(ctx, m)
            expect_reject(ctx, "code sha256")
        check("changed feature/model code hash -> REJECTED", t_code)

        # ==================================================================
        # Hardening round 2: full code closure, whole model path, mandatory
        # caller context, expected dataset.
        # ==================================================================

        # ---- per-file coverage: each model-determining source is hashed ---
        def t_closure_covers(rel):
            def t():
                ctx = build_fixture(os.path.join(tmp, "cov_" + rel.replace(
                    "/", "_")))
                files = ctx["man"]["code_files_sha256"]
                assert rel in files, "%s is NOT in code_files_sha256" % rel
                assert files[rel] == ca.sha256_file(
                    os.path.join(ROOT, rel.replace("/", os.sep))), rel
                # and tampering with just that one file must be detected
                m = dict(ctx["man"])
                m["code_files_sha256"] = dict(files)
                m["code_files_sha256"][rel] = "0" * 64
                write_man(ctx, m)
                expect_reject(ctx, "model-determining source")
            check("%s is hashed and a change to it is detected" % rel, t)

        for rel in ("experiments/exp07_fault_windows/aggregate.py",
                    "experiments/exp07_fault_windows/grouping.py",
                    "experiments/exp07_fault_windows/parallel.py",
                    "experiments/exp07_fault_windows/common.py",
                    "experiments/exp07_fault_windows/modes.py",
                    "experiments/exp07_fault_windows/window_features.py",
                    "experiments/exp07_fault_windows/window_dataset.py",
                    "baseline/features.py", "baseline/localize.py",
                    "evaluation/metrics.py"):
            t_closure_covers(rel)

        def t_runner_model_path():
            ctx = build_fixture(os.path.join(tmp, "modelpath"))
            write_man(ctx)
            m = dict(ctx["man"])
            m["runner_model_path_sha256"] = "0" * 64
            write_man(ctx, m)
            expect_reject(ctx, "model path")
        check("change to the model-determining runner region -> REJECTED",
              t_runner_model_path)

        def t_model_region_covers_steps():
            """The hashed region must really contain every model step."""
            lines = open(reuse.RUNNER_PATH, encoding="utf-8").readlines()
            txt = "".join(reuse._model_region_lines(lines))
            need = ("for f in fold_ids:", "mA = lgb_label().fit",
                    "mA.predict_proba", "verify_stacking", "payloads.append",
                    "par.run_inner_fits", "mw.fit(", "mw.predict_proba",
                    "grp.predict_runs", "grp.aggregate_block",
                    "Xagg_tr[assign == j] = sub[assign == j]",
                    "Xb_tr = pd.DataFrame", "mB = lgb_label().fit",
                    "mB.predict_proba", "oof_proba[B_KEY][va] = pB")
            for k in need:
                assert k in txt, "model region is missing %r" % k
            for k in ("np.save(", "to_csv", "json.dump", "honest_manifest"):
                assert k not in txt, \
                    "model region wrongly includes artifact writer %r" % k
        check("model region covers fits/aggregation/B-features, not writing",
              t_model_region_covers_steps)

        def t_region_not_found():
            """Fail closed: an unhashable runner must be a rejection."""
            bad = os.path.join(tmp, "broken_runner.py")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("def run():\n    return 1\n")
            try:
                reuse.runner_model_path_sha256(path=bad)
            except reuse.ArtifactRejected:
                return
            raise AssertionError("unlocatable model region was not rejected")
        check("unlocatable model region -> REJECTED (fail closed)",
              t_region_not_found)

        def t_yi_none():
            ctx = build_fixture(os.path.join(tmp, "yi_none"))
            write_man(ctx)
            try:
                reuse.load(system=SYSTEM, mode=MD.CV, yi=None,
                           train_run_ids=ctx["run_ids"],
                           artifact_dir=ctx["dir"])
            except reuse.ArtifactRejected as e:
                assert "yi is REQUIRED" in str(e), e
                return
            raise AssertionError("yi=None was ACCEPTED")
        check("yi=None -> REJECTED", t_yi_none)

        def t_rids_none():
            ctx = build_fixture(os.path.join(tmp, "rids_none"))
            write_man(ctx)
            try:
                reuse.load(system=SYSTEM, mode=MD.CV, yi=ctx["yi"],
                           train_run_ids=None, artifact_dir=ctx["dir"])
            except reuse.ArtifactRejected as e:
                assert "train_run_ids is REQUIRED" in str(e), e
                return
            raise AssertionError("train_run_ids=None was ACCEPTED")
        check("train_run_ids=None -> REJECTED", t_rids_none)

        def t_short_yi():
            ctx = build_fixture(os.path.join(tmp, "short_yi"))
            write_man(ctx)
            try:
                reuse.load(system=SYSTEM, mode=MD.CV, yi=ctx["yi"][:9999],
                           train_run_ids=ctx["run_ids"],
                           artifact_dir=ctx["dir"])
            except reuse.ArtifactRejected as e:
                assert "len(yi)" in str(e), e
                return
            raise AssertionError("len(yi)!=10000 was ACCEPTED")
        check("len(yi) != 10000 -> REJECTED", t_short_yi)

        def t_short_rids():
            ctx = build_fixture(os.path.join(tmp, "short_rids"))
            write_man(ctx)
            try:
                reuse.load(system=SYSTEM, mode=MD.CV, yi=ctx["yi"],
                           train_run_ids=ctx["run_ids"][:9999],
                           artifact_dir=ctx["dir"])
            except reuse.ArtifactRejected as e:
                assert "len(train_run_ids)" in str(e), e
                return
            raise AssertionError("len(train_run_ids)!=10000 was ACCEPTED")
        check("len(train_run_ids) != 10000 -> REJECTED", t_short_rids)

        def t_manifest_nruns():
            ctx = build_fixture(os.path.join(tmp, "bad_nruns"))
            m = dict(ctx["man"])
            m["n_runs"] = 9999
            write_man(ctx, m)
            expect_reject(ctx, "n_runs")
        check("manifest n_runs != 10000 -> REJECTED", t_manifest_nruns)

        def t_unverified_reader():
            """The exploratory reader must exist and flag itself."""
            import warnings
            d = MD.outdir(MD.CV)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = reuse.load_unverified(system=SYSTEM, mode=MD.CV)
            assert r["ok"] is False, "load_unverified must not claim ok"
            assert r["provenance"] is None, r["provenance"]
            assert r["proba"] is not None and r["proba"].shape == (10000, 7), \
                r["proba"].shape
            assert w, "load_unverified must warn"
        check("load_unverified() reads but flags ok=False", t_unverified_reader)

        def t_real_artifact():
            """The REAL Exp07 CV artifact must be ACCEPTED."""
            train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                                usecols=["run_id", "label"])
            ys = train["label"].values.astype(object)
            yi = np.array([L2I[l] for l in ys], dtype=int)
            rids = [str(x) for x in train["run_id"]]
            r = reuse.load(system=SYSTEM, mode=MD.CV, yi=yi,
                           train_run_ids=rids)
            assert r["ok"], r
            assert r["proba"].shape == (10000, 7), r["proba"].shape
            assert r["labels"].shape == (10000,), r["labels"].shape
            assert r["n_runs"] == 10000, r["n_runs"]
            assert np.isfinite(r["proba"]).all()
            assert (r["proba"].argmax(1) == r["labels"]).all()
        check("real Exp07 CV artifact -> ACCEPTED", t_real_artifact)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    npass = sum(1 for _, ok, _ in results if ok)
    print("\n%d/%d passed" % (npass, len(results)))
    for name, ok, err in results:
        if not ok:
            print("  FAILED: %s -> %s" % (name, err))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())