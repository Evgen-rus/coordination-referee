"""Fail-closed tests for the Exp08 peak loader.

A test that passes because the artifact was ACCEPTED is a failed test.  Each
case copies the REAL verified artifact set into a temp directory, tampers with
exactly one thing, and asserts ``load_exp07.load`` rejects it - OR, for the
"manifest without peak hashes" case, that it rejects even though every
run-level artifact is byte-perfect.

Uses the real 10 000-row ``train.csv`` so shapes, fold derivation and run order
are the production ones; no model is fitted and nothing is written outside the
temp directory.
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
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
sys.path.insert(0, HERE)
sys.path.insert(0, EXP07)

import modes as MD      # noqa: E402
import load_exp07 as L  # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}

ARTIFACTS = ("oof_proba_B_plus_window.npy", "oof_B_plus_window.npy",
             "oof_runid.json", "honest_manifest.json", "results.json",
             "window_peak_pos.npy", "window_peak_prob.npy",
             "window_oof_run.npy", "window_oof_y.npy", "window_oof_pred.npy")

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
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                        usecols=["run_id", "label"])
    yi = np.array([L2I[l] for l in train["label"].values.astype(object)], dtype=int)
    return yi, [str(x) for x in train["run_id"]]


def stage(src, tmp, name):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    for fn in ARTIFACTS:
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(d, fn))
    return d


def man_of(d):
    with open(os.path.join(d, "honest_manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def put_man(d, m):
    with open(os.path.join(d, "honest_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def expect_reject(d, yi, rids, must_mention=None):
    try:
        L.load(yi=yi, train_run_ids=rids, artifact_dir=d)
    except (L.PeakRejected, L.reuse.ArtifactRejected) as e:
        msg = str(e)
        if must_mention and must_mention.lower() not in msg.lower():
            raise AssertionError("rejected, but message %r does not mention %r"
                                 % (msg[:300], must_mention))
        return msg
    raise AssertionError("artifact was ACCEPTED but should have been rejected")


def main():
    src = MD.outdir(MD.CV)
    yi, rids = ctx()
    tmp = tempfile.mkdtemp(prefix="exp08_peak_test_")
    try:
        # ---- 0. the real, untouched artifacts must be ACCEPTED -------------
        def t0():
            r = L.load(yi=yi, train_run_ids=rids, artifact_dir=src)
            assert r["ok"] is True
            assert r["peak_pos"].shape == (10000, 7), r["peak_pos"].shape
            assert r["peak_prob"].shape == (10000, 7), r["peak_prob"].shape
            assert r["proba"].shape == (10000, 7), r["proba"].shape
            assert (r["proba"].argmax(1) == r["labels"]).all()
            assert int(r["peak_pos"].min()) >= 0
        check("real Exp07 artifacts + sealed peaks -> ACCEPTED", t0)

        # ---- 0b. a staged byte-identical copy is also accepted -------------
        def t0b():
            d = stage(src, tmp, "clean")
            r = L.load(yi=yi, train_run_ids=rids, artifact_dir=d)
            assert r["peak_pos"].shape == (10000, 7)
        check("staged byte-identical copy -> ACCEPTED", t0b)

        # ---- 1. manifest with NO peak hashes at all -------------------------
        def t1():
            d = stage(src, tmp, "nopeak")
            m = man_of(d)
            m.pop("window_peak_sha256", None)
            put_man(d, m)
            expect_reject(d, yi, rids, "unsealed")
        check("manifest without window_peak_sha256 -> REJECTED (unsealed)", t1)

        # ---- 2. a peak key removed -----------------------------------------
        def t2():
            d = stage(src, tmp, "missingkey")
            m = man_of(d)
            m["window_peak_sha256"].pop("pos")
            put_man(d, m)
            expect_reject(d, yi, rids, "missing required key")
        check("window_peak_sha256 missing 'pos' -> REJECTED", t2)

        # ---- 3. one flipped peak position ----------------------------------
        def t3():
            d = stage(src, tmp, "tampered_pos")
            p = np.load(os.path.join(d, "window_peak_pos.npy"))
            p[4242, 3] = p[4242, 3] + 1
            np.save(os.path.join(d, "window_peak_pos.npy"), p)
            expect_reject(d, yi, rids, "sha256 mismatch")
        check("one flipped peak position -> REJECTED", t3)

        # ---- 4. one flipped peak probability -------------------------------
        def t4():
            d = stage(src, tmp, "tampered_prob")
            q = np.load(os.path.join(d, "window_peak_prob.npy"))
            q[7, 1] = 0.999
            np.save(os.path.join(d, "window_peak_prob.npy"), q)
            expect_reject(d, yi, rids, "sha256 mismatch")
        check("one flipped peak probability -> REJECTED", t4)

        # ---- 5. peak file deleted ------------------------------------------
        def t5():
            d = stage(src, tmp, "nopos")
            os.remove(os.path.join(d, "window_peak_pos.npy"))
            expect_reject(d, yi, rids, "missing peak artifact")
        check("deleted window_peak_pos.npy -> REJECTED", t5)

        # ---- 6. dtype rewritten (array hash catches it) --------------------
        def t6():
            d = stage(src, tmp, "dtype")
            p = np.load(os.path.join(d, "window_peak_pos.npy"))
            np.save(os.path.join(d, "window_peak_pos.npy"),
                    p.astype(np.int64))     # same values, different dtype
            msg = expect_reject(d, yi, rids)
            assert "dtype" in msg.lower() or "sha256" in msg.lower(), msg
        check("peak_pos re-saved as int64 -> REJECTED", t6)

        # ---- 7. a run never validated (negative peak) ----------------------
        def t7():
            d = stage(src, tmp, "unvalidated")
            p = np.load(os.path.join(d, "window_peak_pos.npy"))
            p[11, 2] = -1
            np.save(os.path.join(d, "window_peak_pos.npy"), p)
            m = man_of(d)          # keep hashes consistent: is the DATA sane?
            import cache as ca
            m["window_peak_sha256"]["pos"] = ca.sha256_file(
                os.path.join(d, "window_peak_pos.npy"))
            m["window_peak_sha256_array"]["pos"] = ca.sha256_array(p)
            put_man(d, m)
            # hashes now agree, so only the CONTENT check can catch this
            expect_reject(d, yi, rids, "never validated")
        check("peak_pos marks an unvalidated run (-1) -> REJECTED by content",
              t7)

        # ---- 8. window_oof_run covers fewer runs ---------------------------
        def t8():
            d = stage(src, tmp, "coverage")
            wr = np.load(os.path.join(d, "window_oof_run.npy"))
            wr = wr[wr != 9999]
            np.save(os.path.join(d, "window_oof_run.npy"), wr)
            expect_reject(d, yi, rids, "sha256 mismatch")
        check("window_oof_run no longer covers run 9999 -> REJECTED", t8)

        # ---- 9. yi / train_run_ids remain mandatory ------------------------
        def t9():
            d = stage(src, tmp, "mandatory")
            for kwargs in ({"yi": None}, {"train_run_ids": None}):
                try:
                    L.load(artifact_dir=d, **kwargs)
                except (L.PeakRejected, L.reuse.ArtifactRejected) as e:
                    assert "is REQUIRED" in str(e), e
                    continue
                raise AssertionError("missing mandatory arg was ACCEPTED")
        check("yi=None / train_run_ids=None -> REJECTED", t9)

        # ---- 10. run-level provenance still enforced -----------------------
        def t10():
            d = stage(src, tmp, "runlevel")
            rids2 = list(rids)
            rids2[0], rids2[1] = rids2[1], rids2[0]
            expect_reject(d, yi, rids2, "order")
        check("reordered train_run_ids -> REJECTED (run-level check)", t10)

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