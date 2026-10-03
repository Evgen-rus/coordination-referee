"""Exp15 smoke check: everything except the long nested CV.

Runs the input loading, the declared-baseline reads, the parameter asserts,
the fold determinism assert, the success-feature leakage probe and a SINGLE
small corrected-agg block, on a 900-row slice of the corpus.  It is a
plumbing check, not a result: nothing it prints is reported.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "exp15_run", os.path.join(HERE, "run_experiment.py"))
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)

import lightgbm as lgb  # noqa: E402


def main():
    t0 = time.time()
    print("imports OK; META_NAMES=%s" % X.META_NAMES, flush=True)
    assert len(X.META_NAMES) == 7
    assert X.N_B == X.N_SUCCESS + X.N_META == 192

    d = X.ENS.load_verified()
    c, Wd = X.ENS._prepare()
    n = c["n"]
    print("verified: runs=%d foundation=%s success=%s windows=%s"
          % (n, c["foundation"].shape, c["success"].shape, Wd["X"].shape),
          flush=True)

    sp = X.success_params()
    print("success params match production: %s"
          % {k: sp[k] for k in sorted(X.SUCCESS_PARAMS_EXPECTED)}, flush=True)

    committed = np.load(os.path.join(X.EXP13, "oof_D_corrected_wg.npy"),
                        allow_pickle=False)
    print("committed Exp13 D sha matches: %s"
          % (X.sha32(committed) == X.EXP13_SHA_D), flush=True)
    print("EXP13_METRICS composite = %.16f"
          % X.EXP13_METRICS["composite"], flush=True)

    probe = X.success_feature_leakage_probe(c["runs"], 60)
    print("success feature leakage probe: identical=%s max|d|=%.1e cols=%d"
          % (probe["identical"], probe["max_abs_diff"], probe["n_columns"]),
          flush=True)

    # one small corrected-agg block, to exercise the whole nested machinery
    yi = np.asarray(d["yi"], np.int64)
    tr = np.arange(0, 900, dtype=np.int64)
    va = np.arange(900, 1200, dtype=np.int64)
    Xwg = X.W.build_matrix(c["runs"])
    Xa, Xb, meta = X.corrected_agg_block(tr, va, c, Wd, "smoke")
    print("corrected_agg_block: %s  meta=%s"
          % (Xa.shape, Xb.shape), flush=True)
    assert Xa.shape == (len(tr), X.ag.N_AGG)
    assert Xb.shape == (len(va), X.ag.N_AGG)
    assert np.isfinite(Xa).all() and np.isfinite(Xb).all()

    X312 = X.label_matrix_312(tr, Xa, c, Xwg)
    print("312 matrix: %s (expected cols %d)"
          % (X312.shape, X.N_LABEL_312), flush=True)
    assert X312.shape[1] == X.N_LABEL_312 == 312

    clf = lgb.LGBMClassifier(**X.ENS.label_params(42)).fit(X312, yi[tr])
    p = clf.predict_proba(X.label_matrix_312(va, Xb, c, Xwg))
    print("label proba: %s  argmax dist=%s"
          % (p.shape, np.bincount(p.argmax(1), minlength=7).tolist()), flush=True)

    proof = X.exclusion_proof(
        tr, [("n0", tr[:600], tr[600:]), ("n1", tr[600:], tr[:600])], n,
        log=print)
    print("exclusion_proof all_excluded=%s" % proof["all_excluded"],
          flush=True)
    assert proof["all_excluded"]

    s_true = np.asarray(c["train"]["success"].values, np.int64)
    X185 = np.asarray(c["success"].values, np.float64)[:1200]
    y1200 = s_true[:1200]
    m = lgb.LGBMClassifier(**sp).fit(X185[:900], y1200[:900])
    print("success smoke F1 = %.6f" % X.prf_succ(y1200, m.predict(X185[900:]))
          ["f1"], flush=True)
    print("smoke OK in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
