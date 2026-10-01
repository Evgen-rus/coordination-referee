"""Exp13 diagnostic - are the 51 window aggregates FOLD-SPECIFIC?

THE QUESTION
------------
My first Exp13 attempt built ONE global (10000, 300) matrix and filled it fold
by fold with ``Xagg[hold] = ...``.  That assignment writes into rows that are
outer-TRAIN for the current fold - but those same global rows are also
outer-TRAIN for the OTHER outer folds (each run is in train for 2 of 3 folds).
So the last fold to touch a row decides that row's 51 train-columns, and the
result is not any fold's honest training matrix.

This script decides whether that invalidates the global representation.  It
fits NO label model and computes NO metric: it only builds, per outer fold, the
51 train-aggregates Exp07 would build, and compares them for runs that are
outer-train in >= 2 folds.

If those rows differ between folds, the global matrix is invalid and Exp13 must
move to the fold-local scheme.  If they were identical, my hypothesis is wrong
and the refit mismatch has another cause.
"""

from __future__ import annotations

import gc
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
EXP08 = os.path.join(ROOT, "experiments", "exp08_window_localizer")
EXP10 = os.path.join(ROOT, "experiments", "exp10_ab_probability_blend")
EXP11 = os.path.join(ROOT, "experiments", "exp11_label_seed_ensemble")
for _p in (EXP11, EXP08, EXP10, EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aggregate as ag                       # noqa: E402
import cache as ca                           # noqa: E402
import grouping as grp                       # noqa: E402
import parallel as par                       # noqa: E402
import runner as R                           # noqa: E402
import window_dataset as wd                   # noqa: E402

import ensemble as ENS                        # noqa: E402  (Exp11, reuse)

N_INNER = ENS.N_INNER
INNER_SEED = ENS.INNER_SEED
CLEAN = 0
OUT = []


def log(m=""):
    print(m)
    OUT.append(str(m))


def agg_hash(vec):
    """Order-sensitive hash of one run's 51 aggregates."""
    return ca.sha256_array(np.ascontiguousarray(
        np.asarray(vec, dtype=np.float64)))


def main():
    print("=" * 78)
    print("Exp13 DIAGNOSTIC - fold-specificity of the 51 window aggregates")
    print("=" * 78)
    log("No label model is fitted and no metric is computed here.  Only the")
    log("train-aggregate path is exercised, exactly as Exp07 builds it.")

    d = ENS.load_verified()
    c, W = ENS._prepare()
    yi, n = c["yi"], c["n"]
    folds = [np.asarray(v, np.int64) for v in d["folds"]]
    found = c["foundation"]
    Xw, yw, rw = W["X"], W["y"], W["run"]
    CW = wd.class_weight_vector()
    log("runs=%d  folds=%d  foundation=%s  windows=%s"
        % (n, len(folds), found.shape, Xw.shape))

    all_ix = np.arange(n, dtype=np.int64)
    fidx = [(all_ix[~np.isin(all_ix, fi)], fi) for fi in folds]

    # per-run aggregate vectors, keyed by (global run, fold in whose TRAIN it is)
    per_run = defaultdict(dict)          # run -> {fold: agg51}
    train_membership = defaultdict(list)  # run -> [folds where it is train]

    for f, (tr, va) in enumerate(fidx):
        tr_runs = np.array(sorted(set(tr.tolist())))
        inner = list(StratifiedKFold(N_INNER, shuffle=True,
                                     random_state=INNER_SEED)
                     .split(tr_runs, yi[tr_runs]))
        assign = np.zeros(len(tr_runs), dtype=int)
        for j, (_, hl) in enumerate(inner):
            assign[hl] = j
        R.verify_stacking(len(tr_runs), assign, N_INNER, "fold %d" % f)

        payloads = []
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            keep = tr_runs[assign != j]
            payloads.append((f, j, Xw, yw, CW, np.isin(rw, keep),
                             np.isin(rw, hold), R.WINDOW_PARAMS))
        probs = par.run_inner_fits(payloads)
        del payloads
        gc.collect()

        Xagg_tr = np.zeros((len(tr_runs), ag.N_AGG))
        slot = {int(r): k for k, r in enumerate(tr_runs)}
        for j in range(N_INNER):
            hold = tr_runs[assign == j]
            blocks = grp.predict_runs(probs[(f, j)], rw[np.isin(rw, hold)],
                                      len(tr_runs))
            sub, _, _ = grp.aggregate_block(blocks, ag, slot, len(tr_runs))
            Xagg_tr[assign == j] = sub[assign == j]
        del probs
        gc.collect()

        for k, r in enumerate(tr_runs):
            per_run[int(r)][f] = Xagg_tr[k].copy()
            train_membership[int(r)].append(f)
        log("  fold %d: train-aggregates built for %d runs" % (f, len(tr_runs)))

    # ------------------------------------------------------------------
    log()
    log("=" * 78)
    log("RESULT 1 - runs whose train aggregates DIFFER between outer folds")
    log("=" * 78)
    multi = [r for r, fs in train_membership.items() if len(fs) >= 2]
    differing, identical = [], []
    for r in multi:
        hs = {f: agg_hash(per_run[r][f]) for f in sorted(per_run[r])}
        if len(set(hs.values())) > 1:
            differing.append(r)
        else:
            identical.append(r)
    log("runs in >=2 outer-train folds        : %d" % len(multi))
    log("  aggregates DIFFER between folds   : %d  (%.2f%%)"
        % (len(differing), 100.0 * len(differing) / max(1, len(multi))))
    log("  aggregates IDENTICAL              : %d" % len(identical))

    log()
    log("  first 10 runs in >=2 train folds, with their per-fold hash:")
    log("  %-8s %-38s %-12s %s" % ("run", "train folds", "hashes equal?", "detail"))
    for r in sorted(multi)[:10]:
        hs = {f: agg_hash(per_run[r][f])[:12] for f in sorted(per_run[r])}
        eq = "yes" if len(set(hs.values())) == 1 else "NO"
        log("  %-8d %-38s %-12s %s"
            % (r, ",".join("f%d" % f for f in sorted(per_run[r])), eq,
               " ".join("f%d=%s" % (f, hs[f]) for f in sorted(hs))))

    # A zero-vector (the Exp07 index quirk) vs a real vector also differs; say so.
    log()
    n_zero_diff = 0
    for r in differing:
        zs = [not per_run[r][f].any() for f in sorted(per_run[r])]
        if len(set(zs)) > 1:
            n_zero_diff += 1
    log("  of the differing runs, %d have an ALL-ZERO 51-vector in one fold"
        % n_zero_diff)
    log("  and a NON-ZERO one in another (Exp07's global-index quirk bites")
    log("  a different third of the rows in each fold)")

    # ------------------------------------------------------------------
    log()
    log("=" * 78)
    log("RESULT 2 - the global X300 I actually built")
    log("=" * 78)
    last_writer = {}
    for f, (tr, va) in enumerate(fidx):
        for r in tr.tolist():
            last_writer[int(r)] = f
    n_bad_global = 0
    for r in differing:
        # in my global matrix, row r holds fold last_writer[r]'s vector
        held = last_writer[r]
        if agg_hash(per_run[r][held]) != agg_hash(
                per_run[r][sorted(per_run[r])[0]]):
            n_bad_global += 1
    log("global matrix stores, per run, only the aggregates of ONE fold.")
    log("runs whose stored fold is NOT the only fold they belong to: %d"
        % n_bad_global)
    log("  -> for a run in train for folds {0,1,2} minus one validation fold,")
    log("     the global matrix silently presents the WRONG honest")
    log("     representation to 1 of the 2 label fits that must see it.")
    log()
    if differing:
        log("VERDICT: the 51 train-aggregates ARE fold-specific.")
        log("         A single global (10000, 300) matrix CANNOT represent")
        log("         Exp07's training path; the hypothesis is CONFIRMED and")
        log("         Exp13 must use fold-local 300-column matrices.")
    else:
        log("VERDICT: aggregates are IDENTICAL across folds - hypothesis")
        log("         REFUTED; look elsewhere for the refit mismatch.")

    with open(os.path.join(HERE, "diag_fold_specific.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())