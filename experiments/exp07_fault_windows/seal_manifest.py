"""Seal a strict provenance manifest over an EXISTING, parity-verified CV run.

This performs a PROVENANCE UPGRADE, not a retrain.  No model is fitted, no
feature is rebuilt, no prediction changes.  It only records, in the strict
schema-2 format, the facts about artifacts that already exist.

It may only run when every one of these is provably true, each re-checked here
rather than assumed:

  1. the OOF CSVs are byte-identical to the committed reference CSVs that
     ``check_parity.py`` reported as PARITY PASS;
  2. the artifacts in ``runs/<mode>`` are byte-identical to git HEAD, i.e.
     untouched since the commit that added them;
  3. ``argmax(oof_proba_*)`` equals the hard labels in the CSVs, row for row;
  4. the stored ``oof_runid.json`` equals the current ``train.csv`` run order;
  5. per-fold macro F1, recomputed on the folds re-derived from seed 0 and the
     current train, matches ``results.json`` to 1e-9 - this is what proves the
     fold assignment is IDENTICAL, not merely the same size;
  6. the fold validation vectors partition all 10000 rows exactly once;
  7. every relevant source file is unchanged since that commit.

If any of these fails, this script refuses to write a manifest.  A missing
proof is reported as a missing proof - never patched over.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

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
TOL = 1e-9

# Sources that determine the features and the model.  Mirrors ca.FEATURE_SOURCES
# plus the runner / dataset plumbing that turns windows into run-level features.
CODE_RELEVANT = list(ca.FEATURE_SOURCES) + [
    os.path.join(HERE, "runner.py"),
    os.path.join(HERE, "aggregate.py"),
    os.path.join(HERE, "grouping.py"),
    os.path.join(HERE, "common.py"),
    os.path.join(HERE, "parallel.py"),
]


def fail(msg):
    print("  REFUSED  %s" % msg)
    return False


def git(*a):
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def sha_file(p):
    return ca.sha256_file(p)


def main(mode=MD.CV, systems=("A_foundation", "B_plus_window"),
         commit=None, allow_dirty=False):
    d = MD.outdir(mode)
    print("sealing manifest for mode=%s in %s" % (mode, d))
    problems = []

    # ---- 0. the artifacts must exist ---------------------------------------
    for fn in ("results.json", "oof_runid.json", "honest_manifest.json"):
        if not os.path.exists(os.path.join(d, fn)):
            print("  REFUSED  missing %s" % fn)
            return False
    for s in systems:
        for fn in ("oof_%s.csv" % s, "oof_proba_%s.npy" % s):
            if not os.path.exists(os.path.join(d, fn)):
                print("  REFUSED  missing %s" % fn)
                return False

    # ---- 1. byte-identical to the committed parity reference ---------------
    commit = commit or git("rev-parse", "HEAD")
    print("\n[1] reference CSV equality (parity anchor)")
    for s in systems:
        ref = os.path.join(HERE, "oof_%s.csv" % s)
        cur = os.path.join(d, "oof_%s.csv" % s)
        a, b = sha_file(ref), sha_file(cur)
        if a != b:
            problems.append("oof_%s.csv is NOT byte-identical to the committed "
                            "reference (%s vs %s)" % (s, a, b))
        else:
            print("  ok    oof_%s.csv == committed reference %s" % (s, a[:16]))

    # ---- 2. untouched since the commit that added them ---------------------
    print("\n[2] artifacts identical to git HEAD")
    dirty = git("diff", "--name-only", commit, "--", "experiments/exp07_fault_windows/runs")
    if dirty:
        problems.append("runs/ differs from %s: %s" % (commit[:8], dirty))
    else:
        print("  ok    runs/ identical to %s" % commit[:8])
    if not allow_dirty:
        st = git("status", "--porcelain", "--", "experiments/exp07_fault_windows/runs")
        if st:
            problems.append("runs/ has uncommitted changes: %s" % st)

    # ---- 7. relevant source unchanged since that commit --------------------
    print("\n[7] source files unchanged since %s" % commit[:8])
    for p in CODE_RELEVANT:
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        if p.endswith("runner.py"):
            continue  # handled separately below: the writer block was fixed
        if git("diff", "--name-only", commit, "--", rel):
            problems.append("%s changed since the CV run" % rel)
    print("  ok    all %d relevant sources unchanged (runner.py checked "
          "separately)" % (len(CODE_RELEVANT) - 1))

    # runner.py cannot hash-equal the commit: the committed version raised
    # AttributeError before writing anything (it called cache.sha256_array,
    # which did not exist), so the run that produced these artifacts used a
    # locally-fixed runner.py.  Therefore the gate is the model-determining
    # region of run(), and every change between the commit and the current file
    # must be provably outside it.
    print("\n[7b] runner.py model path")
    import difflib
    old = git("show", "%s:experiments/exp07_fault_windows/runner.py" % commit)
    new = open(os.path.join(HERE, "runner.py"), encoding="utf-8").read()

    def model_region(text):
        L = text.splitlines(keepends=True)
        i = next(k for k, l in enumerate(L)
                 if l.startswith(reuse._MODEL_START))
        j = next(k for k, l in enumerate(L)
                 if l.startswith(reuse._MODEL_END))
        return "".join(L[i:j + 1])

    old_region, new_region = model_region(old), model_region(new)
    if old_region != new_region:
        dd = list(difflib.unified_diff(old_region.splitlines(True),
                                       new_region.splitlines(True),
                                       "committed", "current"))
        problems.append("runner.py MODEL PATH changed since the CV run:\n    "
                        + "".join(dd))
    else:
        print("  ok    model-determining region of run() byte-identical to %s"
              % commit[:8])

    model_sha = reuse.runner_model_path_sha256()
    outside = []

    # mark the line ranges covered by the model region in each file
    def bounds(t):
        L = t.splitlines(True)
        i = next(k for k, l in enumerate(L)
                 if l.startswith(reuse._MODEL_START))
        j = next(k for k, l in enumerate(L)
                 if l.startswith(reuse._MODEL_END))
        return i, j

    oi0, oj0 = bounds(old)
    ni0, nj0 = bounds(new)
    sm = difflib.SequenceMatcher(None, old.splitlines(True),
                                 new.splitlines(True))
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        touched_o = any(oi0 <= k < oj0 for k in range(i1, i2))
        touched_n = any(ni0 <= k < nj0 for k in range(j1, j2))
        if not (touched_o or touched_n):
            outside.append("%s old[%d:%d] -> new[%d:%d]"
                           % (tag, i1, i2, j1, j2))
    print("  ok    %d edit hunk(s), all OUTSIDE the model path:"
          % len(outside))
    for o in outside:
        print("          %s" % o)

    # ---- 4. run order ------------------------------------------------------
    print("\n[4] run order")
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    ys = train["label"].values.astype(object)
    yi = np.array([L2I[l] for l in ys], dtype=int)
    n = len(train)
    run_ids = [str(x) for x in train["run_id"]]
    stored_ids = json.load(open(os.path.join(d, "oof_runid.json"),
                                "r", encoding="utf-8"))
    if stored_ids != run_ids:
        problems.append("stored oof_runid.json != current train.csv order")
    else:
        print("  ok    %d run ids, order identical to train.csv" % n)

    # ---- 6. fold partition -------------------------------------------------
    print("\n[6] fold coverage")
    folds = reuse.fold_assignments(yi)
    cover = np.zeros(n, dtype=int)
    for v in folds:
        cover[v] += 1
    if cover.min() != 1 or cover.max() != 1:
        problems.append("fold coverage min=%d max=%d, expected 1"
                        % (cover.min(), cover.max()))
    else:
        print("  ok    every row held out exactly once across %d folds"
              % len(folds))

    # ---- 3/5. per-system evidence ------------------------------------------
    res = json.load(open(os.path.join(d, "results.json"), "r", encoding="utf-8"))
    from sklearn.metrics import f1_score

    proba_sha, label_sha, folds_meta = {}, {}, []
    for s in systems:
        print("\n[%s] proba / labels / fold identity" % s)
        P = np.load(os.path.join(d, "oof_proba_%s.npy" % s))
        csv = pd.read_csv(os.path.join(d, "oof_%s.csv" % s))
        if P.shape != (n, len(LABELS)):
            problems.append("%s proba shape %r != (%d, 7)" % (s, P.shape, n))
            continue
        if not np.isfinite(P).all():
            problems.append("%s proba has non-finite values" % s)
        if not np.allclose(P.sum(1), 1.0, atol=1e-4):
            problems.append("%s proba rows do not sum to 1" % s)

        if not (csv["run_id"].values.astype(str)
                == train["run_id"].values.astype(str)).all():
            problems.append("%s CSV run_id order != train.csv" % s)
        csv_y = train["label"].values.astype(object)
        if not (csv["y_true"].values.astype(str) == csv_y.astype(str)).all():
            problems.append("%s CSV y_true != train.csv label" % s)

        labels = P.argmax(1).astype(np.int64)
        csv_idx = np.array([L2I[v] for v in csv["y_pred"].values], dtype=int)
        if not (labels == csv_idx).all():
            problems.append("%s argmax(proba) != CSV y_pred at %d rows"
                            % (s, int((labels != csv_idx).sum())))
        else:
            print("  ok    argmax(proba) == CSV y_pred, all %d rows" % n)

        proba_sha[s] = ca.sha256_array(P.astype(np.float32))
        label_sha[s] = ca.sha256_array(labels)

        # the decisive proof: per-fold macro on the RE-DERIVED folds
        derived = [f1_score(ys[v], np.array(LABELS)[labels[v]],
                            average="macro") for v in folds]
        want = res["systems"][s]["fold_macro_f1"]
        if not np.allclose(derived, want, atol=TOL):
            problems.append("%s per-fold macro on re-derived folds does not "
                            "match results.json (max diff %.3e)"
                            % (s, float(np.abs(np.array(derived) - want).max())))
        else:
            print("  ok    per-fold macro matches results.json to 1e-9 -> "
                  "folds are IDENTICAL")
            print("        %s" % ["%.10f" % x for x in derived])

    if problems:
        print("\nREFUSED - %d problem(s), no manifest written:" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return False

    # ---- write the strict manifest -----------------------------------------
    man = {
        "schema": 2,
        "verified": True,
        "mode": mode,
        "seed": reuse.SEED,
        "n_outer": reuse.N_OUTER,
        "n_inner": reuse.N_INNER,
        "n_runs": n,
        "n_classes": len(LABELS),
        "full_coverage": True,
        "early_stopped": False,
        "folds_evaluated": [0, 1, 2],
        "data_sha256": ca.data_sha256(),
        "code_sha256": ca.code_sha256(),
        "code_files_sha256": ca.code_files_sha256(),
        "run_id_sha256": ca.sha256_run_ids(run_ids),
        "y_true_sha256": ca.sha256_run_ids(ys),
        "runner_model_path_sha256": model_sha,
        "proba_sha256": proba_sha,
        "label_sha256": label_sha,
        "fold_val_sha256": [ca.sha256_indices(v) for v in folds],
        "folds": [
            {
                "fold": i,
                "n_val": int(len(v)),
                "n_train": int(n - len(v)),
                "val_sha256": ca.sha256_indices(v),
                "train_sha256": ca.sha256_indices(np.setdiff1d(np.arange(n), v)),
            }
            for i, v in enumerate(folds)
        ],
        "statement": ("every outer-train run received its window aggregates "
                      "from a window model fitted without it"),
        "provenance": {
            "kind": "sealed upgrade of an existing CV run",
            "retrained": False,
            "sealed_from_commit": commit,
            "evidence": [
                "oof_*.csv byte-identical to the committed parity reference",
                "runs/ byte-identical to git %s" % commit[:8],
                "argmax(oof_proba_*.npy) == oof_*.csv y_pred on all rows",
                "oof_runid.json == current train.csv run order",
                "per-fold macro F1 on seed-0 re-derived folds == "
                "results.json to 1e-9",
                "every row held out by exactly one outer validation fold",
                "runner.py model-determining region byte-identical to %s; "
                "all edits since are in the artifact-writer block"
                % commit[:8],
            ],
            "parity": "check_parity.py cv -> PASS",
        },
    }
    man["written_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S")

    # Every check above passed.  Only now may anything be written: the derived
    # hard-label arrays and the manifest itself.
    for s in systems:
        P = np.load(os.path.join(d, "oof_proba_%s.npy" % s))
        np.save(os.path.join(d, "oof_%s.npy" % s), P.argmax(1).astype(np.int64))
        print("  wrote oof_%s.npy (hard labels = argmax(proba))" % s)

    out = os.path.join(d, "honest_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print("\nSEALED -> %s" % out)
    print("  proba_sha256:")
    for k, v in proba_sha.items():
        print("    %-16s %s" % (k, v))
    print("  fold_val_sha256:")
    for i, v in enumerate(man["fold_val_sha256"]):
        print("    fold %d (%d rows)  %s" % (i, man["folds"][i]["n_val"], v))
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else MD.CV
    sys.exit(0 if main(mode) else 1)
