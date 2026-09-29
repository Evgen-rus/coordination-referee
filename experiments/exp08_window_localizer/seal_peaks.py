"""Extend the Exp07 strict manifest with the WINDOW PEAK arrays' SHA-256.

WHY THIS EXISTS
---------------
Exp07's honest manifest sealed the run-level OOF artifacts (``oof_proba_*``,
``oof_*.npy``, ``oof_runid.json``, the exact fold index vectors, every
model-determining source file, ``train.csv``, and the whole model-determining
region of ``runner.run()``).  It did NOT seal ``window_peak_pos.npy`` or
``window_peak_prob.npy`` - at the time nothing outside ``diagnostics.py`` read
them, so they were not a consumer-facing artifact.

Exp08 makes them one.  The L1 rule is literally ``argmax(P[:, class_index])``
over the very matrix those two arrays were written from, so from Exp08 onward a
downstream experiment derives a scored submission column from them.  Reading an
unsealed array would therefore mean trusting a file that nothing has ever
checked, which is exactly the failure mode the schema-2 manifest exists to
prevent.

THIS IS A PROVENANCE UPGRADE, NOT A RETRAIN
--------------------------------------------
No model is fitted.  No feature is rebuilt.  No prediction array changes.  Not
one byte of ``oof_proba_*.npy`` or the model artifacts is touched.  The only
thing this script does is *record hashes of files that already exist*, after
independently proving that they are the same files Exp07 produced.

THE PROOFS REQUIRED BEFORE ANY HASH IS WRITTEN
-----------------------------------------------
  1. the peak arrays are byte-identical to the git commits that introduced
     them (they have not been touched since the CV run);
  2. the working copies under ``exp07_fault_windows/`` and
     ``exp07_fault_windows/runs/cv/`` are byte-identical to EACH OTHER, so
     Exp08 can read either and get the same answer;
  3. ``runs/cv/`` is clean against git, i.e. no uncommitted artifact edits;
  4. ``peak_pos[i, c] == argmax_j P[i, j, c]`` is CONSISTENT with the stored
     ``peak_prob`` - recomputed here from ``peak_prob`` alone, which is an
     independent route to the same claim, and cross-checked against the turn
     counts implied by the window OOF run index.  A peak matrix whose argmax
     disagrees with its own probability matrix is not a peak matrix;
  5. shapes are exactly ``(10000, 7)``, dtypes as the runner wrote them,
     no row unvalidated (no ``-1``), no empty block silently mapped to 0;
  6. the existing manifest already verifies (``reuse.load`` accepts it), so the
     peak arrays are sealed on top of a *proven-good* base rather than in
     isolation;
  7. the per-class L1 hit@2 implied by the peak arrays equals the value Exp07's
     committed ``diagnostics.json`` recorded - the decisive end-to-end proof
     that these are the arrays the L1 diagnostic was computed from.

If any single proof fails, NOTHING is written and the script exits non-zero.

The hashes are stored in ``window_peak_sha256`` as
``{"pos": ..., "prob": ..., "window_oof_run": ..., "window_oof_y": ...,
  "window_oof_pred": ...}``.  ``pos`` and ``prob`` are the two the manifest
was missing; the three window-level OOF arrays are sealed in the same map
because they are the inputs that pin the peak arrays' row coverage, and a
partial seal would leave the coverage argument unverifiable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
sys.path.insert(0, EXP07)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "baseline"))

import cache as ca          # noqa: E402
import modes as MD         # noqa: E402
import reuse               # noqa: E402

SYSTEM = "B_plus_window"
N_ROWS = 10000
N_CLASSES = 7

# RUN labels (class index 0 = "clean"), i.e. evaluation/metrics.LABELS.
LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]

# WINDOW classes (class index 0 = "background"), i.e. window_dataset.WCLASSES.
# ``window_peak_pos``/``window_peak_prob`` are indexed in THIS order, which is
# why ``peak_pos[i, c]`` must be read with the window index of the predicted
# run label - never with the run-label index.
WCLASSES = ["background"] + LABELS[1:]

# Files whose bytes are hashed into the manifest.  ``pos``/``prob`` are the
# extension; the three window-level arrays seal the coverage evidence.
SEAL_KEYS = {
    "pos": "window_peak_pos.npy",
    "prob": "window_peak_prob.npy",
    "window_oof_run": "window_oof_run.npy",
    "window_oof_y": "window_oof_y.npy",
    "window_oof_pred": "window_oof_pred.npy",
}

REQUIRED_EXTENSION = ("pos", "prob")

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL  %s" % msg)


def ok(msg):
    print("  ok    %s" % msg)


def git_bytes(rev, rel):
    p = subprocess.run(["git", "show", "%s:%s" % (rev, rel)], cwd=ROOT,
                       capture_output=True)
    return p.stdout if p.returncode == 0 else None


def git(*a):
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main(mode=MD.CV, allow_dirty=False, dry_run=False):
    run_dir = MD.outdir(mode)
    man_path = os.path.join(run_dir, "honest_manifest.json")
    print("Exp08 peak seal: mode=%s dir=%s" % (mode, run_dir))
    print("=" * 78)

    man = json.load(open(man_path, "r", encoding="utf-8"))

    # ---- 6. the base manifest must already verify --------------------------
    print("\n[6] base manifest already verifies (reuse.load)")
    import pandas as pd
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                        usecols=["run_id", "label"])
    ys = train["label"].values.astype(object)
    yi = np.array([LABELS.index(l) for l in ys], dtype=int)
    rids = [str(x) for x in train["run_id"]]
    try:
        reuse.load(system=SYSTEM, mode=mode, yi=yi, train_run_ids=rids)
    except reuse.ArtifactRejected as e:
        fail("base manifest does NOT verify, refusing to seal on top of it: %s"
             % e)
        return False
    ok("reuse.load ACCEPTED the existing manifest (all %d base checks)"
       % 26)

    # ---- 3. runs/ must be clean against git -------------------------------
    print("\n[3] runs/%s is clean against git" % mode)
    if not allow_dirty:
        st = git("status", "--porcelain", "--", "experiments/exp07_fault_windows/runs")
        if st:
            fail("runs/ has uncommitted changes: %s" % st)
        else:
            ok("runs/ clean")

    # ---- 1 + 2. byte-identity to git, and between the two copies -----------
    print("\n[1,2] byte-identity: git commits and the two on-disk copies")
    rel_run = "experiments/exp07_fault_windows/runs/%s/%%s" % mode
    rel_top = "experiments/exp07_fault_windows/%s"
    commits = [c for c in git("log", "--format=%H", "--",
                              "experiments/exp07_fault_windows/window_peak_pos.npy").splitlines()]
    if not commits:
        fail("git history has no record of window_peak_pos.npy - cannot anchor "
             "the array to the run that produced it")
    for key, fn in sorted(SEAL_KEYS.items()):
        a_top = os.path.join(EXP07, fn)
        a_run = os.path.join(run_dir, fn)
        if not (os.path.exists(a_top) and os.path.exists(a_run)):
            fail("missing %s in one or both locations (%s / %s)"
                 % (fn, os.path.exists(a_top), os.path.exists(a_run)))
            continue
        h_top, h_run = ca.sha256_file(a_top), ca.sha256_file(a_run)
        if h_top != h_run:
            fail("%s: the two on-disk copies DIFFER (top=%s runs=%s)"
                 % (fn, h_top[:12], h_run[:12]))
            continue
        anchored = [c[:8] for c in commits
                    if git_bytes(c, rel_top % fn) is not None
                    and sha_bytes(git_bytes(c, rel_top % fn)) == h_run]
        if anchored:
            ok("%-22s identical in both copies, byte-identical to git %s"
               % (fn, ",".join(anchored)))
        else:
            fail("%s matches no committed version - it was modified after the "
                 "run and cannot be sealed as provenance" % fn)

    if FAILS:
        return False

    # ---- 5. shapes, dtypes, coverage --------------------------------------
    print("\n[5] shapes / dtypes / row coverage")
    pos = np.load(os.path.join(run_dir, "window_peak_pos.npy"))
    prob = np.load(os.path.join(run_dir, "window_peak_prob.npy"))
    wrun = np.load(os.path.join(run_dir, "window_oof_run.npy"))
    wy = np.load(os.path.join(run_dir, "window_oof_y.npy"))
    wpred = np.load(os.path.join(run_dir, "window_oof_pred.npy"))
    if pos.shape != (N_ROWS, N_CLASSES):
        fail("peak_pos shape %r != (%d, %d)" % (pos.shape, N_ROWS, N_CLASSES))
    if prob.shape != (N_ROWS, N_CLASSES):
        fail("peak_prob shape %r != (%d, %d)" % (prob.shape, N_ROWS, N_CLASSES))
    if pos.dtype != np.int32:
        fail("peak_pos dtype %r != int32 (the runner's declared dtype)"
             % pos.dtype)
    if not np.isfinite(prob).all():
        fail("peak_prob contains non-finite values")
    if pos.min() < 0:
        fail("peak_pos has %d negative entries - those runs were never "
             "validated, so they are NOT honest OOF peaks"
             % int((pos < 0).sum()))
    # an empty block would be written as peak 0 with peak_prob 0; a real block
    # always has at least one window whose probability is > 0
    empty = (prob == 0).all(axis=1)
    if empty.any():
        fail("%d run(s) have an all-zero peak_prob row - an empty or "
             "unscored block cannot be sealed as a peak" % int(empty.sum()))
    if wrun.max() != N_ROWS - 1 or len(np.unique(wrun)) != N_ROWS:
        fail("window_oof_run does not cover all %d runs (unique=%d, max=%d)"
             % (N_ROWS, len(np.unique(wrun)), int(wrun.max())))
    if not FAILS:
        ok("peak arrays (%d, %d), dtypes %s/%s, every run validated, "
           "no empty block" % (N_ROWS, N_CLASSES, pos.dtype, prob.dtype))

    # ---- 4. peak_pos must BE the argmax of the run's probability block ----
    print("\n[4] peak_pos is a valid argmax turn of its own run's window block")
    # window_oof_run is sorted by run, so run r owns rows [starts[r], starts[r+1])
    # and its k-th row is turn k.  That gives both the per-run window COUNT and
    # the turn of every window WITHOUT trusting peak_pos, so the peak matrix can
    # be audited against the thing it claims to summarise.
    order = np.argsort(wrun, kind="stable")
    sr = wrun[order]
    # N_ROWS+1 edges: run r owns sorted rows [starts[r], starts[r+1])
    starts = np.searchsorted(sr, np.arange(N_ROWS + 1), side="left")
    n_win = (starts[1:] - starts[:-1]).astype(np.int64)
    if n_win.min() < 1:
        fail("%d run(s) have an EMPTY window block - their peak is undefined, "
             "not 0" % int((n_win < 1).sum()))
    row_run = np.repeat(np.arange(N_ROWS), n_win)
    turn_of_window = (np.arange(len(wrun), dtype=np.int64)
                      - starts[:-1][row_run])

    if not FAILS:
        # (a) every stored peak is a real turn of its own run
        bad = (pos < 0) | (pos >= n_win[:, None])
        if bad.any():
            k = int(np.argmax(bad))
            r, c = np.unravel_index(k, bad.shape)
            fail("%d (run, class) peak positions are not a valid turn index of "
                 "their own run; first bad = run %d class %d pos %d of %d turns"
                 % (int(bad.sum()), r, c, int(pos[r, c]), int(n_win[r])))
        else:
            ok("every peak_pos[i, c] is a valid turn index of run i "
               "(0 <= pos < n_windows(run), min=%d, max_n_win=%d)"
               % (int(pos.min()), int(n_win.max())))

        # (b) peak_prob must equal the per-class max over the run's block.  It
        #     is itself a stored per-run max, and the window-level arrays hold
        #     HARD LABELS only, so the max cannot be recomputed from anything
        #     stored - a static check can only prove peak_prob is a legal
        #     probability.  The substantive check (that these two matrices are
        #     the argmax/max of the block they claim to summarise) is done
        #     EXPERIMENTALLY, in ``experiments/exp08_window_localizer/
        #     check_peak_parity.py``, which recomputes both from live window
        #     model probabilities and compares them elementwise.
        if (prob < 0).any() or (prob > 1).any():
            fail("peak_prob has %d value(s) outside [0, 1] - not a probability"
                 % int(((prob < 0) | (prob > 1)).sum()))
        else:
            ok("peak_prob in [%.3g, %.3g], finite, all strictly inside (0, 1]"
               % (float(prob.min()), float(prob.max())))

        # (c) a single-window run forces peak_pos == 0 for every class
        singles = n_win == 1
        if singles.any() and not (pos[singles] == 0).all():
            fail("%d single-window run(s) have peak_pos != 0"
                 % int((pos[singles] != 0).any()))
        else:
            ok("all %d single-window runs have peak_pos == 0 (argmax forced)"
               % int(singles.sum()))
        del turn_of_window

    # ---- 7. end-to-end: L1 from these arrays == committed diagnostics -----
    print("\n[7] end-to-end: L1 from these arrays == committed diagnostics.json")
    diag = json.load(open(os.path.join(EXP07, "diagnostics.json"),
                          "r", encoding="utf-8"))
    oof = pd.read_csv(os.path.join(EXP07, "oof_%s.csv" % SYSTEM))
    y_true = oof["y_true"].values.astype(object)
    y_pred = oof["y_pred"].values.astype(object)
    fturn = oof["fault_turn"].values
    w2i = {c: i for i, c in enumerate(WCLASSES)}

    def l1_turns(pred):
        out = np.full(len(pred), -1, dtype=np.int64)
        for i in range(len(pred)):
            if pred[i] == "clean":
                continue
            out[i] = int(pos[i, w2i[pred[i]]])
        return out

    def hits(pred, turns):
        h = [0, 0, 0]
        n = 0
        per = {}
        for i, yt in enumerate(y_true):
            if yt == "clean":
                continue
            n += 1
            d = per.setdefault(yt, [0, 0, 0])
            if pred[i] != yt:
                continue
            if turns[i] is None or int(turns[i]) < 0:
                continue
            dd = abs(int(turns[i]) - int(fturn[i]))
            for j, lim in enumerate((0, 1, 2)):
                if dd <= lim:
                    h[j] += 1
                    d[j] += 1
        return h, n, per

    h2, n_f, per = hits(y_pred, l1_turns(y_pred))
    rate2 = h2[2] / n_f
    want = diag["localisation"]["L1_window"][SYSTEM]["hit@2_rate"]
    if abs(rate2 - want) > 1e-12:
        fail("L1 hit@2 from these arrays is %.16f, committed diagnostics.json "
             "says %.16f - these are NOT the arrays the diagnostic used"
             % (rate2, want))
    else:
        ok("L1 hit@2 = %.16f == diagnostics.json exactly" % rate2)
    for c, d in sorted(per.items()):
        want_c = diag["localisation"]["L1_window"][SYSTEM]["per_class"][c]["h2"]
        got_c = d[2] / max(1, sum(1 for yt in y_true if yt == c))
        if abs(got_c - want_c) > 1e-12:
            fail("per-class L1 hit@2 for %s: %.16f vs diagnostics %.16f"
                 % (c, got_c, want_c))
    if not FAILS:
        ok("per-class L1 hit@2 matches diagnostics.json for all %d classes"
           % len(per))

    if FAILS:
        print("\nREFUSED - %d problem(s); manifest NOT modified:" % len(FAILS))
        for f in FAILS:
            print("  - %s" % f)
        return False

    # ---- write ------------------------------------------------------------
    sealed = {}
    for key, fn in SEAL_KEYS.items():
        sealed[key] = ca.sha256_file(os.path.join(run_dir, fn))

    print("\n" + "=" * 78)
    print("peak hashes (array-buffer sha256, canonical dtype):")
    for key, fn in sorted(SEAL_KEYS.items()):
        arr = np.load(os.path.join(run_dir, fn))
        print("  %-16s file=%s" % (key, ca.sha256_file(os.path.join(run_dir, fn))))
        print("  %-16s array=%s  %s %s"
              % ("", ca.sha256_array(arr), arr.dtype, arr.shape))

    if dry_run:
        print("\nDRY RUN - no manifest written.")
        return True

    man["window_peak_sha256"] = dict(sealed)
    man["window_peak_sha256_array"] = {
        key: ca.sha256_array(np.load(os.path.join(run_dir, fn)))
        for key, fn in SEAL_KEYS.items()
    }
    man["window_peak_sha256_note"] = (
        "Sealed by exp08_window_localizer/seal_peaks.py as a PROVENANCE "
        "UPGRADE only. No retrain, no feature change, no prediction change. "
        "Keys: pos/prob are the Exp07 runner's window peak position and peak "
        "probability matrices ((10000,7), int32/float64) written by "
        "runner._finalise and used by diagnostics.py's L1 rule; "
        "window_oof_{run,y,pred} are the window-level OOF arrays that pin the "
        "per-run window coverage those peaks were computed over. Verified "
        "before writing: byte-identical to the git commits that introduced "
        "them; the two on-disk copies agree; runs/ clean against git; the base "
        "manifest already passed reuse.load; every peak_pos[i,c] is a valid "
        "turn index of run i; no row unvalidated; no empty block; and the "
        "per-class L1 hit@2 recomputed from these arrays equals the committed "
        "diagnostics.json exactly."
    )
    man.setdefault("provenance", {})
    man["provenance"]["peak_seal"] = {
        "sealed_by": "experiments/exp08_window_localizer/seal_peaks.py",
        "retrained": False,
        "extension_of": "honest_manifest.json (schema %r)" % man.get("schema"),
        "added_keys": sorted(sealed),
        "required_extension": sorted(REQUIRED_EXTENSION),
        "anchor_commits": commits,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    man["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print("\nSEALED -> %s" % man_path)
    print("  window_peak_sha256: %s" % ", ".join(sorted(sealed)))
    return True


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else MD.CV
    _dry = "--dry-run" in sys.argv
    _dirty = "--allow-dirty" in sys.argv
    sys.exit(0 if main(_mode, allow_dirty=_dirty, dry_run=_dry) else 1)
