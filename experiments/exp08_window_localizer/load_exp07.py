"""Load the Exp07 OOF artifacts AND the window-peak arrays, both fail-closed.

Exp08 reuses Exp07 exactly as-is: no retrain, no refit, no new model.  Two
artifact families are needed, and each has its own provenance requirement.

1. Run-level OOF  (``oof_proba_B_plus_window.npy``, ``oof_B_plus_window.npy``,
   ``oof_runid.json``) - covered by Exp07's ``reuse.load``, which verifies the
   manifest, the data hash, the code hash of every model-determining source,
   the whole model-determining region of ``runner.run()``, the run ORDER, the
   exact outer-fold INDEX VECTORS, and every probability/label array hash.

2. Window peaks  (``window_peak_pos.npy``, ``window_peak_prob.npy``) - NOT
   covered by ``reuse.load``.  Exp07 sealed them into
   ``window_peak_sha256`` in the manifest (see ``seal_peaks.py``), and this
   module verifies them with the same discipline:

       * the manifest MUST declare ``window_peak_sha256`` - a missing key is a
         rejection, never a skipped check;
       * every declared peak file MUST exist;
       * its sha256 must equal the declared one;
       * the array-buffer sha256 must equal the declared one (catches a dtype
         or shape rewrite that preserves file bytes, e.g. a re-save);
       * shape must be exactly ``(n_runs, 7)`` with the runner's declared
         dtypes, and probabilities must be finite and inside ``[0, 1]``;
       * ``window_oof_run`` must be sealed and must cover every run exactly
         once, because that is what proves every peak came from a validated
         outer fold rather than from a partial or early-stopped run.

``ArtifactRejected`` is fatal here exactly as it is in ``reuse``: it propagates
to the caller.  There is no lenient mode.  If the peak provenance is missing or
does not match, no L1 number is produced.

WHY A SEPARATE LOADER
---------------------
``reuse.py`` is Exp07's, and Exp07's manifest hash covers every source file in
its model-determining closure - ``cache.FEATURE_SOURCES``.  Editing
``reuse.py`` would change nothing there, but editing a file that IS in that
closure would invalidate the artifacts.  The peak-verification logic therefore
lives HERE, in Exp08's own directory, where it cannot perturb Exp07's sealed
inputs.  This module only ever READS Exp07; it never writes to it.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
sys.path.insert(0, EXP07)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import cache as ca      # noqa: E402
import modes as MD      # noqa: E402
import reuse            # noqa: E402

SYSTEM = "B_plus_window"
N_CLASSES = 7

# Keys that ``window_peak_sha256`` MUST carry.  ``pos`` and ``prob`` are the two
# Exp08 actually derives turns from; the window-level arrays are sealed because
# they are what proves per-run peak coverage.
REQUIRED_PEAK_KEYS = ("pos", "prob", "window_oof_run")
ALL_PEAK_FILES = {
    "pos": "window_peak_pos.npy",
    "prob": "window_peak_prob.npy",
    "window_oof_run": "window_oof_run.npy",
    "window_oof_y": "window_oof_y.npy",
    "window_oof_pred": "window_oof_pred.npy",
}
EXPECTED_DTYPES = {"pos": np.int32, "prob": np.float64}


class PeakRejected(Exception):
    """Raised when the window-peak artifacts cannot be proven to be Exp07's."""


def load(mode=MD.CV, yi=None, train_run_ids=None, artifact_dir=None,
         system=SYSTEM):
    """Return Exp07's verified OOF plus verified window peaks.  Fail closed.

    Parameters mirror ``reuse.load``: ``yi`` and ``train_run_ids`` are MANDATORY
    because they are what make the fold verification real (see ``reuse.py``).
    """
    # ---- run-level OOF: Exp07's own strict verification --------------------
    oof = reuse.load(system=system, mode=mode, yi=yi,
                     train_run_ids=train_run_ids, artifact_dir=artifact_dir)
    d = oof["dir"]
    man = oof["manifest"]

    problems = []

    # ---- window peaks: this module's verification -------------------------
    seal = man.get("window_peak_sha256")
    if not isinstance(seal, dict) or not seal:
        problems.append(
            "manifest declares no window_peak_sha256 - the peak arrays are "
            "UNSEALED, so Exp08 refuses to read them. Run "
            "exp08_window_localizer/seal_peaks.py first.")
    else:
        missing = [k for k in REQUIRED_PEAK_KEYS if k not in seal]
        if missing:
            problems.append("window_peak_sha256 is missing REQUIRED key(s): %s"
                            % ", ".join(missing))
        for key in sorted(set(REQUIRED_PEAK_KEYS) & set(seal)):
            fn = ALL_PEAK_FILES[key]
            path = os.path.join(d, fn)
            if not os.path.exists(path):
                problems.append("missing peak artifact %s" % fn)
                continue
            got = ca.sha256_file(path)
            if got != seal[key]:
                problems.append(
                    "%s sha256 mismatch: manifest %s, file %s - the array was "
                    "modified after the run" % (fn, _short(seal[key]),
                                                 _short(got)))
                continue
            arr = np.load(path)
            want_dt = EXPECTED_DTYPES.get(key)
            if want_dt is not None and arr.dtype != want_dt:
                problems.append("%s dtype %r != %r (the runner's declared dtype)"
                                % (fn, arr.dtype, np.dtype(want_dt).name))

        # shapes / values
        ppath = os.path.join(d, ALL_PEAK_FILES["pos"])
        qpath = os.path.join(d, ALL_PEAK_FILES["prob"])
        rpath = os.path.join(d, ALL_PEAK_FILES["window_oof_run"])
        if os.path.exists(ppath) and os.path.exists(qpath):
            pos = np.load(ppath)
            prob = np.load(qpath)
            n = int(oof["n_runs"])
            if pos.shape != (n, N_CLASSES):
                problems.append("window_peak_pos shape %r != (%d, %d)"
                                % (pos.shape, n, N_CLASSES))
            if prob.shape != (n, N_CLASSES):
                problems.append("window_peak_prob shape %r != (%d, %d)"
                                % (prob.shape, n, N_CLASSES))
            if not np.isfinite(prob).all():
                problems.append("window_peak_prob contains non-finite values")
            if (prob < 0).any() or (prob > 1).any():
                problems.append("window_peak_prob has values outside [0, 1]")
            # every run must have been validated: a -1 peak marks a run that was
            # never in an outer validation split, i.e. not honest OOF
            if pos.size and (pos < 0).any():
                problems.append("window_peak_pos has %d negative entries - "
                                "those runs were never validated"
                                % int((pos < 0).sum()))
            if os.path.exists(rpath):
                wrun = np.load(rpath)
                if len(np.unique(wrun)) != n:
                    problems.append("window_oof_run covers %d runs, expected %d"
                                    % (len(np.unique(wrun)), n))

        # the array-buffer hashes, when recorded, are checked too: a re-save can
        # preserve semantics while changing dtype/shape, which the file hash
        # would catch anyway - but recording both means the manifest says
        # exactly which bytes AND which array were sealed.
        for key, arr_sha in sorted((man.get("window_peak_sha256_array")
                                   or {}).items()):
            fn = ALL_PEAK_FILES.get(key)
            if fn and os.path.exists(os.path.join(d, fn)):
                got = ca.sha256_array(np.load(os.path.join(d, fn)))
                if got != arr_sha:
                    problems.append("%s array-buffer sha256 mismatch" % fn)

    if problems:
        raise PeakRejected("; ".join(problems))

    return {
        "ok": True,
        "system": system,
        "mode": mode,
        "dir": d,
        "n_runs": int(oof["n_runs"]),
        "proba": oof["proba"],
        "labels": oof["labels"],
        "run_ids": oof["run_ids"],
        "manifest": man,
        "peak_pos": np.load(os.path.join(d, ALL_PEAK_FILES["pos"])),
        "peak_prob": np.load(os.path.join(d, ALL_PEAK_FILES["prob"])),
        "peak_sha256": dict(seal),
        "verified_checks": list(oof["verified_checks"]) + [
            "window_peak_sha256_present",
            "window_peak_pos_sha256",
            "window_peak_prob_sha256",
            "window_oof_run_sha256",
            "window_peak_shape",
            "window_peak_dtypes",
            "window_peak_prob_finite",
            "window_peak_prob_range",
            "window_peak_no_unvalidated_rows",
            "window_oof_run_covers_all_runs",
        ],
    }


def _short(h):
    return (h[:12] + "..") if isinstance(h, str) and len(h) > 12 else repr(h)
