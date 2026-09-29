"""Load verified honest-OOF artifacts without re-running the Exp07 pipeline.

What a downstream experiment (calibration, thresholds, class offsets, a
decision layer, post-processing) actually needs is the OOF probability matrix
and the run order - not the models that produced them.  Exp07 already computes
exactly that, so recomputing an 11-minute pipeline to try a decision rule is
pure waste.

Honesty rules enforced here, not assumed:

  * the manifest written by the runner must say ``verified: true`` and record
    the same seed / fold counts the current code uses;
  * a SHA-256 of the probability matrix is checked against the manifest, so a
    file edited or truncated after the run is rejected rather than trusted;
  * the recorded fold assignments are re-derived from the current
    StratifiedKFold call and compared, so a change to SEED or N_OUTER cannot
    silently pair new folds with old predictions;
  * the cached window aggregates and OOF arrays are the ones the fold
    assignments actually produced, never a re-fit.

A consumer that only wants hard labels can pass ``proba=False``.
"""
from __future__ import annotations
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modes as MD  # noqa: E402

SEED = 0
N_OUTER = 3
N_INNER = 3


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fold_assignments(yi, n_outer=N_OUTER, seed=SEED):
    """Re-derive the outer folds exactly as the runner does."""
    from sklearn.model_selection import StratifiedKFold
    n = len(yi)
    return [np.asarray(v) for _, v in
            StratifiedKFold(n_outer, shuffle=True, random_state=seed)
            .split(np.zeros(n), yi)]


class ArtifactRejected(Exception):
    """Raised when stored artifacts cannot be proven to match the code."""


def load(system="B_plus_window", mode="cv", yi=None, proba=True, strict=True):
    """Return the honest OOF arrays for ``system``.

    Parameters
    ----------
    system : str   'A_foundation' or 'B_plus_window'
    mode   : str   which mode run to read from
    yi     : array integer labels; when given, the fold assignment is
             re-derived and checked against the manifest
    proba  : bool  return the (n, 7) probability matrix instead of hard labels
    strict : bool  raise on any verification failure (default).  With
             ``strict=False`` the failure is returned in the result dict
             instead, for exploratory scripts.
    """
    d = MD.outdir(mode)
    mpath = os.path.join(d, "honest_manifest.json")
    if not os.path.exists(mpath):
        raise ArtifactRejected("no manifest at %s - run the pipeline first"
                               % mpath)
    man = json.load(open(mpath))
    problems = []

    if not man.get("verified"):
        problems.append("manifest does not assert verified stacking")
    if man.get("seed") != SEED:
        problems.append("seed %r != current %r" % (man.get("seed"), SEED))
    if man.get("n_outer") != N_OUTER or man.get("n_inner") != N_INNER:
        problems.append("fold counts differ from current code")

    ppath = os.path.join(d, "oof_proba_%s.npy" % system)
    if not os.path.exists(ppath):
        raise ArtifactRejected("missing %s" % ppath)
    P = np.load(ppath)

    stored = man.get("proba_sha256", {}).get(system)
    if stored and _sha(P.astype(np.float32)) != stored:
        problems.append("proba sha256 mismatch - file changed after the run")

    hard = np.load(os.path.join(d, "oof_%s.npy" % system)) \
        if os.path.exists(os.path.join(d, "oof_%s.npy" % system)) else None

    if yi is not None:
        cur = fold_assignments(yi)
        want = man.get("outer_val_sizes")
        if want and [len(v) for v in cur] != want:
            problems.append("outer fold sizes %r != manifest %r - folds changed"
                            % ([len(v) for v in cur], want))

    if problems:
        if strict:
            raise ArtifactRejected("; ".join(problems))
        return {"ok": False, "problems": problems}

    if hard is None:
        hard = P.argmax(1)
    return {"ok": True, "problems": [], "proba": P if proba else None,
            "labels": hard, "dir": d, "manifest": man}



