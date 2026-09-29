"""Load verified honest-OOF artifacts without re-running the Exp07 pipeline.

What a downstream experiment (calibration, thresholds, class offsets, a
decision layer, post-processing) actually needs is the OOF probability matrix
and the run order - not the models that produced them.  Exp07 already computes
exactly that, so recomputing an 11-minute pipeline to try a decision rule is
pure waste.

HONESTY RULES
--------------
The previous version of this module used the pattern ``if stored: compare``.
That makes every check *optional*: an old or truncated manifest silently
degrades verification down to "the file exists and has the right shape".  A
consumer that cannot tell a verified artifact from an unverified one must not
be handed the unverified one, so every check below is MANDATORY.  If a field is
missing the artifact is REJECTED.

``yi`` and ``train_run_ids`` are MANDATORY arguments to ``load()``.  They are
what turn two of the checks from a formality into evidence:

  * without ``yi`` the outer folds cannot be re-derived, so fold verification
    collapses to "the recorded sizes sum to n" - which cannot detect a fold
    that was reshuffled while keeping its sizes;
  * without ``train_run_ids`` the stored run ORDER cannot be compared to the
    rows the probabilities will be attached to, so a permuted artifact would
    pass.

Neither has a fallback in ``load()``.  ``load_unverified()`` is the separate,
deliberately unchecked reader, named so the loss of provenance is obvious at the
call site.

What is verified, every time, before a single array is returned:

  * the caller supplied the current ``yi`` and ``train_run_ids``;
  * manifest exists, parses, and declares ``verified: true``;
  * ``seed`` / ``n_outer`` / ``n_inner`` equal the current code's;
  * ``data_sha256`` - the exact ``train.csv`` bytes the run consumed;
  * ``code_sha256`` - every model-determining source file, byte for byte;
  * ``runner_model_path_sha256`` - the whole model-determining region of
    ``runner.run()``: fold loop, both classifiers, inner-fit payloads, the
    window fits, the aggregation and the B feature matrix;
  * ``run_id_sha256`` - the run ORDER, compared against the stored
    ``oof_runid.json`` and against the current train, not just the count;
  * ``fold_val_sha256`` - the exact outer-validation INDEX VECTOR of every fold,
    re-derived from the current ``StratifiedKFold`` call.  Equal fold *sizes*
    with different members are rejected;
  * ``proba_sha256`` - every ``oof_proba_*`` matrix;
  * ``label_sha256`` - hard labels, cross-checked against ``argmax(proba)``;
  * ``n_runs == len(yi) == len(train_run_ids) == 10000``;
  * proba shape exactly ``(10000, 7)``, labels exactly ``(10000,)``, all
    probabilities finite, rows summing to 1;
  * every row covered by exactly one outer validation fold;
  * ``full_coverage`` - the run was CV mode and was not early-stopped, so no
    row is an un-scored zero.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cache as ca  # noqa: E402
import modes as MD  # noqa: E402

SEED = 0
N_OUTER = 3
N_INNER = 3
N_CLASSES = 7
EXPECTED_ROWS = 10000
SCHEMA = 2

SYSTEMS = ("A_foundation", "B_plus_window")

# Fields the manifest MUST carry.  Missing any of these is a rejection, never a
# skipped check.
REQUIRED_FIELDS = (
    "verified", "seed", "n_outer", "n_inner", "schema", "mode",
    "n_runs", "n_classes", "data_sha256", "code_sha256", "run_id_sha256",
    "proba_sha256", "label_sha256", "fold_val_sha256", "full_coverage",
    "runner_model_path_sha256", "folds",
)


class ArtifactRejected(Exception):
    """Raised when stored artifacts cannot be proven to match the code."""


def _sha(a):
    return ca.sha256_array(a)


def sha_indices(ix):
    return ca.sha256_indices(ix)


def fold_assignments(yi, n_outer=N_OUTER, seed=SEED):
    """Re-derive the outer folds exactly as the runner does."""
    from sklearn.model_selection import StratifiedKFold
    n = len(yi)
    return [np.asarray(v, dtype=np.int64) for _, v in
            StratifiedKFold(n_outer, shuffle=True, random_state=seed)
            .split(np.zeros(n), yi)]


# The runner's artifact-WRITING block was fixed after the CV run (the committed
# version crashed before it could write anything), so runner.py as a whole can
# never hash-equal the commit that added the artifacts.  What must not have
# changed is the part of ``run()`` that DECIDES THE MODEL, and that region now
# runs from the outer fold loop all the way to the ``_finalise(...)`` hand-off,
# covering the inner-fit payloads, the window fits, the aggregation, the B
# feature matrix and both classifiers' fit/predict_proba.
#
# Bounds must be exact.  A loose prefix match could bind the region to a comment
# or to a similarly-indented line in another function, silently hashing the
# wrong text; so the markers are matched exactly, the search is scoped to
# ``run()``, and anything unexpected is a rejection rather than a best guess.
RUNNER_PATH = os.path.join(HERE, "runner.py")
_MODEL_START = "    for f in fold_ids:"
_MODEL_END = "    return _finalise("
_RUN_FN = "def run("


def _model_region_lines(lines, path=RUNNER_PATH):
    """Locate the model-determining region, or raise.  Never guesses.

    Markers are matched as an exact line OR as an exact prefix of a single line
    (the end marker is the first call's opening text).  The search is scoped to
    the body of ``run()`` so the marker constants defined at module level can
    never be mistaken for the real code.
    """
    def find(marker, after):
        m = marker.strip()
        for k in range(after + 1, len(lines)):
            s = lines[k].strip()
            if s == m or s.startswith(m):
                return k
        return None

    try:
        fn = next(k for k, l in enumerate(lines) if l.startswith(_RUN_FN))
    except StopIteration:
        raise ArtifactRejected("no %r in %s - cannot locate run() to verify"
                               % (_RUN_FN, path))
    i = find(_MODEL_START, fn)
    if i is None:
        raise ArtifactRejected(
            "model-region start %r not found after run() in %s - refusing to "
            "treat an unhashed runner as verified" % (_MODEL_START, path))
    j = find(_MODEL_END, i)
    if j is None:
        raise ArtifactRejected(
            "model-region end %r not found after %r in %s - refusing to treat "
            "an unhashed runner as verified" % (_MODEL_END, _MODEL_START, path))
    if j <= i:
        raise ArtifactRejected("model-region bounds are inverted in %s" % path)
    # the end marker is the hand-off line itself, not part of the region
    return lines[i:j]


def runner_model_path_sha256(path=RUNNER_PATH):
    """SHA-256 of the model-determining region of ``runner.run()``."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    region = "".join(_model_region_lines(lines, path))
    if not region.strip():
        raise ArtifactRejected("model-path region of %s is empty" % path)
    return hashlib.sha256(region.encode("utf-8")).hexdigest()


def _read_manifest(path):
    if not os.path.exists(path):
        raise ArtifactRejected("no manifest at %s - run the pipeline first"
                               % path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            man = json.load(fh)
    except ValueError as e:
        raise ArtifactRejected("manifest %s is not valid JSON: %s" % (path, e))
    if not isinstance(man, dict):
        raise ArtifactRejected("manifest %s is not a JSON object" % path)
    return man


def _check_manifest_fields(man, path):
    """MANDATORY: every required field must be present.  Fail closed."""
    problems = []
    missing = [f for f in REQUIRED_FIELDS if f not in man]
    if missing:
        problems.append("manifest %s missing REQUIRED field(s): %s"
                        % (os.path.basename(path), ", ".join(missing)))
    return problems


def _check_provenance(man, system, d, yi, train_run_ids):
    """Return a list of problems.  Empty list means every check passed.

    ``yi`` and ``train_run_ids`` are already validated as non-None by ``load()``.
    """
    problems = []

    # ---- manifest identity -------------------------------------------------
    if man.get("verified") is not True:
        problems.append("manifest does not assert verified stacking "
                        "(verified=%r)" % (man.get("verified"),))
    if man.get("schema") != SCHEMA:
        problems.append("manifest schema %r != required %r - this manifest "
                        "predates the strict format and cannot be trusted"
                        % (man.get("schema"), SCHEMA))
    if man.get("seed") != SEED:
        problems.append("seed %r != current %r" % (man.get("seed"), SEED))
    if man.get("n_outer") != N_OUTER or man.get("n_inner") != N_INNER:
        problems.append("fold counts (%r, %r) differ from current code (%d, %d)"
                        % (man.get("n_outer"), man.get("n_inner"),
                           N_OUTER, N_INNER))
    if man.get("n_classes") != N_CLASSES:
        problems.append("n_classes %r != %d" % (man.get("n_classes"),
                                                N_CLASSES))
    if man.get("full_coverage") is not True:
        problems.append("run was not a complete CV run (mode=%r "
                        "full_coverage=%r) - rows may be unscored"
                        % (man.get("mode"), man.get("full_coverage")))
    if man.get("mode") != MD.CV:
        problems.append("artifacts are from mode %r, expected %r"
                        % (man.get("mode"), MD.CV))

    # ---- data hash ---------------------------------------------------------
    cur_data = ca.data_sha256()
    if man.get("data_sha256") != cur_data:
        problems.append("train.csv sha256 mismatch: manifest %s, current %s"
                        % (_short(man.get("data_sha256")),
                           _short(cur_data)))

    # ---- code hash: aggregate + every individual file --------------------
    cur_code = ca.code_sha256()
    if man.get("code_sha256") != cur_code:
        problems.append("Exp07 feature/model code sha256 mismatch: manifest "
                        "%s, current %s"
                        % (_short(man.get("code_sha256")), _short(cur_code)))

    # The per-file map is checked INDEPENDENTLY of the combined hash.  Doing it
    # only inside the combined-mismatch branch would mean that tampering with a
    # single entry of code_files_sha256 - while leaving code_sha256 alone - is
    # never noticed, and it is exactly that map that names the changed file.
    stored_files = man.get("code_files_sha256") or {}
    if not isinstance(stored_files, dict) or not stored_files:
        problems.append("manifest has no usable code_files_sha256 map - "
                        "cannot attribute a code change to a file")
    else:
        changed = []
        for rel, want in sorted(stored_files.items()):
            got = ca.sha256_file(os.path.join(ROOT, rel.replace("/", os.sep)))
            if got is None:
                problems.append("code file %s recorded in the manifest no "
                                "longer exists" % rel)
            elif got != want:
                changed.append(rel)
        # and the reverse direction: a file now covered but absent from the
        # manifest means the provenance set itself grew after the run
        cur_files = ca.code_files_sha256()
        for rel in sorted(cur_files):
            if rel not in stored_files:
                problems.append("code file %s is now part of the model-"
                                "determining set but is absent from the "
                                "manifest" % rel)
        if changed:
            problems.append("Exp07 model-determining source(s) changed since "
                            "the run: %s" % ", ".join(changed))

    # ---- runner model path -------------------------------------------------
    try:
        cur_rp = runner_model_path_sha256()
    except ArtifactRejected as e:
        problems.append(str(e))
    else:
        if man.get("runner_model_path_sha256") != cur_rp:
            problems.append("runner.py model path sha256 mismatch: manifest "
                            "%s, current %s - the fold loop, estimators or "
                            "seeds have changed since the run"
                            % (_short(man.get("runner_model_path_sha256")),
                               _short(cur_rp)))

    # ---- run order ---------------------------------------------------------
    rid_path = os.path.join(d, "oof_runid.json")
    if not os.path.exists(rid_path):
        problems.append("missing oof_runid.json - cannot verify run order")
    else:
        stored_ids = json.load(open(rid_path, "r", encoding="utf-8"))
        if len(stored_ids) != EXPECTED_ROWS:
            problems.append("oof_runid.json has %d rows, expected %d"
                            % (len(stored_ids), EXPECTED_ROWS))
        if ca.sha256_run_ids(stored_ids) != man.get("run_id_sha256"):
            problems.append("oof_runid.json sha256 != manifest run_id_sha256 "
                            "- run order was modified after the run")
        if len(train_run_ids) != len(stored_ids):
            problems.append("current train has %d runs, artifacts have %d"
                            % (len(train_run_ids), len(stored_ids)))
        elif train_run_ids != stored_ids:
            n_bad = sum(1 for a, b in zip(train_run_ids, stored_ids)
                        if a != b)
            problems.append("run_id ORDER differs from current train.csv at "
                            "%d position(s) - probabilities would be attached "
                            "to the wrong rows" % n_bad)

    # ---- fold assignment: exact indices, not sizes ------------------------
    want_folds = man.get("fold_val_sha256")
    man_folds = man.get("folds") or []
    if not isinstance(want_folds, list) or len(want_folds) != N_OUTER:
        problems.append("fold_val_sha256 must list %d per-fold hashes, got %r"
                        % (N_OUTER, want_folds))
    if len(man_folds) != N_OUTER:
        problems.append("manifest records %d folds, expected %d"
                        % (len(man_folds), N_OUTER))

    # yi is mandatory (validated in load()), so folds are ALWAYS re-derived and
    # compared index-by-index.  There is deliberately no size-only fallback.
    cur = fold_assignments(yi)
    if isinstance(want_folds, list) and len(want_folds) == N_OUTER:
        got = [ca.sha256_indices(v) for v in cur]
        if got != list(want_folds):
            bad = [i for i, (a, b) in enumerate(zip(got, list(want_folds)))
                   if a != b]
            problems.append("outer fold %s validation INDICES differ from "
                            "the current StratifiedKFold(3, seed=0) - "
                            "predictions do not belong to these folds" % bad)
    # every row held out exactly once -> true OOF, no unscored rows
    cover = np.zeros(len(yi), dtype=int)
    for v in cur:
        cover[v] += 1
    if cover.min() != 1 or cover.max() != 1:
        problems.append("fold coverage min=%d max=%d, expected exactly 1 - "
                        "some rows are never validated"
                        % (cover.min(), cover.max()))
    tot = sum(int(f.get("n_val", 0)) for f in man_folds)
    if tot != EXPECTED_ROWS:
        problems.append("recorded fold validation sizes sum to %d, expected %d"
                        % (tot, EXPECTED_ROWS))

    # ---- artifact presence -------------------------------------------------
    for fn in ("oof_proba_%s.npy" % system, "oof_%s.npy" % system):
        if not os.path.exists(os.path.join(d, fn)):
            problems.append("missing required artifact %s" % fn)

    return problems


def _short(h):
    return (h[:12] + "..") if isinstance(h, str) and len(h) > 12 else repr(h)


def load(system="B_plus_window", mode=MD.CV, yi=None, train_run_ids=None,
         proba=True, artifact_dir=None):
    """Return the honest OOF arrays for ``system``.  STRICT, fail closed.

    Parameters
    ----------
    system         : str   'A_foundation' or 'B_plus_window'
    mode           : str   which mode run to read from (must be a full CV run)
    yi             : array  integer labels of the CURRENT train.  MANDATORY.
                     This is what makes fold verification real: the outer folds
                     are re-derived from it and compared index-by-index to the
                     manifest.  Passing None used to silently downgrade the check
                     to "fold sizes sum to n", which cannot detect a fold that
                     was reshuffled but kept its sizes - so it is now a
                     rejection, not a fallback.
    train_run_ids  : sequence of run ids of the CURRENT train.  MANDATORY, for
                     the same reason: without it the stored run ORDER cannot be
                     compared against the rows the probabilities will be attached
                     to, and a permuted artifact would be silently accepted.
    proba          : bool  return the (n, 7) probability matrix
    artifact_dir   : str   read this directory instead of ``runs/<mode>``.
                     Used by the provenance tests to exercise tampered copies;
                     it does not weaken any check.

    Raises ``ArtifactRejected`` on any failure.  There is no lenient mode here;
    for interactive poking use ``load_unverified()``, which is named to make
    the loss of provenance obvious at the call site.
    """
    if system not in SYSTEMS:
        raise ArtifactRejected("unknown system %r, expected one of %s"
                               % (system, SYSTEMS))
    # ---- mandatory caller context --------------------------------------
    if yi is None:
        raise ArtifactRejected(
            "yi is REQUIRED: without the current integer labels the outer "
            "folds cannot be re-derived, so fold verification would degrade to "
            "size-only. Pass yi=np.array([L2I[l] for l in train['label']]).")
    yi = np.asarray(yi)
    if train_run_ids is None:
        raise ArtifactRejected(
            "train_run_ids is REQUIRED: without the current run order the "
            "stored oof_runid.json cannot be compared position-by-position, so "
            "a permuted artifact would be accepted silently. Pass "
            "train_run_ids=[str(x) for x in train['run_id']].")
    train_run_ids = [str(x) for x in train_run_ids]

    # ---- expected dataset ------------------------------------------------
    if yi.ndim != 1:
        raise ArtifactRejected("yi must be 1-D, got shape %r" % (yi.shape,))
    if len(yi) != EXPECTED_ROWS:
        raise ArtifactRejected("len(yi) is %d, expected %d - this is not the "
                               "dataset these artifacts were produced from"
                               % (len(yi), EXPECTED_ROWS))
    if len(train_run_ids) != EXPECTED_ROWS:
        raise ArtifactRejected("len(train_run_ids) is %d, expected %d"
                               % (len(train_run_ids), EXPECTED_ROWS))

    d = artifact_dir or MD.outdir(mode)
    mpath = os.path.join(d, "honest_manifest.json")
    man = _read_manifest(mpath)

    problems = _check_manifest_fields(man, mpath)
    problems += _check_provenance(man, system, d, yi, train_run_ids)

    if man.get("n_runs") != EXPECTED_ROWS:
        problems.append("manifest n_runs is %r, expected %d"
                        % (man.get("n_runs"), EXPECTED_ROWS))

    ppath = os.path.join(d, "oof_proba_%s.npy" % system)
    lpath = os.path.join(d, "oof_%s.npy" % system)
    if os.path.exists(ppath):
        P = np.load(ppath)
        want = (man.get("proba_sha256") or {}).get(system)
        if want is None:
            problems.append("manifest has no proba_sha256 for %r" % system)
        elif _sha(P.astype(np.float32)) != want:
            problems.append("proba sha256 mismatch for %r - the .npy changed "
                            "after the run" % system)

        # shape / finiteness are mandatory, not advisory
        if P.shape != (EXPECTED_ROWS, N_CLASSES):
            problems.append("proba shape %r, expected exactly (%d, %d)"
                            % (P.shape, EXPECTED_ROWS, N_CLASSES))
        if P.dtype != np.float32:
            problems.append("proba dtype is %r, expected float32" % P.dtype)
        if not np.isfinite(P).all():
            problems.append("proba contains non-finite values")
        if P.ndim == 2 and P.shape[1] > 0 and not np.allclose(P.sum(axis=1),
                                                              1.0, atol=1e-3):
            problems.append("proba rows do not sum to 1")
    else:
        P = None

    labels = None
    if os.path.exists(lpath):
        labels = np.load(lpath).astype(np.int64)
        want = (man.get("label_sha256") or {}).get(system)
        if want is None:
            problems.append("manifest has no label_sha256 for %r" % system)
        elif _sha(labels) != want:
            problems.append("hard-label sha256 mismatch for %r" % system)
        if labels.shape != (EXPECTED_ROWS,):
            problems.append("labels shape %r != (%d,)"
                            % (labels.shape, EXPECTED_ROWS))
        if labels.min() < 0 or labels.max() >= N_CLASSES:
            problems.append("labels out of range [0, %d)" % N_CLASSES)
        # only compare when BOTH sides have the right shape: a mis-shaped array
        # is already a rejection, and broadcasting it against argmax would raise
        # ValueError instead of reporting a clean, actionable rejection.
        if (P is not None and P.shape == (EXPECTED_ROWS, N_CLASSES)
                and labels.shape == (EXPECTED_ROWS,)):
            # cross-check: the stored labels must be the argmax of the stored
            # probabilities.  A mismatch means one of the two is not from this
            # run.
            bad = int((P.argmax(1) != labels).sum())
            if bad:
                problems.append("argmax(proba) != stored hard labels at %d/%d "
                                "rows" % (bad, len(labels)))
    else:
        problems.append("missing required artifact %s" % lpath)

    if problems:
        raise ArtifactRejected("; ".join(problems))

    return {"ok": True, "problems": [], "system": system, "mode": mode,
            "dir": d, "n_runs": int(P.shape[0]),
            "proba": P if proba else None, "labels": labels,
            "run_ids": json.load(open(os.path.join(d, "oof_runid.json"),
                                      "r", encoding="utf-8")),
            "manifest": man,
            "verified_checks": ["verified", "schema", "seed", "n_outer",
                                "n_inner", "n_classes", "full_coverage",
                                "mode", "expected_rows", "n_runs",
                                "data_sha256", "code_sha256",
                                "runner_model_path_sha256", "run_id_sha256",
                                "fold_val_sha256", "row_coverage",
                                "proba_sha256", "label_sha256",
                                "argmax_equals_labels", "shape", "finite",
                                "rowsum1"]}


def load_all(mode=MD.CV, **kw):
    """Both systems at once, still fully verified."""
    return {k: load(system=k, mode=mode, **kw) for k in SYSTEMS}


def load_unverified(system="B_plus_window", mode=MD.CV, artifact_dir=None,
                    **why):
    """EXPLORATORY ONLY - reads artifacts with NO provenance checking.

    This deliberately performs no verification at all: no manifest hashes, no
    fold re-derivation, no run-order comparison, no data/code checks.  It exists
    so that a developer staring at an array can do so without pretending the
    array is trustworthy.

    The returned dict is stamped ``ok=False`` and ``provenance=None`` and prints
    a warning, so it cannot be mistaken for a verified result downstream.  Any
    number that reaches a report, a submission or a comparison MUST come from
    ``load()``.  If you find yourself calling this in an experiment script, you
    want ``load()`` and you are missing an argument.
    """
    import warnings
    d = artifact_dir or MD.outdir(mode)
    mpath = os.path.join(d, "honest_manifest.json")
    msg = ("load_unverified(): reading %s/%s with NO provenance checks. "
           "Results are NOT trustworthy." % (system, d))
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    print("  [WARNING] %s%s" % (msg, (" (%s)" % why) if why else ""),
          file=sys.stderr)
    ppath = os.path.join(d, "oof_proba_%s.npy" % system)
    lpath = os.path.join(d, "oof_%s.npy" % system)
    rid = os.path.join(d, "oof_runid.json")
    return {"ok": False, "provenance": None, "system": system, "mode": mode,
            "dir": d, "warning": msg,
            "proba": np.load(ppath) if os.path.exists(ppath) else None,
            "labels": (np.load(lpath).astype(np.int64)
                       if os.path.exists(lpath) else None),
            "run_ids": json.load(open(rid, "r", encoding="utf-8"))
            if os.path.exists(rid) else None,
            "manifest": json.load(open(mpath, "r", encoding="utf-8"))
            if os.path.exists(mpath) else None}
