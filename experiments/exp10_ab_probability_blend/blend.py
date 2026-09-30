"""Exp10 - probability blend of the Exp07 A and B label heads.  Reuse only.

THE ONE HYPOTHESIS
------------------
Exp07 fitted two label heads on the same 3 outer folds and both are still on
disk as verified honest OOF:

    A = ``A_foundation``   249 features (Exp06b ``full - age``)
    B = ``B_plus_window``  300 features = the same 249 + 51 window aggregates

Exp08 ships B.  Exp10 asks whether a plain probability-level ensemble of the two
improves the official composite over B alone:

    P_blend = (1 - alpha) * P_A + alpha * P_B
    pred    = argmax(P_blend)

``alpha = 0`` is pure A, ``alpha = 1`` is exactly Exp08's B.  ONE scalar.  No
class-specific alpha, no offsets, no temperature, no geometric or logarithmic
blend, no thresholds, no stacking model, no new feature, no new model, no
retrain of anything.

Because the label can move, ``fault_turn`` moves with it: the Exp08 L1 rule is
re-applied to the BLENDED predicted label,

    blended pred == clean          ->  fault_turn = -1
    otherwise                     ->  fault_turn = peak_pos[i, W2I[pred]]

so hit@2 is re-derived, never held fixed, and every candidate is scored on the
FULL official composite - macro AND robustness AND hit@2.  Optimising macro
alone would be optimising 0.50 of the objective and silently paying for it in
the other 0.50.

FIXED SEARCH SPACE, DECLARED BEFORE ANY NUMBER WAS SEEN
-------------------------------------------------------
    alpha in {0.00, 0.05, 0.10, ..., 1.00}   (21 values, step 0.05)
    objective     official ``evaluation.metrics.composite()``
    tie-break     1) higher official composite
                  2) if the gap is <= 1e-12, the alpha closer to 1.0
                     (i.e. the incumbent Exp08 baseline)
                  3) if still tied, the LARGER alpha

The grid is not widened, not refined, and never re-run at finer resolution
after a result is visible.  21 evaluations is the whole search.

DETERMINISM
-----------
Blending happens in float64 on arrays converted once at load, and the 21 grid
points are exact multiples of 0.05 representable as ``k / 20``.  ``alpha = 0``
gives ``1.0 * P_A + 0.0 * P_B == P_A`` bit-for-bit, and ``alpha = 1`` gives
``0.0 * P_A + 1.0 * P_B == P_B`` bit-for-bit, so the two endpoints are exact by
construction rather than by tolerance.  Reruns are bit-identical.

WHAT IS NOT CLAIMED
-------------------
This is OOF meta-CV: a cross-fitted scalar sitting on top of Exp07's existing
OOF probabilities.  It is NOT a fully nested retrain of the Exp07 base models -
those probabilities were produced once, and the alpha chosen for a fold never
sees that fold.  So the cross-fitted number is an honest estimate of the BLEND
LAYER's contribution, not an independent validation of the whole system.  The
public leaderboard remains the external validation, and its score is recorded
here as a reference only - the search never reads it.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP07 = os.path.join(ROOT, "experiments", "exp07_fault_windows")
EXP08 = os.path.join(ROOT, "experiments", "exp08_window_localizer")
for _p in (HERE, EXP08, EXP07, os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import composite, f1_per_class, macro_f1      # noqa: E402
from metrics import fault_turn_hit_at_k                   # noqa: E402
import load_exp07 as L                                     # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
# WINDOW classes: index 0 is ``background``, so a RUN label maps through W2I.
# Reading a peak with a LABEL index would shift every class by one and still
# produce plausible numbers - this is the mapping, not a convenience.
W2I = {c: i for i, c in enumerate(["background"] + LABELS[1:])}

SYSTEM_A = "A_foundation"
SYSTEM_B = "B_plus_window"
N_CLASSES = 7
CLEAN = 0
FREE_CLASSES = tuple(range(1, N_CLASSES))       # clean (0) has no window class
N_RUNS = 10000
TOL = 1e-12

SUCCESS_F1_PINNED = 0.8643757406010988   # Exp03 B, untouched by any decision rule

# ---- the FROZEN search space ------------------------------------------------
N_STEPS = 20
GRID = tuple(k / float(N_STEPS) for k in range(N_STEPS + 1))   # 0.00 .. 1.00
ALPHA_MIN = 0.0
ALPHA_MAX = 1.0

# local cache (robustness mask only; never a probability, never a model)
CACHE = os.path.join(HERE, ".cache")


def check_alpha(a):
    """Hard assertion: alpha is a real scalar inside the declared [0, 1] box."""
    v = float(a)
    if not (v == v):                       # NaN
        raise AssertionError("alpha is NaN")
    if not (ALPHA_MIN - TOL <= v <= ALPHA_MAX + TOL):
        raise AssertionError("alpha %r outside the declared [0, 1] box" % v)
    return v


# ---------------------------------------------------------------------------
# 1. verified inputs - strict, fail closed
# ---------------------------------------------------------------------------

def load_verified(train_csv=None):
    """Return BOTH verified Exp07 OOF matrices plus the sealed window peaks.

    ``ArtifactRejected`` / ``PeakRejected`` propagate to the caller.  There is
    no lenient path and no fallback: an unverified array must never reach the
    search, because an unverified array would make every number below it a
    guess dressed as a measurement.
    """
    train_csv = train_csv or os.path.join(ROOT, "data", "train.csv")
    train = pd.read_csv(train_csv)
    ys = train["label"].values.astype(object)
    yi = np.array([L2I[l] for l in ys], dtype=int)
    rids = [str(x) for x in train["run_id"]]
    t0 = time.time()
    artA = L.load(system=SYSTEM_A, yi=yi, train_run_ids=rids)   # raises
    artB = L.load(system=SYSTEM_B, yi=yi, train_run_ids=rids)   # raises
    # The peaks are sealed against the B manifest; A's manifest declares the
    # same seal because both systems were produced by the same Exp07 run.  If
    # they ever disagreed, one of the two probability matrices would be attached
    # to peaks it was not produced with.
    if not np.array_equal(artA["peak_pos"], artB["peak_pos"]):
        raise L.PeakRejected(
            "A and B disagree on window_peak_pos - the two OOF matrices do not "
            "come from the same Exp07 run and must not be blended")
    return {
        "train": train, "ys": ys, "yi": yi, "run_ids": rids,
        "proba_A": artA["proba"], "labels_A": artA["labels"],
        "proba_B": artB["proba"], "labels_B": artB["labels"],
        "peak_pos": artB["peak_pos"], "peak_prob": artB["peak_prob"],
        "manifest": artB["manifest"], "checks": artB["verified_checks"],
        "load_sec": round(time.time() - t0, 2),
    }


def fold_ids(yi, manifest):
    """Outer folds, asserted equal to the sealed manifest's index hashes.

    Equal fold SIZES with different members is precisely the substitution a
    size-only check cannot see, so the comparison is on the index vector.
    """
    folds = L.reuse.fold_assignments(yi)
    got = [L.ca.sha256_indices(np.asarray(v, dtype=np.int64)) for v in folds]
    want = list(manifest["fold_val_sha256"])
    return folds, got, want


# ---------------------------------------------------------------------------
# 2. the robustness mask - cached, deterministic, never a prediction
# ---------------------------------------------------------------------------

def _mask_fingerprint(train_csv):
    h = hashlib.sha256()
    h.update(hashlib.sha256(open(train_csv, "rb").read()).digest())
    for p in (os.path.join(ROOT, "baseline", "features.py"),):
        h.update(hashlib.sha256(open(p, "rb").read()).digest())
    return h.hexdigest()[:16]


def robustness_mask(train_csv=None, use_cache=True):
    """Exp07's ``hard/robustness`` slice mask, byte-for-byte its definition.

        nmsg > median(nmsg)  &  (topo_mesh > 0 | topo_blackboard > 0)

    It is a property of the DATA, not of any prediction, so it is identical for
    every alpha.  Cached under a fingerprint of train.csv + the feature module,
    so a stale cache can never be silently reused.
    """
    train_csv = train_csv or os.path.join(ROOT, "data", "train.csv")
    fp = _mask_fingerprint(train_csv)
    path = os.path.join(CACHE, "robustness_mask.%s.npy" % fp)
    if use_cache and os.path.exists(path):
        m = np.load(path)
        if m.shape == (N_RUNS,) and m.dtype == np.bool_:
            print("  robustness mask: cache hit (%s)" % os.path.basename(path))
            return m
    from features import extract_features, parse_run
    t0 = time.time()
    train = pd.read_csv(train_csv)
    runs = [parse_run(r) for r in train.to_dict("records")]
    Xb = pd.DataFrame([extract_features(r) for r in runs]).fillna(0.0)
    nmsg = Xb["n_messages"].values
    m = (nmsg > np.median(nmsg)) & ((Xb["topo_mesh"].values > 0)
                                    | (Xb["topo_blackboard"].values > 0))
    m = np.asarray(m, dtype=bool)
    if use_cache:
        os.makedirs(CACHE, exist_ok=True)
        np.save(path, m)
    print("  robustness mask: built in %.1fs, %d rows" % (time.time() - t0,
                                                          int(m.sum())))
    return m


# ---------------------------------------------------------------------------
# 3. the Corpus: everything a candidate evaluation needs, precomputed ONCE
# ---------------------------------------------------------------------------

class Corpus(object):
    """Immutable precomputed state for the two-system blend.

    Candidate evaluation touches nothing else: no CSV, no JSON, no Exp07 call,
    no feature.  That is the whole reason this class exists - the search is
    21 evaluations per fold and each must be a few vector ops over 10000 rows.
    """

    def __init__(self, proba_A, proba_B, yi, fturn, peak_pos, rob_mask,
                 labels_B=None):
        PA = np.asarray(proba_A, dtype=np.float64)
        PB = np.asarray(proba_B, dtype=np.float64)
        if PA.shape != (N_RUNS, N_CLASSES):
            raise ValueError("P_A shape %r != (%d, %d)"
                             % (PA.shape, N_RUNS, N_CLASSES))
        if PB.shape != (N_RUNS, N_CLASSES):
            raise ValueError("P_B shape %r != (%d, %d)"
                             % (PB.shape, N_RUNS, N_CLASSES))
        # A probability matrix that is not finite or not a distribution would
        # make every downstream number meaningless, and argmax on NaN returns
        # a silently wrong class rather than raising.
        for tag, P in (("P_A", PA), ("P_B", PB)):
            if not np.isfinite(P).all():
                raise ValueError("%s contains non-finite values" % tag)
            if (P < 0).any() or (P > 1).any():
                raise ValueError("%s has values outside [0, 1]" % tag)
        self.n = N_RUNS
        self.PA = PA
        self.PB = PB
        self.yi = np.asarray(yi, dtype=np.int64)
        self.fturn = np.asarray(fturn, dtype=np.int64)
        self.rob = np.asarray(rob_mask, dtype=bool)
        # alpha = 1 is Exp08, so the reference label vector is argmax(P_B)
        self.labels_B = (np.asarray(labels_B, dtype=np.int64)
                         if labels_B is not None else PB.argmax(1))
        self.labels_A = PA.argmax(1)

        # ---- candidate L1 turn matrix (n, 7) --------------------------------
        # column 0 (clean) is -1 by rule; column c >= 1 is the SEALED window peak
        # for that class, i.e. peak_pos[:, W2I[LABELS[c]]].
        T = np.empty((self.n, N_CLASSES), dtype=np.int64)
        T[:, CLEAN] = -1
        for c in FREE_CLASSES:
            T[:, c] = np.asarray(peak_pos[:, W2I[LABELS[c]]], dtype=np.int64)
        self.T = T
        if (self.T[:, FREE_CLASSES] < 0).any():
            raise ValueError("sealed window peaks contain a negative position")

    # ---- the blend itself -------------------------------------------------
    def blend(self, alpha, rows=None):
        """``(1 - alpha) * P_A + alpha * P_B`` in float64.

        Endpoints are exact by construction: ``1.0 * P_A + 0.0 * P_B`` is
        ``P_A`` and ``0.0 * P_A + 1.0 * P_B`` is ``P_B``, bit for bit, because
        IEEE multiply/accumulate by exactly 0.0 or 1.0 is the identity.
        """
        a = check_alpha(alpha)
        if rows is None:
            return (1.0 - a) * self.PA + a * self.PB
        r = np.asarray(rows, dtype=np.int64)
        return (1.0 - a) * self.PA[r] + a * self.PB[r]

    def predict(self, alpha, rows=None):
        """argmax(P_blend) - the ONLY decision rule in this experiment."""
        return np.argmax(self.blend(alpha, rows), axis=1).astype(np.int64)

    def turns_for(self, pred, rows=None):
        """L1 turns for a predicted-label vector, using the SEALED peaks.

        This is the Exp08 rule, unchanged, applied to whatever label vector it
        is handed - which is the point: the turn belongs to the BLENDED label.
        """
        r = np.arange(self.n) if rows is None else np.asarray(rows, dtype=np.int64)
        return self.T[r, np.asarray(pred, dtype=np.int64)]

    def name(self, idx):
        return [LABELS[i] for i in np.asarray(idx)]


# ---------------------------------------------------------------------------
# 4. Objective: the official composite over a fixed row subset
# ---------------------------------------------------------------------------

class Objective(object):
    """The official composite, restricted to a fixed row subset.

    The arithmetic is the same as ``evaluation.metrics`` - ``composite()`` is
    itself called, not reimplemented - but the class counts, the robustness
    subset and the faulty-run subset are extracted ONCE per Objective so a
    candidate costs a handful of vector ops.  ``run_experiment.py`` asserts this
    fast path against the official ``macro_f1`` / ``fault_turn_hit_at_k`` on
    real, non-trivial candidates before the search is allowed to run.
    """

    __slots__ = ("corpus", "ix", "n", "PA", "PB", "y", "row_counts", "present",
                 "rob_pos", "rob_y", "rob_row_counts", "rob_present", "n_rob",
                 "fpos", "frows", "fy", "Tf", "fturn_f", "n_faulty")

    def __init__(self, corpus, ix):
        ix = np.asarray(ix, dtype=np.int64)
        if ix.size == 0:
            raise ValueError("an Objective over zero rows has no objective")
        self.corpus = corpus
        self.ix = ix
        self.n = int(len(ix))
        self.PA = np.ascontiguousarray(corpus.PA[ix])          # (m, 7)
        self.PB = np.ascontiguousarray(corpus.PB[ix])          # (m, 7)

        y = corpus.yi[ix].astype(np.int64)
        self.y = y
        self.row_counts = np.bincount(y, minlength=N_CLASSES)
        self.present = self.row_counts > 0

        r = np.where(corpus.rob[ix])[0]                          # positions
        self.rob_pos = r
        self.rob_y = y[r]
        self.rob_row_counts = np.bincount(self.rob_y, minlength=N_CLASSES)
        self.rob_present = self.rob_row_counts > 0
        self.n_rob = int(len(r))

        f = np.where(corpus.yi[ix] != CLEAN)[0]   # POSITIONS within this subset
        self.fpos = f
        self.frows = ix[f]                        # GLOBAL row indices
        self.fy = y[f]
        self.Tf = np.ascontiguousarray(corpus.T[self.frows])     # (q, 7)
        self.fturn_f = corpus.fturn[self.frows]
        self.n_faulty = int(len(f))

    # ---- the five quantities --------------------------------------------
    def predict(self, alpha):
        a = check_alpha(alpha)
        return np.argmax((1.0 - a) * self.PA + a * self.PB, axis=1).astype(np.int64)

    def _macro_from_conf(self, conf, row_counts, present):
        col = conf.sum(axis=0)
        tp = np.diag(conf)
        denom = row_counts + col                       # 2tp + fp + fn
        f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
        return float(f1[present].mean())

    def metrics(self, alpha, pred=None):
        if pred is None:
            pred = self.predict(alpha)
        # ---- macro F1 --------------------------------------------------
        conf = np.bincount(self.y * N_CLASSES + pred,
                           minlength=N_CLASSES * N_CLASSES).reshape(
                               N_CLASSES, N_CLASSES)
        mf = self._macro_from_conf(conf, self.row_counts, self.present)
        # ---- robustness F1 --------------------------------------------
        if self.n_rob:
            cr = np.bincount(self.rob_y * N_CLASSES + pred[self.rob_pos],
                             minlength=N_CLASSES * N_CLASSES).reshape(
                                 N_CLASSES, N_CLASSES)
            rf = self._macro_from_conf(cr, self.rob_row_counts,
                                       self.rob_present)
        else:
            rf = 0.0
        # ---- hit@2 with the peak of the BLENDED PREDICTED class ---------
        if self.n_faulty:
            turn = self.Tf[np.arange(self.n_faulty), pred[self.fpos]]
            ok = ((pred[self.fpos] == self.fy) & (turn >= 0)
                  & (np.abs(turn - self.fturn_f) <= 2))
            h2 = float(ok.sum()) / self.n_faulty
        else:
            h2 = 0.0
        return {"macro_f1": mf, "robustness_f1": rf,
                "success_f1": SUCCESS_F1_PINNED, "fault_turn_hit2": h2,
                "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                        "success_f1": SUCCESS_F1_PINNED,
                                        "fault_turn_hit2": h2})}

    def composite(self, alpha):
        return self.metrics(alpha)["composite"]

    # ---- official cross-check (slow, used by the gate only) -------------
    def official(self, alpha):
        """Recompute with ``evaluation.metrics`` itself, no fast path.

        NOTE on alignment: ``fault_turn_hit_at_k`` receives ONLY the truly
        faulty rows, for labels AND turns together.  Its ``zip`` truncates to
        the shortest input, so passing full-length label lists next to
        faulty-only turn lists would pair row i's label with row i's turn and
        silently produce a different number - the same class of bug that made
        Exp07 ship hit@2 0.1843 against the true 0.4258.  Skipping the clean
        rows is a no-op for the metric (it ``continue``s on them) and removes
        the hazard entirely.
        """
        pred = self.predict(alpha)
        yt = [LABELS[i] for i in self.y]
        yp = [LABELS[i] for i in pred]
        pf = pred[self.fpos]                    # predictions on faulty rows
        turns = self.Tf[np.arange(self.n_faulty), pf]
        mf = macro_f1(yt, yp, LABELS)
        rf = (macro_f1([yt[j] for j in self.rob_pos],
                       [yp[j] for j in self.rob_pos], LABELS)
              if self.n_rob else 0.0)
        h2 = fault_turn_hit_at_k([LABELS[i] for i in self.fy],
                                 [LABELS[i] for i in pf],
                                 [int(t) for t in self.fturn_f],
                                 [int(t) for t in turns], k=2)
        return {"macro_f1": mf, "robustness_f1": rf,
                "success_f1": SUCCESS_F1_PINNED, "fault_turn_hit2": h2,
                "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                        "success_f1": SUCCESS_F1_PINNED,
                                        "fault_turn_hit2": h2})}


# ---------------------------------------------------------------------------
# 5. the pre-declared selection rule - exactly the declared tie-break
# ---------------------------------------------------------------------------

def _better(comp_a, alpha_a, comp_b, alpha_b):
    """Strict weak ordering implementing the FROZEN tie-break.

    1) a strictly higher official composite wins;
    2) within 1e-12 the alpha CLOSER TO 1.0 wins - the incumbent Exp08 baseline
       is kept unless the blend beats it outright, so "no evidence" resolves
       to "do not ship";
    3) still tied -> the LARGER alpha wins.

    Written as a strict order (never "less than") so that scanning the grid in
    order cannot make the outcome depend on scan order.
    """
    a, b = float(alpha_a), float(alpha_b)
    if comp_a > comp_b + TOL:
        return True
    if abs(comp_a - comp_b) <= TOL:
        return a > b          # (2) and (3) agree in direction on [0, 1]
    return False


def select_alpha(obj, grid=GRID):
    """Best alpha for one Objective.  Returns ``(alpha, table)``.

    The full 21-row table is returned so the report can show the curve rather
    than assert a winner.
    """
    table = []
    best_a, best_c = None, None
    for a in grid:
        m = obj.metrics(a)
        table.append({"alpha": float(a), "composite": m["composite"],
                      "macro_f1": m["macro_f1"],
                      "robustness_f1": m["robustness_f1"],
                      "fault_turn_hit2": m["fault_turn_hit2"]})
        if best_a is None or _better(m["composite"], a, best_c, best_a):
            best_a, best_c = float(a), m["composite"]
    return best_a, table
