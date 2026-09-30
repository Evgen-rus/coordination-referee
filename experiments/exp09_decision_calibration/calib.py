"""Exp09 - decision-layer calibration on top of Exp08.  Reuse only, no training.

WHAT THIS IS
------------
Exp08 shipped ``pred_label = argmax(P)`` and ``fault_turn = argmax_t P[run, t,
class(pred_label)]``.  Exp09 changes exactly one statement and nothing else:

    score_c = log(max(P_c, 1e-15)) + delta_c
    pred    = argmax_c score_c

``delta`` is a single vector of 7 ADDITIVE CLASS OFFSETS in LOG space.  There is
no new model, no new feature, no threshold, no temperature, no probability
multiplication, no per-class fault-turn logic, no ensemble.  ``delta[0]``
(clean) is PINNED to 0: it is the reference level, and letting it move would
only be a reparametrisation of the other six.

Because the decision changes, ``fault_turn`` changes with it: the L1 window peak
is read for the NEW predicted class.  Every candidate is therefore scored on the
full official composite, macro AND robustness AND hit@2, never on macro alone.

PRE-DECLARED SEARCH SPACE, FIXED BEFORE ANY NUMBER WAS SEEN
----------------------------------------------------------
    start            delta = [0, 0, 0, 0, 0, 0, 0]
    clean            pinned to 0
    bounds           [-0.40, +0.40] on the six fault offsets
    step schedule    [0.20, 0.10, 0.05, 0.025, 0.0125]
    rule             deterministic coordinate descent: try +step and -step on
                     each of the six offsets, take the single best move by
                     TRAINING composite, apply, repeat until no legal move
                     improves the objective, then halve the step
    tie-break        1) higher composite  2) if within 1e-12, smaller
                     sum(delta^2)  3) first in the fixed class order

Bounds are never widened after seeing a result, the step schedule is never
changed, and no second search space is tried.

DETERMINISM IS EXACT, NOT APPROXIMATE
-------------------------------------
Every step in the schedule is an exact integer multiple of the finest step
0.0125 (16/8/4/2/1), and 0.40 is exactly 32 of them.  The search therefore
runs on an INTEGER vector of "units" and converts with ``units * 0.0125``.  No
accumulated floating-point drift, so a rerun cannot land on a different vertex
of the grid, and the tie-break is a comparison of integers rather than of
nearly-equal floats.

WHAT IS NOT CLAIMED
-------------------
This is OOF meta-CV: a cross-fitted decision layer sitting on top of Exp07's
already-existing OOF probabilities.  It is NOT a fully nested retrain of the
Exp07 base model - the base probabilities were produced once, and the offsets
never see them during a fold's own inner/outer training.  So the cross-fitted
number is an honest estimate of the OFFSET layer's contribution, not an
independent validation of the whole system.  The public leaderboard remains the
external validation.
"""

from __future__ import annotations

import hashlib
import json
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

from metrics import composite, f1_per_class, macro_f1   # noqa: E402
from metrics import fault_turn_hit_at_k                # noqa: E402
import load_exp07 as L                                  # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
# WINDOW classes: index 0 is background, so a RUN label maps through W2I.
W2I = {c: i for i, c in enumerate(["background"] + LABELS[1:])}

SYSTEM = "B_plus_window"
N_CLASSES = 7
CLEAN = 0
N_RUNS = 10000
EPS = 1e-15
TOL = 1e-12

SUCCESS_F1_PINNED = 0.8643757406010988   # Exp03 B, untouched by any decision rule

# ---- the pre-declared, frozen search space --------------------------------
UNIT = 0.0125                    # finest step
BOUND = 0.40                     # |delta| bound on each fault offset
BOUND_UNITS = int(round(BOUND / UNIT))          # 32
STEP_SCHEDULE = (0.20, 0.10, 0.05, 0.025, 0.0125)
STEP_UNITS = tuple(int(round(s / UNIT)) for s in STEP_SCHEDULE)   # 16,8,4,2,1
FREE_CLASSES = tuple(range(1, N_CLASSES))      # clean (0) is pinned

# local-cache dir (robustness mask only; never a probability, never a model)
CACHE = os.path.join(HERE, ".cache")


# ---------------------------------------------------------------------------
# 1. verified inputs - strict, fail closed
# ---------------------------------------------------------------------------

def load_verified(train_csv=None):
    """Return the verified Exp07 OOF + sealed window peaks.  No lenient mode.

    ``ArtifactRejected`` / ``PeakRejected`` propagate: there is no fallback that
    would let an unverified artifact reach the search.
    """
    train_csv = train_csv or os.path.join(ROOT, "data", "train.csv")
    train = pd.read_csv(train_csv)
    ys = train["label"].values.astype(object)
    yi = np.array([L2I[l] for l in ys], dtype=int)
    rids = [str(x) for x in train["run_id"]]
    t0 = time.time()
    art = L.load(yi=yi, train_run_ids=rids)          # raises on any problem
    return {
        "train": train, "ys": ys, "yi": yi, "run_ids": rids,
        "proba": art["proba"], "labels": art["labels"],
        "peak_pos": art["peak_pos"], "peak_prob": art["peak_prob"],
        "manifest": art["manifest"], "checks": art["verified_checks"],
        "load_sec": round(time.time() - t0, 2),
    }


def fold_ids(yi, manifest):
    """Outer folds, asserted equal to the sealed manifest's index hashes.

    Equal fold SIZES with different members is exactly the substitution a
    size-only check cannot see, so the comparison is on the index vector.
    """
    folds = L.reuse.fold_assignments(yi)
    got = [L.ca.sha256_indices(np.asarray(v, dtype=np.int64)) for v in folds]
    want = list(manifest["fold_val_sha256"])
    return folds, got, want


# ---------------------------------------------------------------------------
# 2. the robustness mask - cached, because it costs ~10s and is deterministic
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
    every candidate.  Cached under a fingerprint of train.csv + the feature
    module, so a stale cache can never be silently reused.
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
    """Immutable precomputed state.  Candidate evaluation touches nothing else.

    The search evaluates ~1500 candidates, so every candidate must be O(n) over
    small integer arrays and must never re-read a CSV, re-parse JSON, re-call
    Exp07 or re-derive a feature.  That is what this class is for.
    """

    def __init__(self, proba, yi, fturn, peak_pos, rob_mask, labels=None):
        self.n = int(proba.shape[0])
        if proba.shape != (self.n, N_CLASSES):
            raise ValueError("proba shape %r != (%d, 7)" % (proba.shape, self.n))
        self.P = np.asarray(proba, dtype=np.float64)
        self.logP = np.log(np.maximum(self.P, EPS))          # (n, 7)
        self.yi = np.asarray(yi, dtype=np.int64)
        self.fturn = np.asarray(fturn, dtype=np.int64)
        self.rob = np.asarray(rob_mask, dtype=bool)
        self.labels = (np.asarray(labels, dtype=np.int64) if labels is not None
                       else self.P.argmax(1))
        # ---- candidate L1 turn matrix (n, 7) -------------------------------
        # column 0 (clean) is -1 by rule; column c>=1 is the sealed window peak
        # for that class, i.e. peak_pos[:, W2I[LABELS[c]]] = peak_pos[:, c].
        T = np.empty((self.n, N_CLASSES), dtype=np.int64)
        T[:, CLEAN] = -1
        for c in FREE_CLASSES:
            T[:, c] = np.asarray(peak_pos[:, W2I[LABELS[c]]], dtype=np.int64)
        self.T = T
        if (self.T[:, FREE_CLASSES] < 0).any():
            raise ValueError("sealed window peaks contain a negative position")

    # -- reporting helpers ------------------------------------------------
    def predict(self, delta):
        """The calibrated decision, on every row: argmax(log P + delta)."""
        return np.argmax(self.logP + np.asarray(delta, dtype=np.float64),
                         axis=1).astype(np.int64)

    def name(self, idx_int):
        return [LABELS[i] for i in idx_int]

    def turns_for(self, pred, idx=None):
        """L1 turns for a predicted-label vector, using the SEALED peaks."""
        rows = np.arange(self.n) if idx is None else np.asarray(idx)
        return self.T[rows, np.asarray(pred, dtype=np.int64)]


# ---------------------------------------------------------------------------
# 4. Objective: composite over a fixed set of rows
# ---------------------------------------------------------------------------

class Objective(object):
    """The official composite, restricted to a fixed row subset.

    Identical arithmetic to ``evaluation.metrics`` - ``composite()`` itself is
    called, not a hand-written copy - but the class counts, the robustness
    subset and the faulty-run subset are extracted ONCE per Objective so that a
    candidate costs a handful of vector ops.

    ``run_experiment.py`` asserts this fast path against the official
    ``macro_f1`` / ``fault_turn_hit_at_k`` on real candidates before any search
    is allowed to run.
    """

    __slots__ = ("LP", "y", "row_counts", "present", "rob_pos", "rob_y",
                 "rob_row_counts", "rob_present", "fpos", "frows", "fy", "Tf",
                 "fturn_f", "n", "n_faulty", "n_rob", "ix", "corpus")

    def __init__(self, corpus, ix):
        ix = np.asarray(ix, dtype=np.int64)
        self.ix = ix
        self.corpus = corpus
        self.n = int(len(ix))
        self.LP = np.ascontiguousarray(corpus.logP[ix])          # (m, 7)

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
        self.frows = np.asarray(ix)[f]            # GLOBAL row indices
        self.fy = y[f]
        self.Tf = np.ascontiguousarray(corpus.T[self.frows])     # (q, 7)
        self.fturn_f = corpus.fturn[self.frows]
        self.n_faulty = int(len(f))

    # ---- the five quantities --------------------------------------------
    def predict(self, delta):
        return np.argmax(self.LP + delta, axis=1).astype(np.int64)

    def _macro_from_conf(self, conf, row_counts, present):
        col = conf.sum(axis=0)
        tp = np.diag(conf)
        denom = row_counts + col                       # 2tp + fp + fn
        f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
        return float(f1[present].mean())

    def metrics(self, delta, pred=None):
        if pred is None:
            pred = self.predict(delta)
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
            rf = self._macro_from_conf(cr, self.rob_row_counts, self.rob_present)
        else:
            rf = 0.0
        # ---- hit@2 with the peak of the PREDICTED class -----------------
        if self.n_faulty:
            turn = self.Tf[np.arange(self.n_faulty), pred[self.fpos]]
            ok = ((pred[self.fpos] == self.fy) & (turn >= 0)
                  & (np.abs(turn - self.fturn_f) <= 2))
            h2 = float(ok.sum()) / self.n_faulty
        else:
            h2 = 0.0
        return {"macro_f1": mf, "robustness_f1": rf, "success_f1": SUCCESS_F1_PINNED,
                "fault_turn_hit2": h2,
                "composite": composite({"macro_f1": mf, "robustness_f1": rf,
                                        "success_f1": SUCCESS_F1_PINNED,
                                        "fault_turn_hit2": h2})}

    def composite(self, delta):
        return self.metrics(delta)["composite"]

    # ---- official cross-check (slow, used by the gate only) -------------
    def official(self, delta):
        """Recompute with ``evaluation.metrics`` itself, no fast path.

        NOTE on alignment: ``fault_turn_hit_at_k`` receives ONLY the truly
        faulty rows, for labels AND turns together.  Its ``zip`` truncates to
        the shortest input, so passing full-length label lists next to
        faulty-only turn lists would pair row i's label with row i's turn and
        silently produce a different number - precisely the bug that made Exp07
        ship hit@2 0.1843 against the true 0.4258.  Skipping the ``clean``
        rows is a no-op for the metric (it ``continue``s on them) and removes
        the hazard entirely.
        """
        pred = self.predict(delta)
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
# 5. the pre-declared deterministic coordinate search
# ---------------------------------------------------------------------------

def _better(comp_a, units_a, comp_b, units_b):
    """Strict weak ordering, exactly the pre-declared tie-break.

    1) strictly higher composite wins;
    2) equal within 1e-12 -> smaller sum(delta^2) wins;
    3) still equal -> the incumbent stays, because the scan order is the fixed
       class order and the FIRST of two identical candidates is kept.
    """
    if comp_a > comp_b + TOL:
        return True
    if abs(comp_a - comp_b) <= TOL:
        sa = int((units_a.astype(np.int64) ** 2).sum())
        sb = int((units_b.astype(np.int64) ** 2).sum())
        return sa < sb
    return False


def coordinate_search(obj, verbose=True, log=print):
    """Deterministic coordinate descent inside the FROZEN search space.

    Returns ``(units, info)``.  ``units`` is the integer grid coordinate; the
    shipped offsets are ``units * 0.0125`` and ``units[0]`` is always 0.
    """
    units = np.zeros(N_CLASSES, dtype=np.int64)
    cur = obj.composite(units * UNIT)
    info = {"start_composite": cur, "steps": [], "n_candidates": 0,
            "units": units.tolist(), "grid_unit": UNIT,
            "step_schedule": list(STEP_SCHEDULE), "bound": BOUND}

    for s_units, s_val in zip(STEP_UNITS, STEP_SCHEDULE):
        n_moves = 0
        while True:
            best = None                       # (comp, units)
            n_tried = 0
            for c in FREE_CLASSES:             # fixed class order
                for sgn in (+1, -1):          # fixed sign order
                    cand = units.copy()
                    cand[c] = units[c] + sgn * s_units
                    if abs(int(cand[c])) > BOUND_UNITS:
                        continue               # BOUND: never exceeded, never clipped
                    n_tried += 1
                    info["n_candidates"] += 1
                    v = obj.composite(cand * UNIT)
                    if best is None or _better(v, cand, best[0], best[1]):
                        best = (v, cand)
            if best is None or best[0] <= cur + TOL:
                if verbose:
                    log("    step %.4f: no legal move improves the training "
                        "composite (%.16f) - halve the step" % (s_val, cur))
                break
            cur, units = best[0], best[1]
            n_moves += 1
            if verbose:
                log("    step %.4f: move #%d -> composite %.16f  units %s"
                    % (s_val, n_moves, cur, units.tolist()))
        info["steps"].append({"step": s_val, "n_moves": n_moves,
                              "composite": cur, "units": units.tolist()})
    info["final_composite"] = cur
    info["final_units"] = units.tolist()
    return units, info


def to_offsets(units):
    """Integer grid -> the shipped float offsets (index 0 pinned to 0.0)."""
    d = np.asarray(units, dtype=np.float64) * UNIT
    d[CLEAN] = 0.0
    return d


def check_bounds(offsets):
    """Hard assertion: clean pinned, every fault offset inside the declared box."""
    d = np.asarray(offsets, dtype=np.float64)
    assert d.shape == (N_CLASSES,), d.shape
    assert d[CLEAN] == 0.0, "clean offset is %r, must be pinned to 0" % d[CLEAN]
    bad = [(LABELS[c], float(d[c])) for c in FREE_CLASSES if abs(d[c]) > BOUND]
    assert not bad, "offset out of the declared bounds: %r" % bad
    return True
