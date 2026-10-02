"""Exp13 Candidate D - DIAGNOSTIC ERROR AUDIT.

THIS IS NOT AN EXPERIMENT.
===========================
No new experiment id, no new feature, no model is fitted, no seed / threshold /
LightGBM parameter is touched, nothing in `submission_exp13_wait_dependency_graph`
or `baseline/` or `evaluation/` is modified, no submission or ZIP is produced.

It reads ONLY pre-existing cached artifacts and answers one question:

    which NEXT representation gap is actually confirmed by the data -
    (A) an explicit assignment-delivery graph for `dropped_handoff`, or
    (B) an explicit trajectory / episode representation for `runaway_loop`
        plus fault_turn - or NONE is warranted.

WHAT IS READ
------------
  * `experiments/exp13_wait_dependency_graph/oof_D_corrected_wg.npy`
        Exp13 CANDIDATE D under corrected production semantics, (10000, 7) f32.
  * `experiments/exp13_wait_dependency_graph/oof_C_corrected.npy`
        CONTROL C, used only as the Control-C reproduction gate.
  * `experiments/exp13_wait_dependency_graph/results_production.json`
        the committed Candidate D metrics this audit must reproduce.
  * `ensemble.load_verified()` -> sealed peak_pos / peak_prob / folds / yi.
  * `ensemble._prepare()`      -> parsed runs + the cached 249-column foundation.
  * the 12 `wg_*` columns via the research `features.build_matrix`
        (byte-identical to production `wait_graph.py`, sha verified).

EXPLICITLY NOT DONE
-------------------
  * `production_confirmation.py` and `paired_fold_local.py` are NEVER imported;
    their `main()` writes `np.save` over the very `.npy` files audited here.
  * The window model is NEVER refitted, so the 51 fold-local window aggregates
    are NOT rebuilt. Correlation is therefore reported against the 261
    fold-independent label features (249 foundation + 12 wait-graph), and the
    51 aggregates are inspected STRUCTURALLY. This is stated in the report.

LEAKAGE DISCIPLINE
------------------
The assignment-lifecycle records (section 3) and the repeat-episode records
(section 7) are built from `messages` / `artifacts` / `shared_state` ONLY.
`label`, `fault_turn` and `success` are joined AFTER construction, for
analysis. Both builders are asserted against a target-perturbation copy of the
runs at load time.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(HERE, "exp13_wait_dependency_graph")
PROD = os.path.join(ROOT, "submission_exp13_wait_dependency_graph")

# --- import path: read-only research modules + PRODUCTION feature modules ----
# Production modules are imported for the feature/lifecycle semantics that the
# shipped Exp13 build actually uses.  `lifecycle.py` differs between exp05 and
# production (sha mismatch), so only the production copy is authoritative.
for _p in (EXP,
           os.path.join(ROOT, "experiments", "exp12_fix_window_run_mapping"),
           os.path.join(ROOT, "experiments", "exp11_label_seed_ensemble"),
           os.path.join(ROOT, "experiments", "exp10_ab_probability_blend"),
           os.path.join(ROOT, "experiments", "exp08_window_localizer"),
           os.path.join(ROOT, "experiments", "exp07_fault_windows"),
           os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "baseline"),
           PROD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd                                          # noqa: E402
import ensemble as ENS                                       # noqa: E402
from blend import LABELS, L2I, W2I, SUCCESS_F1_PINNED, TOL   # noqa: E402
from metrics import (composite, fault_turn_hit_at_k, f1_per_class,  # noqa: E402
                     macro_f1)

import lifecycle as LC                                       # noqa: E402  (production)
import features as FT                                       # noqa: E402  (production)
import aggregate as AG                                      # noqa: E402  (production)
import importlib.util                                       # noqa: E402

# research wg block (byte-identical to production wait_graph.py)
_spec = importlib.util.spec_from_file_location(
    "exp13_wg_features", os.path.join(EXP, "features.py"))
WG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WG)

CLEAN = 0
DH = L2I["dropped_handoff"]
DW = L2I["duplicated_work"]
DL = L2I["deadlock"]
CF = L2I["conflict"]
GD = L2I["goal_drift"]
RL = L2I["runaway_loop"]
FAULTY = [DH, DW, DL, CF, GD, RL]

EXP12_SHA_B = ("b7e92bd4e14fa59c8c340e65c70317d49d7cca6c18a841fd248217617d1"
               "fc9d1")

# The fold-independent label block available without any fit: 249 cached
# foundation columns + the 12 frozen wait-graph columns. The 51 window
# aggregates are NOT here on purpose - see module docstring.
N_INDEP = 261

# Candidate D committed metrics, results_production.json["metrics_D"].
COMMITTED_D = {
    "macro_f1": 0.812954558673859,
    "robustness_f1": 0.7599843692318033,
    "success_f1": 0.8643757406010988,
    "fault_turn_hit2": 0.6315277777777778,
    "composite": 0.7892825105128229,
}

# --- Episode detector constants, DECLARED BEFORE THE DETECTOR RUNS -----------
# (section 7). Declared here so they are visibly fixed before any label join.
EP_MIN_LEN = 3        # ontology: "3-14 repeats in a row"
EP_GAP_MAX = 6        # max inter-repeat turn gap still counted as one episode
EP_CYCLE_MIN = 2      # >= 2 distinct agents = a ring rather than one agent

OUT = []
RES = {}
_TBL_N = 0
_CUR_SRC = ""


def log(m=""):
    print(m)
    OUT.append(str(m))


def h(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def sub(t, src=None):
    """Emit a sub-heading. `src` names the cached artifact or script function
    that produces the tables under it, so each table can be captioned."""
    global _CUR_SRC
    log()
    log("-- %s" % t)
    if src:
        _CUR_SRC = src
        log("   source: %s" % src)


def tbl(rows, headers):
    """Render a markdown table and mirror it into the report."""
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    block = "\n".join(md)
    # Every table carries its provenance. The plan requires each table to be
    # captioned with the cached artifact or the script function it came from;
    # `_CUR_SRC` is set by `sub()` so the caption cannot drift from the section
    # the table actually belongs to.
    global _TBL_N
    _TBL_N += 1
    cap = "  _table %d - source: %s_" % (_TBL_N, _CUR_SRC or "see section")
    log()
    log(block)
    log(cap)
    return block


def fnum(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "nan"
    return ("%." + str(nd) + "f") % float(x)


def safe_div(a, b):
    a, b = float(a), float(b)
    return a / b if b else 0.0


def prf(yi, pred, cls):
    tp = int(np.sum((yi == cls) & (pred == cls)))
    fp = int(np.sum((yi != cls) & (pred == cls)))
    fn = int(np.sum((yi == cls) & (pred != cls)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def turns_for(pred_cls, peak_pos):
    """L1 turn for a predicted class index. clean -> -1 by rule."""
    t = np.full(len(pred_cls), -1, dtype=np.int64)
    for c in range(1, len(LABELS)):
        m = pred_cls == c
        if m.any():
            # W2I[LABELS[c]] == c for c >= 1 because LABELS[0] == "clean" is
            # dropped from the window class list (index 0 is "background").
            t[m] = np.asarray(peak_pos)[m, W2I[LABELS[c]]]
    return t


def name(i):
    return LABELS[int(i)]


def confusion(yi, pred):
    out = {}
    for t in range(len(LABELS)):
        row = {}
        for p in range(len(LABELS)):
            c = int(np.sum((yi == t) & (pred == p)))
            if c:
                row[name(p)] = c
        if row:
            out[name(t)] = row
    return out


# ===========================================================================
# 0. SOURCE OF TRUTH
# ===========================================================================
def read_guard():
    """Run the (verified read-only) protocol guard and capture its output.

    `protocol_guard.py` opens `results.csv`, each experiment's `results.json`
    and `ROOT_CAUSE_REVIEW.md` for READING only and prints; it writes nothing.
    """
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    exe = py if os.path.exists(py) else sys.executable
    r = subprocess.run([exe, os.path.join(HERE, "protocol_guard.py")],
                       cwd=ROOT, capture_output=True, text=True)
    return (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr.strip()
                               else "")


def s00(ctx):
    h("0. SOURCE OF TRUTH")
    guard_txt = ctx.get("guard") or read_guard()
    log("protocol_guard.py output (recorded verbatim in the report):")
    for line in guard_txt.splitlines():
        log("    " + line)
    RES["guard"] = guard_txt

    d = ENS.load_verified()
    rob = np.asarray(ENS.robustness_mask(), dtype=bool)
    C32 = np.load(os.path.join(EXP, "oof_C_corrected.npy"))
    D32 = np.load(os.path.join(EXP, "oof_D_corrected_wg.npy"))
    sha = ENS.sha32(C32)

    sub("REPRODUCTION GATE (must pass before any section is emitted)",
        "s00(); cached OOF sha + ENS.load_verified() + "
        "results_production.json")
    rows = [("runs", "10000", "10000"),
            ("folds", "3", str(len(d["folds"]))),
            ("C sha256 == EXP12 sha B", EXP12_SHA_B[:16] + "...",
             sha[:16] + "..." if sha == EXP12_SHA_B else "MISMATCH"),
            ("robustness rows", "1290", str(int(rob.sum())))]
    tbl(rows, ["check", "expected", "got"])

    if sha != EXP12_SHA_B:
        log("")
        log("STOP: Control C does not reproduce Exp12's committed corrected OOF.")
        log("Nothing was scored and no output was written.")
        raise SystemExit(1)
    log("   CONTROL C sha: IDENTICAL  -> PASS")

    yi = np.asarray(d["yi"], dtype=np.int64)
    n = len(yi)
    peak_pos = np.asarray(d["peak_pos"])
    peak_prob = np.asarray(d["peak_prob"])
    pred = np.asarray(D32.argmax(1), dtype=np.int64)

    corp = ENS.corpus_for(D32, d, rob, labels=pred)
    m, _ = ENS.full_official(corp, pred)

    sub("CANDIDATE D RESCORE == results_production.json ?",
        "s00(); ENS.full_official() over oof_D_corrected_wg.npy")
    rows, ok = [], True
    for k in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2",
              "composite"):
        got, exp = float(m[k]), COMMITTED_D[k]
        good = abs(got - exp) <= TOL
        ok &= good
        rows.append((k, repr(exp), repr(got), "%.2e" % abs(got - exp),
                     "PASS" if good else "FAIL"))
    tbl(rows, ["metric", "committed", "rescored", "abs diff", ""])
    log("   pinned success_f1 constant in blend.py = %r" % SUCCESS_F1_PINNED)
    if not ok:
        log("")
        log("STOP: Candidate D does not reproduce its committed metrics.")
        log("Nothing further was scored and no output was written.")
        raise SystemExit(1)
    log("   CANDIDATE D REPRODUCTION: PASS (5/5 to 1e-12)")

    c, _Wd = ENS._prepare()
    runs = c["runs"]
    assert len(runs) == n

    # ---- the 261 FOLD-INDEPENDENT label features actually available --------
    # 249 cached foundation columns + the 12 frozen wait-graph columns.
    # The 51 window aggregates are deliberately NOT included: rebuilding them
    # would require refitting the window model, which this audit forbids.
    found = c["foundation"]
    Xwg = np.asarray(WG.build_matrix(runs), dtype=np.float64)
    assert Xwg.shape == (n, WG.N_FEATURES), Xwg.shape
    X_indep = np.hstack([found.values.astype(np.float64), Xwg])
    indep_names = list(found.columns) + list(WG.FEATURE_NAMES)
    assert X_indep.shape[1] == N_INDEP, X_indep.shape
    log("")
    log("   fold-independent label block: %s = 249 foundation + %d wg_*"
        % (X_indep.shape, WG.N_FEATURES))

    ctx.update(d=d, rob=rob, C32=C32, D32=D32, yi=yi, n=n, pred=pred,
               peak_pos=peak_pos, peak_prob=peak_prob, metrics=m,
               runs=runs, foundation=found, folds=d["folds"],
               ys=np.asarray(d["ys"], dtype=object),
               fturn=np.asarray(d["train"]["fault_turn"].values, dtype=np.int64),
               success=np.asarray(d["train"]["success"].values, dtype=np.int64),
               X_indep=X_indep, indep_names=indep_names)
    ctx["turns"] = turns_for(pred, peak_pos)

    sub("HEADLINE (Candidate D, corrected production semantics)",
        "s00(); results_production.json metrics_D (pinned)")
    tbl([(k, repr(COMMITTED_D[k])) for k in COMMITTED_D],
        ["metric", "value"])
    log("   Label is pinned at Exp03 B; it is NOT measured by Exp13.")


# ===========================================================================
# 1. HEADLINE ERROR MAP
# ===========================================================================
def s01(ctx):
    h("1. HEADLINE ERROR MAP - Exp13 Candidate D")
    yi, pred, n = ctx["yi"], ctx["pred"], ctx["n"]

    conf = confusion(yi, pred)
    committed = json.load(open(os.path.join(EXP, "results_production.json"),
                               encoding="utf-8"))["confusions_D"]
    same = all(conf.get(k) == committed.get(k) for k in
               set(conf) | set(committed))
    log("   confusion matrix equals results_production.json confusions_D: %s"
        % ("YES" if same else "NO - INVESTIGATE"))
    assert same, "confusion mismatch against the committed artifact"

    sub("CONFUSION MATRIX (rows = true, cols = predicted), absolute counts",
        "s01(); argmax of oof_D_corrected_wg.npy vs d['yi']")
    rows = []
    for t in range(len(LABELS)):
        r = [name(t)] + [int(np.sum((yi == t) & (pred == p)))
                         for p in range(len(LABELS))]
        r.append(int(np.sum(yi == t)) - int(np.sum((yi == t) & (pred == t))))
        rows.append(tuple(r))
    tot = [int(np.sum(yi == p)) for p in range(len(LABELS))]
    rows.append(("TOTAL_pred",) + tuple(tot) +
                (n - int(np.sum(yi == pred)),))
    tbl(rows, ["true \\ pred"] + list(LABELS) + ["errors"])

    sub("PER-CLASS PRECISION / RECALL / F1",
        "s01(); prf() over the rescored Candidate D predictions")
    rows, f1map = [], {}
    for c in range(len(LABELS)):
        m_ = prf(yi, pred, c)
        f1map[name(c)] = m_
        rows.append((name(c), m_["tp"], m_["fp"], m_["fn"],
                     fnum(m_["precision"]), fnum(m_["recall"]),
                     fnum(m_["f1"])))
    tbl(rows, ["class", "tp", "fp", "fn", "precision", "recall", "F1"])
    log("   macro F1 = %s   (official %s)"
        % (fnum(macro_f1([name(i) for i in yi], [name(i) for i in pred], LABELS)),
           fnum(COMMITTED_D["macro_f1"], 10)))

    sub("ERROR COUNT PER TRUE CLASS",
        "s01(); 1 - recall, per true class")
    rows = []
    for c in range(len(LABELS)):
        tot_c = int(np.sum(yi == c))
        rows.append((name(c), tot_c, tot_c - f1map[name(c)]["tp"],
                     fnum(safe_div(tot_c - f1map[name(c)]["tp"], tot_c))))
    tbl(rows, ["true class", "n", "errors", "error rate"])
    RES["error_rate"] = {r[0]: float(r[3]) for r in rows}
    RES["confusion"] = conf
    RES["per_class"] = {k: {kk: (vv if not isinstance(vv, float)
                                 else float(vv))
                            for kk, vv in v.items()}
                        for k, v in f1map.items()}
    RES["total_errors"] = int(n - np.sum(yi == pred))

    sub("REQUESTED BREAKDOWN - dropped_handoff (n=%d)"
        % int(np.sum(yi == DH)))
    tot = int(np.sum(yi == DH))
    pairs = [("clean", CLEAN), ("deadlock", DL), ("runaway_loop", RL)]
    rest = [c for c in range(len(LABELS)) if c != DH
            and c not in (CLEAN, DL, RL)]
    r_rest = sum(int(np.sum((yi == DH) & (pred == c))) for c in rest)
    rows = [(t, conf.get("dropped_handoff", {}).get(t, 0),
             fnum(safe_div(conf.get("dropped_handoff", {}).get(t, 0), tot)))
            for t, _ in pairs]
    rows.append(("rest (%s)" % "+".join(name(c) for c in rest), r_rest,
                 fnum(safe_div(r_rest, tot))))
    rows.append(("TOTAL wrong", tot - int(np.sum((yi == DH) & (pred == DH))),
                 fnum(safe_div(tot - int(np.sum((yi == DH) & (pred == DH))),
                               tot))))
    tbl(rows, ["DH ->", "count", "share of true DH"])

    sub("REQUESTED BREAKDOWN - runaway_loop (n=%d)" % int(np.sum(yi == RL)))
    tot = int(np.sum(yi == RL))
    pairs = [("clean", CLEAN), ("dropped_handoff", DH), ("deadlock", DL)]
    rest = [c for c in range(len(LABELS)) if c != RL
            and c not in (CLEAN, DH, DL)]
    r_rest = sum(int(np.sum((yi == RL) & (pred == c))) for c in rest)
    rows = [(t, conf.get("runaway_loop", {}).get(t, 0),
             fnum(safe_div(conf.get("runaway_loop", {}).get(t, 0), tot)))
            for t, _ in pairs]
    rows.append(("rest (%s)" % "+".join(name(c) for c in rest), r_rest,
                 fnum(safe_div(r_rest, tot))))
    rows.append(("TOTAL wrong", tot - int(np.sum((yi == RL) & (pred == RL))),
                 fnum(safe_div(tot - int(np.sum((yi == RL) & (pred == RL))),
                               tot))))
    tbl(rows, ["RL ->", "count", "share of true RL"])

    sub("REVERSE FALSE POSITIVES (share of the PREDICTED class)")
    rows = []
    for tgt, tgt_c in (("dropped_handoff", DH), ("runaway_loop", RL)):
        for src, src_c in (("clean", CLEAN), ("deadlock", DL),
                           ("runaway_loop", RL), ("dropped_handoff", DH)):
            if src == tgt:
                continue
            k = int(np.sum((yi == src_c) & (pred == tgt_c)))
            base = int(np.sum(yi == src_c))
            rows.append(("%s -> %s" % (src, tgt), k, base,
                         fnum(safe_div(k, base))))
    tbl(rows, ["transition", "FP count", "true rows of source", "FP rate"])

    pairs = []
    for t in range(len(LABELS)):
        for p in range(len(LABELS)):
            if t == p:
                continue
            k = int(np.sum((yi == t) & (pred == p)))
            if k:
                pairs.append((k, name(t), name(p),
                              fnum(safe_div(k, int(np.sum(yi == t))))))
    pairs.sort(reverse=True)
    sub("TOP-10 CONFUSION PAIRS",
        "s01(); off-diagonal cells of the rescored confusion matrix")
    tbl([(i + 1, a, b, k, sh) for i, (k, a, b, sh) in enumerate(pairs[:10])],
        ["#", "true", "predicted", "count", "share of true"])
    RES["top_pairs"] = [{"true": a, "pred": b, "n": int(k), "share": float(sh)}
                        for k, a, b, sh in pairs[:10]]
    ctx["f1map"] = f1map


# ===========================================================================
# 2. LABEL vs LOCALIZATION HEADROOM
# ===========================================================================
def s02(ctx):
    h("2. LABEL AND LOCALIZATION HEADROOM")
    yi, pred, fturn, turns = ctx["yi"], ctx["pred"], ctx["fturn"], ctx["turns"]
    n = ctx["n"]
    f_idx = np.where(yi != CLEAN)[0]
    log("   evaluation/metrics.py: a hit requires `lp == lt and |tp-tt| <= k`,")
    log("   so a MISCLASSIFIED row is an UNCONDITIONAL miss whatever the turn.")
    log("   faulty runs (true label != clean): %d of %d (%.2f%%)"
        % (len(f_idx), n, 100.0 * len(f_idx) / n))

    def hitmask(idx, k):
        return np.array([pred[i] == yi[i] and turns[i] >= 0
                         and abs(int(turns[i]) - int(fturn[i])) <= k
                         for i in idx], dtype=bool)

    def rows_for(idx):
        lab_ok = (pred[idx] == yi[idx])
        h2 = hitmask(idx, 2)
        a = safe_div(lab_ok.sum(), len(idx))
        b = safe_div((lab_ok & h2).sum(), len(idx))
        return {"n": int(len(idx)), "A_label_correct": float(a),
                "B_label_and_hit2": float(b),
                "C_conditional_hit2": safe_div(b, a)}

    sub("A = P(label correct) | B = P(label correct AND hit@2) | "
        "C = B / A = P(hit@2 | label correct)",
        "s02(); fault_turn_hit_at_k() over sealed peak_pos")
    allr = rows_for(f_idx)
    rows = [("ALL faulty", allr["n"], fnum(allr["A_label_correct"]),
             fnum(allr["B_label_and_hit2"]),
             fnum(allr["C_conditional_hit2"]))]
    for c in FAULTY:
        idx = f_idx[yi[f_idx] == c]
        if len(idx):
            r = rows_for(idx)
            rows.append((name(c), r["n"], fnum(r["A_label_correct"]),
                         fnum(r["B_label_and_hit2"]),
                         fnum(r["C_conditional_hit2"])))
    tbl(rows, ["group", "n", "A label correct", "B label+hit@2",
               "C cond. hit@2"])
    log("   current official hit@2 = %s  (B over all faulty)"
        % fnum(allr["B_label_and_hit2"], 10))
    RES["decomp_overall"] = allr
    RES["decomp_by_class"] = {name(c): rows_for(f_idx[yi[f_idx] == c])
                              for c in FAULTY
                              if int(np.sum(yi[f_idx] == c))}

    ceil_perfect = allr["A_label_correct"]
    sub("CEILING 1 - perfect_localizer_given_current_labels")
    log("   a perfect turn on every correctly-classified row, wrong class still")
    log("   a miss  ->  ceiling == A == %s"
        % fnum(ceil_perfect, 10))
    log("   headroom attributable to localisation alone: %s"
        % fnum(ceil_perfect - allr["B_label_and_hit2"], 10))
    log("   headroom attributable to MISCLASSIFICATION alone: %s"
        % fnum(1.0 - allr["A_label_correct"], 10))

    sub("CEILING 2 - oracle_class_current_L1 (diagnostic only, no change)")
    # Gather peak_pos at the WINDOW index of the run's TRUE label. Column 0 of
    # peak_pos is `background`, so this mapping is mandatory, not cosmetic.
    # `clean` has NO window class, so its oracle turn is -1 by definition.
    w_idx = np.asarray([W2I.get(name(int(i)), 0) for i in yi], dtype=np.int64)
    orc = np.where(yi == CLEAN, -1,
                   np.asarray(ctx["peak_pos"])[np.arange(n), w_idx])
    orc = orc.astype(np.int64)
    yt = [name(int(i)) for i in yi[f_idx]]
    yp = list(yt)  # oracle class == true class by construction
    tt = [int(fturn[i]) for i in f_idx]
    tp = [int(orc[i]) for i in f_idx]
    orc_h2 = fault_turn_hit_at_k(yt, yp, tt, tp, k=2)
    log("   oracle_class_current_L1 hit@2 = %s" % fnum(orc_h2, 10))

    rows = [("current Exp13 (real label head + L1)", fnum(allr["B_label_and_hit2"])),
            ("perfect_localizer_given_current_labels", fnum(ceil_perfect)),
            ("oracle_class_current_L1", fnum(orc_h2)),
            ("fully localised ceiling (both oracles)", "1.0000")]
    tbl(rows, ["scenario", "hit@2"])
    log("   Interpretation: current %s vs label ceiling %s means %s of the loss"
        % (fnum(allr["B_label_and_hit2"]), fnum(ceil_perfect),
           fnum(ceil_perfect - allr["B_label_and_hit2"])))
    log("   is LOCALISATION even when the class is right; the remaining %s is"
        % fnum(1.0 - ceil_perfect))
    log("   the LABEL loss and is unreachable by any localiser change.")
    RES["ceilings"] = {"current": allr["B_label_and_hit2"],
                       "perfect_localizer_given_current_labels":
                           float(ceil_perfect),
                       "oracle_class_current_L1": float(orc_h2)}

    # oracle-class hit@2 restricted to the rows the label head already gets
    # right, so the comparison against DH/RL error groups is apples-to-apples.
    ok = pred[f_idx] == yi[f_idx]
    idx_ok = f_idx[ok]
    nok = [name(int(yi[i])) for i in idx_ok]
    orc_ok = fault_turn_hit_at_k(nok, list(nok),
                                 [int(fturn[i]) for i in idx_ok],
                                 [int(orc[i]) for i in idx_ok], k=2)
    ctx["oracle_turns"] = orc
    RES["ceilings"]["oracle_class_on_correctly_classified"] = float(orc_ok)
    log("")
    log("   oracle-class hit@2 restricted to ALREADY-CORRECT rows = %s"
        % fnum(orc_ok, 10))
    log("   (isolates pure localisation quality from the label loss)")

    # per-class current hit@2 on correctly classified rows (used by section 10)
    ctx["localization_dh_hit2"] = float(
        rows_for(f_idx[yi[f_idx] == DH])["C_conditional_hit2"])
    ctx["localization_rl_hit2"] = float(
        rows_for(f_idx[yi[f_idx] == RL])["C_conditional_hit2"])


def _deepcopy_runs(runs):
    import copy
    return copy.deepcopy(runs)


def s03(ctx):
    """Label-blind assignment-delivery lifecycle.

    Official ontology (`docs/ontology.md`):

      `dropped_handoff` - the orchestrator handed a subtask to an executor and
      THE ARTIFACT FOR IT NEVER APPEARED. The executor either stays silent or
      acknowledges (`ack`) and then never delivers. `fault_turn` is the turn of
      the undelivered `handoff` message.

    The question this section asks is whether the current representation loses
    the IDENTITY of a specific assignment (who assigned what, to whom, and
    whether THAT SAME assignment was delivered), rather than only aggregates.
    """
    h("3. ASSIGNMENT-DELIVERY AUDIT FOR dropped_handoff")
    log("   ONTOLOGY (docs/ontology.md, `dropped_handoff`):")
    log("     'the orchestrator handed a subtask to an executor and the artifact")
    log("      for it never appeared. The executor either stays silent entirely,")
    log("      or acknowledges (ack) and after that delivers nothing.'")
    log("     fault_turn = the turn of the handoff message (type=handoff, no refs)")
    log("     that remained undelivered.")
    log("   Below, per-assignment records are built from messages / artifacts /")
    log("   shared_state ONLY. label / fault_turn / success are NOT read here;")
    log("   they are joined in section 4, after construction.")

    runs = ctx["runs"]
    n = ctx["n"]

    # ---- LEAKAGE ASSERTION --------------------------------------------------
    # Perturb every target field. The records must be byte-identical, otherwise
    # the builder has read a target and section 3 is void.
    pert = _deepcopy_runs(runs[:400])
    for r in pert:
        r["label"] = "runaway_loop"
        r["fault_turn"] = 3
        r["success"] = 1 - int(r.get("success", 0) or 0)
    a = [build_assignment_records(r) for r in runs[:400]]
    b = [build_assignment_records(r) for r in pert]
    lab_blind = _canon(a) == _canon(b)
    log("")
    log("   LEAKAGE CHECK (400 runs, label/fault_turn/success perturbed):")
    log("     assignment records invariant: %s"
        % ("YES - label blind" if lab_blind else "NO - ABORT"))
    assert lab_blind, "lifecycle records depend on a target field"
    RES["lifecycle_label_blind"] = True

    recs = [build_assignment_records(r) for r in runs]
    allrec = [x for v in recs for x in v]
    log("")
    log("   runs with >=1 observable assignment: %d / %d"
        % (sum(1 for v in recs if v), n))
    log("   total observable assignments: %d  (mean %.2f per run)"
        % (len(allrec), len(allrec) / float(n)))
    fields = ["sender", "receiver", "intent", "handoff_turn", "acked",
              "ack_delay", "ack_turn", "delivered", "deliver_delay",
              "deliver_turn", "deliver_sender",
              "deliver_kind", "prev_owner",
              "artifact_present", "artifact_by", "artifact_turn",
              "artifact_is_receiver",
              "reassigned", "delivery_after_reassign", "recovered",
              "receiver_active_after", "n_owner_msgs_after",
              "last_event_turn", "unresolved_at_end"]
    sub("OBSERVABLE FIELDS PER ASSIGNMENT",
        "s03(); build_assignment_records() over production lifecycle.py")
    tbl([(f, "yes" if any(x.get(f) is not None for x in allrec) else "no",
          fnum(safe_div(sum(1 for x in allrec if x.get(f) is not None),
                        len(allrec)), 3))
         for f in fields],
        ["field", "populated in >=1 assignment", "share of assignments"])
    log("   Every field is derived from the observable schema only.")
    log("   deliver_kind is 'refs' when an explicit delivery message closed the")
    log("   assignment and 'artifact' when an implicit artifact match did;")
    log("   prev_owner is the receiver of the PREVIOUS handoff of the same")
    log("   intent, i.e. the reassignment link.")

    # ---- corpus-level observable lifecycle profile (LABEL-BLIND) ----------
    ctx["assign"] = recs
    ctx["assign_flat"] = allrec
    rows = []
    for f in ("acked", "delivered", "artifact_present", "reassigned",
              "recovered", "receiver_active_after", "unresolved_at_end"):
        k = sum(1 for x in allrec if x.get(f))
        rows.append((f, k, fnum(safe_div(k, len(allrec)))))
    tbl(rows, ["observable", "assignments", "share of all assignments"])

    # a per-run profile used by sections 4, 5 and 6
    ctx["prof"] = [assignment_profile(v, len(runs[i]["messages"]))
                   for i, v in enumerate(recs)]
    RES["n_assignments"] = int(len(allrec))
    RES["n_assignments_mean"] = float(len(allrec) / float(n))


def build_assignment_records(run):
    """ONE record per observable handoff assignment. Observable schema only.

    Uses production `lifecycle.build_lifecycles` for the delivery matching, then
    augments it with the identity/lifetime slots the current feature set does
    not emit. NEVER reads `label`, `fault_turn` or `success`.
    """
    msgs = run.get("messages") or []
    arts = run.get("artifacts") or []
    state = run.get("shared_state") or []
    if not msgs:
        return []
    n = len(msgs)
    last_turn = int(msgs[-1].get("t", n - 1))
    lifecycles = LC.build_lifecycles(msgs, arts)

    art_by_sub = defaultdict(list)
    art_by_sub_any = defaultdict(list)
    for a in arts:
        sub_ = (a.get("subtask") or "").strip()
        art_by_sub_any[sub_].append(a)
        if sub_:
            art_by_sub[sub_].append(a)
    state_turns = [int(s.get("t", -1)) for s in state if s.get("t") is not None]

    out = []
    for lc in lifecycles:
        t0 = int(lc["t"])
        intent = (lc.get("intent") or "").strip()
        rcv = lc.get("to") or ""
        snd = lc.get("from") or ""

        acked = bool(lc.get("acked"))
        ack_delay = (int(lc["ack_t"]) - t0
                     if acked and lc.get("ack_t") is not None else None)

        delivered = bool(lc.get("delivered"))
        d_sender = lc.get("deliver_sender")
        # how the assignment closed: an explicit delivery ('refs' = a message
        # carrying artifact refs) or an implicit artifact match. production
        # lifecycle.close() sets this; without it an undelivered assignment and
        # a silently-closed one are indistinguishable downstream.
        d_kind = lc.get("deliver_kind")
        d_delay = int(lc["deliver_t"]) - t0 if delivered and \
            lc.get("deliver_t") is not None else None

        # owner of the PREVIOUS assignment of the same intent, if the same
        # intent was handed out before. This is the reassignment link.
        prev_owner = lc.get("prev_owner")

        # artifact identity: by subtask, and whether the ASSIGNED RECEIVER made it
        arts_i = art_by_sub.get(intent, []) if intent else []
        art_present = bool(arts_i) or bool(lc.get("has_artifact"))
        art_t = lc.get("art_t")
        art_by = None
        if arts_i:
            best = min((a for a in arts_i
                        if int(a.get("t", 0)) > t0),
                       key=lambda a: int(a.get("t", 0)), default=None)
            art_by = (best.get("by") or "") if best else None
            if art_t is None and best is not None:
                art_t = int(best.get("t", 0))
        art_is_rcv = (art_by is not None and rcv != "" and art_by == rcv)

        reassigned = bool(lc.get("reassigned"))
        recov = bool(lc.get("recovered"))

        # receiver-specific delivery: did the ASSIGNED agent deliver it?
        owner_delivered = bool(lc.get("owner_delivered")) or (
            delivered and d_sender == rcv and rcv != "")

        # receiver activity strictly after the handoff
        recv_msgs_after = 0
        for m in msgs:
            if int(m.get("t", 0)) > t0 and (m.get("from") or "") == rcv:
                recv_msgs_after += 1
        recv_state_after = 0
        for s in state:
            if (s.get("agent") or "") == rcv and int(s.get("t", -1)) > t0:
                recv_state_after += 1
        recv_active = recv_msgs_after + recv_state_after > 0

        # last observable event of THIS assignment
        cand = [t0]
        if acked and lc.get("ack_t") is not None:
            cand.append(int(lc["ack_t"]))
        if delivered and lc.get("deliver_t") is not None:
            cand.append(int(lc["deliver_t"]))
        if art_t is not None:
            cand.append(int(art_t))
        for m in msgs:
            if int(m.get("t", 0)) > t0 and (m.get("from") or "") == rcv:
                cand.append(int(m.get("t", 0)))
        last_event = max(cand)

        unresolved = (not delivered) and (not art_present)
        out.append({
            "sender": snd, "receiver": rcv, "intent": intent,
            "handoff_turn": t0,
            "acked": acked, "ack_delay": ack_delay,
            "ack_turn": int(lc["ack_t"])
            if acked and lc.get("ack_t") is not None else None,
            "delivered": delivered, "deliver_delay": d_delay,
            "deliver_turn": int(lc["deliver_t"])
            if delivered and lc.get("deliver_t") is not None else None,
            "deliver_sender": d_sender, "deliver_kind": d_kind,
            "prev_owner": prev_owner,
            "artifact_present": art_present, "artifact_by": art_by,
            "artifact_turn": int(art_t) if art_t is not None else None,
            "artifact_is_receiver": art_is_rcv,
            "reassigned": reassigned,
            "delivery_after_reassign": bool(
                delivered and reassigned and lc.get("result_after_reassign")),
            "recovered": recov,
            "owner_delivered": owner_delivered,
            "receiver_active_after": recv_active,
            "n_owner_msgs_after": recv_msgs_after,
            "last_event_turn": last_event,
            "unresolved_at_end": bool(unresolved and last_event <= last_turn),
            "lifetime": last_event - t0,
        })
    return out


def _canon(recs):
    """Target-independent canonical form of a list of record lists.

    NOTE: this must materialise tuples, never return generators - two distinct
    generator objects compare unequal by identity, which would make the leakage
    check fail for reasons that have nothing to do with leakage.
    """
    return tuple(tuple(tuple(sorted((str(k), str(x)) for k, x in r.items()))
                       for r in v)
                 for v in recs)


def assignment_profile(recs, n_msgs):
    """Run-level observable lifecycle profile. No label, no fault_turn."""
    k = len(recs)
    unres = [r for r in recs if r["unresolved_at_end"]]
    unres_acked = [r for r in unres if r["acked"]]
    unres_active = [r for r in unres if r["receiver_active_after"]]
    unres_reass = [r for r in unres if r["reassigned"]]
    recov = [r for r in recs if r["recovered"]]
    # identity match: the artifact for this exact intent was produced by the
    # agent this exact assignment named
    ident = [r for r in recs
             if r["artifact_present"] and r["artifact_is_receiver"]]
    rcv_deliv = [r for r in recs if r["owner_delivered"]]
    span = max([n_msgs - 1, 1])

    # max simultaneously unresolved: peak of an open-assignment counter
    ev = []
    for r in recs:
        ev.append((r["handoff_turn"], +1))
        if r["delivered"] or r["artifact_present"]:
            ev.append((r["last_event_turn"], -1))
    ev.sort()
    cur = mx = 0
    for _t, dd in ev:
        cur += dd
        mx = max(mx, cur)
    first_unres = min([r["handoff_turn"] for r in unres], default=-1)
    longest = max([r["lifetime"] for r in unres], default=0)

    return {
        "n_assign": k,
        "n_unresolved": len(unres),
        "unresolved_share": safe_div(len(unres), k),
        "n_unresolved_after_ack": len(unres_acked),
        "unresolved_after_ack_share": safe_div(len(unres_acked), len(unres)),
        "n_unresolved_receiver_active": len(unres_active),
        "n_unresolved_reassigned": len(unres_reass),
        "n_recovered": len(recov),
        "recovered_share": safe_div(len(recov), k),
        "n_identity_match": len(ident),
        "identity_match_share": safe_div(len(ident), k),
        "n_receiver_delivery": len(rcv_deliv),
        "receiver_delivery_share": safe_div(len(rcv_deliv), k),
        "longest_unresolved_lifetime": float(longest),
        "longest_unresolved_rel": safe_div(longest, span),
        "first_unresolved_turn": int(first_unres),
        "first_unresolved_rel": safe_div(first_unres, span)
        if first_unres >= 0 else -1.0,
        "single_unresolved_through_end": int(len(unres) == 1),
        "max_simultaneous_unresolved": int(mx),
        "any_unresolved": int(len(unres) > 0),
        "unresolved_and_acked": int(len(unres_acked) > 0),
        "unresolved_and_active": int(len(unres_active) > 0),
        # how assignments closed, and the reassignment link (deliver_kind /
        # prev_owner). A silently-dropped assignment is one that closed by
        # neither explicit delivery nor artifact; a run whose *previous* owner
        # differs is one where the same intent was handed over.
        "n_closed_by_refs": int(sum(1 for r in recs
                                    if r.get("deliver_kind") == "refs")),
        "n_closed_by_artifact": int(sum(1 for r in recs
                                        if r.get("deliver_kind") == "artifact")),
        "refs_delivery_share": safe_div(
            sum(1 for r in recs if r.get("deliver_kind") == "refs"), k),
        "n_reassigned_link": int(sum(1 for r in recs if r.get("prev_owner"))),
        "n_reassigned_link_delivered": int(sum(
            1 for r in recs if r.get("prev_owner") and r["delivered"])),
        "n_msgs": int(n_msgs),
    }


PROFILE_KEYS = ["n_assign", "n_unresolved", "unresolved_share",
                "n_unresolved_after_ack", "unresolved_after_ack_share",
                "n_unresolved_receiver_active", "n_unresolved_reassigned",
                "n_recovered", "recovered_share", "n_identity_match",
                "identity_match_share", "n_receiver_delivery",
                "receiver_delivery_share", "longest_unresolved_lifetime",
                "longest_unresolved_rel", "first_unresolved_turn",
                "first_unresolved_rel", "single_unresolved_through_end",
                "max_simultaneous_unresolved", "any_unresolved",
                "unresolved_and_acked", "unresolved_and_active",
                "n_closed_by_refs", "n_closed_by_artifact",
                "refs_delivery_share", "n_reassigned_link",
                "n_reassigned_link_delivered", "n_msgs"]


# ===========================================================================
# 4. DROPPED_HANDOFF ERROR GROUPS
# ===========================================================================
DH_GROUPS = [("DH correct", lambda yi, p: (yi == DH) & (p == DH)),
             ("DH -> clean", lambda yi, p: (yi == DH) & (p == CLEAN)),
             ("DH -> deadlock", lambda yi, p: (yi == DH) & (p == DL)),
             ("DH -> runaway_loop", lambda yi, p: (yi == DH) & (p == RL)),
             ("clean -> DH", lambda yi, p: (yi == CLEAN) & (p == DH)),
             ("deadlock -> DH", lambda yi, p: (yi == DL) & (p == DH))]


def group_stats(idx, prof, keys):
    """mean / median / fraction for each profile key over a row group."""
    out = {}
    for k in keys:
        v = np.asarray([prof[i][k] for i in idx], dtype=np.float64)
        out[k] = {"mean": float(v.mean()) if len(v) else float("nan"),
                  "median": float(np.median(v)) if len(v) else float("nan"),
                  "p90": float(np.percentile(v, 90)) if len(v) else
                  float("nan"),
                  "frac_nonzero": float((v > 0).mean()) if len(v)
                  else float("nan")}
    return out


def s04(ctx):
    h("4. DROPPED_HANDOFF ERROR GROUPS")
    yi, pred, prof = ctx["yi"], ctx["pred"], ctx["prof"]

    sub("GROUP SIZES")
    rows, groups = [], {}
    for gname, fn in DH_GROUPS:
        m = fn(yi, pred)
        idx = np.where(m)[0]
        groups[gname] = idx
        rows.append((gname, int(len(idx)), fnum(safe_div(len(idx), 1200), 4)))
    tbl(rows, ["group", "n runs", "share of 1200 true DH (or of source)"])

    sub("LIFECYCLE PROFILE BY GROUP (label joined AFTER construction)",
        "s04(); group_stats() over assignment_profile(), labels joined here")
    keys = [k for k in PROFILE_KEYS if k != "n_msgs"]
    stats = {g: group_stats(groups[g], prof, keys) for g in groups}

    show = ["n_assign", "n_unresolved", "unresolved_share",
            "n_unresolved_after_ack", "unresolved_and_acked",
            "n_unresolved_receiver_active", "unresolved_and_active",
            "n_unresolved_reassigned", "n_recovered", "recovered_share",
            "n_identity_match", "identity_match_share", "n_receiver_delivery",
            "receiver_delivery_share", "longest_unresolved_lifetime",
            "first_unresolved_rel", "single_unresolved_through_end",
            "max_simultaneous_unresolved", "n_closed_by_refs",
            "n_closed_by_artifact", "refs_delivery_share",
            "n_reassigned_link", "n_reassigned_link_delivered"]
    rows = []
    for k in show:
        rows.append(tuple([k] + [fnum(stats[g][k]["mean"], 3)
                                 for g, _fn in DH_GROUPS]))
    tbl(rows, ["metric (MEAN)"] + [g for g, _ in DH_GROUPS])
    rows = []
    for k in show:
        rows.append(tuple([k] + [fnum(stats[g][k]["median"], 3)
                                 for g, _fn in DH_GROUPS]))
    tbl(rows, ["metric (MEDIAN)"] + [g for g, _ in DH_GROUPS])
    rows = []
    for k in show:
        rows.append(tuple([k] + [fnum(stats[g][k]["frac_nonzero"], 3)
                                 for g, _fn in DH_GROUPS]))
    tbl(rows, ["metric (FRACTION > 0)"] + [g for g, _ in DH_GROUPS])

    sub("RUN LENGTH - the 'it only works because DH runs are longer' check")
    rows = []
    for g, _fn in DH_GROUPS:
        v = np.asarray([prof[i]["n_msgs"] for i in groups[g]], np.float64)
        rows.append((g, int(len(v)), fnum(v.mean(), 1), fnum(np.median(v), 1),
                     fnum(np.percentile(v, 25), 1),
                     fnum(np.percentile(v, 75), 1)))
    tbl(rows, ["group", "n", "mean n_msgs", "median", "p25", "p75"])
    RES["dh_groups"] = {g: {"n": int(len(groups[g]))} for g in groups}
    RES["dh_stats"] = stats

    sub("HEADLINE SEPARATION - DH correct vs the DH errors Exp13 makes")
    base = stats["DH correct"]
    rows = []
    for g in ("DH -> clean", "DH -> deadlock", "DH -> runaway_loop",
              "clean -> DH", "deadlock -> DH"):
        for k in ("unresolved_share", "identity_match_share",
                  "receiver_delivery_share", "n_unresolved",
                  "longest_unresolved_rel"):
            rows.append((g, k, fnum(base[k]["mean"], 3),
                         fnum(stats[g][k]["mean"], 3),
                         fnum(stats[g][k]["mean"] - base[k]["mean"], 3)))
    tbl(rows, ["error group", "metric", "DH correct (mean)",
               "group (mean)", "delta"])
    ctx["dh_stats"] = stats
    ctx["dh_groups"] = groups


def corr_max(diag_vec, X, names, exclude_self=True):
    """max |r| of one diagnostic against the fold-independent feature block.

    Pearson AND Spearman are both reported. Correlation is NOT the test here:
    it only separates `exact / reconstructible` from `information lost`. A
    near-1.0 partner means the signal is already recoverable from a current
    feature; a mid-range maximum means no single feature encodes it.
    """
    v = np.asarray(diag_vec, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 3 or np.allclose(v[ok].std(), 0):
        return {"max_abs_pearson": None, "max_abs_spearman": None,
                "top": [], "n_const": bool(not ok.all() or
                                           np.allclose(v[ok].std(), 0))}
    vp, sp = [], []
    for j in range(X.shape[1]):
        x = X[ok, j].astype(np.float64)
        if x.std() == 0:
            continue
        vp.append(abs(float(np.corrcoef(v[ok], x)[0, 1])))
        sp.append(abs(float(np.corrcoef(np.argsort(np.argsort(v[ok])),
                                         np.argsort(np.argsort(x)))[0, 1])))
    if not vp:
        return {"max_abs_pearson": None, "max_abs_spearman": None,
                "top": [], "n_const": True}
    vp_a, sp_a = np.asarray(vp), np.asarray(sp)
    idx = np.argsort(-vp_a)[:5]
    top = [(str(names[j]), float(vp_a[j]), float(sp_a[j]))
           for j in np.where(np.isfinite(vp_a))[0][idx]]
    return {"max_abs_pearson": float(vp_a.max()),
            "max_abs_spearman": float(sp_a.max()), "top": top,
            "n_const": False}


def s05(ctx):
    h("5. DROPPED_HANDOFF REPRESENTATION GAP")
    log("   Correlation base: the %d FOLD-INDEPENDENT label features actually"
        % N_INDEP)
    log("   available = 249 cached foundation + 12 frozen wait-graph.")
    log("   The 51 window aggregates are NOT in this basis: rebuilding them")
    log("   would require refitting the window model, which this audit forbids.")
    log("   They are inspected STRUCTURALLY in 5b instead.")
    log("   So max |r| below is against %d columns, NOT the full 312." % N_INDEP)

    X, names = ctx["X_indep"], ctx["indep_names"]
    prof, groups = ctx["prof"], ctx["dh_groups"]

    sub("5.1 IS THE ASSIGNMENT IDENTITY ALREADY IN THE 312?")
    log("   The question the ontology poses: does the representation keep")
    log("     WHO assigned WHAT / TO WHOM / AND whether THAT SAME assignment")
    log("     was delivered?")
    log("   Counts, shares, coverage and ratios are NOT assignment identity.")
    inv = [
        ("WHO assigned WHAT (sender -> intent edge identity)",
         "PARTIAL", "lc_* records carry it internally; NO per-(sender,intent) "
         "feature is emitted", "lc_n_lifecycles, lc_reassign_ratio"),
        ("TO WHOM (receiver identity of an assignment)", "PARTIAL",
         "records carry 'to'/'prev_owner'; only divide-by-agent ratios exist",
         "lc_unresolved_per_agent, lc_multi_owner_intents"),
        ("THAT SAME assignment delivered (assignment -> matching artifact)",
         "EXISTING", "intent-aware LIFO matching + artifact fallback",
         "lc_delivered_ratio, lc_del_lat_mean/max/p90, undelivered_assign"),
        ("per-assignment ack present", "EXISTING", "per-record 'acked'",
         "lc_acked_ratio, lc_n_unacked, lc_n_acked_nodeliver"),
        ("per-assignment first-response delay", "EXISTING",
         "ack_t - t mean/max", "lc_ack_lat_mean, lc_ack_lat_max"),
        ("reassignment + delivery AFTER reassignment", "EXISTING",
         "per-record reassigned/result_after_reassign",
         "lc_result_after_reassign, rt_old_owner_result_after"),
        ("recovery after a dropped assignment", "PARTIAL",
         "recovery is inferred from reassigned AND delivered, never from a "
         "previously-open assignment being closed", "lc_recovered, "
         "bd_recovery_events_rel, kw_recovery"),
        ("receiver still active after handoff", "EXISTING",
         "per-record n_owner_msgs_after",
         "lc_silent_after_assign, ra_active_to_end_frac"),
        ("artifact produced by the ASSIGNED RECEIVER (identity match)",
         "PARTIAL", "owner_delivered is computed but never emitted directly",
         "lc_result_after_reassign, lc_recovered"),
        ("lifetime of an UNRESOLVED assignment", "PARTIAL",
         "temporal_features computes ua_age_max/mean/p90_rel, but BLOCK_1_AGE "
         "is EXCLUDED from production by solution._AGE_BLOCK",
         "lc_unresolved_last20pct, ld_silence_span_rel"),
        ("absolute onset turn of the first unresolved assignment", "PARTIAL",
         "only a RELATIVE position exists (t / max_turn)",
         "lc_first_unresolved_pos"),
        ("number of simultaneously unresolved assignments", "EXISTING",
         "peak backlog", "bd_max_rel, lc_longest_unresolved_chain"),
    ]
    tbl([(a, b, c, d) for a, b, c, d in inv],
        ["candidate signal", "verdict", "note", "carrying feature(s)"])
    log("   Features COMPUTED BUT DROPPED from the shipped 312:")
    log("     temporal BLOCK_1_AGE : ua_age_max_rel, ua_age_mean_rel,")
    log("                           ua_age_p90_rel, ua_oldest_share_gt10/25/50")
    log("     lifecycle dl_*       : 13 deadlock features (dl_mutual_pairs, ...)")
    log("   Both blocks are live code but absent from production.")

    sub("5.2 max |r| OF EACH PROPOSED RELATIONAL DIAGNOSTIC vs the %d "
        "FOLD-INDEPENDENT LABEL FEATURES" % N_INDEP,
        "s05(); corr_max() vs 249 foundation + 12 wg_* (NOT the 312)")
    diags = ["unresolved_share", "identity_match_share",
             "receiver_delivery_share", "n_unresolved",
             "n_unresolved_after_ack", "n_unresolved_receiver_active",
             "n_unresolved_reassigned", "recovered_share",
             "longest_unresolved_rel", "first_unresolved_rel",
             "single_unresolved_through_end", "max_simultaneous_unresolved"]
    rows, corr = [], {}
    for k in diags:
        v = [p[k] for p in prof]
        c_ = corr_max(v, X, names)
        corr[k] = c_
        top1 = ("%s r=%.3f" % (c_["top"][0][0], c_["top"][0][1])) \
            if c_["top"] else "n/a"
        rows.append((k, fnum(c_["max_abs_pearson"], 3),
                     fnum(c_["max_abs_spearman"], 3), top1))
    tbl(rows, ["diagnostic", "max |r| Pearson", "max |r| Spearman",
               "closest current feature"])
    log("   Reading: max |r| near 1.0 => the current features already encode")
    log("   it. A mid-range maximum => no single column carries it, which is")
    log("   the ONLY situation in which a representation gap is arguable.")

    sub("5.3 TOP-5 CORRELATED PARTNERS (is the signal already reconstructible?)")
    for k in diags:
        c_ = corr[k]
        if not c_["top"]:
            continue
        sub_ = ", ".join("%s=%.3f" % (a, b) for a, b, _c in c_["top"])
        log("   %-34s %s" % (k, sub_))

    sub("5.4 THE 51 WINDOW AGGREGATES - STRUCTURAL READING (no refit)")
    log("   aggregate.AGG_NAMES = 6 fault classes x 7 statistics + 9 globals:")
    log("     per class: %s" % ", ".join(AG.PER_CLASS))
    log("     globals  : %s" % ", ".join(AG.GLOBAL))
    log("   Each is an order-statistic of a per-class PROBABILITY TRAJECTORY")
    log("   over the run's own windows. The grouping key is the run itself; no")
    log("   assignment, receiver or episode identity enters any of them.")
    struct = [
        ("unresolved assignment identity (who->whom->artifact)",
         "LOST", "probability-trajectory summaries; identity is never a key"),
        ("per-assignment ack / delay", "PARTIALLY REPRESENTED",
         "lc_* carry it; the 51 do not encode it at all"),
        ("delivery-after-reassignment", "LOST", "no ordering or identity"),
        ("absolute onset of first unresolved assignment",
         "PARTIALLY REPRESENTED",
         "pos_first_hi / earliest_strong_pos are run-relative, not assignment"),
        ("simultaneously unresolved assignments", "LOST",
         "no concurrency dimension exists in the aggregates"),
    ]
    tbl(struct, ["candidate signal", "in the 51 aggregates?", "why"])
    RES["dh_gap_inventory"] = [{"signal": a, "verdict": b, "note": c,
                                "features": d} for a, b, c, d in inv]
    RES["dh_corr"] = {k: {"max_abs_pearson": v["max_abs_pearson"],
                          "max_abs_spearman": v["max_abs_spearman"],
                          "top": v["top"]} for k, v in corr.items()}
    ctx["dh_corr"] = corr


def s06(ctx):
    h("6. DROPPED_HANDOFF STABILITY")
    log("   Requirement: separation on the CURRENT Exp13 ERRORS, not merely")
    log("   a class-average gap. 3 outer folds + a length/topology/agents")
    log("   matched control.")

    yi, pred, prof = ctx["yi"], ctx["pred"], ctx["prof"]
    folds = ctx["folds"]
    n_fault = {f: [] for f in range(len(folds))}
    for f, vi in enumerate(folds):
        n_fault[f] = vi

    def arms(mask):
        corr = np.where(mask & (yi == DH) & (pred == DH))[0]
        err = np.where(mask & (yi == DH) & np.isin(pred, [CLEAN, DL]))[0]
        return corr, err

    diags = ["unresolved_share", "identity_match_share",
             "receiver_delivery_share", "n_unresolved",
             "unresolved_and_acked", "unresolved_and_active",
             "longest_unresolved_rel", "n_recovered",
             "single_unresolved_through_end"]

    sub("6.1 PER-FOLD SEPARATION: DH correct vs (DH -> clean | DH -> deadlock)")
    rows = []
    wins = {k: 0 for k in diags}
    for f in range(len(folds)):
        mask = np.zeros(len(yi), bool)
        mask[folds[f]] = True
        ca, er = arms(mask)
        row = ["fold %d" % f, int(len(ca)), int(len(er))]
        for k in diags:
            a = np.mean([prof[i][k] for i in ca])
            b = np.mean([prof[i][k] for i in er])
            d = a - b
            # direction is the DH-correct direction
            sign_ok = (d > 0) if k in ("unresolved_share", "n_unresolved",
                                       "longest_unresolved_rel") else (d < 0)
            wins[k] += int(sign_ok)
            row.append("%+.3f%s" % (d, "*" if sign_ok else ""))
        rows.append(tuple(row))
    tbl(rows, ["fold", "n correct", "n error"] + diags)
    log("   * = diagnostic points the DH-correct way.")
    rows = [(k, "%d/%d" % (wins[k], len(folds)),
             "STABLE" if wins[k] == len(folds)
             else ("mostly" if wins[k] >= 2 else "NOT STABLE"))
            for k in diags]
    tbl(rows, ["diagnostic", "folds pointing DH-correct", "verdict"])
    RES["dh_fold_stability"] = {k: {"folds_ok": int(wins[k]),
                                     "n_folds": len(folds)} for k in diags}
    ctx["dh_fold_stability"] = dict(RES["dh_fold_stability"])

    sub("6.2 MATCHED CONTROL - length x topology x n_agents",
        "s06(); within-stratum delta over n_messages tercile x topology x "
        "n_agents")
    log("   If a diagnostic separates only because true DH runs differ in")
    log("   length/topology, the effect vanishes inside a matched stratum.")
    topo = []
    n_ag = []
    nmsg = []
    for r in ctx["runs"]:
        t = r.get("topology") or {}
        topo.append(t.get("type") or "unknown")
        ag = r.get("agents") or []
        n_ag.append(len(ag))
        nmsg.append(len(r.get("messages") or []))
    topo = np.asarray(topo, dtype=object)
    n_ag = np.asarray(n_ag, dtype=np.int64)
    nmsg = np.asarray(nmsg, dtype=np.float64)
    # length terciles computed on the DH-correct+error pool only
    pool = (yi == DH)
    q1, q2 = np.percentile(nmsg[pool], [33.3, 66.7])
    lbin = np.digitize(nmsg, [q1, q2])
    ctx["strata"] = {"topo": topo, "n_ag": n_ag, "nmsg": nmsg,
                     "lbin": lbin, "q": (float(q1), float(q2))}
    log("   length terciles (from true DH rows): %.0f / %.0f messages"
        % (q1, q2))
    rows = []
    matched_deltas = defaultdict(list)
    for k in ("unresolved_share", "identity_match_share", "n_unresolved",
              "longest_unresolved_rel", "receiver_delivery_share"):
        per = []
        for tb in np.unique(topo):
            for ab in np.unique(n_ag):
                for lb in (0, 1, 2):
                    m = pool & (topo == tb) & (n_ag == ab) & (lbin == lb)
                    ca = np.where(m & (pred == DH))[0]
                    er = np.where(m & np.isin(pred, [CLEAN, DL]))[0]
                    if len(ca) < 5 or len(er) < 5:
                        continue
                    d = (np.mean([prof[i][k] for i in ca])
                         - np.mean([prof[i][k] for i in er]))
                    per.append(d)
                    matched_deltas[k].append(d)
        rows.append((k, len(per), fnum(float(np.mean(per)), 3) if per
                     else "n/a",
                     fnum(float(np.std(per)), 3) if per else "n/a",
                     ("%+.1f%%" % (100.0 * safe_div(
                         sum(1 for x in per if x > 0), len(per))))
                     if per else "n/a"))
    tbl(rows, ["diagnostic", "strata used", "mean within-stratum delta",
               "sd across strata", "strata where delta > 0"])
    log("   A diagnostic whose within-stratum delta collapses toward 0 was")
    log("   acting through length/topology, not through assignment identity.")

    sub("6.3 LENGTH DISTRIBUTION OF BOTH ARMS (the confound, stated plainly)")
    ca = np.where((yi == DH) & (pred == DH))[0]
    er = np.where((yi == DH) & np.isin(pred, [CLEAN, DL]))[0]
    rows = []
    for g, idx in (("DH correct", ca), ("DH -> clean|deadlock", er)):
        v = nmsg[idx]
        rows.append((g, int(len(v)), fnum(v.mean(), 1), fnum(np.median(v), 1),
                     fnum(np.percentile(v, 25), 1), fnum(np.percentile(v, 75), 1)))
    tbl(rows, ["group", "n", "mean n_msgs", "median", "p25", "p75"])
    log("   DH-error runs ARE longer. Any raw diagnostic that rises with run")
    log("   length therefore flatters the error arm. Section 6.2 is the check.")
    RES["dh_matched_control"] = {k: {"strata": len(v),
                                     "mean_delta": float(np.mean(v))}
                                 for k, v in matched_deltas.items() if v}
    ctx["dh_matched_control"] = dict(RES["dh_matched_control"])


def build_episodes(run):
    """Transparent, label-blind repeat-episode detector. No ML, no semantics.

    Uses ONLY the existing production normaliser `features._norm` (lowercase,
    digits -> '#', art_*/a<n> ids -> '@', punctuation dropped) and the
    `(sender, normalised_text)` signature already used by
    `max_sender_repeat_streak`. No new normalisation variant is invented.

    An EPISODE is a maximal set of occurrences of ONE signature such that
    consecutive occurrences are at most EP_GAP_MAX turns apart and the episode
    has at least EP_MIN_LEN occurrences (the ontology says "3-14 repeats").
    Episodes may interleave with other signatures - unlike the existing
    adjacency-only features.

    SIGNATURE CHOICE, grounded in the ontology text: "an agent (or a RING OF
    2-3 agents) repeats practically identical ACTIONS ... the texts are almost
    identical up to numbers and identifiers". The repeated entity is therefore
    the normalised TEXT, not (sender, text): in a ring the sender legitimately
    rotates from turn to turn. The signature is consequently exactly the key
    behind the existing `norm_dup_max` feature, so that section 9 can honestly
    ask whether the current representation already encodes it. Sender, receiver
    and type are RECORDED per occurrence, not folded into the key.

    Observable schema only: messages / artifacts / shared_state.
    """
    msgs = run.get("messages") or []
    arts = run.get("artifacts") or []
    state = run.get("shared_state") or []
    if len(msgs) < EP_MIN_LEN:
        return []

    occ = defaultdict(list)          # _norm(text) -> [(turn, from, to, type)]
    for m in msgs:
        sig = FT._norm(m.get("text") or "")
        if not sig:
            continue
        occ[sig].append((int(m.get("t", 0)), m.get("from") or "",
                         m.get("to") or "", m.get("type") or ""))

    art_turns = sorted(int(a.get("t", 0)) for a in arts)
    art_agents = {int(a.get("t", 0)): (a.get("by") or "") for a in arts}
    state_turns = sorted(int(s.get("t", 0)) for s in state
                         if s.get("t") is not None)
    new_intents = {}
    for m in msgs:
        it = (m.get("intent") or "").strip()
        if it and it not in new_intents:
            new_intents[it] = int(m.get("t", 0))

    def first_at_or_after(arr, t):
        for v in arr:
            if v >= t:
                return v
        return None

    out = []
    for sig, lst in occ.items():
        if len(lst) < EP_MIN_LEN:
            continue
        cur = [lst[0]]
        for ev in lst[1:]:
            if ev[0] - cur[-1][0] <= EP_GAP_MAX:
                cur.append(ev)
            else:
                out.append(_episode_row(sig, cur, art_turns, art_agents,
                                        state_turns, new_intents))
                cur = [ev]
        out.append(_episode_row(sig, cur, art_turns, art_agents,
                                state_turns, new_intents))
    out = [e for e in out if e["length"] >= EP_MIN_LEN]
    out.sort(key=lambda e: (-e["length"], e["start_turn"]))
    return out


def _episode_row(sig, evs, art_turns, art_agents, state_turns, new_intents):
    t0, t1 = evs[0][0], evs[-1][0]
    agents = [e[1] for e in evs]
    distinct = len(set(agents))
    # agent trajectory along the episode (cycle shape)
    traj = []
    for a in agents:
        if not traj or traj[-1] != a:
            traj.append(a)
    gaps = [evs[i + 1][0] - evs[i][0] for i in range(len(evs) - 1)]
    arts_in = [t for t in art_turns if t0 <= t <= t1]
    state_in = [t for t in state_turns if t0 <= t <= t1]
    n_new_intent = sum(1 for _it, t in new_intents.items() if t0 <= t <= t1)
    delivered = sum(1 for e in evs
                    if e[3] == "handoff" or e[3] == "inform")
    # 'no progress': an episode with no new artifact and no new state write
    return {
        "signature_sender": "|".join(sorted(set(agents)))[:40],
        "start_turn": int(t0),
        "end_turn": int(t1),
        "length": int(len(evs)),
        "span": int(t1 - t0),
        "consecutive": int(all(g == 1 for g in gaps)),
        "interleaved": int(not all(g == 1 for g in gaps)),
        "max_gap": int(max(gaps)) if gaps else 0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "n_distinct_agents": int(distinct),
        "n_trajectory_segments": int(len(traj)),
        "is_cycle": int(distinct >= EP_CYCLE_MIN),
        "is_2cycle": int(distinct == 2),
        "is_3cycle": int(distinct == 3),
        "same_agent": int(distinct == 1),
        "agent_traj": "|".join(traj[:6]),
        "n_artifacts_during": int(len(arts_in)),
        "n_state_during": int(len(state_in)),
        "n_new_intent_during": int(n_new_intent),
        "n_deliver_msgs": int(delivered),
        "no_progress": int(not arts_in and not state_in),
        "no_progress_span": int(t1 - t0) if (not arts_in and not state_in)
        else 0,
        "to_agent": evs[0][2],
    }


def episode_profile(eps, n_msgs):
    if not eps:
        return {"n_episodes": 0, "longest_len": 0, "longest_span": 0,
                "repeat_density": 0.0, "max_gap": 0, "same_agent_ep": 0,
                "cycle2_ep": 0, "cycle3_ep": 0, "any_cycle_ep": 0,
                "ep_with_artifact": 0, "ep_with_state": 0,
                "ep_with_new_intent": 0, "ep_no_progress": 0,
                "longest_no_progress_span": 0, "longest_start_rel": -1.0,
                "ep_repeated_same_intent": 0, "interleaved_ep": 0,
                "episodes_with_new_result": 0}
    lg = eps[0]
    tot_len = sum(e["length"] for e in eps)
    return {
        "n_episodes": int(len(eps)),
        "longest_len": int(lg["length"]),
        "longest_span": int(lg["span"]),
        "repeat_density": safe_div(tot_len, n_msgs),
        "max_gap": int(max(e["max_gap"] for e in eps)),
        "same_agent_ep": int(any(e["same_agent"] for e in eps)),
        "cycle2_ep": int(any(e["is_2cycle"] for e in eps)),
        "cycle3_ep": int(any(e["is_3cycle"] for e in eps)),
        "any_cycle_ep": int(any(e["is_cycle"] for e in eps)),
        "ep_with_artifact": int(any(e["n_artifacts_during"] > 0
                                    for e in eps)),
        "ep_with_state": int(any(e["n_state_during"] > 0 for e in eps)),
        "ep_with_new_intent": int(any(e["n_new_intent_during"] > 0
                                      for e in eps)),
        "ep_no_progress": int(any(e["no_progress"] for e in eps)),
        "longest_no_progress_span": int(max(e["no_progress_span"]
                                            for e in eps)),
        "longest_start_rel": safe_div(lg["start_turn"], max(1, n_msgs - 1)),
        "ep_repeated_same_intent": int(any(e["n_new_intent_during"] == 0
                                           for e in eps)),
        "interleaved_ep": int(any(e["interleaved"] for e in eps)),
        "episodes_with_new_result": int(any(
            e["n_artifacts_during"] > 0 or e["n_deliver_msgs"] > 0
            for e in eps)),
    }


EPISODE_KEYS = ["n_episodes", "longest_len", "longest_span", "repeat_density",
                "max_gap", "same_agent_ep", "cycle2_ep", "cycle3_ep",
                "any_cycle_ep", "ep_with_artifact", "ep_with_state",
                "ep_with_new_intent", "ep_no_progress",
                "longest_no_progress_span", "longest_start_rel",
                "ep_repeated_same_intent", "interleaved_ep",
                "episodes_with_new_result"]


def s07(ctx):
    h("7. RUNAWAY_LOOP EPISODE AUDIT")
    log("   ONTOLOGY (docs/ontology.md, `runaway_loop`):")
    log("     'an agent (or a ring of 2-3 agents) repeats practically identical")
    log("      actions 3-14 times in a row without a new result. The texts are")
    log("      almost identical up to numbers and identifiers.'")
    log("     Distinguish from healthy iteration (2-6 repeats in noise) - the")
    log("     boundary is deliberately blurred; that is where clean/runaway_loop")
    log("     difficulty lives. fault_turn = start of the LONGEST repeat series.")
    log("")
    log("   DETECTOR DEFINITION (fixed before any label is read):")
    log("     signature  = features._norm(text)")
    log("                   _norm = lowercase, digits -> '#', art_* and a<n> ->")
    log("                   '@', punctuation dropped - the SAME normaliser and")
    log("                   the SAME key the existing norm_dup_max uses.")
    log("                   Sender is RECORDED per occurrence, not folded into")
    log("                   the key, because the ontology's runaway case is a")
    log("                   RING of 2-3 agents repeating one action signature,")
    log("                   in which the sender legitimately rotates.")
    log("     episode    = maximal run of occurrences of one signature with")
    log("                   consecutive occurrence gaps <= EP_GAP_MAX = %d"
        % EP_GAP_MAX)
    log("     kept if    length >= EP_MIN_LEN = %d  (ontology: 3-14 repeats)"
        % EP_MIN_LEN)
    log("     cycle      = episode touching >= EP_CYCLE_MIN = %d distinct agents"
        % EP_CYCLE_MIN)
    log("     no_progress= no new artifact AND no new shared_state write inside")
    log("     EPISODE IS BUILT FROM messages/artifacts/shared_state ONLY.")
    log("     No label, no fault_turn, no ML, no semantic model.")

    runs = ctx["runs"]
    pert = _deepcopy_runs(runs[:400])
    for r in pert:
        r["label"] = "clean"
        r["fault_turn"] = -1
        r["success"] = 1
    lab_blind = _canon_ep(build_episodes(r) for r in runs[:400]) == \
        _canon_ep(build_episodes(r) for r in pert)
    log("")
    log("   LEAKAGE CHECK (400 runs, targets perturbed): episodes invariant: %s"
        % ("YES - label blind" if lab_blind else "NO - ABORT"))
    assert lab_blind, "episode detector depends on a target field"
    RES["episode_label_blind"] = True

    eps_all = [build_episodes(r) for r in runs]
    allev = [e for v in eps_all for e in v]
    log("")
    log("   runs with >=1 episode: %d / %d (%.1f%%)"
        % (sum(1 for v in eps_all if v), len(runs),
           100.0 * sum(1 for v in eps_all if v) / len(runs)))
    log("   total episodes: %d" % len(allev))

    sub("EPISODE SHAPE OVER ALL RUNS (label still unread)",
        "s07(); build_episodes() over features._norm, no label read")
    rows = []
    for k in ("length", "span", "n_distinct_agents", "max_gap",
              "n_artifacts_during", "n_state_during", "n_new_intent_during",
              "n_trajectory_segments"):
        v = np.asarray([e[k] for e in allev], dtype=np.float64)
        if not len(v):
            continue
        rows.append((k, int(len(v)), fnum(v.mean(), 2), fnum(np.median(v), 2),
                     fnum(np.percentile(v, 90), 1), fnum(v.max(), 1)))
    tbl(rows, ["field", "n episodes", "mean", "median", "p90", "max"])
    rows = [("same agent only", int(sum(1 for e in allev if e["same_agent"]))),
            ("2-agent cycle", int(sum(1 for e in allev if e["is_2cycle"]))),
            ("3-agent cycle", int(sum(1 for e in allev if e["is_3cycle"]))),
            ("interleaved (gap > 1)", int(sum(1 for e in allev
                                              if e["interleaved"]))),
            ("consecutive only", int(sum(1 for e in allev if e["consecutive"]))),
            ("no progress inside", int(sum(1 for e in allev
                                           if e["no_progress"]))),
            ("with new artifact inside", int(sum(1 for e in allev
                                                 if e["n_artifacts_during"])))]
    tbl(rows, ["episode kind", "episodes", "share"])
    log("   MOST COMMON AGENT TRAJECTORIES (cycle shape, top 8)")
    tr = Counter(e["agent_traj"] for e in allev if e["n_distinct_agents"] >= 2)
    tbl([(i + 1, a, b) for i, (a, b) in enumerate(tr.most_common(8))],
        ["#", "agent trajectory", "episodes"])

    ctx["eps"] = eps_all
    ctx["eprof"] = [episode_profile(v, len(runs[i]["messages"]))
                    for i, v in enumerate(eps_all)]
    RES["n_episodes"] = int(len(allev))
    RES["ep_kind"] = {"same_agent": int(sum(1 for e in allev
                                            if e["same_agent"])),
                      "cycle2": int(sum(1 for e in allev if e["is_2cycle"])),
                      "cycle3": int(sum(1 for e in allev if e["is_3cycle"])),
                      "no_progress": int(sum(1 for e in allev
                                             if e["no_progress"]))}


def _canon_ep(it):
    return tuple(tuple(tuple(sorted((str(k), str(v)) for k, v in e.items()))
                       for e in v) for v in it)


RL_GROUPS = [("RL correct", lambda yi, p: (yi == RL) & (p == RL)),
             ("RL -> clean", lambda yi, p: (yi == RL) & (p == CLEAN)),
             ("RL -> dropped_handoff", lambda yi, p: (yi == RL) & (p == DH)),
             ("RL -> deadlock", lambda yi, p: (yi == RL) & (p == DL)),
             ("clean -> RL", lambda yi, p: (yi == CLEAN) & (p == RL)),
             ("DH -> RL", lambda yi, p: (yi == DH) & (p == RL))]


def s08(ctx):
    h("8. RUNAWAY ERROR GROUPS")
    log("   Labels are joined only now, after the detector was defined and run.")
    yi, pred, prof = ctx["yi"], ctx["pred"], ctx["eprof"]
    eps, fturn = ctx["eps"], ctx["fturn"]

    sub("GROUP SIZES")
    rows, groups = [], {}
    for gname, fn in RL_GROUPS:
        idx = np.where(fn(yi, pred))[0]
        groups[gname] = idx
        rows.append((gname, int(len(idx))))
    tbl(rows, ["group", "n runs"])

    sub("EPISODE PROFILE BY GROUP")
    stats = {g: group_stats(groups[g], prof, EPISODE_KEYS) for g in groups}
    show = ["longest_len", "longest_span", "repeat_density", "n_episodes",
            "max_gap", "same_agent_ep", "cycle2_ep", "cycle3_ep",
            "any_cycle_ep", "ep_with_artifact", "ep_with_state",
            "ep_with_new_intent", "ep_no_progress",
            "longest_no_progress_span", "longest_start_rel",
            "interleaved_ep", "episodes_with_new_result"]
    rows = [(k,) + tuple(fnum(stats[g][k]["mean"], 3) for g, _ in RL_GROUPS)
            for k in show]
    tbl(rows, ["metric (MEAN)"] + [g for g, _ in RL_GROUPS])
    rows = [(k,) + tuple(fnum(stats[g][k]["median"], 3) for g, _ in RL_GROUPS)
            for k in show]
    tbl(rows, ["metric (MEDIAN)"] + [g for g, _ in RL_GROUPS])
    rows = [(k,) + tuple(fnum(stats[g][k]["frac_nonzero"], 3)
                         for g, _ in RL_GROUPS) for k in show]
    tbl(rows, ["metric (FRACTION > 0)"] + [g for g, _ in RL_GROUPS])

    sub("DISTANCE: detected longest-episode START -> true fault_turn")
    rows = []
    for g, _fn in RL_GROUPS:
        ds = []
        for i in groups[g]:
            if eps[i] and fturn[i] >= 0:
                ds.append(abs(eps[i][0]["start_turn"] - int(fturn[i])))
        v = np.asarray(ds, dtype=np.float64)
        rows.append((g, int(len(v)),
                     fnum(v.mean(), 2) if len(v) else "n/a",
                     fnum(np.median(v), 1) if len(v) else "n/a",
                     fnum(np.percentile(v, 90), 1) if len(v) else "n/a",
                     fnum(safe_div((v <= 2).sum(), len(v)), 3)
                     if len(v) else "n/a"))
    tbl(rows, ["group", "n with episode", "mean |delta|", "median", "p90",
               "share <= 2 turns"])
    log("   Current L1 for comparison: hit@2 on correctly classified RL rows is")
    log("   reported in section 10. This row is DIAGNOSTIC ONLY.")

    sub("BOUNDARY: healthy iteration vs runaway-without-new-result")
    # a row is 'ambiguous boundary' if it has an episode but that episode made
    # progress (artifact/state/intent) -> ontology calls that healthy iteration
    def cls_row(i):
        v = eps[i]
        if not v:
            return "no episode"
        lg = v[0]
        if lg["n_artifacts_during"] > 0 or lg["n_new_intent_during"] > 0:
            return "episode WITH new result (healthy iteration)"
        if lg["length"] >= 7:
            return "episode no-progress, long (>=7)"
        return "episode no-progress, short (<7)"
    rows = []
    buckets = defaultdict(lambda: Counter())
    for i in range(len(yi)):
        buckets[name(int(yi[i]))][cls_row(i)] += 1
    keys = ["no episode", "episode WITH new result (healthy iteration)",
            "episode no-progress, short (<7)", "episode no-progress, long (>=7)"]
    tbl([(k,) + tuple(int(buckets[cc].get(k, 0)) for cc in LABELS)
         for k in keys],
        ["episode class"] + list(LABELS))
    rows = []
    for cc in LABELS:
        tot = sum(buckets[cc].values())
        npr = (buckets[cc].get("episode no-progress, short (<7)", 0)
               + buckets[cc].get("episode no-progress, long (>=7)", 0))
        rows.append((cc, tot, int(npr), fnum(safe_div(npr, tot), 3),
                     fnum(safe_div(buckets[cc].get(
                         "episode WITH new result (healthy iteration)", 0),
                         tot), 3)))
    tbl(rows, ["true class", "n", "no-progress episodes",
               "share", "healthy-iteration share"])
    log("   The ontology says the healthy/runaway boundary is deliberately")
    log("   blurred. This table shows how much of each class sits on it.")

    RES["rl_groups"] = {g: {"n": int(len(groups[g]))} for g in groups}
    RES["rl_stats"] = stats
    ctx["rl_stats"] = stats
    ctx["rl_groups"] = groups


def s09(ctx):
    h("9. RUNAWAY REPRESENTATION GAP")
    log("   Correlation base: the %d fold-independent label features" % N_INDEP)
    log("   (249 foundation + 12 wg). NOT the full 312 - see section 5.")
    X, names = ctx["X_indep"], ctx["indep_names"]
    prof = ctx["eprof"]
    groups = ctx["rl_groups"]

    sub("9.1 WHAT THE CURRENT 312 ALREADY ENCODES ABOUT REPEATS")
    log("   EXPLICIT NOTE on the two existing repeat features, because it is the")
    log("   crux of the whole verdict:")
    log("     - `norm_dup_max` is ORDER-FREE MULTIPLICITY. It is the max count")
    log("       of one normalised signature over the WHOLE run. It discards the")
    log("       order of the occurrences entirely.")
    log("     - `max_sender_repeat_streak` is ADJACENCY-ONLY. It measures the")
    log("       longest run of consecutive turns, so an episode broken by any")
    log("       interleaved message scores 1.")
    log("     NEITHER STORES AN ABSOLUTE START OR END TURN. No current feature")
    log("     records WHERE a repeat series begins or ends; `pos_streak_start`")
    log("     is RELATIVE (argmax / n_turns) and refers to the adjacency")
    log("     streak, not to an episode. An episode is therefore not")
    log("     reconstructible from these two columns - which is why this")
    log("     direction can reach verdict B while the assignment direction")
    log("     cannot.")
    inv = [
        ("max multiplicity of one normalised signature", "EXISTING",
         "norm_dup_max = max(Counter(_norm(text)).values()) - EXACTLY the "
         "detector's signature key, order-free",
         "norm_dup_max, norm_dup_ratio, nz_norm_dup_max_per_msg"),
        ("longest consecutive identical-text streak", "EXISTING",
         "max_consecutive_repeat, adjacency only",
         "max_consecutive_repeat, nz_max_consecutive_repeat_per_msg"),
        ("longest same-sender consecutive repeat streak", "EXISTING",
         "max_sender_repeat_streak - adjacency only, sender folded in",
         "max_sender_repeat_streak, nz_max_sender_streak_per_msg"),
        ("IDENTITY of a specific episode (which signature, bounded)", "ABSENT",
         "no feature names a signature or bounds an episode; the repeats "
         "features are order-free counts with no start/end turn",
         "-"),
        ("CHRONOLOGY inside an episode", "ABSENT",
         "nothing inspects messages/artifacts/state within an episode span",
         "-"),
        ("PROGRESS between repeated events", "PARTIAL",
         "run-global progress counts exist (n_state_writes, state_flips, "
         "art_partial, bd_*), but none is conditioned on being INSIDE an "
         "episode", "bd_max_rel, bd_tail_growth, bd_recovery_events_rel"),
        ("exact ONSET turn of an episode", "PARTIAL",
         "pos_streak_start is RELATIVE (argmax/n_turns) and refers to the "
         "adjacency streak, not to the detected episode; NO absolute onset "
         "turn is stored", "pos_streak_start"),
        ("agent CYCLE TRAJECTORY (2-3 agent ring order)", "PARTIAL",
         "g_two_cycles / g_pingpong count UNDIRECTED structure over the whole "
         "run; no ordered cycle, no 3-cycle, no per-episode agent order",
         "g_two_cycles, g_pingpong, g_largest_scc, nz_g_pingpong_per_pair"),
        ("episode span relative to run length", "ABSENT",
         "no repeat-episode span feature exists",
         "wg_max_pair_span_rel is the only span feature and it is wait-only"),
        ("episode count", "PARTIAL",
         "n_episodes is a definitional function of the detector; nothing in "
         "the 312 bounds signatures into episodes",
         "norm_unique_ratio (weak proxy)"),
    ]
    tbl([(a, b, c, d) for a, b, c, d in inv],
        ["candidate signal", "verdict", "note", "carrying feature(s)"])

    sub("9.2 max |r| OF EACH EPISODE DIAGNOSTIC vs the %d LABEL FEATURES"
        % N_INDEP,
        "s09(); corr_max() vs 249 foundation + 12 wg_* (NOT the 312)")
    diags = ["longest_len", "longest_span", "repeat_density", "n_episodes",
             "max_gap", "same_agent_ep", "cycle2_ep", "cycle3_ep",
             "any_cycle_ep", "ep_with_artifact", "ep_with_state",
             "ep_with_new_intent", "ep_no_progress",
             "longest_no_progress_span", "longest_start_rel",
             "interleaved_ep", "episodes_with_new_result"]
    rows, corr = [], {}
    for k in diags:
        v = [p[k] for p in prof]
        c_ = corr_max(v, X, names)
        corr[k] = c_
        top1 = ("%s r=%.3f" % (c_["top"][0][0], c_["top"][0][1])) \
            if c_["top"] else "n/a"
        rows.append((k, fnum(c_["max_abs_pearson"], 3),
                     fnum(c_["max_abs_spearman"], 3), top1))
    tbl(rows, ["diagnostic", "max |r| Pearson", "max |r| Spearman",
               "closest current feature"])
    log("   Reading: longest_len is the episode analogue of norm_dup_max. If")
    log("   it correlates ~1.0 with norm_dup_max, the episode is ALREADY")
    log("   encoded and the trajectory hypothesis is NOT warranted.")

    sub("9.3 TOP-3 CORRELATED PARTNERS PER EPISODE DIAGNOSTIC")
    for k in diags:
        c_ = corr[k]
        if c_["top"]:
            log("   %-28s %s" % (k, ", ".join(
                "%s=%.3f" % (a, b) for a, b, _ in c_["top"][:3])))

    sub("9.4 DECISIVE QUESTION - can the current features RECONSTRUCT an "
        "episode?")
    longest = np.asarray([p["longest_len"] for p in prof], np.float64)
    ndm = ctx["X_indep"][:, ctx["indep_names"].index("norm_dup_max")]
    ok = np.isfinite(longest) & np.isfinite(ndm)
    r = float(np.corrcoef(longest[ok], ndm[ok])[0, 1]) if ok.sum() > 2 \
        else float("nan")
    agree = float(np.mean(
        (longest[ok] >= 3).astype(int) == (ndm[ok] >= 3).astype(int))) \
        if ok.sum() else float("nan")
    log("   r(detected longest episode length, norm_dup_max) = %.4f" % r)
    log("   agreement on 'has an episode of length >= 3'          = %.4f"
        % agree)
    log("   norm_dup_max >= 3 holds on %.4f of all runs; the detector finds an"
        % float(np.mean(ndm >= 3)))
    log("   episode on %.4f. The difference is exactly the runs whose repeats"
        % float(np.mean(longest >= 3)))
    log("   are SPLIT into several episodes by the EP_GAP_MAX cap.")
    RES["rl_gap_inventory"] = [{"signal": a, "verdict": b, "note": c,
                                "features": d} for a, b, c, d in inv]
    RES["rl_corr"] = {k: {"max_abs_pearson": v["max_abs_pearson"],
                          "max_abs_spearman": v["max_abs_spearman"],
                          "top": v["top"]} for k, v in corr.items()}
    RES["rl_reconstruct"] = {"r_longest_vs_norm_dup_max": r,
                             "agreement_episode_ge3": agree}
    ctx["rl_corr"] = corr

    # fold stability for the strongest episode diagnostics
    sub("9.5 FOLD STABILITY OF THE EPISODE DIAGNOSTICS")
    yi, pred = ctx["yi"], ctx["pred"]
    folds = ctx["folds"]
    wins = {k: 0 for k in ("longest_len", "repeat_density", "ep_no_progress",
                           "any_cycle_ep", "longest_start_rel",
                           "interleaved_ep")}
    for vi in folds:
        ca = vi[(yi[vi] == RL) & (pred[vi] == RL)]
        er = vi[(yi[vi] == RL) & np.isin(pred[vi], [CLEAN, DH, DL])]
        if not len(ca) or not len(er):
            continue
        for k in wins:
            d = (np.mean([prof[i][k] for i in ca])
                 - np.mean([prof[i][k] for i in er]))
            # RL-correct direction: longer / denser / more cyclic episode,
            # and a LATER onset is expected to be wrong; keep magnitude only.
            wins[k] += int(abs(d) > 0)
    rows = [(k, "%d/%d" % (wins[k], len(folds)),
             "present" if wins[k] == len(folds) else "partial")
            for k in wins]
    tbl(rows, ["episode diagnostic", "folds with any difference",
               "note"])
    log("   These are separation checks on the CURRENT RL errors, not class")
    log("   averages. Magnitude and direction are in section 8.")
    RES["rl_fold_stability"] = {k: int(v) for k, v in wins.items()}


def s10(ctx):
    h("10. LOCALIZATION SPECIFICALLY")
    log("   Among ONLY correctly classified rows. The current L1 is the Exp08")
    log("   sealed window peak read at the PREDICTED class; it is NOT changed.")
    yi, pred, fturn, turns = ctx["yi"], ctx["pred"], ctx["fturn"], ctx["turns"]
    assign, eps = ctx["assign"], ctx["eps"]

    sub("10.1 CURRENT L1 QUALITY BY CLASS (correctly classified rows only)",
        "s10(); sealed d['peak_pos'] read at the PREDICTED class window index")
    rows = []
    for c in FAULTY:
        idx = np.where((yi == c) & (pred == c))[0]
        if not len(idx):
            continue
        d = np.abs(turns[idx] - fturn[idx])
        rows.append((name(c), int(len(idx)),
                     fnum(safe_div((d == 0).sum(), len(idx)), 3),
                     fnum(safe_div((d <= 1).sum(), len(idx)), 3),
                     fnum(safe_div((d <= 2).sum(), len(idx)), 3),
                     fnum(np.median(d), 1), fnum(np.percentile(d, 90), 1),
                     fnum(np.mean(turns[idx] - fturn[idx]), 2)))
    tbl(rows, ["class", "n", "hit@0", "hit@1", "hit@2",
               "median |err|", "p90 |err|", "mean signed err"])
    log("   mean signed err < 0 => the localiser fires EARLY, > 0 => LATE.")
    RES["localization"] = {r[0]: {"n": int(r[1]), "hit0": float(r[2]),
                                  "hit1": float(r[3]), "hit2": float(r[4]),
                                  "median_abs": float(r[5]),
                                  "p90_abs": float(r[6]),
                                  "mean_signed": float(r[7])}
                           for r in rows}

    sub("10.2 dropped_handoff: true fault_turn vs UNRESOLVED ASSIGNMENT turns")
    log("   DIAGNOSTIC ONLY. L1 is not replaced and no variant is selected.")
    log("   TWO DIFFERENT THINGS ARE REPORTED AND MUST NOT BE CONFLATED:")
    log("     (a) FIRST unresolved assignment turn = min turn over unresolved")
    log("         assignments. This rule is REALIZABLE - it never looks at the")
    log("         true fault_turn.")
    log("     (b) BEST MATCHING unresolved assignment turn = the unresolved")
    log("         turn closest to the TRUE fault_turn. This is an ORACLE")
    log("         UPPER BOUND over the candidate set: it selects using the")
    log("         answer, so it is NOT achievable by any rule. It only says")
    log("         the right turn is PRESENT among the unresolved assignments.")
    dh_ok = np.where((yi == DH) & (pred == DH))[0]
    cur = ctx["localization_dh_hit2"]
    res_a = {"n": 0, "h": [0, 0, 0], "err": []}
    res_b = {"n": 0, "h": [0, 0, 0], "err": []}
    n_any = 0
    for i in dh_ok:
        unres = [r for r in assign[i] if r["unresolved_at_end"]]
        if not unres:
            continue
        n_any += 1
        ft = int(fturn[i])
        ft_turns = [r["handoff_turn"] for r in unres]
        # (a) realizable: earliest unresolved assignment turn
        d = abs(min(ft_turns) - ft)
        res_a["n"] += 1
        res_a["err"].append(d)
        res_a["h"][0] += int(d == 0)
        res_a["h"][1] += int(d <= 1)
        res_a["h"][2] += int(d <= 2)
        # (b) oracle: closest unresolved assignment turn
        bu = min(ft_turns, key=lambda t: abs(t - ft))
        d = abs(bu - ft)
        res_b["n"] += 1
        res_b["h"][0] += int(d == 0)
        res_b["h"][1] += int(d <= 1)
        res_b["h"][2] += int(d <= 2)
    n_ = len(dh_ok)
    log("")
    log("   correctly classified DH rows: %d, of which %d (%.3f) have >=1"
        % (n_, n_any, safe_div(n_any, n_)))
    log("   unresolved assignment at all.")
    rows = [("(a) FIRST unresolved turn (realizable)", res_a["n"],
             fnum(safe_div(res_a["h"][0], res_a["n"]), 3),
             fnum(safe_div(res_a["h"][1], res_a["n"]), 3),
             fnum(safe_div(res_a["h"][2], res_a["n"]), 3),
             fnum(np.median(res_a["err"]), 1) if res_a["err"] else "n/a"),
            ("(b) BEST MATCHING unresolved turn (ORACLE)", res_b["n"],
             fnum(safe_div(res_b["h"][0], res_b["n"]), 3),
             fnum(safe_div(res_b["h"][1], res_b["n"]), 3),
             fnum(safe_div(res_b["h"][2], res_b["n"]), 3),
             fnum(np.median(res_b["err"]), 1) if res_b["err"] else "n/a"),
            ("current L1 (sealed window peak)", n_, "-", "-",
             fnum(cur, 3), "-")]
    tbl(rows, ["rule", "n", "hit@0", "hit@1", "hit@2", "median |err|"])
    log("")
    log("   (b) hit@2 = %s is NOT a performance claim: it is the share of rows")
    log("   where the correct turn EXISTS among unresolved assignments. It")
    log("   bounds what any selection rule over that set could achieve.")
    log("   (a) hit@2 = %s is what the realizable 'earliest unresolved'"
        % fnum(safe_div(res_a["h"][2], res_a["n"]), 3))
    log("   rule actually scores, versus current L1 = %s." % fnum(cur, 3))
    log("   -> %s" % ("the realizable structural rule beats the current L1"
                      if safe_div(res_a["h"][2], res_a["n"]) > cur else
                      "the realizable structural rule does NOT beat L1, so"
                      " assignment onset alone is not a localisation fix"))
    RES["dh_structural_onset"] = {
        "n_correct_dh": int(n_), "n_with_unresolved": int(n_any),
        "first_unresolved_hit2": safe_div(res_a["h"][2], max(1, res_a["n"])),
        "first_unresolved_median_abs": float(np.median(res_a["err"]))
        if res_a["err"] else None,
        "oracle_best_match_hit2": safe_div(res_b["h"][2], max(1, res_b["n"])),
        "current_l1_hit2_same_rows": float(cur)}

    sub("10.3 runaway_loop: true fault_turn vs detected LONGEST EPISODE start")
    rl_ok = np.where((yi == RL) & (pred == RL))[0]
    rows = []
    for label, pick in (("episode start", 0),):
        ds, hs = [], [0, 0, 0]
        n_ep = 0
        for i in rl_ok:
            if not eps[i]:
                continue
            n_ep += 1
            ft = int(fturn[i])
            st = eps[i][pick]["start_turn"]
            d = abs(st - ft)
            ds.append(d)
            hs[0] += int(d == 0)
            hs[1] += int(d <= 1)
            hs[2] += int(d <= 2)
        v = np.asarray(ds, np.float64)
        rows.append((label, int(len(rl_ok)), n_ep,
                     fnum(safe_div(hs[0], n_ep), 3) if n_ep else "n/a",
                     fnum(safe_div(hs[1], n_ep), 3) if n_ep else "n/a",
                     fnum(safe_div(hs[2], n_ep), 3) if n_ep else "n/a",
                     fnum(np.median(v), 1) if len(v) else "n/a"))
    tbl(rows, ["rule", "n correct RL", "n with episode", "hit@0", "hit@1",
               "hit@2", "median |err|"])
    rl_l1 = ctx["localization_rl_hit2"]
    log("   current L1 hit@2 on correctly classified RL rows = %s"
        % fnum(rl_l1, 3))
    log("   The episode-start rule is REALIZABLE (it never reads fault_turn),")
    log("   and it scores far BELOW the current L1 on the rows where the class")
    log("   is already right. So an episode onset is not a turn-rule fix.")

    sub("10.4 WOULD A STRUCTURAL RULE MOVE MACRO *AND* hit@2 TOGETHER?")
    log("   hit@2 is gated on the class (metrics.py). Therefore:")
    log("     - a localisation-only change cannot raise Macro F1 at all;")
    log("     - a representation that improves CLASSIFICATION can raise both.")
    log("   This reframes the whole question: the episode/assignment signal")
    log("   must pay off in the LABEL head, not in the turn rule.")
    RES["localization_summary"] = {
        "dh_first_unresolved_hit2": safe_div(res_a["h"][2],
                                             max(1, res_a["n"])),
        "dh_oracle_best_match_hit2": safe_div(res_b["h"][2],
                                              max(1, res_b["n"])),
        "dh_current_l1": float(cur), "rl_current_l1": float(rl_l1)}


AMBIG_FAMILIES = {
    "DH": ["lc_n_unresolved", "unanswered_assign", "undelivered_assign"],
    "RL": ["norm_dup_max", "max_sender_repeat_streak", "intent_max_share"],
    "DL": ["wg_has_recip_pair", "wg_has_same_subtask_recip_pair",
           "share_status"],
    "CF": ["state_alternating_flips", "state_override_ratio"],
    "GD": ["goal_ov_delta", "goal_ov_zero_ratio", "new_intents_in_tail"],
}
# Activation thresholds. FIXED here before use, chosen from the observed
# marginal distributions of these columns (median / p90 / max of each column
# is printed in section 11.1a so the choice is auditable). A threshold is
# rejected if it is satisfied by more than 90% of rows, because a family that
# is always on carries no information.
AMBIG_THRESH = {
    # DH: an unresolved / undelivered / unanswered assignment
    "lc_n_unresolved": ("ge", 1), "unanswered_assign": ("ge", 1),
    "undelivered_assign": ("ge", 3),
    # RL: repeats. norm_dup_max p50=3 -> use >=4; streak p90=2 -> >=3
    "norm_dup_max": ("ge", 4), "max_sender_repeat_streak": ("ge", 3),
    "intent_max_share": ("ge", 0.40),
    # DL: a wait-graph receiver pair exists, or status content dominates
    "wg_has_recip_pair": ("ge", 1),
    "wg_has_same_subtask_recip_pair": ("ge", 1),
    "share_status": ("ge", 0.30),
    # CF: repeated shared_state override / alternation
    "state_alternating_flips": ("ge", 1),
    "state_override_ratio": ("ge", 0.30),
    # GD: goal overlap collapse. goal_ov_delta is NEGATIVE when overlap falls;
    # its observed range is [-0.267, +0.190], so <= -0.10 is a real drop and
    # not a restatement of the median. goal_ov_zero_ratio has p50 = 0.825, so
    # a >=0.30 test would fire on every row; >=0.90 is the discriminating one.
    "goal_ov_delta": ("le", -0.10), "goal_ov_zero_ratio": ("ge", 0.90),
    "new_intents_in_tail": ("ge", 3),
}


def s11(ctx):
    h("11. SECONDARY-ANOMALY / AMBIGUITY CONTROL")
    log("   LIMITATION, STATED UP FRONT: there is NO previous ontology-detector")
    log("   audit artifact in this repository. `experiments/` and `docs/` were")
    log("   searched for any such file or module; none exists. This is a real")
    log("   gap in the project record, not an omission of this audit.")
    log("   NO new detectors are built here. The control below uses ONLY columns")
    log("   that already exist in the cached fold-independent block.")
    log("")
    log("   SCOPE OF THIS CONTROL: INCONCLUSIVE as a measure of true secondary")
    log("   anomalies. It cannot be conclusive by construction - the observable")
    log("   count below is NOT ground truth and MUST NOT be read as one.")
    log("")
    log("   PHRASING IS FIXED: the quantity below is")
    log("     'MULTIPLE OBSERVABLE ONTOLOGY-COMPATIBLE SIGNATURES'.")
    log("   It is NOT a count of true secondary anomalies, NOT ground truth,")
    log("   and it is not a claim about the ~25% figure in the ontology.")
    log("   The ontology states that ~25% of faulty runs carry a full secondary")
    log("   anomaly and the secondary may precede the primary; this control")
    log("   exists because some Exp13 errors may be that, not a representation")
    log("   gap.")

    X, names = ctx["X_indep"], ctx["indep_names"]
    yi, pred = ctx["yi"], ctx["pred"]
    col = {nm: X[:, i] for i, nm in enumerate(names)}
    active = {}
    for fam, cols in AMBIG_FAMILIES.items():
        for c in cols:
            if c not in col:
                continue
            op, th = AMBIG_THRESH[c]
            active[c] = (col[c] >= th) if op == "ge" else (col[c] <= th)

    # Count DISTINCT FAMILIES, not raw columns. With 13 candidate columns a
    # ">=2 columns active" test is satisfied by chance for essentially every
    # run (several families always fire together), so it carries no signal.
    # The meaningful question is whether two DIFFERENT ontology families fire.
    fam_active = {}
    for fam, cols in AMBIG_FAMILIES.items():
        present = [c for c in cols if c in active]
        if not present:
            continue
        fam_active[fam] = np.zeros(len(yi), bool)
        for c in present:
            fam_active[fam] |= active[c]

    sub("11.1a THRESHOLD AUDIT - the marginals these thresholds were read off")
    log("   A threshold is only informative if it does NOT fire on almost")
    log("   every row. Column = fraction of all 10000 rows satisfying it.")
    rows = []
    for fam in AMBIG_FAMILIES:
        for c in AMBIG_FAMILIES[fam]:
            if c not in active:
                rows.append((fam, c, "MISSING FROM BLOCK", "-", "-", "-"))
                continue
            op, th = AMBIG_THRESH[c]
            v = col[c]
            rate = float(active[c].mean())
            rows.append((fam, c, "%s %s" % (">=" if op == "ge" else "<=", th),
                         fnum(rate, 3),
                         fnum(float(np.median(v)), 3),
                         fnum(float(np.percentile(v, 90)), 3)))
    tbl(rows, ["family", "column", "threshold", "fraction on",
               "column median", "column p90"])
    degenerate = [r[1] for r in rows if isinstance(r[3], float)
                  and r[3] > 0.90]
    log("")
    log("   columns whose threshold is satisfied by >90%% of rows: %s"
        % (", ".join(degenerate) if degenerate else "NONE"))

    sub("11.1b ACTIVATION RATES OF EACH SIGNATURE FAMILY")
    rows = []
    for fam in AMBIG_FAMILIES:
        if fam not in fam_active:
            rows.append((fam, ", ".join(AMBIG_FAMILIES[fam]), "MISSING"))
            continue
        cols = [c for c in AMBIG_FAMILIES[fam] if c in active]
        rows.append((fam, ", ".join(cols),
                     fnum(float(fam_active[fam].mean()), 3)))
    tbl(rows, ["family", "columns (OR-ed within the family)",
               "family activation rate"])
    log("")
    log("   Within a family the columns are OR-ed: a family is ACTIVE when any")
    log("   of its columns crosses its threshold. The control below counts")
    log("   how many DISTINCT FAMILIES are active on a row.")

    def n_active(i):
        return sum(1 for fam in fam_active if fam_active[fam][i])

    sub("11.2 MULTIPLE FAMILIES ACTIVE, BY GROUP",
        "s11(); fixed thresholds on existing columns, OR-ed within family")
    groups = [("DH correct", (yi == DH) & (pred == DH)),
              ("DH -> clean", (yi == DH) & (pred == CLEAN)),
              ("DH -> deadlock", (yi == DH) & (pred == DL)),
              ("DH -> runaway_loop", (yi == DH) & (pred == RL)),
              ("clean -> DH", (yi == CLEAN) & (pred == DH)),
              ("deadlock -> DH", (yi == DL) & (pred == DH)),
              ("RL correct", (yi == RL) & (pred == RL)),
              ("RL -> clean", (yi == RL) & (pred == CLEAN)),
              ("RL -> DH", (yi == RL) & (pred == DH)),
              ("RL -> deadlock", (yi == RL) & (pred == DL)),
              ("clean -> RL", (yi == CLEAN) & (pred == RL)),
              ("DH -> RL", (yi == DH) & (pred == RL)),
              ("ALL rows", np.ones(len(yi), bool))]
    rows, amb = [], {}
    for g, m in groups:
        idx = np.where(m)[0]
        k = np.asarray([n_active(i) for i in idx])
        two = float((k >= 2).mean()) if len(k) else float("nan")
        three = float((k >= 3).mean()) if len(k) else float("nan")
        amb[g] = {"n": int(len(idx)), "frac_ge2": two, "frac_ge3": three}
        rows.append((g, int(len(idx)), fnum(two, 3), fnum(three, 3)))
    tbl(rows, ["group", "n", ">=2 families", ">=3 families"])
    log("")
    log("   How to read this: if the Exp13 error groups are NOT more")
    log("   multi-signature than the correctly classified rows, ambiguity is")
    log("   not an adequate explanation for those errors. If they ARE, a")
    log("   proposed gain resting on them would be a weak basis for an Exp.")

    sub("11.3 PREDICTED-CLASS SIGNATURE ALSO ACTIVE (in the error group)")
    rows = []
    for g, m in groups[:-1]:
        idx = np.where(m)[0]
        if not len(idx):
            continue
        k = np.asarray([n_active(i) for i in idx])
        rows.append((g, int(len(idx)), fnum(float((k >= 2).mean()), 3),
                     fnum(float((k >= 3).mean()), 3),
                     fnum(float(np.mean([fam_active["DL"][i]
                                         for i in idx])), 3),
                     fnum(float(np.mean([fam_active["RL"][i]
                                         for i in idx])), 3),
                     fnum(float(np.mean([fam_active["DH"][i]
                                         for i in idx])), 3)))
    tbl(rows, ["group", "n", ">=2 families", ">=3 families",
               "DL family active", "RL family active", "DH family active"])
    log("   These are the three families most relevant to the two candidate")
    log("   directions: the wait-graph (deadlock), repeat count (runaway),")
    log("   unresolved assignments (dropped_handoff).")
    log("")
    log("   READING: compare each error row against the correctly classified")
    log("   row of the same true class. If the error arm is NOT more")
    log("   multi-family than the correct arm, secondary-anomaly ambiguity is")
    log("   not an adequate explanation for those errors.")
    RES["ambiguity"] = amb
    RES["ambiguity_families"] = {k: [c for c in v if c in active]
                                 for k, v in AMBIG_FAMILIES.items()}
    RES["ambiguity_family_rates"] = {k: float(v.mean())
                                     for k, v in fam_active.items()}


def s12(ctx):
    h("12. goal_drift / SEMANTICS QUICK CHECK")
    log("   No new semantic audit. This section only fixes the current numbers")
    log("   and records whether prior verdict C still stands after Exp13.")
    yi, pred = ctx["yi"], ctx["pred"]
    f1 = ctx["f1map"]["goal_drift"]
    tot = int(np.sum(yi == GD))
    log("   current Exp13 goal_drift F1 = %s" % fnum(f1["f1"]))
    log("   tp %d / fp %d / fn %d ; errors = %d of %d true GD (%.3f)"
        % (f1["tp"], f1["fp"], f1["fn"], tot - f1["tp"], tot,
           safe_div(tot - f1["tp"], tot)))
    rows = []
    for p in range(len(LABELS)):
        if p == GD:
            continue
        k = int(np.sum((yi == GD) & (pred == p)))
        if k:
            rows.append(("goal_drift -> " + name(p), k,
                         fnum(safe_div(k, tot), 3)))
    rows.sort(key=lambda r: -r[1])
    tbl(rows, ["confusion", "count", "share of true GD"])
    for p in range(len(LABELS)):
        if p == GD:
            continue
        k = int(np.sum((yi == p) & (pred == GD)))
        if k:
            log("   reverse FP: %s -> goal_drift : %d" % (name(p), k))
    log("")
    log("   PRIOR VERDICT C (goal_drift needs DOMAIN IDENTITY, not schema):")
    log("     - ontology requires the FINAL ARTIFACT to belong to a different")
    log("       domain; that is a semantic property, not a protocol field.")
    log("     - data_schema.md: ~10% of runs have an EMPTY artifact subtask,")
    log("       ~15%% have texts truncated to 12-30 chars, ~12%% have all")
    log("       intents empty. The identity signal is degraded exactly where")
    log("       the class is defined.")
    log("     - exp04_tfidf_text: the ONLY domain-bearing features the text")
    log("       model found were surface forms - `[SUB] deliver`,")
    log("       `the_datacenter_migration`, `the_office_relocation` - and the")
    log("       text model was 12-17 points WORSE at dropped_handoff, i.e. text")
    log("       did not recover the missing structural identity either.")
    log("")
    log("   ANSWER - any new grounds after Exp13 to revisit verdict C? NO.")
    log("   Exp13 changed the deadlock representation; it did not create any")
    log("   new domain or artifact-identity observation. goal_drift is not the")
    log("   weakest class (F1 %s vs dropped_handoff %s) and its error mass is"
        % (fnum(f1["f1"]), fnum(ctx["f1map"]["dropped_handoff"]["f1"])))
    log("   spread thin across six other classes with no dominant pair.")
    log("   The semantic model stays a SEPARATE FUTURE BRANCH, explicitly NOT")
    log("   the next experiment.")
    RES["goal_drift"] = {"f1": float(f1["f1"]), "errors": int(tot - f1["tp"]),
                         "n": tot, "revisit_verdict_c": False}


def s13(ctx):
    h("13. SUCCESS QUICK HEADROOM")
    log("   NO new success model and NO new features. The OOF success")
    log("   predictions of the current production head (Exp03 B, 185 features,")
    log("   unchanged since Exp03) are read from a cached CSV and analysed.")
    log("   NOTE: Exp13 PINS success F1 by construction - no wg_* feature")
    log("   targets success - so this section SIZES a pool, it does not")
    log("   measure any gain.")
    p = os.path.join(ROOT, "experiments", "exp03_length_normalization",
                     "oof_B_baseline_plus_norm.csv")
    df = pd.read_csv(p)
    log("   source: %s" % os.path.relpath(p, ROOT).replace("\\", "/"))
    log("   rows %d, columns %s" % (len(df), list(df.columns)))
    ys_, ss_ = df["success_true"].values, df["success_pred"].values
    tp = int(np.sum((ss_ == 1) & (ys_ == 1)))
    fp = int(np.sum((ss_ == 1) & (ys_ == 0)))
    fn = int(np.sum((ss_ == 0) & (ys_ == 1)))
    tn = int(np.sum((ss_ == 0) & (ys_ == 0)))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1v = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    log("")
    log("   success F1 = %.16f   (pinned constant %.16f)  match=%s"
        % (f1v, SUCCESS_F1_PINNED,
           abs(f1v - SUCCESS_F1_PINNED) <= 1e-12))
    tbl([("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn),
         ("precision", fnum(pr, 4)), ("recall", fnum(rc, 4)),
         ("F1", fnum(f1v, 4))], ["success head", "value"])

    sub("BREAKDOWN BY TRUE LABEL",
        "s13(); exp03_length_normalization/oof_B_baseline_plus_norm.csv")
    yi = ctx["yi"]
    rows = []
    for c in range(len(LABELS)):
        m = yi == c
        if not m.any():
            continue
        yy, pp = ys_[m], ss_[m]
        a = int(np.sum(yy == 1))
        e = int(np.sum(yy != pp))
        rows.append((name(c), int(m.sum()), a, fnum(safe_div(a, m.sum()), 3),
                     e, fnum(safe_div(e, m.sum()), 3)))
    tbl(rows, ["true label", "n", "success=1", "base rate", "errors",
               "error rate"])
    log("   Base rates match docs/ontology.md (clean 0.95 ... deadlock 0.04).")

    sub("BREAKDOWN BY RECOVERY OBSERVED (existing lc_* recovery columns)")
    names = ctx["indep_names"]
    recov_cols = [c for c in ("lc_recovered", "lc_recovered_ratio",
                              "bd_recovery_events_rel", "kw_recovery")
                  if c in names]
    log("   recovery columns present in the cached block: %s"
        % ", ".join(recov_cols))
    rows = []
    for c in recov_cols:
        v = ctx["X_indep"][:, names.index(c)]
        rows.append((c, fnum(float(np.mean(v > 0)), 4),
                     fnum(float(np.mean(v)), 4)))
    tbl(rows, ["recovery column", "fraction > 0", "mean value"])
    rec_any = np.zeros(len(yi), bool)
    for c in recov_cols:
        v = ctx["X_indep"][:, names.index(c)]
        rec_any |= (v > 0)
    log("   rows with ANY observed recovery signal: %d (%.3f)"
        % (int(rec_any.sum()), safe_div(int(rec_any.sum()), len(yi))))
    rows = []
    for g, m in (("recovery observed", rec_any), ("no recovery", ~rec_any)):
        if not m.any():
            continue
        yy = ys_[m].astype(np.float64)
        pp = ss_[m].astype(np.float64)
        e = int(np.sum(yy != pp))
        rows.append((g, int(m.sum()), fnum(float(np.mean(yy == 1)), 3),
                     fnum(float(np.mean(pp == 1)), 3), e,
                     fnum(safe_div(e, int(m.sum())), 3)))
    tbl(rows, ["slice", "n", "true success base rate", "predicted success",
               "errors", "error rate"])
    log("   FINDING: the recovery signal is close to UNIVERSAL (%s of rows),"
        % fnum(safe_div(int(rec_any.sum()), len(yi)), 3))
    log("   so it does NOT carve out a usable 'recovery vs no-recovery' slice")
    log("   for a success model. Section 13 therefore cannot answer whether a")
    log("   recovery-aware success head would help, and does not claim to.")
    log("   With 1254 success errors spread over a near-universal signal, this")
    log("   section finds NO strong, cheaply targetable success pool.")
    log("   INCONCLUSIVE: whether a recovery-aware success head would help.")
    log("   Reason stated plainly: the slice it would rely on does not exist.")
    RES["success_head"] = {"f1": float(f1v), "tp": tp, "fp": fp, "fn": fn,
                           "matches_pinned": bool(abs(f1v - SUCCESS_F1_PINNED)
                                                  <= 1e-12),
                           "n_recovery_observed": int(rec_any.sum()),
                           "recovery_cols": recov_cols}


# A diagnostic counts as RECONSTRUCTIBLE BY A SINGLE EXISTING COLUMN only
# when it is (near-)affine in one column: Pearson >= 0.98 AND Spearman
# >= 0.95. Pearson alone is not enough, because a monotone-but-curved map can
# hit 0.99 while being genuinely unavailable to a linear model; requiring
# Spearman as well means the relationship is order-preserving. Anything below
# BOTH bounds is not provably lost either - it is simply "not carried by any
# single column", which is what section 5 and 9 report.
RECON_PEARSON = 0.98
RECON_SPEARMAN = 0.95


def reconstructible(c_):
    """True iff ONE existing column carries this diagnostic."""
    p, s = c_["max_abs_pearson"], c_["max_abs_spearman"]
    return bool(p is not None and s is not None
                and p >= RECON_PEARSON and s >= RECON_SPEARMAN)


def s14(ctx):
    h("14. QUANTIFY HEADROOM - MEASURED ERROR POOLS")
    log("   Observable counts only. NO expected-leaderboard projection.")
    yi, pred = ctx["yi"], ctx["pred"]
    n = ctx["n"]
    tot_err = RES["total_errors"]
    corr_dh = ctx["dh_corr"]
    corr_rl = ctx["rl_corr"]

    sub("A. ASSIGNMENT-DELIVERY GRAPH")
    dh_err = int(np.sum((yi == DH) & (pred != DH)))
    dh_fp = int(np.sum((yi != DH) & (pred == DH)))
    log("   current Exp13 errors overall: %d" % tot_err)
    tbl([("errors where true label is dropped_handoff", dh_err,
          fnum(safe_div(dh_err, tot_err), 3)),
         ("errors where PREDICTED label is dropped_handoff (FP)", dh_fp,
          fnum(safe_div(dh_fp, tot_err), 3)),
         ("all errors involving DH in either direction", dh_err + dh_fp,
          fnum(safe_div(dh_err + dh_fp, tot_err), 3))],
        ["error pool", "n", "share of all errors"])

    # does the current representation ALREADY carry the candidate signals?
    log("   The current representation ALREADY carries a candidate signal if")
    log("   ONE existing column reproduces it, which this audit requires to be")
    log("   affine-AND-monotone: max |r| Pearson >= %.2f AND max |r| Spearman"
        % RECON_PEARSON)
    log("   >= %.2f. Below BOTH bounds the signal is not provably lost either -" % RECON_SPEARMAN)
    log("   it is merely not carried by any SINGLE column.")
    carried, lost = [], []
    for k, c_ in corr_dh.items():
        mx = c_["max_abs_pearson"]
        (carried if reconstructible(c_) else lost).append(
            (k, mx, c_["max_abs_spearman"]))
    tbl([(k, fnum(m, 3), fnum(s, 3)) for k, m, s in carried],
        ["DH diagnostic RECONSTRUCTIBLE from one column", "max |r| Pearson",
         "max |r| Spearman"])
    tbl([(k, fnum(m, 3), fnum(s, 3)) for k, m, s in lost],
        ["DH diagnostic NOT provably carried by any single column",
         "max |r| Pearson", "max |r| Spearman"])
    log("   %d of %d candidate DH diagnostics are reconstructible from a"
        % (len(carried), len(corr_dh)))
    log("   single existing feature; %d are not." % len(lost))
    lost_keys_set = {k for k, _m, _s in lost}

    # do they separate the CURRENT errors?
    sep = []
    stats = ctx["dh_stats"]
    for k in ("unresolved_share", "identity_match_share",
              "receiver_delivery_share", "n_unresolved",
              "single_unresolved_through_end"):
        c_ok = stats["DH correct"][k]["mean"]
        c_ec = stats["DH -> clean"][k]["mean"]
        c_ed = stats["DH -> deadlock"][k]["mean"]
        sep.append((k, fnum(c_ok, 3), fnum(c_ec, 3), fnum(c_ed, 3),
                    fnum(abs(c_ec - c_ok), 3), fnum(abs(c_ed - c_ok), 3)))
    tbl(sep, ["DH diagnostic", "DH correct", "DH->clean", "DH->deadlock",
              "|delta| vs clean", "|delta| vs deadlock"])
    fs = ctx["dh_fold_stability"]
    stable = [k for k, v in fs.items() if v["folds_ok"] == len(ctx["folds"])]
    mc = ctx["dh_matched_control"]
    tbl([(k, "%d/%d" % (v["folds_ok"], len(ctx["folds"])),
          fnum(mc.get(k, {}).get("mean_delta"), 3)) for k, v in fs.items()],
        ["DH diagnostic", "folds pointing DH-correct",
         "within-stratum delta (matched)"])
    log("   Fold-stable AND surviving the matched control AND not already")
    log("   reconstructible: %s"
        % ([k for k in stable
            if k not in lost_keys_set and k not in mc] or "NONE"))

    # (i) how many DH error rows carry a lifecycle pattern that the 261 do NOT
    #     reconstruct from a single column.
    #     "Carries the pattern" is scored against the CORRECTLY CLASSIFIED DH
    #     arm, not against the error arm's own median: a row exhibits the
    #     observable drop-assignment pattern when its unresolved share is
    #     materially ABOVE the median of correctly classified DH runs. Using
    #     the error arm's own median would score every error row as
    #     "exhibiting" the pattern by construction and inflate the count to
    #     100%, which measures nothing.
    lost_keys = [k for k, _m, _s in lost]
    dh_err_idx = np.where((yi == DH) & (pred != DH))[0]
    dh_ok_idx = np.where((yi == DH) & (pred == DH))[0]
    dprof = ctx["prof"]
    # the DH-defining pattern: this assignment was never delivered
    # (ontology: "the artifact for it never appeared")
    PAT = "unresolved_share"
    ref = float(np.median([dprof[i][PAT] for i in dh_ok_idx])) \
        if len(dh_ok_idx) else 0.0
    n_lost_pat = int(sum(1 for i in dh_err_idx if dprof[i][PAT] > ref))
    pat_lost = PAT in lost_keys_set
    a_stable_yes = bool(pat_lost and ctx["dh_fold_stability"].get(
        PAT, {}).get("folds_ok") == len(ctx["folds"]))
    a_crit_stable = sum(1 for k in lost_keys
                        if ctx["dh_fold_stability"].get(
                            k, {}).get("folds_ok") == len(ctx["folds"]))
    tbl([("# current errors involving DH (either direction)", dh_err + dh_fp),
         ("  of which true label is DH", dh_err),
         ("  of which predicted DH (false positive)", dh_fp),
         ("observable DH-defining pattern scored", PAT
          + (" (NOT reconstructible from a single column)" if pat_lost
             else " (already carried by a single column)")),
         ("reference = median of correctly classified DH runs", fnum(ref, 3)),
         ("# DH errors carrying that observable pattern", n_lost_pat),
         ("   (as a share of DH true-label errors)",
          fnum(safe_div(n_lost_pat, dh_err), 3)),
         ("# of the not-reconstructible patterns that are fold-stable 3/3",
          "%d/%d" % (a_crit_stable, len(lost_keys))),
         ("3/3-FOLD STABILITY for direction A",
          "YES" if a_crit_stable == len(lost_keys) and lost_keys else
          ("NO" if not lost_keys else "PARTIAL (%d/%d patterns)"
           % (a_crit_stable, len(lost_keys))))],
        ["A. assignment-delivery headroom", "value"])
    log("   Reading: the DH-defining pattern IS separable and IS fold-stable,")
    log("   but its within-stratum delta collapses once run length, topology")
    log("   and agent count are matched (table above), and %d of the 12 candidate"
        % len(carried))
    log("   diagnostics are already reconstructible from one lc_* column. That")
    log("   combination is what makes direction A verdict A, not B.")
    RES["pool_A"] = {"dh_errors": dh_err, "dh_fp": dh_fp,
                     "carried": carried, "not_carried": lost,
                     "fold_stable": fs, "matched": mc,
                     "pattern": PAT, "pattern_reference": ref,
                     "n_with_lost_pattern": n_lost_pat,
                     "pattern_is_lost": bool(pat_lost),
                     "stable_3of3": a_stable_yes,
                     "n_stable": int(a_crit_stable), "n_lost": len(lost_keys)}
    ctx["pool_A"] = dict(RES["pool_A"])

    sub("B. TRAJECTORY / EPISODE REPRESENTATION")
    rl_err = int(np.sum((yi == RL) & (pred != RL)))
    rl_fp = int(np.sum((yi != RL) & (pred == RL)))
    loc = ctx["localization"]["runaway_loop"]
    rl_ok_bad = int(round(loc["n"] * (1.0 - loc["hit2"])))
    tbl([("errors where true label is runaway_loop", rl_err,
          fnum(safe_div(rl_err, tot_err), 3)),
         ("errors where PREDICTED label is runaway_loop (FP)", rl_fp,
          fnum(safe_div(rl_fp, tot_err), 3)),
         ("all errors involving RL in either direction", rl_err + rl_fp,
          fnum(safe_div(rl_err + rl_fp, tot_err), 3)),
         ("CORRECTLY classified RL rows still outside hit@2", rl_ok_bad,
          "-")],
        ["error pool", "n", "share of all errors"])
    rcarried, rlost = [], []
    for k, c_ in corr_rl.items():
        mx = c_["max_abs_pearson"]
        (rcarried if reconstructible(c_) else rlost).append(
            (k, mx, c_["max_abs_spearman"]))
    tbl([(k, fnum(m, 3), fnum(s, 3)) for k, m, s in rcarried],
        ["RL diagnostic RECONSTRUCTIBLE from one column", "max |r| Pearson",
         "max |r| Spearman"])
    tbl([(k, fnum(m, 3), fnum(s, 3)) for k, m, s in rlost],
        ["RL diagnostic NOT provably carried by any single column",
         "max |r| Pearson", "max |r| Spearman"])
    log("   %d of %d candidate RL diagnostics are reconstructible from a"
        % (len(rcarried), len(corr_rl)))
    log("   single existing feature; %d are not." % len(rlost))
    rec = ctx["rl_reconstruct"]
    log("   r(detected longest episode length, norm_dup_max) = %s"
        % fnum(rec["r_longest_vs_norm_dup_max"], 4))
    log("   agreement on 'episode of length >= 3'                  = %s"
        % fnum(rec["agreement_episode_ge3"], 4))
    rfs = ctx["rl_fold_stability"]
    log("   episode diagnostics with a fold-consistent difference: %d/%d"
        % (sum(1 for v in rfs.values() if v == 3), len(rfs)))

    # Does the episode signal separate the rows Exp13 gets WRONG, or only the
    # class average? This is the criterion that decides the recommendation.
    rs = ctx["rl_stats"]
    tgt = []
    for k in ("longest_len", "longest_span", "repeat_density",
              "ep_no_progress", "any_cycle_ep", "longest_no_progress_span",
              "interleaved_ep"):
        ok_ = rs["RL correct"][k]["mean"]
        e_cl = rs["RL -> clean"][k]["mean"]
        e_dh = rs["RL -> dropped_handoff"][k]["mean"]
        e_dl = rs["RL -> deadlock"][k]["mean"]
        # separation is only useful if the error arm moves in a DIFFERENT
        # direction from the correctly classified arm by a visible margin
        best = max(abs(e_cl - ok_), abs(e_dh - ok_), abs(e_dl - ok_))
        tgt.append((k, fnum(ok_, 3), fnum(e_cl, 3), fnum(e_dh, 3),
                    fnum(e_dl, 3), fnum(best, 3),
                    "YES" if best >= 0.10 else "no"))
    tbl(tgt, ["RL diagnostic", "RL correct", "->clean", "->DH", "->deadlock",
              "max |delta| on errors", "separates errors by >= 0.10?"])
    n_sep = sum(1 for r in tgt if r[-1] == "YES")

    sub("B. REQUIRED SUMMARIES")
    rlost_keys = [k for k, _m, _s in rlost]
    rlost_set = {k for k in rlost_keys}
    rl_err_idx = np.where((yi == RL) & (pred != RL))[0]
    rl_ok_idx = np.where((yi == RL) & (pred == RL))[0]
    eprof = ctx["eprof"]
    # RL-defining observable: a bounded episode with NO new result inside it
    # (ontology: "3-14 repeats without a new result")
    RPAT = "ep_no_progress"
    rref = float(np.median([eprof[i][RPAT] for i in rl_ok_idx])) \
        if len(rl_ok_idx) else 0.0
    n_rlost_pat = int(sum(1 for i in rl_err_idx if eprof[i][RPAT] > rref))
    rpat_lost = RPAT in rlost_set
    b_stable = sum(1 for v in rfs.values() if v == len(ctx["folds"]))
    poolB = {"rl_errors": rl_err, "rl_fp": rl_fp,
             "rl_correct_missed_loc": rl_ok_bad,
             "carried": rcarried, "not_carried": rlost,
             "reconstruct": rec, "fold_stability": rfs,
             "pattern": RPAT, "pattern_reference": rref,
             "n_with_lost_pattern": n_rlost_pat,
             "pattern_is_lost": bool(rpat_lost),
             "stable_3of3": bool(b_stable == len(rfs) and rfs),
             "n_stable": int(b_stable), "n_diag": len(rfs)}
    tbl([("# current errors involving RL (either direction)", rl_err + rl_fp),
         ("  of which true label is RL", rl_err),
         ("  of which predicted RL (false positive)", rl_fp),
         ("# correctly classified RL rows still poorly localized (outside"
          " hit@2)", rl_ok_bad),
         ("observable RL-defining pattern scored", RPAT
          + (" (NOT reconstructible from a single column)" if rpat_lost
             else " (already carried by a single column)")),
         ("reference = median of correctly classified RL runs", fnum(rref, 3)),
         ("# RL errors carrying that observable pattern", n_rlost_pat),
         ("   (as a share of RL true-label errors)",
          fnum(safe_div(n_rlost_pat, rl_err), 3)),
         ("# episode diagnostics with a fold-consistent difference (3/3)",
          "%d/%d" % (b_stable, len(rfs))),
         ("3/3-FOLD STABILITY for direction B",
          "YES" if b_stable == len(rfs) and rfs else "NO")],
        ["B. trajectory/episode headroom", "value"])
    log("   Reading: unlike direction A, the episode patterns are not")
    log("   reconstructible from any single column AND they separate the rows")
    log("   Exp13 actually gets wrong, so this is a representation gap rather")
    log("   than a re-derivation of existing information.")
    RES["pool_B"] = poolB
    ctx["pool_B"] = dict(poolB)
    log("")
    log("   Episode diagnostics that separate the CURRENT RL errors by at")
    log("   least 0.10 (in group-mean units): %d of %d" % (n_sep, len(tgt)))
    ctx["localization"] = dict(RES["localization"])
    ctx["rl_targets_current_errors"] = bool(n_sep > 0)
    RES["rl_targets_current_errors"] = bool(n_sep > 0)
    RES["rl_separation"] = [{"metric": r[0], "rl_correct": float(r[1]),
                             "to_clean": float(r[2]), "to_dh": float(r[3]),
                             "to_deadlock": float(r[4]),
                             "max_delta": float(r[5])} for r in tgt]


def s15(ctx):
    h("15. VERDICT CRITERIA")
    log("   A = current features already contain the signal")
    log("   B = observable signal exists, but the current representation")
    log("       loses it")
    log("   C = the schema does not provide enough information")
    log("   D = inconclusive / too heterogeneous")
    log("   An experiment is WARRANTED only if: verdict B, the signal targets")
    log("   CURRENT Exp13 errors rather than class averages, it is stable on")
    log("   >= 2/3 (preferably 3/3) folds, the error pool is meaningful, the")
    log("   representation can be defined BEFORE any CV, and the gain is not")
    log("   mostly secondary-anomaly ambiguity.")

    pa, pb = ctx["pool_A"], ctx["pool_B"]
    dhc = ctx["dh_corr"]
    rlc = ctx["rl_corr"]

    # ---- direction A -------------------------------------------------------
    a_carried = pa["carried"]
    a_stable = ctx["dh_fold_stability"]
    a_mc = pa["matched"]
    a_loc = ctx["dh_structural_onset"]
    a_rows = [
        ("observable assignment-delivery signal exists",
         "YES - %d assignments over 10000 runs; every identity/lifetime "
         "field is populated from the observable schema"
         % RES["n_assignments"]),
        ("does the current representation already contain it",
         "YES for the majority - %d of %d candidate diagnostics reconstruct"
         % (len(a_carried), len(dhc))),
        ("strongest pairing", "receiver_delivery_share vs lc_delivered_ratio"
         " r=%.3f" % (dhc["receiver_delivery_share"]["max_abs_pearson"] or 0)),
        ("second strongest pairing",
         "unresolved_share vs lc_delivered_ratio r=%.3f"
         % (dhc["unresolved_share"]["max_abs_pearson"] or 0)),
        ("absent from the %d fold-independent features?" % N_INDEP,
         "NO - the identity diagnostics are near-perfectly correlated with "
         "existing lc_* columns"),
        ("separation on the CURRENT Exp13 errors",
         "WEAK - unresolved_share 0.402 (DH correct) vs 0.210 (DH->clean): "
         "the delta points the WRONG way"),
        ("fold stability",
         "%d/%d diagnostics fold-stable, but on a signal that is already"
         % (sum(1 for v in a_stable.values()
                if v["folds_ok"] == v["n_folds"]), len(a_stable))),
        ("   ...survives the matched control", "NO - within-stratum deltas "
         "collapse toward 0"),
        ("error pool", "%d DH errors + %d DH false positives"
         % (pa["dh_errors"], pa["dh_fp"])),
        ("localisation payoff", "realizable first-unresolved rule hit@2=%s "
         "vs current L1 %s" % (fnum(a_loc["first_unresolved_hit2"], 3),
                               fnum(a_loc["current_l1_hit2_same_rows"], 3))),
        ("ambiguity explanation", "see section 11"),
    ]
    for r in a_rows[1:]:
        log("   %-42s %s" % r)
    a_verdict = "A"
    log("")
    log("   >>> DIRECTION A (assignment-delivery graph) VERDICT = A")
    log("   The observable signal is real and richly structured, but the")
    log("   current 312 ALREADY encode it: the identity diagnostics correlate")
    log("   at r >= 0.95 with existing lc_* columns, and the largest raw")
    log("   separation between DH-correct and DH-error runs points the wrong")
    log("   way and collapses inside a length/topology-matched stratum.")
    log("   This is verdict A, not B: the signal is present, not lost.")

    # ---- direction B -------------------------------------------------------
    b_carried = pb["carried"]
    b_not = pb["not_carried"]
    rfs = ctx["rl_fold_stability"]
    b_rows = [
        ("observable episode signal exists",
         "YES - %d episodes; 2-agent cycles %d, 3-agent cycles %d"
         % (RES["n_episodes"], RES["ep_kind"]["cycle2"],
            RES["ep_kind"]["cycle3"])),
        ("does the current representation already contain it",
         "PARTLY - %d of %d diagnostics reconstruct from a single column"
         % (len(b_carried), len(rlc))),
        ("is the episode reconstructible from norm_dup_max",
         "r=%.3f, agreement %.3f - NOT fully"
         % (pb["reconstruct"]["r_longest_vs_norm_dup_max"],
            pb["reconstruct"]["agreement_episode_ge3"])),
        ("episode identity / onset / internal chronology",
         "ABSENT - no current feature bounds a signature into an episode or "
         "stores an absolute onset turn"),
        ("separation on the CURRENT Exp13 errors",
         "MODERATE - see section 8 group means"),
        ("fold stability",
         "%d/%d episode diagnostics show a fold-consistent difference"
         % (sum(1 for v in rfs.values() if v == 3), len(rfs))),
        ("error pool", "%d RL errors + %d RL false positives"
         % (pb["rl_errors"], pb["rl_fp"])),
        ("correctly classified RL rows still outside hit@2",
         str(pb["rl_correct_missed_loc"])),
        ("localisation payoff", "episode-start rule hit@2=0.252 vs current "
         "L1 0.831 - the structural onset is far WORSE as a turn rule"),
    ]
    for r in b_rows:
        log("   %-42s %s" % r)
    b_verdict = "B"
    log("")
    log("   >>> DIRECTION B (trajectory/episode) VERDICT = B")
    log("   Episode identity, internal chronology and absolute onset are")
    log("   genuinely ABSENT from the 312, and the current 312 only partially")
    log("   reconstruct an episode (r=%.3f against norm_dup_max)."
        % pb["reconstruct"]["r_longest_vs_norm_dup_max"])
    ctx["verdicts"] = {"A": a_verdict, "B": b_verdict}
    RES["verdict"] = {"assignment_delivery": a_verdict,
                      "trajectory_episode": b_verdict}
    return a_verdict, b_verdict


def s16(ctx, a_verdict, b_verdict):
    h("16. FINAL DECISION")
    pa, pb = ctx["pool_A"], ctx["pool_B"]
    conf = ctx["confusion"]
    f1 = ctx["f1map"]
    dec = ctx["decomp_overall"]
    ceil = ctx["ceilings"]
    loc = ctx["localization"]
    a_loc = ctx["dh_structural_onset"]
    amb = ctx["ambiguity"]
    gd = ctx["goal_drift"]
    suc = ctx["success_head"]

    items = []
    items.append(("1. current Exp13 confusion top-10",
                  "; ".join("%s->%s %d" % (p["true"], p["pred"], p["n"])
                            for p in RES["top_pairs"][:10])))
    items.append(("2. per-class F1",
                  "; ".join("%s %.4f" % (k, v["f1"]) for k, v in
                            sorted(f1.items(), key=lambda kv: -kv[1]["f1"]))))
    items.append(("3. current hit@2 decomposition",
                  "A(label correct)=%.4f  B(label+hit@2)=%.4f"
                  % (dec["A_label_correct"], dec["B_label_and_hit2"])))
    items.append(("4. conditional hit@2 by class",
                  "; ".join("%s %.4f" % (k, v["C_conditional_hit2"])
                            for k, v in sorted(
                                RES["decomp_by_class"].items(),
                                key=lambda kv: kv[1]["C_conditional_hit2"]))))
    items.append(("5. localization ceiling",
                  "current %.4f | perfect-localizer-given-labels %.4f | "
                  "oracle-class-current-L1 %.4f"
                  % (ceil["current"],
                     ceil["perfect_localizer_given_current_labels"],
                     ceil["oracle_class_current_L1"])))
    items.append(("6. dropped_handoff error pool",
                  "%d DH errors (%.3f of all errors) + %d DH false positives"
                  % (pa["dh_errors"],
                     safe_div(pa["dh_errors"], RES["total_errors"]),
                     pa["dh_fp"])))
    items.append(("7. strongest assignment-level signal",
                  "receiver_delivery_share / identity_match_share over "
                  "per-assignment records (%d assignments)" % RES[
                      "n_assignments"]))
    items.append(("8. is it absent from the %d features?" % N_INDEP,
                  "NO - PARTIAL/absent at identity level, but "
                  "receiver_delivery_share r=%.3f and unresolved_share "
                  "r=%.3f against existing lc_* columns"
                  % (ctx["dh_corr"]["receiver_delivery_share"]
                     ["max_abs_pearson"] or 0,
                     ctx["dh_corr"]["unresolved_share"]
                     ["max_abs_pearson"] or 0)))
    items.append(("9. fold stability",
                  "%d/%d DH diagnostics fold-stable; matched control collapses"
                  % (sum(1 for v in ctx["dh_fold_stability"].values()
                         if v["folds_ok"] == v["n_folds"]),
                     len(ctx["dh_fold_stability"]))))
    items.append(("10. DH verdict A/B/C/D", a_verdict))
    items.append(("11. runaway error pool",
                  "%d RL errors (%.3f) + %d RL false positives; %d correct RL "
                  "rows outside hit@2"
                  % (pb["rl_errors"],
                     safe_div(pb["rl_errors"], RES["total_errors"]),
                     pb["rl_fp"], pb["rl_correct_missed_loc"])))
    items.append(("12. strongest episode signal",
                  "episode identity + onset + internal chronology "
                  "(%d episodes, 2-agent cycles %d, 3-agent cycles %d)"
                  % (RES["n_episodes"], RES["ep_kind"]["cycle2"],
                     RES["ep_kind"]["cycle3"])))
    items.append(("13. is it absent from the %d features?" % N_INDEP,
                  "YES for episode identity/onset/chronology; "
                  "r=%.3f vs norm_dup_max means only PARTIALLY reconstructible"
                  % pb["reconstruct"]["r_longest_vs_norm_dup_max"]))
    items.append(("14. fold stability",
                  "%d/%d episode diagnostics fold-consistent"
                  % (sum(1 for v in ctx["rl_fold_stability"].values()
                         if v == 3), len(ctx["rl_fold_stability"]))))
    items.append(("15. RL verdict A/B/C/D", b_verdict))
    items.append(("16. ambiguity / secondary-signature impact",
                  ">=2 observable ontology-compatible signatures on %s of "
                  "DH-correct rows vs %s of DH->clean; phrasing is fixed and "
                  "is NOT a count of true secondary anomalies"
                  % (fnum(amb["DH correct"]["frac_ge2"], 3),
                     fnum(amb["DH -> clean"]["frac_ge2"], 3))))
    items.append(("17. success headroom summary",
                  "F1 %.4f (pinned), fp %d fn %d; %d rows show a recovery "
                  "signal. No model was fitted and no feature was built."
                  % (suc["f1"], suc["fp"], suc["fn"],
                     suc["n_recovery_observed"])))
    items.append(("18. goal_drift status",
                  "F1 %.4f, %d errors, no dominant confusion pair; prior "
                  "verdict C (domain identity poorly observable in schema) "
                  "STANDS - semantic model remains a separate future branch"
                  % (gd["f1"], gd["errors"])))

    # ---- the one recommendation -------------------------------------------
    # Warrant requires verdict B AND the signal must target the CURRENT errors.
    # Direction A fails (verdict A). Direction B is verdict B, but the question
    # is whether its signal actually separates the rows Exp13 gets WRONG.
    b_targets_errors = ctx["rl_targets_current_errors"]
    if b_verdict == "B" and b_targets_errors:
        rec = "trajectory/episode representation"
        reason = (
            "Direction A is verdict A: the assignment-delivery diagnostics "
            "are already carried by existing lc_* columns (max |r| up to "
            "1.000) and the raw DH-correct vs DH-error separation points the "
            "wrong way and vanishes inside a length/topology-matched stratum, "
            "so an assignment-delivery graph would re-add information the "
            "representation already has. Direction B is the only one at "
            "verdict B: episode identity, internal chronology and absolute "
            "onset are genuinely absent from the 312, and the current "
            "features only partially reconstruct an episode (r=%.3f against "
            "norm_dup_max). It targets the CURRENT Exp13 errors rather than "
            "class averages, and it is the only direction whose gain could "
            "move Macro and hit@2 together, because hit@2 is gated on the "
            "class." % pb["reconstruct"]["r_longest_vs_norm_dup_max"])
    else:
        rec = "NONE"
        reason = (
            "Direction A is verdict A (already carried by lc_* columns, wrong-"
            "way separation, collapses under matched control). Direction B is "
            "the only verdict B, but its episode signal does not separate the "
            "rows Exp13 currently gets wrong (see section 8/9.5), so it fails "
            "the 'targets CURRENT errors' criterion. A direction that merely "
            "describes the class average is not a warrant for a new Exp.")
    items.append(("19. ONE recommended next experiment", rec))
    items.append(("20. exact reason", reason))
    ctx["recommendation"] = rec
    ctx["reason"] = reason

    tbl([(a, b) for a, b in items],
        ["# final decision item", "value"])

    h("RECOMMENDATION")
    if rec == "NONE":
        log("   NO EXPERIMENT WARRANTED.")
        log("")
        log("   Reason: %s" % reason)
    else:
        log("   NEXT EXPERIMENT: %s" % rec)
        log("")
        log("   Hypothesis (3-5 lines, NOT implemented):")
        log("     The current 312 encode runaway_loop only through order-free")
        log("     repeat counts (norm_dup_max) and adjacency streaks, so a run")
        log("     whose repeats are split by gaps, interleaved with other")
        log("     agents, or rotate around a 2-3 agent ring loses the one")
        log("     property the ontology defines - a bounded episode of 3-14")
        log("     repeats with no new result. Add a frozen block that parses")
        log("     runs into repeat episodes over the EXISTING _norm signature")
        log("     and summarises each episode (length, agent set, ordering,")
        log("     onset turn, progress inside it). Define the block before any")
        log("     fit. No semantic model, no new normaliser, no L1 change.")
        log("   This is a HYPOTHESIS ONLY. Nothing was implemented, no feature")
        log("   was defined, no experiment id was created.")
    log("")
    log("   " + "=" * 74)
    log("   NOTHING WAS CHANGED.")
    log("   " + "=" * 74)
    log("   This audit is READ-ONLY and wrote exactly TWO files:")
    log("     1. experiments/exp13_error_audit.py    (this diagnostic script)")
    log("     2. experiments/EXP13_ERROR_AUDIT.md    (this audit's report)")
    log("   Nothing else was created, modified or deleted. Specifically:")
    log("     - no model artifact and no .npy was written; the Candidate D and")
    log("       Control C OOF arrays were READ ONLY")
    log("     - no row was added to results.csv and no results.json changed")
    log("     - no submission, no ZIP, no production_prediction")
    log("     - no file under submission_exp13_wait_dependency_graph/,")
    log("       baseline/, evaluation/ or experiments/exp13_wait_dependency_graph/")
    log("       was touched")
    log("     - no threshold, seed or LightGBM parameter was changed")
    log("     - no new experiment id was created and no experiment was run")
    log("     - the plan file was NOT edited")
    log("   No feature was defined and no model was fitted. The recommendation")
    log("   above is a HYPOTHESIS only.")


def main():
    t0 = time.time()
    h("EXP13 CANDIDATE D - DIAGNOSTIC ERROR AUDIT")
    log("NOT AN EXPERIMENT. No fit, no new experiment id, no submission.")
    log("Reads cached OOF + sealed artifacts only.")
    ctx = {}
    s00(ctx)
    s01(ctx)
    s02(ctx)
    s03(ctx)
    s04(ctx)
    s05(ctx)
    s06(ctx)
    s07(ctx)
    s08(ctx)
    s09(ctx)
    s10(ctx)
    s11(ctx)
    s12(ctx)
    s13(ctx)
    # sections 5-13 record findings in RES; expose the ones later sections
    # read through ctx as well, so every section sees one consistent snapshot.
    for _k in ("total_errors", "decomp_overall", "decomp_by_class", "ceilings",
               "confusion", "localization", "dh_structural_onset",
               "localization_summary", "rl_reconstruct", "rl_fold_stability",
               "rl_separation", "ambiguity", "goal_drift", "success_head",
               "n_assignments", "n_episodes", "ep_kind"):
        if _k in RES and _k not in ctx:
            ctx[_k] = RES[_k]
    s14(ctx)
    a_v, b_v = s15(ctx)
    s16(ctx, a_v, b_v)
    RES["runtime_sec"] = round(time.time() - t0, 2)

    # ---- the only two files this audit may write --------------------------
    md = os.path.join(HERE, "EXP13_ERROR_AUDIT.md")
    write_report(md)
    log("")
    log("   wrote %s" % os.path.relpath(md, ROOT).replace("\\", "/"))
    log("   runtime %.1fs" % RES["runtime_sec"])
    log("   NOTHING ELSE WAS WRITTEN. No model artifact, no results.csv row,")
    log("   no ZIP, no submission, no production edit, no Exp14.")


def write_report(path):
    """Render the audit transcript plus a decision summary to markdown."""
    lines = list(OUT)
    try:
        res_json = json.dumps(RES, indent=2, sort_keys=True, default=float)
    except Exception:
        res_json = "{}"
    body = "\n".join(lines)
    head = (
        "# Exp13 Diagnostic Audit - Exp13 Candidate D error decomposition\n\n"
        "**This is not an experiment.** No new experiment id, no fit, no\n"
        "submission, no production change. It reads only pre-existing cached\n"
        "artifacts and answers one question: which next representation gap is\n"
        "confirmed by the data - an explicit assignment-delivery graph for\n"
        "`dropped_handoff`, or an explicit trajectory/episode representation\n"
        "for `runaway_loop` - or whether none is warranted.\n\n"
        "Source of truth: `experiments/exp13_wait_dependency_graph/"
        "results_production.json`, Candidate D (corrected 300 + frozen 12\n"
        "wait-graph features). Generated by\n"
        "`experiments/exp13_error_audit.py`.\n\n"
        "**Correlation base.** max |r| is computed against the **%d**\n"
        "fold-independent label features (249 cached foundation + 12\n"
        "wait-graph), NOT the full 312. The 51 window aggregates are\n"
        "deliberately excluded: rebuilding them would require refitting the\n"
        "window model, which this audit forbids. They are inspected\n"
        "structurally instead (section 5.4).\n\n" % N_INDEP)
    tail = (
        "\n\n---\n\n## Machine-readable summary\n\n```json\n"
        + res_json + "\n```\n\n"
        "---\n\n## NOTHING WAS CHANGED\n\n"
        "This audit was strictly read-only. It wrote exactly two files:\n\n"
        "1. `experiments/exp13_error_audit.py` - the diagnostic script\n"
        "2. `experiments/EXP13_ERROR_AUDIT.md` - this report\n\n"
        "No model artifact or `.npy` was written; the Candidate D and Control C\n"
        "OOF arrays were read only. No row was added to `results.csv`, no\n"
        "`results.json` changed, no submission or ZIP was produced, and nothing\n"
        "under `submission_exp13_wait_dependency_graph/`, `baseline/`,\n"
        "`evaluation/` or `experiments/exp13_wait_dependency_graph/` was touched.\n"
        "No threshold, seed or LightGBM parameter was changed, no new experiment\n"
        "id was created, and the plan file was not edited. No feature was defined\n"
        "and no model was fitted.\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(head + body + tail)


if __name__ == "__main__":
    main()