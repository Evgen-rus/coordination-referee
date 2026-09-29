"""Honest baseline: run the REAL exp06b logic, timed, writing into .perf/.

Same code path as the experiment (same build(), same folds, same lgb_label(),
same metric calls) -- only the output directory differs, so the repo stays clean.

    python .perf/bench_real_baseline.py base 0
"""
from __future__ import annotations

import gc
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "experiments", "exp06b_stability")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for p in (os.path.join(ROOT, "baseline"),
          os.path.join(ROOT, "evaluation"),
          os.path.join(ROOT, "experiments", "exp03_length_normalization"),
          os.path.join(ROOT, "experiments", "exp05_handoff_lifecycle")):
    sys.path.insert(0, p)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)

_c = _load("common", os.path.join(EXP, "common.py"))
from metrics import fault_turn_hit_at_k          # noqa: E402
from localize import localize                      # noqa: E402

COLS = _c.COLS
KEYS = list(COLS.keys())
macro_f1 = _c.macro_f1
prf = _c.prf
LABELS = _c.LABELS


def sl_macro(ys, pred, mask, min_n=30):
    idx = np.where(mask)[0]
    if len(idx) < min_n:
        return None
    return macro_f1([ys[i] for i in idx], [pred[i] for i in idx])


def run_one_seed(build, seed, outdir, label):
    t_start = time.perf_counter()
    c = build()
    t_build = time.perf_counter() - t_start
    runs, mats, yi, ys = c["runs"], c["mats"], c["yi"], c["ys"]
    n, nmsg, fturn = c["n"], c["nmsg"], c["fturn"]
    rob_m = c["SL"]["hard/robustness"]
    dhlong_m = c["DH_SL"]["long(>p66)"]

    t0 = time.perf_counter()
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed)
                  .split(np.zeros(n), yi))
    oof = {k: np.zeros(n, dtype=int) for k in KEYS}
    fold_m = {k: [] for k in KEYS}
    for f, (tr, va) in enumerate(splits):
        for key in KEYS:
            m = _c.lgb_label().fit(mats[key].iloc[tr], yi[tr])
            p = m.predict(mats[key].iloc[va])
            oof[key][va] = p
            fold_m[key].append(macro_f1([ys[i] for i in va],
                                       [LABELS[j] for j in p]))
            del m
        gc.collect()
    t_models = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = {}
    for key in KEYS:
        pred = [LABELS[i] for i in oof[key]]
        po = np.array(pred, dtype=object)
        turns = [localize(runs[i], pred[i]) for i in range(n)]
        h2 = fault_turn_hit_at_k(list(ys), pred, list(fturn), turns)
        res[key] = {"macro_f1": macro_f1(ys, pred),
                    "robustness_f1": sl_macro(ys, po, rob_m),
                    "fault_turn_hit2": h2,
                    "fold_macro_f1": [float(x) for x in fold_m[key]]}
        pd.DataFrame({"run_id": c["train"]["run_id"], "y_true": ys,
                      "y_pred": pred,
                      "n_messages": nmsg}).to_csv(
            os.path.join(outdir, "%s_oof_%s_seed%d.csv" % (label, key, seed)),
            index=False)
    t_post = time.perf_counter() - t0
    total = t_build + t_models + t_post
    print("[%s] seed=%d  build %.1fs | models %.1fs | post %.1fs | TOTAL %.1fs"
          % (label, seed, t_build, t_models, t_post, total), flush=True)
    return res, oof, total


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "base"
    seeds = [int(x) for x in sys.argv[2:]] or [0]
    outdir = os.path.join(HERE, "out_" + label)
    os.makedirs(outdir, exist_ok=True)
    import profile_exp06b as P
    grand = 0.0
    for s in seeds:
        _, _, tot = run_one_seed(P.do_build, s, outdir, label)
        grand += tot
    print("[%s] SEEDS %s TOTAL %.1fs" % (label, seeds, grand), flush=True)


if __name__ == "__main__":
    main()
