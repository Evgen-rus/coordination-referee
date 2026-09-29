"""Profiling harness for the exp06b pipeline. Writes nothing into the repo.

Replicates ``exp06b_stability/run_validation.py`` / ``common.build()`` call for
call, but instrumented per stage, so time can be attributed without editing any
experiment file.

Usage:
    python .perf/profile_exp06b.py --phase
    python .perf/profile_exp06b.py --stage build
    python .perf/profile_exp06b.py --stage lgb
    python .perf/profile_exp06b.py --stage localize
    python .perf/profile_exp06b.py --phase --seeds 0
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import io
import os
import pstats
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "experiments", "exp06b_stability")
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# exp06b's `common` must win over exp06's identically named module.
for p in (os.path.join(ROOT, "baseline"),
          os.path.join(ROOT, "evaluation"),
          os.path.join(ROOT, "experiments", "exp03_length_normalization"),
          os.path.join(ROOT, "experiments", "exp05_handoff_lifecycle")):
    sys.path.insert(0, p)
sys.path.insert(0, EXP)

_c = _load("common", os.path.join(EXP, "common.py"))
COLS, N_FEATURES, N_FOLDS = _c.COLS, _c.N_FEATURES, _c.N_FOLDS
lgb_label, log, macro_f1, prf, LABELS = (_c.lgb_label, _c.log, _c.macro_f1,
                                         _c.prf, _c.LABELS)
from features import extract_features, parse_run             # noqa: E402
from metrics import binary_f1, fault_turn_hit_at_k            # noqa: E402
from localize import localize                                 # noqa: E402
import new_features as nf                                     # noqa: E402
import lifecycle as lc                                        # noqa: E402
ft = _load("exp06_features",
           os.path.join(ROOT, "experiments", "exp06_temporal_dynamics", "features.py"))

KEYS = list(COLS.keys())
T = {}


def tick(name, t0):
    T[name] = T.get(name, 0.0) + (time.perf_counter() - t0)
    return time.perf_counter()


def do_build(verbose=True):
    """Exactly common.build()'s body, but stage-instrumented."""
    t = time.perf_counter()
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    t = tick("1. read_csv", t)
    ys = train["label"].values.astype(object)
    yi = np.array([LABELS.index(l) for l in ys], dtype=int)
    t = tick("2. labels", t)

    runs = [parse_run(r) for r in train.to_dict("records")]
    t = tick("3. parse_run (json.loads)", t)

    base_rows = [extract_features(r) for r in runs]
    t = tick("4. extract_features (122)", t)
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    t = tick("5. DataFrame(122)", t)

    Xn = nf.build_matrix(runs, base_rows)
    t = tick("6. exp03 norm features (63)", t)
    del base_rows

    XA = pd.concat([Xb, Xn, lc.build_matrix(runs)[lc.LIFECYCLE_NAMES]], axis=1)
    t = tick("7. exp05 lifecycle (54)", t)
    X6 = ft.build_matrix(runs)
    t = tick("8. exp06 temporal (29)", t)

    mats = {k: (pd.concat([XA, X6[c]], axis=1) if c else XA)
            for k, c in COLS.items()}
    for k, m in mats.items():
        assert m.shape[1] == N_FEATURES[k], (k, m.shape[1], N_FEATURES[k])
    t = tick("9. concat -> 4 matrices", t)

    e3 = pd.read_csv(os.path.join(EXP, "..", "exp03_length_normalization",
                                  "oof_B_baseline_plus_norm.csv"))
    assert (e3["run_id"].values == train["run_id"].values).all()
    success_f1 = float(binary_f1(list(train["success"].values),
                                 list(e3["success_pred"].values)))
    t = tick("10. pin success_f1", t)

    nmsg = Xb["n_messages"].values
    q33, q66 = np.quantile(nmsg, 1 / 3), np.quantile(nmsg, 2 / 3)
    SL = {"hard/robustness": (nmsg > np.median(nmsg)) &
          ((Xb["topo_mesh"].values > 0) | (Xb["topo_blackboard"].values > 0)),
          "topo_mesh": Xb["topo_mesh"].values > 0}
    DH_SL = {"long(>p66)": (ys == "dropped_handoff") & (nmsg > q66)}
    t = tick("11. slice masks", t)

    c = {"train": train, "runs": runs, "mats": mats, "yi": yi, "ys": ys,
         "n": len(yi), "success_f1": success_f1,
         "fturn": train["fault_turn"].values, "nmsg": nmsg,
         "SL": SL, "DH_SL": DH_SL}
    if verbose:
        report()
    return c


def report():
    total = sum(v for v in T.values() if v > 0)
    log("=== PHASE PROFILE (total %.2fs) ===" % total)
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        if v <= 0:
            continue
        log("  %-40s %8.2fs  %5.1f%%" % (k, v, 100 * v / total))
    return total


def phase_profile(seeds, write_oof=False):
    c = do_build(verbose=False)
    runs, mats, yi, ys = c["runs"], c["mats"], c["yi"], c["ys"]
    n, nmsg, fturn = c["n"], c["nmsg"], c["fturn"]
    rob_m = c["SL"]["hard/robustness"]
    dhlong_m = c["DH_SL"]["long(>p66)"]

    for seed in seeds:
        t = time.perf_counter()
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
                      .split(np.zeros(n), yi))
        oof = {k: np.zeros(n, dtype=int) for k in KEYS}
        fold_m = {k: [] for k in KEYS}
        t = tick("12. StratifiedKFold splits", t)
        for f, (tr, va) in enumerate(splits):
            for key in KEYS:
                m = lgb_label().fit(mats[key].iloc[tr], yi[tr])
                t = tick("13. LGB .fit()", t)
                p = m.predict(mats[key].iloc[va])
                t = tick("14. LGB .predict()", t)
                oof[key][va] = p
                fold_m[key].append(macro_f1([ys[i] for i in va],
                                           [LABELS[j] for j in p]))
                del m
            gc.collect()
        t = tick("15. gc.collect()", t)
        for key in KEYS:
            t = time.perf_counter()
            pred = [LABELS[i] for i in oof[key]]
            po = np.array(pred, dtype=object)
            idx = np.where(rob_m)[0]
            rf = macro_f1([ys[i] for i in idx], [po[i] for i in idx])
            t = tick("16. macro_f1 (python loop)", t)
            turns = [localize(runs[i], pred[i]) for i in range(n)]
            t = tick("17. localize() x10000", t)
            h2 = fault_turn_hit_at_k(list(ys), pred, list(fturn), turns)
            t = tick("18. fault_turn_hit_at_k", t)
            prf(ys, po, "dropped_handoff")
            prf(ys, po, "dropped_handoff", dhlong_m)
            prf(ys, po, "deadlock")
            idx2 = np.where(c["SL"]["topo_mesh"])[0]
            macro_f1([ys[i] for i in idx2], [po[i] for i in idx2])
            t = tick("19. prf / slice macro", t)
            if write_oof:
                pd.DataFrame({"run_id": c["train"]["run_id"], "y_true": ys,
                              "y_pred": pred, "n_messages": nmsg}).to_csv(
                    os.path.join(HERE, "_prof_oof_%s_seed%d.csv" % (key, seed)),
                    index=False)
                t = tick("20. write oof csv", t)
    report()


def cprof(fn, label, sort="tottime", n=30):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(n)
    log("=== cProfile %s (sort=%s) ===" % (label, sort))
    print(s.getvalue(), flush=True)


def stage_build():
    def go():
        T.clear()
        do_build(verbose=False)
    cprof(go, "build() feature generation", "tottime", 30)


def stage_lgb():
    c = do_build(verbose=False)
    t = time.perf_counter()
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
                  .split(np.zeros(c["n"]), c["yi"]))
    tr, va = splits[0]
    def go():
        for key in KEYS:
            m = lgb_label().fit(c["mats"][key].iloc[tr], c["yi"][tr])
            m.predict(c["mats"][key].iloc[va])
    cprof(go, "4 LGB fits + predicts (one fold)", "tottime", 20)
    log("4 fits+predicts took %.2fs (setup excluded)" % (time.perf_counter() - t))


def stage_localize():
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["run_id", "agents", "shared_state", "messages", "artifacts", "topology", "goal"])
    runs = [parse_run(r) for r in train.to_dict("records")]
    def go():
        for lab in ("clean", "dropped_handoff", "deadlock", "runaway_loop"):
            for r in runs:
                localize(r, lab)
    cprof(go, "localize x10000 x4 labels", "tottime", 25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", action="store_true")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    if a.stage == "build":
        stage_build()
    elif a.stage == "lgb":
        stage_lgb()
    elif a.stage == "localize":
        stage_localize()
    else:
        phase_profile(a.seeds)


if __name__ == "__main__":
    main()
