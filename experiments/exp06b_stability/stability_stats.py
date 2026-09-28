"""Paired significance for Exp06b, computed from the saved OOF predictions.

Three levels, deliberately reported side by side because they answer different
questions:

1. ``seed_level``      paired t-test / Wilcoxon over the 5 seed-level deltas.
   n=5 is tiny, so this is a *consistency* check, not a real p-value.
2. ``pooled_bootstrap`` paired bootstrap over runs, pooling all 5 seeds' OOF
   predictions.  Much tighter, but the 5 seeds resample the same 10 000 runs, so
   rows are correlated and the CI is optimistic.
3. ``median``          median seed-level delta, which is robust to the single
   outlier seed that dominates several of the means below.

None of the three can repair the study's real limitation: the candidates were
chosen on this demo data after seeing Exp06's single 3-fold OOF run.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import LABELS, REFERENCE, SEEDS, build, log  # noqa: E402

N_BOOT = 20000
CHUNK = 2000
RNG = np.random.default_rng(20260928)
K = len(LABELS)
METRICS = [("macro_f1", "Macro F1"), ("robustness_f1", "Robustness F1"),
           ("composite", "composite"), ("dropped_handoff_f1", "dh F1 overall"),
           ("long_dh_recall", "long dh RECALL"), ("deadlock_f1", "deadlock F1"),
           ("topo_mesh_macro_f1", "topo_mesh Macro F1")]
CANDIDATES = ["exp06_ALL", "full_minus_age", "full_minus_age_minus_reassign"]


def macro_from_conf(conf):
    """conf is (n_classes, n_classes): rows true, cols pred."""
    tp = np.diag(conf)
    denom = tp + conf.sum(0) + conf.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-12), 0.0)
    return f1.mean()


def main():
    with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
        res = json.load(f)

    c = build()
    ys, nmsg = c["ys"], c["nmsg"]
    n = len(ys)
    yi = np.array([LABELS.index(l) for l in ys], dtype=int)
    hard = c["SL"]["hard/robustness"]
    mesh = c["SL"]["topo_mesh"]
    dh = ys == "dropped_handoff"
    long = dh & (nmsg > np.quantile(nmsg, 2 / 3))
    dli = LABELS.index("dropped_handoff")
    dl = LABELS.index("deadlock")
    log("n=%d  hard=%d  mesh=%d  long_dh=%d" % (n, hard.sum(), mesh.sum(), long.sum()))

    # index arrays per system/seed, plus per-slice restricted pairs
    pi, wts = {}, {}
    for k in CANDIDATES + [REFERENCE]:
        for s in SEEDS:
            df = pd.read_csv(os.path.join(HERE, "oof_%s_seed%d.csv" % (k, s)))
            assert (df["y_true"].values == ys).all(), "row order mismatch %s/%d" % (k, s)
            pi[(k, s)] = np.array([LABELS.index(x) for x in df["y_pred"].values])
            wts[(k, s)] = df["y_pred"].notna().values.astype(float)
    n_eff = int(wts[(REFERENCE, SEEDS[0])].sum())
    log("effective rows per OOF = %d" % n_eff)

    # one-hot codes for the slices, used to mask the confusion matrix
    hard_i, mesh_i, long_i = np.where(hard)[0], np.where(mesh)[0], np.where(long)[0]

    def boot_dist(sel_idx=None, drop_i=None, class_only=None):
        """Paired bootstrap of Macro F1 per system, averaged over the 5 seeds.

        resamples are drawn once and reused across all systems (paired), via a
        multinomial weight vector - this is exactly equivalent to sampling runs
        with replacement but costs one bincount instead of a full pass.
        """
        out = {k: np.empty(N_BOOT) for k in CANDIDATES + [REFERENCE]}
        if sel_idx is None:
            idx = np.arange(n)
            y_local = yi
            p_local = {kk: np.stack([pi[(kk, s)] for s in SEEDS]) for kk in out}
        else:
            idx = sel_idx
            y_local = yi[sel_idx]
            p_local = {kk: np.stack([pi[(kk, s)][sel_idx] for s in SEEDS])
                       for kk in out}
        done = 0
        while done < N_BOOT:
            b = min(CHUNK, N_BOOT - done)
            w = RNG.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=b)
            for kk in out:
                acc = np.zeros(b)
                for si in range(len(SEEDS)):
                    code = y_local * K + p_local[kk][si]
                    conf = np.zeros((b, K * K))
                    for r in range(b):
                        conf[r] = np.bincount(code, weights=w[r], minlength=K * K)
                    acc += np.array([macro_from_conf(conf[r].reshape(K, K))
                                     for r in range(b)])
                out[kk][done:done + b] = acc / len(SEEDS)
            done += b
            log("  bootstrap %d/%d" % (done, N_BOOT))
        return out

    out = {"n_runs": int(n), "n_boot": N_BOOT, "seed_level": {}, "bootstrap": {}}

    # ---------------- level 1: seed-level paired tests ----------------------
    for mk, mname in METRICS:
        row = {}
        ref_v = [res["runs"][str(s)][REFERENCE][mk] for s in SEEDS]
        for k in CANDIDATES:
            v = [res["runs"][str(s)][k][mk] for s in SEEDS]
            d = [a - b for a, b in zip(v, ref_v)]
            t, p = stats.ttest_rel(v, ref_v)
            try:
                w_p = float(stats.wilcoxon(d).pvalue) if any(d) else 1.0
            except ValueError:
                w_p = 1.0
            row[k] = {"per_seed": v, "deltas": d, "mean": float(np.mean(d)),
                      "std": float(np.std(d, ddof=1)), "median": float(np.median(d)),
                      "min": float(np.min(d)), "max": float(np.max(d)),
                      "wins": int(sum(1 for x in d if x > 0)),
                      "ttest_p": float(p), "wilcoxon_p": w_p}
        out["seed_level"][mname] = row
    log("seed-level tests done")

    # ---------------- level 2: pooled paired bootstrap ----------------------
    # (a) overall Macro F1
    log("bootstrap A: overall macro")
    ba = boot_dist()
    for k in CANDIDATES:
        d = ba[k] - ba[REFERENCE]
        out["bootstrap"]["overall_macro"] = out["bootstrap"].get("overall_macro", {})
        out["bootstrap"]["overall_macro"][k] = {
            "delta": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}

    # (b) Macro F1 on the robustness / topo_mesh slices
    for tag, sidx in (("robustness", hard_i), ("topo_mesh", mesh_i)):
        log("bootstrap: %s" % tag)
        bs = boot_dist(sel_idx=sidx)
        for k in CANDIDATES:
            d = bs[k] - bs[REFERENCE]
            out["bootstrap"][tag] = out["bootstrap"].get(tag, {})
            out["bootstrap"][tag][k] = {
                "delta": float(d.mean()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}

    # (c) long-slice dh RECALL - the primary target, on a positives-only slice
    log("bootstrap: long dh recall")
    br = np.empty((N_BOOT, len(CANDIDATES) + 1))
    done = 0
    while done < N_BOOT:
        b = min(CHUNK, N_BOOT - done)
        w = RNG.multinomial(len(long_i), np.full(len(long_i), 1.0 / len(long_i)), size=b)
        for j, kk in enumerate(CANDIDATES + [REFERENCE]):
            acc = np.zeros(b)
            for s in SEEDS:
                hit = (pi[(kk, s)][long_i] == dli).astype(float)
                acc += (w * hit).sum(1) / w.sum(1)
            br[done:done + b, j] = acc / len(SEEDS)
        done += b
    for j, k in enumerate(CANDIDATES):
        d = br[:, j] - br[:, -1]
        out["bootstrap"]["long_dh_recall"] = out["bootstrap"].get("long_dh_recall", {})
        out["bootstrap"]["long_dh_recall"][k] = {
            "delta": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}

    with open(os.path.join(HERE, "stability_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)

    L = ["## Exp06b - paired significance of the seed-level deltas", "",
         "Seed-level tests: paired t-test and Wilcoxon over the 5 CV splits. n=5 is",
         "tiny, so read these as *consistency* evidence, not as real p-values.",
         "Pooled bootstrap: 20k resamples of runs, all 5 seeds averaged (correlated",
         "rows, so the CI is optimistic). `median` is robust to the outlier seed.", "",
         "| metric | system | mean d | std | median d | wins | t-test p | Wilcoxon p | pooled-boot CI95 | p |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for mk, mname in METRICS:
        for k in CANDIDATES:
            s_ = out["seed_level"][mname][k]
            bt = out["bootstrap"].get({"macro_f1": "overall_macro",
                                       "robustness_f1": "robustness",
                                       "long_dh_recall": "long_dh_recall",
                                       "topo_mesh_macro_f1": "topo_mesh"}.get(mk, ""), {})
            b_ = bt.get(k)
            ci = ("[%+.4f, %+.4f]" % (b_["ci95"][0], b_["ci95"][1])) if b_ else "-"
            bp = ("%.4f" % b_["p"]) if b_ else "-"
            L.append("| %s | %s | %+.4f | %.4f | %+.4f | %d/5 | %.4f | %.4f | %s | %s |"
                     % (mname, k, s_["mean"], s_["std"], s_["median"], s_["wins"],
                        s_["ttest_p"], s_["wilcoxon_p"], ci, bp))
    txt = "\n".join(L)
    with open(os.path.join(HERE, "STATS.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
