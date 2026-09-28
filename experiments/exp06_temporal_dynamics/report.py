"""Console report for Exp06."""

from __future__ import annotations

from common import DH, DROPS, INCR, LABELS, ft, log


def show(res, s, objs, ys, c):
    ref = res["systems"]["0_B_exp05_B"]
    log("")
    log("=" * 100)
    log("HEADLINE (3-fold OOF; Success F1 fixed; fault_turn = official localiser)")
    log("=" * 100)
    log("%-20s %5s %8s %9s %9s %8s %8s %9s"
        % ("variant", "feats", "MacroF1", "RobustF1", "composite", "dMacro",
           "dh_F1", "d_dh_long"))
    for k in INCR + DROPS:
        d = res["systems"][k]
        log("%-20s %5d %8.4f %9.4f %9.4f %+8.4f %8.4f %+9.4f"
            % (k, d["n_features"], d["macro_f1"], d["robustness_f1"], d["composite"],
               d["d_macro"], d["dh_overall"]["f1"], d["d_dh_long"]))

    log("")
    log("DROPPED_HANDOFF BY RUN LENGTH  (main diagnostic target)")
    log("%-22s %6s %s" % ("sub-slice", "n",
                           "".join("%13s" % k.split("_")[0] for k in INCR)))
    for sname, m in c["DH_SL"].items():
        if int(m.sum()) < 30:
            log("%-22s %6d  (too small, skipped)" % (sname, int(m.sum())))
            continue
        log("%-22s %6d %s" % (sname, int(m.sum()), "".join(
            "%13.4f" % res["systems"][k]["dh_subslices"][sname]["f1"] for k in INCR)))
    log("  columns: 0=B 1=age 2=backlog 3=receiver 4=reassign 5=tail 6=ALL")

    log("")
    log("  long-slice precision/recall:")
    for k in INCR:
        d = res["systems"][k]["dh_subslices"]["long(>p66)"]
        log("    %-20s P=%.4f R=%.4f F1=%.4f"
            % (k, d["precision"], d["recall"], d["f1"]))

    log("")
    log("PER-CLASS F1 (foundation vs full)")
    log("%-18s %10s %10s %8s" % ("class", "0_B", "6_B_ALL", "delta"))
    for cl in LABELS:
        a0 = ref["per_class"][cl]
        a6 = res["systems"]["6_B_ALL"]["per_class"][cl]
        log("%-18s %10.4f %10.4f %+8.4f" % (cl, a0, a6, a6 - a0))

    log("")
    log("SLICE Macro F1 (foundation vs full)")
    log("%-20s %10s %10s %8s" % ("slice", "0_B", "6_B_ALL", "delta"))
    for sl in c["SL"]:
        a0 = ref["slices"][sl]
        a6 = res["systems"]["6_B_ALL"]["slices"][sl]
        if a0 is None or a6 is None:
            continue
        log("%-20s %10.4f %10.4f %+8.4f" % (sl, a0, a6, a6 - a0))

    log("")
    log("FOLD-BY-FOLD Macro F1")
    for k in INCR:
        f = res["systems"][k]["fold_macro_f1"]
        d = [f[i] - ref["fold_macro_f1"][i] for i in range(3)]
        log("  %-20s %s | d = %s | wins %d/3"
            % (k, ", ".join("%.4f" % x for x in f),
               ", ".join("%+.4f" % x for x in d), sum(1 for x in d if x > 0)))

    log("")
    log("CONFUSION counts (B -> ALL)")
    for a, b in [(DH, "clean"), (DH, "deadlock"), (DH, "runaway_loop"),
                 ("deadlock", "clean"), ("duplicated_work", "clean")]:
        v0 = int(((ys == a) & (objs["0_B_exp05_B"] == b)).sum())
        v6 = int(((ys == a) & (objs["6_B_ALL"] == b)).sum())
        log("  %-18s -> %-13s n=%4d  B=%4d  ALL=%4d  (%+d)"
            % (a, b, int((ys == a).sum()), v0, v6, v6 - v0))

    log("")
    log("=" * 100)
    log("ABLATION - INCREMENTAL (primary)")
    log("=" * 100)
    for k in INCR[1:]:
        d = res["systems"][k]
        log("  %-20s dMacro=%+.4f d_dh=%+.4f d_dh_long=%+.4f dRob=%+.4f dComp=%+.4f"
            % (k, d["d_macro"], d["d_dh"], d["d_dh_long"], d["d_rob"], d["d_comp"]))
    log("")
    log("ABLATION - DROP-ONE (reference only: correlated within blocks)")
    for k in DROPS:
        d = res["systems"][k]
        log("  %-20s oof=%.4f (dMacro=%+.4f) d_dh_long=%+.4f"
            % (k, d["macro_f1"], d["d_macro"], d["d_dh_long"]))

    imp = res["importance"]
    log("")
    log("FEATURE IMPORTANCE (full model)")
    log("  new: gain share %.2f%% | non-zero %d/29 | top-20 %d | top-50 %d"
        % (100 * imp["gain_share"], imp["n_used"], imp["in_top20"], imp["in_top50"]))
    for bn, bc in ft.BLOCKS.items():
        log("    %-12s %2d feats  %6.2f%% gain"
            % (bn, len(bc), 100 * imp["by_block"][bn]))
    log("")
    log("  top 15 new features:")
    for x, v in list(imp["top_new"].items())[:15]:
        log("    %-34s %6.3f%%" % (x, 100 * v))
    log("")
    log("EXP06 COMPLETE - all 12 variants, 3 folds")
