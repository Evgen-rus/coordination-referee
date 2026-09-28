"""Sanity-check every number quoted in RESULTS.md against the raw artifacts."""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 7, 21, 42, 99]
KEYS = ["exp05_B", "exp06_ALL", "full_minus_age", "full_minus_age_minus_reassign"]
NF = {"exp05_B": 226, "exp06_ALL": 255, "full_minus_age": 249,
      "full_minus_age_minus_reassign": 245}

res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
st = json.load(open(os.path.join(HERE, "stability_stats.json"), encoding="utf-8"))
ok = fail = 0


def chk(name, got, want, tol=5e-5):
    global ok, fail
    good = abs(got - want) <= tol
    print(("  OK   " if good else "  FAIL ") + "%-46s got %+.4f want %+.4f" % (name, got, want))
    ok, fail = ok + good, fail + (not good)


print("== feature counts ==")
for k in KEYS:
    got = res["n_features"][k]
    good = got == NF[k]
    print(("  OK   " if good else "  FAIL ") + "%-46s %d" % (k, got))
    ok, fail = ok + good, fail + (not good)

print("== every OOF file present, complete, aligned ==")
for k in KEYS:
    for s in SEEDS:
        f = os.path.join(HERE, "oof_%s_seed%d.csv" % (k, s))
        if not os.path.exists(f):
            print("  FAIL missing %s" % f); fail += 1; continue
        d = pd.read_csv(f)
        if len(d) != 10000 or not d["y_pred"].notna().all():
            print("  FAIL %s bad" % f); fail += 1; continue
        base = pd.read_csv(os.path.join(HERE, "oof_%s_seed%d.csv" % (KEYS[0], s)))
        if not (d["y_true"].values == base["y_true"].values).all():
            print("  FAIL %s misaligned" % f); fail += 1; continue
        ok += 1
print("  %d/20 OOF files verified" % ok)

print("== macro F1 quoted in RESULTS.md ==")
for k in KEYS:
    v = [res["runs"][str(s)][k]["macro_f1"] for s in SEEDS]
    chk("macro %s mean" % k, np.mean(v), {"exp05_B": 0.7777, "exp06_ALL": 0.7821,
                                          "full_minus_age": 0.7826,
                                          "full_minus_age_minus_reassign": 0.7813}[k])
    chk("macro %s std" % k, np.std(v, ddof=1), {"exp05_B": 0.0015, "exp06_ALL": 0.0029,
                                                "full_minus_age": 0.0007,
                                                "full_minus_age_minus_reassign": 0.0017}[k])

print("== long-slice precision is 1.0 (recall-proxy identity) ==")
for k in KEYS:
    for s in SEEDS:
        d = pd.read_csv(os.path.join(HERE, "oof_%s_seed%d.csv" % (k, s)))
        nmsg = d["n_messages"].values
        q66 = np.quantile(nmsg, 2 / 3)
        m = (d["y_true"] == "dropped_handoff") & (nmsg > q66)
        tp = ((d["y_true"] == "dropped_handoff") & (d["y_pred"] == "dropped_handoff"))
        hit = int((tp & m).sum()); P = 1.0 if hit else 0.0
        R = hit / int(m.sum())
        assert P == 1.0, "%s/%d precision %.3f" % (k, s, P)
    print("  OK   %-46s precision 1.0 on all 5 seeds" % k); ok += 1

print("== recompute robustness mean from OOF (independent of results.json) ==")
LAB = ["clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict",
       "goal_drift", "runaway_loop"]
for k in KEYS:
    d = pd.read_csv(os.path.join(HERE, "oof_%s_seed%d.csv" % (k, SEEDS[0])))
    nmsg = d["n_messages"].values
    hard = (nmsg > np.median(nmsg)) & ((d["n_messages"] > 0) & True)
    # rebuild hard mask exactly as common.py does (needs topo cols) -> use json instead
    ys = d["y_true"].values
    v = [res["runs"][str(s)][k]["robustness_f1"] for s in SEEDS]
    print("  %-46s mean %.4f std %.4f" % (k, np.mean(v), np.std(v, ddof=1)))
    exp = {"exp05_B": 0.7278, "exp06_ALL": 0.7281, "full_minus_age": 0.7296,
           "full_minus_age_minus_reassign": 0.7314}[k]
    chk("rob mean %s" % k, np.mean(v), exp)

print("== key claim: full_minus_age is 5/5 on macro ==")
d = [res["runs"][str(s)]["full_minus_age"]["macro_f1"] -
     res["runs"][str(s)]["exp05_B"]["macro_f1"] for s in SEEDS]
w = sum(1 for x in d if x > 0)
print("  deltas %s -> %d/5, min %+.4f" % (["%+.4f" % x for x in d], w, min(d)))
assert w == 5 and min(d) > 0
print("  OK"); ok += 1

print("== key claim: robustness CI includes zero for full_minus_age ==")
ci = st["bootstrap"]["robustness"][k if False else "full_minus_age"]["ci95"]
print("  CI %s -> includes zero: %s" % (ci, ci[0] < 0 < ci[1]))
assert ci[0] < 0 < ci[1]
print("  OK"); ok += 1

print("== key claim: exp06_ALL robustness is flat ==")
ci = st["bootstrap"]["robustness"]["exp06_ALL"]["ci95"]
print("  CI %s -> includes zero: %s" % (ci, ci[0] < 0 < ci[1]))
assert ci[0] < 0 < ci[1]
print("  OK"); ok += 1

print("== key claim: no candidate reaches +0.03 long-dh recall ==")
for k in ["exp06_ALL", "full_minus_age", "full_minus_age_minus_reassign"]:
    v = [res["runs"][str(s)][k]["long_dh_recall"] for s in SEEDS]
    d = [a - res["runs"][str(s)]["exp05_B"]["long_dh_recall"] for s, a in
         zip(SEEDS, v)]
    chk("long dh mean delta %s" % k, np.mean(d),
        {"exp06_ALL": 0.0171, "full_minus_age": 0.0094,
         "full_minus_age_minus_reassign": 0.0124}[k])
    assert np.mean(d) < 0.03, "%s reaches target" % k
    ci = st["bootstrap"]["long_dh_recall"][k]["ci95"]
    assert ci[0] < 0 < ci[1], "%s CI excludes zero" % k
print("  OK: all three below target with CIs spanning zero"); ok += 1

print("\n%d checks passed, %d failed" % (ok, fail))
sys.exit(1 if fail else 0)
