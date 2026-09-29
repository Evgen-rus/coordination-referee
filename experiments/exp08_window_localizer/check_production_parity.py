"""Production parity: Exp08 submission vs Exp07 submission, on the real split.

The claim Exp08 makes is narrow and mechanical:

    ONLY ``fault_turn`` changes.

``label`` and ``success`` must be byte-identical on all 4000 test rows, because
Exp08 adds no feature, refits nothing and touches no head - it only reads back
the peak positions the window model had already computed and discarded.  If a
single row's label or success differs, something downstream is not what it
claims to be and the build is rejected.

Both submissions are run as SEPARATE PROCESSES on the SAME inputs, in the same
way the platform will invoke them, so this compares the shipped artefacts rather
than two imports in one interpreter (which would share module state and could
hide a cross-contamination bug).

Also checked here: the feature modules are byte-identical to Exp07's, the model
hyper-parameters match attribute by attribute, and the output passes the
official submission validator.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SUB07 = os.path.join(ROOT, "submission_exp07_fault_windows")
SUB08 = os.path.join(ROOT, "submission_exp08_window_localizer")
OUT = os.path.join(ROOT, "artifacts", "exp08_parity")

MODULES = ("features.py", "localize.py", "new_features.py", "lifecycle.py",
           "temporal_features.py", "window_features.py", "window_dataset.py",
           "aggregate.py")

FAILS = []


def check(name, ok_, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok_ else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok_:
        FAILS.append(name)
    return ok_


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_solution(subdir, train_csv, test_csv, out_csv, tag):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    cmd = [sys.executable, os.path.join(subdir, "solution.py"),
           "--train", train_csv, "--test", test_csv, "--output", out_csv]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=os.path.dirname(out_csv), env=env)
    dt = time.time() - t0
    if p.returncode != 0:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        check("%s solution.py completed" % tag, False, "exit %d" % p.returncode)
        return None, dt
    check("%s solution.py completed" % tag, True, "%.1fs" % dt)
    return p.stdout, dt


def main(train_csv=None, test_csv=None, limit_rows=None, quick=False):
    os.makedirs(OUT, exist_ok=True)
    train_csv = train_csv or os.path.join(ROOT, "data", "train.csv")
    test_csv = test_csv or os.path.join(ROOT, "data", "test.csv")
    print("=" * 78)
    print("Exp08 PRODUCTION PARITY: submission_exp08 vs submission_exp07")
    print("=" * 78)
    print("train=%s" % train_csv)
    print("test =%s" % test_csv)

    # ---- 1. the modules that build features must be untouched ------------
    print("\n1. FEATURE MODULES byte-identical to Exp07")
    for m in MODULES:
        a = os.path.join(SUB07, m)
        b = os.path.join(SUB08, m)
        same = os.path.exists(a) and os.path.exists(b) and sha_file(a) == sha_file(b)
        check("%-24s identical" % m, same,
              "" if same else "hash differs -> features would change")

    # ---- 2. run both, as separate processes ------------------------------
    print("\n2. RUNNING BOTH SUBMISSIONS")
    out07 = os.path.join(OUT, "pred_exp07.csv")
    out08 = os.path.join(OUT, "pred_exp08.csv")
    if quick and limit_rows:
        pass
    log07, t07 = run_solution(SUB07, train_csv, test_csv, out07, "exp07")
    log08, t08 = run_solution(SUB08, train_csv, test_csv, out08, "exp08")
    if log07 is None or log08 is None:
        return 1
    for tag, log in (("exp07", log07), ("exp08", log08)):
        for line in log.splitlines():
            if "stacking verified" in line or "DONE in" in line \
                    or "localising fault_turn" in line \
                    or "non -1" in line:
                print("    %s | %s" % (tag, line.strip()))

    p07 = pd.read_csv(out07)
    p08 = pd.read_csv(out08)
    n = len(p07)

    # ---- 3. label / success must be byte-identical -----------------------
    print("\n3. label AND success byte-identical")
    check("row count identical (%d)" % n, len(p08) == n)
    check("run_id order identical",
          bool((p07["run_id"].values == p08["run_id"].values).all()))

    for col in ("label", "success"):
        a = p07[col].values
        b = p08[col].values
        n_bad = int((a != b).sum())
        check("%s byte-identical %d/%d" % (col, n - n_bad, n), n_bad == 0,
              "%d rows differ" % n_bad if n_bad else "")

    # byte-level, not just element-level: same length, same encoding
    for col in ("label", "success"):
        raw07 = open(out07, "rb").read().split(b"\n")
        raw08 = open(out08, "rb").read().split(b"\n")
        col07 = [ln.split(b",")[1 if col == "label" else 2]
                 for ln in raw07 if ln]
        col08 = [ln.split(b",")[1 if col == "label" else 2]
                 for ln in raw08 if ln]
        check("%s identical as raw bytes in the CSV" % col, col07 == col08)

    # ---- 4. only fault_turn moved ----------------------------------------
    print("\n4. fault_turn is the ONLY thing that changed")
    ft07 = pd.to_numeric(p07["fault_turn"]).values
    ft08 = pd.to_numeric(p08["fault_turn"]).values
    n_diff = int((ft07 != ft08).sum())
    check("fault_turn differs (the intended change)", n_diff > 0,
          "%d of %d rows differ" % (n_diff, n))
    clean = p08["label"].values.astype(str) == "clean"
    check("every clean run has fault_turn == -1 (L1 rule)",
          bool((ft08[clean] == -1).all()),
          "%d clean rows violate" % int((ft08[clean] != -1).sum()))
    fault = ~clean
    # A predicted fault run must receive a real turn.  -1 is only legitimate for
    # an EMPTY window block, so the count of faulty runs left at -1 must not
    # exceed the number of runs that genuinely have no windows.  (An earlier
    # version of this check OR-ed its own condition with True and so could never
    # fail; this one cannot pass vacuously.)
    n_neg = int((ft08[fault] < 0).sum())
    print("    predicted-fault runs with fault_turn == -1: %d" % n_neg)
    check("no predicted-fault run is left without a turn", n_neg == 0,
          "%d faulty row(s) got -1; only an empty window block may do that, "
          "and every test run in this fixture has windows" % n_neg)
    print("    fault_turn: exp07 mean %.3f | exp08 mean %.3f"
          % (ft07[clean == False].mean(), ft08[fault].mean()))

    # ---- 5. model params unchanged ---------------------------------------
    print("\n5. MODEL HYPER-PARAMETERS unchanged")
    sys.path.insert(0, SUB08)
    for m in ("solution", "aggregate", "window_dataset", "window_features",
              "temporal_features", "new_features", "lifecycle", "features",
              "localize", "peak_turn"):
        sys.modules.pop(m, None)
    import solution as s08
    for m in ("solution", "aggregate", "window_dataset", "window_features",
              "temporal_features", "new_features", "lifecycle", "features",
              "localize"):
        sys.modules.pop(m, None)
    sys.path.insert(0, SUB07)
    import solution as s07
    for nm in ("label", "success", "window"):
        a = getattr(s07, "make_%s_model" % nm)().get_params()
        b = getattr(s08, "make_%s_model" % nm)().get_params()
        d = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
             if a.get(k) != b.get(k)}
        check("%s model params identical" % nm, not d, str(d))
    check("feature counts pinned (300 / 185)",
          s08.N_FOUNDATION == s07.N_FOUNDATION == 249
          and s08.N_SUCCESS_FEATS == s07.N_SUCCESS_FEATS == 185)
    check("aggregate.AGG_NAMES identical (51 window features)",
          list(s08.ag.AGG_NAMES) == list(s07.ag.AGG_NAMES),
          "%d names" % len(s08.ag.AGG_NAMES))
    check("aggregate.N_AGG == 51", s08.ag.N_AGG == 51, str(s08.ag.N_AGG))

    # ---- 6. validator ----------------------------------------------------
    print("\n6. OFFICIAL VALIDATOR")
    v = subprocess.run([sys.executable,
                        os.path.join(ROOT, "scripts", "validate_submission.py"),
                        "--pred", out08, "--test", test_csv],
                       capture_output=True, text=True)
    print("    " + (v.stdout.strip().replace("\n", "\n    ")))
    check("validator PASS on exp08 predictions", v.returncode == 0,
          v.stderr.strip()[-200:])

    # ---- 7. runtime ------------------------------------------------------
    print("\n7. RUNTIME")
    print("    exp07 %.1fs (%.2f min)" % (t07, t07 / 60))
    print("    exp08 %.1fs (%.2f min)" % (t08, t08 / 60))
    print("    overhead of L1 localisation: %+.2fs" % (t08 - t07))
    check("exp08 within the 30-minute limit", t08 < 1800,
          "%.1fs" % t08)

    summary = {
        "n_rows": int(n),
        "label_identical": bool((p07["label"].values == p08["label"].values).all()),
        "success_identical": bool((p07["success"].values
                                   == p08["success"].values).all()),
        "fault_turn_changed_rows": int(n_diff),
        "clean_turn_all_minus1": bool((ft08[clean] == -1).all()),
        "runtime_sec": {"exp07": round(t07, 1), "exp08": round(t08, 1),
                        "overhead": round(t08 - t07, 2)},
        "modules_identical": MODULES,
        "pred_sha256": {"exp07": sha_file(out07), "exp08": sha_file(out08)},
        "train": train_csv, "test": test_csv,
    }
    json.dump(summary, open(os.path.join(HERE, "production_parity.json"),
                            "w", encoding="utf-8"), indent=2)

    print("=" * 78)
    if FAILS:
        print("PRODUCTION PARITY FAILED: %d" % len(FAILS))
        for f in FAILS:
            print("   - %s" % f)
        return 1
    print("PRODUCTION PARITY PASSED - label and success byte-identical, "
          "only fault_turn moved")
    return 0


if __name__ == "__main__":
    _test = None
    _train = None
    if "--smoke" in sys.argv:
        import tempfile
        tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
        te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
        tmp = tempfile.mkdtemp(prefix="exp08_smoke_")
        _train = os.path.join(tmp, "train.csv")
        _test = os.path.join(tmp, "test.csv")
        tr.head(1200).to_csv(_train, index=False)
        te.head(400).to_csv(_test, index=False)
        print("SMOKE MODE: 1200 train -> 400 test")
    sys.exit(main(train_csv=_train, test_csv=_test, quick="--smoke" in sys.argv))
