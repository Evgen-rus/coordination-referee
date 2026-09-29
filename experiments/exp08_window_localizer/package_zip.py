"""Build and verify the standalone Exp08 submission ZIP.

The archive is the artefact that is actually uploaded, so it is verified as a
file, not as a directory:

  1. it contains exactly the runtime modules plus RESULTS.md - no ``__pycache__``,
     no test scripts, no data, no ``__pycache__`` of a test;
  2. ``solution.py`` sits at the ARCHIVE ROOT (the platform runs
     ``python solution.py ...`` from the extracted directory);
  3. every module inside is byte-identical to the in-repo submission, so the ZIP
     cannot quietly ship a different pipeline than the one that was tested;
  4. it is extracted to a temp dir OUTSIDE the repo and run there, producing
     byte-identical predictions to the in-repo run - which also proves it is
     self-contained and does not import anything from the repository.

The comparison in (4) is against ``artifacts/exp08_parity/pred_exp08.csv``,
written by ``check_production_parity.py``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SUB = os.path.join(ROOT, "submission_exp08_window_localizer")
ZIP = os.path.join(ROOT, "submission_exp08_window_localizer.zip")
REF_PRED = os.path.join(ROOT, "artifacts", "exp08_parity", "pred_exp08.csv")

# Runtime modules that solution.py imports, plus the report.  check_parity.py is
# deliberately EXCLUDED: it imports the experiment tree, which the archive does
# not ship, so including it would make the archive depend on the repo.
RUNTIME = ("solution.py", "peak_turn.py", "features.py", "localize.py",
           "new_features.py", "lifecycle.py", "temporal_features.py",
           "window_features.py", "window_dataset.py", "aggregate.py")
ALLOWED = set(RUNTIME) | {"RESULTS.md"}

FAILS = []


def check(name, ok_, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok_ else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok_:
        FAILS.append(name)
    return ok_


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(smoke=False):
    print("=" * 78)
    print("Exp08 STANDALONE ZIP")
    print("=" * 78)

    # ---- build -----------------------------------------------------------
    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(ALLOWED):
            src = os.path.join(SUB, fn)
            if not os.path.exists(src):
                check("%s present in the submission dir" % fn, False)
                return 1
            z.write(src, fn)                  # flat: no directory prefix
    size = os.path.getsize(ZIP)
    print("  wrote %s (%d bytes, %.1f KB)" % (ZIP, size, size / 1024))

    # ---- 1/2. contents and layout ---------------------------------------
    print("\n1. ARCHIVE CONTENTS")
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(z.namelist())
    for n in names:
        print("    %s" % n)
    check("no __pycache__ / .pyc in the archive",
          not any("__pycache__" in n or n.endswith(".pyc") for n in names))
    check("no test scripts in the archive",
          not any("check_parity" in n or "test_" in n for n in names))
    check("no data files in the archive",
          not any(n.endswith((".csv", ".npy", ".json", ".gz")) for n in names))
    check("solution.py is at the archive root", "solution.py" in names)
    check("every entry is flat (no directories)",
          all("/" not in n for n in names),
          str([n for n in names if "/" in n]))
    check("archive holds exactly the runtime modules + RESULTS.md",
          set(names) == ALLOWED,
          "extra=%s missing=%s" % (sorted(set(names) - ALLOWED),
                                   sorted(ALLOWED - set(names))))

    # ---- 3. bytes identical to the tested submission ---------------------
    print("\n2. ARCHIVE BYTES == TESTED SUBMISSION BYTES")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(td)
        for fn in sorted(ALLOWED):
            a = sha(os.path.join(SUB, fn))
            b = sha(os.path.join(td, fn))
            if not check("%-22s identical" % fn, a == b):
                break

        # ---- 4. run standalone, outside the repo ------------------------
        print("\n3. STANDALONE RUN (extracted outside the repository)")
        if smoke:
            n_train, n_test = 1200, 400
            tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv")).head(n_train)
            te = pd.read_csv(os.path.join(ROOT, "data", "test.csv")).head(n_test)
        else:
            n_train = n_test = None
            tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
            te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
        trp = os.path.join(td, "train.csv")
        tep = os.path.join(td, "test.csv")
        outp = os.path.join(td, "standalone_predictions.csv")
        tr.to_csv(trp, index=False)
        te.to_csv(tep, index=False)

        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        cmd = [sys.executable, os.path.join(td, "solution.py"),
               "--train", trp, "--test", tep, "--output", outp]
        print("  running: python solution.py  (%d -> %d)" % (len(tr), len(te)))
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=td, env=env)
        dt = time.time() - t0
        if p.returncode != 0:
            print(p.stdout[-3000:])
            print(p.stderr[-3000:])
            check("standalone solution.py completed", False,
                  "exit %d" % p.returncode)
            return 1
        check("standalone solution.py completed", True, "%.1fs" % dt)
        for line in p.stdout.splitlines():
            if "stacking verified" in line or "DONE in" in line \
                    or "fault_turn:" in line:
                print("      " + line.strip())
        check("standalone runtime within 30 min", dt < 1800, "%.1fs" % dt)

        sp = pd.read_csv(outp)
        check("standalone output has %d rows" % len(te), len(sp) == len(te))

        # validator on the standalone output
        v = subprocess.run([sys.executable,
                            os.path.join(ROOT, "scripts", "validate_submission.py"),
                            "--pred", outp, "--test", tep],
                           capture_output=True, text=True)
        check("validator PASS on standalone output", v.returncode == 0,
              v.stdout.strip()[:200])

        # ---- parity against the in-repo run ----------------------------
        if not smoke and os.path.exists(REF_PRED):
            ref = pd.read_csv(REF_PRED)
            check("standalone row count == in-repo run", len(sp) == len(ref))
            for col in ("label", "success"):
                n_bad = int((sp[col].values != ref[col].values).sum())
                check("standalone %s byte-identical to in-repo run" % col,
                      n_bad == 0, "%d of %d differ" % (n_bad, len(ref)))
            ft = pd.to_numeric(sp["fault_turn"]).values
            ft_ref = pd.to_numeric(ref["fault_turn"]).values
            n_bad = int((ft != ft_ref).sum())
            check("standalone fault_turn byte-identical to in-repo run",
                  n_bad == 0, "%d of %d differ" % (n_bad, len(ref)))
        elif smoke:
            print("      (smoke mode: parity against the full run not compared)")

        # fallback path
        fb = os.path.join(td, "only_ids.csv")
        te[["run_id"]].to_csv(fb, index=False)
        pf = subprocess.run([sys.executable, os.path.join(td, "solution.py"),
                             "--train", trp, "--test", fb,
                             "--output", os.path.join(td, "fb.csv")],
                            capture_output=True, text=True, cwd=td, env=env)
        good = pf.returncode == 0 and os.path.exists(
            os.path.join(td, "fb.csv"))
        check("fallback path (run_id-only test) emits valid output", good,
              (pf.stderr or "")[-200:])

    print("\n4. ZIP DIGEST")
    print("    %s" % ZIP)
    print("    %d bytes, sha256 %s" % (size, sha(ZIP)))

    print("=" * 78)
    if FAILS:
        print("ZIP VERIFICATION FAILED: %d" % len(FAILS))
        for f in FAILS:
            print("   - %s" % f)
        return 1
    print("STANDALONE ZIP VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main(smoke="--smoke" in sys.argv))