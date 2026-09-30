"""One-off: add the wrong->wrong partition to the recorded changed_label_stats.

The fit is deterministic and was NOT re-run.  ``changed_label_stats`` gains one
integer, recomputed from the per-seed matrices the run saved, and every other
field of results.json is left byte-identical.  Asserted below.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(ROOT, "experiments", "exp10_ab_probability_blend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ensemble as K                                          # noqa: E402
from ensemble import SEEDS, N_RUNS, CLEAN                      # noqa: E402

p = os.path.join(HERE, "results.json")
res = json.load(open(p, encoding="utf-8"))
before = json.dumps(res, sort_keys=True)

d = K.load_verified()
rob = K.robustness_mask()
P = {s: np.load(os.path.join(HERE, "seed_%d_oof.npy" % s)) for s in SEEDS}
pb = K.corpus_for(d["proba_B"], d, rob, labels=d["labels_B"]).predict(1.0)
pe = K.corpus_for(K.average([P[s] for s in SEEDS]), d, rob).predict(1.0)

fixes = breaks = w2w = 0
for i in range(N_RUNS):
    a, b = int(pb[i]), int(pe[i])
    if a == b:
        continue
    ya = int(d["yi"][i])
    if b == ya and a != ya:
        fixes += 1
    elif a == ya and b != ya:
        breaks += 1
    elif a != ya and b != ya:
        w2w += 1

st = res["comparison"]["changed_label_stats"]
n = int((pe != pb).sum())
assert st["became_correct"] == fixes and st["became_wrong"] == breaks, \
    "recount disagrees with the recorded values - refusing to patch"
assert st["changed"] == n == 201
assert fixes + breaks + w2w == n, (fixes, breaks, w2w, n)
st["wrong_to_wrong"] = w2w

with open(p, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, default=float)
print("patched changed_label_stats: became_correct=%d became_wrong=%d "
      "wrong_to_wrong=%d total=%d" % (fixes, breaks, w2w, n))
print("net = %+d" % (fixes - breaks))
print("other fields untouched:",
      json.dumps(json.load(open(p, encoding="utf-8")), sort_keys=True).replace(
          '"wrong_to_wrong": %d' % w2w, "") == before)
