"""Cache the exp06b feature matrices + fold splits to .npz so repeated LightGBM
micro-benchmarks do not each pay the ~17s feature build.

Writes only into .perf/ (outside the repo's tracked tree).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import profile_exp06b as P  # noqa: E402

OUT = os.path.join(HERE, "matrices.npz")


def main():
    c = P.do_build(verbose=True)
    mats, yi = c["mats"], c["yi"]
    d = {}
    for k, m in mats.items():
        d["X__" + k] = m.to_numpy(dtype=np.float64)
    d["y"] = yi
    n = c["n"]
    for seed in (0, 7, 21, 42, 99):
        sp = list(StratifiedKFold(3, shuffle=True, random_state=seed)
                  .split(np.zeros(n), yi))
        for f, (tr, va) in enumerate(sp):
            d["tr__%d_%d" % (seed, f)] = tr.astype(np.int32)
            d["va__%d_%d" % (seed, f)] = va.astype(np.int32)
    d["names"] = np.array(list(mats.keys()))
    np.savez_compressed(OUT, **d)
    print("wrote", OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")


def load():
    z = np.load(OUT, allow_pickle=False)
    keys = [str(x) for x in z["names"]]
    return {k: z["X__" + k] for k in keys}, z["y"]


def split(z, seed, f):
    return z["tr__%d_%d" % (seed, f)].astype(np.int64), \
        z["va__%d_%d" % (seed, f)].astype(np.int64)


if __name__ == "__main__":
    main()
