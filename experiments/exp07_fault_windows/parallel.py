"""Safe process-level parallelism for INDEPENDENT LightGBM fits.

Only the inner window fits are run this way.  Those are genuinely independent:
each is trained on a different ``keep`` subset of outer-train runs and predicts
a disjoint ``hold`` subset, each carries its own ``random_state=42``, and none
can see another one's held-out rows.  The honest-stacking guarantee therefore
does not depend on execution order.

The run-level classifiers (system A and B) stay sequential on purpose: they
are fitted after all inner folds complete.

Thread count is NOT a safe knob for the window model.  Measured on this data:
with 300 trees, 83 features and ``sample_weight``, ``n_jobs=8`` gives
``max|dP| = 3.3e-14`` against the ``n_jobs=-1`` reference (only 1..12 threads
that divide evenly keep it at exactly 0).  Those last-bit differences flip
``argmax`` on near-tied windows, and because the 51 aggregates are built from
the argmax-derived positions, 49 of 10000 system-B predictions changed.

So the window model is pinned to ``n_jobs=-1`` -- the original setting -- and
process-level parallelism is used only where the per-process thread count is
irrelevant to the result.  The 600-tree label model, by contrast, was verified
bit-identical for every thread count tried, but it is left at ``-1`` too so that
FAST / CV / FINAL cannot drift apart from each other.
"""
from __future__ import annotations

import os

# Pinned to the original ``n_jobs=-1``: see the module docstring.  Changing this
# changes the model, so it is a correctness setting, not a tuning knob.
DEFAULT_THREADS = -1



def _worker(payload):
    import lightgbm as lgb
    (Xw, yw, cw, sub, hold, params, nj, key) = payload
    m = lgb.LGBMClassifier(n_jobs=nj, **params)
    m.fit(Xw[sub], yw[sub], sample_weight=cw[yw[sub]])
    return key, m.predict_proba(Xw[hold])


def run_inner_fits(payloads, threads=DEFAULT_THREADS, workers=None):
    """Fit several independent window models; return {(fold, j): probs}."""
    import lightgbm as lgb

    if workers is None:
        workers = max(1, min(3, (os.cpu_count() or 4) // 3))
    # -1 means "all cores" to LightGBM and must be passed through untouched.
    # Clamping it to a concrete number changes the floating-point reduction
    # order, which moves window probabilities in the last bits and can flip
    # argmax on near-tied windows.
    if threads is not None and threads > 0:
        threads = min(threads, os.cpu_count() or 1)

    if workers <= 1:
        out = {}
        for (f, j, Xw, yw, cw, sub, hold, params) in payloads:
            m = lgb.LGBMClassifier(n_jobs=threads, **params)
            m.fit(Xw[sub], yw[sub], sample_weight=cw[yw[sub]])
            out[(f, j)] = m.predict_proba(Xw[hold])
        return out

    from joblib import Parallel, delayed
    args = [(p[2], p[3], p[4], p[5], p[6], p[7], threads, (p[0], p[1]))
            for p in payloads]
    res = Parallel(n_jobs=workers, backend="loky")(
        delayed(_worker)(a) for a in args)
    return {r[0]: r[1] for r in res}
