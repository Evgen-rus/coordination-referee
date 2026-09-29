"""Content-addressed cache for the expensive, deterministic Exp07 inputs."""
from __future__ import annotations
import hashlib, json, os, pickle, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE_DIR = os.path.join(HERE, ".cache")

FEATURE_SOURCES = [
    os.path.join(ROOT, "baseline", "features.py"),
    os.path.join(ROOT, "experiments", "exp03_length_normalization",
                 "new_features.py"),
    os.path.join(ROOT, "experiments", "exp05_handoff_lifecycle",
                 "lifecycle.py"),
    os.path.join(ROOT, "experiments", "exp06_temporal_dynamics", "features.py"),
    os.path.join(HERE, "window_features.py"),
    os.path.join(HERE, "window_dataset.py"),
]
DATA_SOURCES = [os.path.join(ROOT, "data", "train.csv")]

_cache_enabled = os.environ.get("EXP07_NO_CACHE", "") != "1"


def set_enabled(on):
    global _cache_enabled
    _cache_enabled = bool(on) and os.environ.get("EXP07_NO_CACHE", "") != "1"


def enabled():
    return _cache_enabled


def _fingerprint():
    h = hashlib.sha256()
    for p in DATA_SOURCES + FEATURE_SOURCES:
        h.update(os.path.basename(p).encode())
        try:
            with open(p, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


def _path(tag):
    return os.path.join(CACHE_DIR, "%s.%s.pkl" % (tag, _fingerprint()))


# ---------------------------------------------------------------------------
# Provenance hashing.
#
# These are the primitives the honest manifest is built from and the consumer
# re-checks.  They are deliberately here, next to the fingerprint that already
# combines data + feature sources, so there is exactly ONE definition of "the
# code that produced these numbers".
# ---------------------------------------------------------------------------

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    """SHA-256 of a file's bytes, or None when the file does not exist."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def sha256_array(a):
    """SHA-256 of an array's raw buffer, dtype and shape preserved.

    The caller is responsible for casting first (e.g. ``.astype(np.float32)``)
    so that the same value hashed on write and on read is the same value.
    """
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_indices(ix):
    """SHA-256 of an exact index vector, order-sensitive.

    Used for fold assignments: two folds with the same *count* but different
    members must hash differently, which is exactly what the old size-only
    check could not detect.
    """
    import numpy as np
    return sha256_array(np.asarray(ix, dtype=np.int64))


def sha256_run_ids(run_ids):
    """SHA-256 of the run order, sensitive to permutation and content."""
    return sha256_bytes(
        json.dumps([str(x) for x in run_ids], separators=(",", ":")).encode())


def code_sha256():
    """One hash over every source file that determines the features and model.

    A consumer recomputes this from the working tree; if any relevant source
    has been edited since the run, the artifacts are stale and rejected.
    """
    h = hashlib.sha256()
    for p in FEATURE_SOURCES:
        h.update(os.path.basename(p).encode())
        h.update((sha256_file(p) or "<missing>").encode())
    return h.hexdigest()


def code_files_sha256():
    """Per-file hashes, so a rejection can name the file that changed."""
    return {os.path.relpath(p, ROOT).replace("\\", "/"): sha256_file(p)
            for p in FEATURE_SOURCES}


def data_sha256():
    """Hash of the exact train.csv the artifacts were produced from."""
    return sha256_file(DATA_SOURCES[0])


def _atomic_write(path, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def get_or_build(tag, builder):
    """Return ``builder()``, using a cache entry when one matches the inputs."""
    path = _path(tag)
    if _cache_enabled and os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            print("[cache] HIT  %s" % tag, flush=True)
            return obj
        except Exception as e:
            print("[cache] BAD  %s (%s)" % (tag, e), flush=True)
    t0 = time.time()
    obj = builder()
    if _cache_enabled:
        _atomic_write(path, obj)
    print("[cache] MISS %s built in %.1fs" % (tag, time.time() - t0), flush=True)
    return obj


def clear():
    if not os.path.isdir(CACHE_DIR):
        return
    for f in os.listdir(CACHE_DIR):
        os.unlink(os.path.join(CACHE_DIR, f))

def status():
    print("cache dir  : %s" % CACHE_DIR)
    print("enabled    : %s" % _cache_enabled)
    print("fingerprint: %s" % _fingerprint())
    if os.path.isdir(CACHE_DIR):
        for f in sorted(os.listdir(CACHE_DIR)):
            p = os.path.join(CACHE_DIR, f)
            print("  %-44s %7.1f MB" % (f, os.path.getsize(p) / 1e6))
