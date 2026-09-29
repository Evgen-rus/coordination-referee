"""Content-addressed cache for the expensive, deterministic Exp07 inputs."""
from __future__ import annotations
import hashlib, os, pickle, tempfile, time

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
