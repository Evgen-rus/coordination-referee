"""Peak-RSS probe for Windows without third-party deps (ctypes -> psapi).

``psutil`` is not installed in this venv and there is no pip, so we call
``GetProcessMemoryInfo`` directly.  Falls back to ``resource`` on POSIX.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import platform
import sys

_IS_WIN = platform.system() == "Windows"


class _PMC(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS_EX: cb, PageFaultCount, then 8 SIZE_Ts, then PrivateUsage."""
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


_KERNEL = None
_PSAPI = None


def _init():
    """Load DLLs once, with correct 64-bit-safe restypes.

    ``GetCurrentProcess`` returns a pseudo-handle of -1.  ctypes defaults
    ``restype`` to ``c_int``, which is harmless here, but the handle must be
    passed to a ``HANDLE``-typed parameter or the 32-bit truncation of a real
    PID handle would corrupt the call.  We request the pointer-sized type.
    """
    global _KERNEL, _PSAPI
    if _KERNEL is None:
        _KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)
        _KERNEL.GetCurrentProcess.restype = ctypes.c_void_p
        _KERNEL.GetCurrentProcess.argtypes = []
        _PSAPI = ctypes.WinDLL("psapi")
        _PSAPI.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_PMC), wt.DWORD]
        _PSAPI.GetProcessMemoryInfo.restype = wt.BOOL


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    if _IS_WIN:
        _init()
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not _PSAPI.GetProcessMemoryInfo(_KERNEL.GetCurrentProcess(),
                                            ctypes.byref(c), c.cb):
            return float("nan")
        return c.PeakWorkingSetSize / (1024.0 * 1024.0)
    if sys.platform == "darwin":                        # pragma: no cover
        try:
            import subprocess
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                 capture_output=True, text=True).stdout.strip()
            return int(out) / 1024.0
        except Exception:
            return float("nan")
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:                                   # pragma: no cover
        return float("nan")


def current_rss_mb() -> float:
    if _IS_WIN:
        return _rss_now()
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") / 1048576.0)
    except Exception:                                   # pragma: no cover
        return float("nan")


def _rss_now() -> float:
    _init()
    c = _PMC()
    c.cb = ctypes.sizeof(_PMC)
    _PSAPI.GetProcessMemoryInfo(_KERNEL.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / (1048576.0)


if __name__ == "__main__":
    print("peak_rss_mb=%.1f  current_rss_mb=%.1f  (py %s)"
          % (peak_rss_mb(), current_rss_mb(), sys.version.split()[0]))
