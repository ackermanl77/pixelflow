"""Crash and file-lock diagnostics (report item R8), Windows-first.

- vc_runtime_report(): when torch fails to import (or the process crashed with
  0xC0000005 before), the usual cause on a fresh Windows is a missing or old
  Microsoft Visual C++ 2015-2022 runtime. This says which of its three DLLs
  can't be loaded, the same audit Comfy Desktop runs.
- locking_processes(path): when a delete fails because the file is open, the
  names of the processes holding it, from the Windows Restart Manager
  (RmStartSession/RmRegisterResources/RmGetList), so the message can say
  "close Photoshop" instead of "access denied".

Both are no-ops returning empty results on other platforms. Standard library only.
"""

import ctypes
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

VC_RUNTIME_DLLS = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
VC_RUNTIME_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def missing_vc_runtime_dlls(loader=None) -> List[str]:
    """The VC++ runtime DLLs that can't be loaded ([] when all load, or off Windows)."""
    if sys.platform != "win32" and loader is None:
        return []
    load = loader or (lambda name: ctypes.WinDLL(name))
    missing = []
    for name in VC_RUNTIME_DLLS:
        try:
            load(name)
        except OSError:
            missing.append(name)
    return missing


def vc_runtime_report(loader=None) -> str:
    """One sentence for a failed torch import: which runtime DLLs are missing
    and where the installer is, or "" when the runtime looks fine."""
    missing = missing_vc_runtime_dlls(loader)
    if not missing:
        return ""
    return (f"the Microsoft Visual C++ 2015-2022 runtime is missing or broken ({', '.join(missing)} can't be loaded); "
            f"install the x64 redistributable from {VC_RUNTIME_URL} and restart PixelFlow")


class _RmUniqueProcess(ctypes.Structure):
    # FILETIME is two DWORDs (4-byte aligned): a c_ulonglong here would shift every later field.
    _fields_ = [("dwProcessId", ctypes.c_ulong), ("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]


class _RmProcessInfo(ctypes.Structure):
    _fields_ = [
        ("Process", _RmUniqueProcess),
        ("strAppName", ctypes.c_wchar * 256),
        ("strServiceShortName", ctypes.c_wchar * 64),
        ("ApplicationType", ctypes.c_int),
        ("AppStatus", ctypes.c_ulong),
        ("TSSessionId", ctypes.c_ulong),
        ("bRestartable", ctypes.c_int),
    ]


def locking_processes(path: Path) -> List[str]:
    """Names (with process ids) of the processes that have `path` open, via the
    Restart Manager; [] off Windows, when nothing holds it, or on any error."""
    if sys.platform != "win32":
        return []
    try:
        rstrtmgr = ctypes.WinDLL("rstrtmgr")
    except OSError:
        return []
    session = ctypes.c_ulong()
    key = ctypes.create_unicode_buffer(64)
    if rstrtmgr.RmStartSession(ctypes.byref(session), 0, key) != 0:
        return []
    try:
        files = (ctypes.c_wchar_p * 1)(str(Path(path).resolve()))
        if rstrtmgr.RmRegisterResources(session, 1, files, 0, None, 0, None) != 0:
            return []
        needed, count, reasons = ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_ulong(0)
        result = rstrtmgr.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), None, ctypes.byref(reasons))
        if needed.value == 0:
            return []
        infos = (_RmProcessInfo * needed.value)()
        count = ctypes.c_uint(needed.value)
        result = rstrtmgr.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), infos, ctypes.byref(reasons))
        if result != 0:
            return []
        return [f"{infos[i].strAppName or 'unknown'} (pid {infos[i].Process.dwProcessId})" for i in range(count.value)]
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        logger.debug("Restart Manager lookup failed for %s", path, exc_info=True)
        return []
    finally:
        rstrtmgr.RmEndSession(session)


def explain_delete_failure(path: Path, error: OSError) -> str:
    """A message for a failed delete that names whoever holds the file."""
    holders = locking_processes(path)
    if holders:
        return f"could not delete {Path(path).name}: it is open in {', '.join(holders)} -- close it there and try again"
    return f"could not delete {Path(path).name}: {error.strerror or error}"
