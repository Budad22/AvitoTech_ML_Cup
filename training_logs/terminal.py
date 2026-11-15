"""Enable modern Windows terminal rendering for this process, then restore it."""
import atexit
import os
import sys

from rich.console import Console


def create_console(format='auto'):
    vt = False
    if os.name == 'nt' and sys.stdout.isatty():
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetStdHandle.restype = wintypes.HANDLE
        handle = kernel.GetStdHandle(-11)
        mode = wintypes.DWORD()
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            original = mode.value
            vt = bool(kernel.SetConsoleMode(handle, original | 4))
            if vt:
                atexit.register(lambda: kernel.SetConsoleMode(handle, original))
    return Console(force_terminal=True if format == 'rich' else None,
                   legacy_windows=False if vt or format == 'rich' else None)
