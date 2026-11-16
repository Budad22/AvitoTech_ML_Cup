"""Enable modern Windows terminal rendering for this process, then restore it."""
import atexit
import os
import sys

from rich.console import Console


def add_refresh_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--watch', type=float, nargs='?', const=1, default=None, metavar='SECONDS',
                       help='Override refresh interval; terminal default is 1 second')
    group.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    parser.add_argument('--no-animation', action='store_true', help='Disable visual interpolation')


def refresh_interval(args):
    if args.once:
        return 0
    if args.watch is not None:
        return args.watch
    if getattr(args, 'logs', 0) or getattr(args, 'verbose', False):
        return 0.
    return 1. if sys.stdout.isatty() and not args.json and args.format != 'plain' else 0.


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
