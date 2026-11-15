"""Animate presentation between observations; measured numbers stay untouched."""
from copy import deepcopy
import threading
import time

from rich.panel import Panel
from rich.spinner import Spinner

from .dashboard import dashboard


class AnimatedDashboard:
    def __init__(self, *, enabled=True, duration=.7):
        self.duration = duration if enabled else 0
        self.lock = threading.RLock()
        self.view = None
        self.details = None
        self.started = 0.
        self.progress_start = self.gpu_start = 0.
        self.waiting = Panel(Spinner('dots', text='Ожидание данных сервера…', style='#56d4dd'),
                             border_style='#48556b')

    def frame(self, now=None):
        with self.lock:
            if self.view is None:
                return None
            view = deepcopy(self.view)
            elapsed = (time.monotonic() if now is None else now) - self.started
            fraction = min(1., max(0., elapsed / self.duration)) if self.duration else 1.
            eased = 1 - (1 - fraction) ** 3
            progress = view.get('progress') or {}
            if progress.get('total'):
                progress['display_current'] = self.progress_start + (progress['current'] - self.progress_start) * eased
            gpu = view.get('gpu')
            if gpu:
                gpu['display_utilization'] = self.gpu_start + (gpu['utilization'] - self.gpu_start) * eased
            return view

    def update(self, view, details=None, *, now=None):
        with self.lock:
            stamp = time.monotonic() if now is None else now
            previous = self.frame(stamp)
            progress = view.get('progress') or {}
            same_epoch = previous and (previous.get('run_id'), previous.get('phase'),
                                      (previous.get('progress') or {}).get('epoch')) == (
                                          view.get('run_id'), view.get('phase'), progress.get('epoch'))
            self.progress_start = ((previous.get('progress') or {}).get('display_current', progress.get('current', 0))
                                   if same_epoch else progress.get('current', 0))
            self.gpu_start = ((previous.get('gpu') or {}).get('display_utilization', (view.get('gpu') or {}).get('utilization', 0))
                              if previous else (view.get('gpu') or {}).get('utilization', 0))
            self.view, self.details, self.started = deepcopy(view), details, stamp

    def __rich_console__(self, console, options):
        with self.lock:
            view, details = self.frame(), self.details
        yield (dashboard(view, width=options.max_width, height=options.max_height, details=details)
               if view is not None else self.waiting)
