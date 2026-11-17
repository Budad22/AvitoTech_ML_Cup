"""Process lifecycle helpers; experiment code owns its state and reports."""
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import time
import traceback

from ftt.data import write_json


def read_json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


@contextmanager
def worker_lock(path):
    """Hold an OS lock and prevent Windows idle sleep until the worker exits."""
    with Path(path).open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        yield


def record_failure(queue, error, reporter=None):
    (queue.root / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    status = 'budget_exhausted' if isinstance(error, TimeoutError) else 'failed'
    if reporter:
        reporter(status=status, error=str(error))
    queue.state.update(status=status, error=str(error), heartbeat=time.time())
    write_json(queue.root / 'status.json', queue.state)


def add_run_arguments(parser, *, hours, experiment):
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hours', type=float, default=hours)
    parser.add_argument('--experiment', default=experiment)
