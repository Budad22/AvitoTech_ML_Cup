"""Build the shared CPU research cache once, independently of the chat process."""
import argparse
import ctypes
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa
import torch

from mlp_research.cache import build
from scripts.mlp_screen import controller_lock
from training_logs import RunLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    pa.set_cpu_count(2)
    args.output.mkdir(parents=True, exist_ok=True)
    with controller_lock(args.output / 'cache.lock'), RunLogger(args.logs, 'EXP-029-cache', 'CPU/cache') as logger:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

        def emit(event, **fields):
            logger.emit(event, phase='prepare', progress=dict(current=fields['current'], total=fields['total'],
                                                             unit='feed'), split=fields['split'], rows=fields['rows'])

        info = build(args.data, args.output, emit)
        logger.emit('cache_completed', phase='completed', sizes=info['splits'])


if __name__ == '__main__':
    main()
