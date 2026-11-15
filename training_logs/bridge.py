"""Observe trusted legacy checkpoints without restarting or modifying training.

Only CPU loads of this repository's own checkpoints are allowed. For new
trainers, use RunLogger directly; this adapter exits when a trainer owns events.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time

from .events import RunLogger


def observe(output, prepared, logger):
    import torch
    checkpoint = output / 'last.pt'
    if not checkpoint.exists():
        return False
    modified = checkpoint.stat().st_mtime_ns
    if logger.state.get('checkpoint_mtime_ns') == modified:
        return False
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    manifest = json.loads((prepared / 'manifest.json').read_text())
    total = sum(math.ceil(shard['feeds'] / config['batch_size']) for shard in manifest['splits']['train']['files'])
    epoch, cursor = state['epoch'], state['cursor']
    metrics = {}
    if state.get('best', -1) >= 0:
        metrics['best_ndcg'] = state['best']
    logger.emit('checkpoint_observed', source='checkpoint', checkpoint_mtime_ns=modified,
                checkpoint_at=datetime.fromtimestamp(modified / 1e9, timezone.utc).isoformat(),
                elapsed_seconds=state['elapsed'], budget_seconds=config['hours'] * 3600,
                global_step=epoch * total + cursor,
                progress=dict(epoch=min(epoch + 1, config['epochs']), epochs=config['epochs'],
                              current=cursor, total=total, unit='батч'), metrics=metrics)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--architecture', default='MLP')
    parser.add_argument('--task-name', default='AvitoMLP')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    with RunLogger(args.run_dir, args.run_dir.name, args.architecture,
                   producer='checkpoint_bridge', echo=False) as logger:
        while True:
            latest = args.run_dir / 'progress.json'
            if latest.exists() and json.loads(latest.read_text()).get('producer') == 'trainer':
                return
            observe(args.run_dir, args.data, logger)
            if args.once:
                return
            state = subprocess.check_output(['powershell', '-NoProfile', '-NonInteractive', '-Command',
                f"(Get-ScheduledTask -TaskName '{args.task_name}').State.ToString()"], text=True).strip()
            if state != 'Running':
                logger.emit('observer_stopped', phase='completed' if (args.run_dir / 'checksums.json').exists() else 'stopped')
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
