"""Autonomous train/evaluate/predict on a frozen prepared dataset (no downloads)."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='EXP-013-r02')
    parser.add_argument('--data', type=Path, default=ROOT / 'artifacts/prepared-full-v2')
    parser.add_argument('--embedding-cache', choices=['ram', 'memmap'], default='ram')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error('run-id must be a directory name')
    output = ROOT / 'runs' / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    if not (args.data / 'manifest.json').is_file():
        raise FileNotFoundError('Prepared dataset is required; no automatic re-preparation')
    stop = output / 'STOP_AFTER_CHECKPOINT'
    def run(*command):
        print('COMMAND', command, flush=True)
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True, stderr=subprocess.STDOUT)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        command = ['-m', 'mlp', 'train', '--data', args.data, '--output', output,
                   '--embedding-cache', args.embedding_cache, '--cache-budget-gib', '16',
                   '--prefetch', '2', '--cpu-threads', '4', '--arrow-threads', '2', '--stop-file', stop]
        if (output / 'last.pt').exists():
            command += ['--resume', output / 'last.pt']
        run(*command)
        if stop.exists():
            print('Paused after checkpoint; evaluation deferred.', flush=True)
            return
        if not (output / 'best.pt').exists():
            print('No evaluated checkpoint; inspect training log.', flush=True)
            return
        run('-m', 'mlp', 'evaluate', '--data', args.data, '--model', output / 'best.pt', '--output', output / 'metrics.json')
        run('-m', 'mlp', 'predict', '--data', args.data, '--model', output / 'best.pt', '--output', output / 'predictions-test.parquet')
        hashes = {}
        for path in output.iterdir():
            if path.suffix in ('.pt', '.json', '.parquet') and path.name != 'checksums.json':
                with path.open('rb') as stream:
                    hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        (output / 'checksums.json').write_text(json.dumps(hashes, indent=2))
        print('Pipeline complete; no online submission performed.', flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
