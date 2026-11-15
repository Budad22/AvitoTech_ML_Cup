"""Autonomous Windows worker. Run through Task Scheduler, never through SSH."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUN = ROOT / 'runs' / 'EXP-013-r01'


def run(*args):
    print('COMMAND', args, flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, stderr=subprocess.STDOUT)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        print('Worker started', time.ctime(), flush=True)
        archive = WORKSPACE / 'competition.zip'
        expected = (WORKSPACE / 'competition.sha256').read_text().strip()
        marker = WORKSPACE / 'extraction.complete'
        # Extraction's marker is written only after SHA-256 and ZIP CRC checks.
        cached = (archive.exists() and marker.exists() and marker.read_text().strip() == expected
                  and marker.stat().st_mtime_ns >= archive.stat().st_mtime_ns)
        if not archive.exists():
            partial = archive.with_suffix('.partial')
            url = 'https://storage.yandexcloud.net/avitotech-mlcup-2026/competition.zip'
            for attempt in range(30):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        append = response.status == 206
                        with partial.open('ab' if append else 'wb') as target:
                            while chunk := response.read(8 * 1024 * 1024):
                                target.write(chunk)
                    partial.replace(archive)
                    break
                except Exception as error:
                    print('Download retry', attempt, repr(error), flush=True)
                    time.sleep(10)
            else:
                raise RuntimeError('Dataset download failed; partial file preserved')
        if cached:
            digest = expected
            print('Reusing verified, extracted archive (unchanged since verification).', flush=True)
        else:
            with archive.open('rb') as source_archive:
                digest = hashlib.file_digest(source_archive, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError('Dataset archive SHA-256 mismatch')
        source = WORKSPACE / 'competition'
        if not marker.exists():
            with zipfile.ZipFile(archive) as compressed:
                compressed.extractall(WORKSPACE)
            marker.write_text(digest)
        with zipfile.ZipFile(archive) as compressed:
            inventory = {'archive_sha256': digest, 'members': [{'name': entry.filename,
                         'bytes': entry.file_size} for entry in compressed.infolist()]}
        (WORKSPACE / 'archive_manifest.json').write_text(json.dumps(inventory))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        print('CUDA', torch.__version__, torch.cuda.get_device_name(), flush=True)
        run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
        prepared = ROOT / 'artifacts' / 'prepared-full-v2'
        if not prepared.exists():
            run('-m', 'recsys_data', '--data', source, '--output', prepared, '--allow-train-duplicate-items')
        last = RUN / 'last.pt'
        args = ['-m', 'mlp', 'train', '--data', prepared, '--output', RUN]
        if last.exists():
            args.extend(['--resume', last])
        run(*args)
        best = RUN / 'best.pt'
        if not best.exists():
            print('No completed epoch; resumable checkpoint retained.', flush=True)
            return
        run('-m', 'mlp', 'evaluate', '--data', prepared, '--model', best, '--output', RUN / 'metrics.json')
        run('-m', 'mlp', 'predict', '--data', prepared, '--model', best, '--output', RUN / 'predictions-test.parquet')
        hashes = {path.name: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                  for path in RUN.iterdir() if path.suffix in ('.pt', '.parquet', '.json')}
        (RUN / 'checksums.json').write_text(json.dumps(hashes, indent=2))
        metrics = json.loads((RUN / 'metrics.json').read_text())
        journal = ROOT / 'experiments.md'
        with journal.open('a', encoding='utf-8') as destination:
            destination.write('\n\n### EXP-013-r01 — measured server results\n\n')
            destination.write('Completed: ' + time.ctime() + '\n\n```json\n')
            destination.write(json.dumps(metrics, indent=2) + '\n```\n')
            destination.write('Artifacts: runs/EXP-013-r01; checksums.json records SHA-256. No online submission.\n')
        print('Pipeline complete', time.ctime(), flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
