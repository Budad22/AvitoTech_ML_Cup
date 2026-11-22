"""Prepare a private packed sample and run the resumable HSTU-lite pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--train-shards', type=int, default=4)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-feeds', type=int, default=2000)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from mlp_next.supervisor import guard_lock
    from mlp_trials.state import write_json

    root = args.output.resolve()
    work = root / 'work'
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, AVITO_ROOT=str(root), AVITO_RAW=str(args.data.resolve()),
               PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    python = [sys.executable, '-X', 'utf8', '-u', '-m']
    with guard_lock(root / 'launcher.lock'):
        with (work / 'worker.log').open('a', encoding='utf-8') as log:
            if (work / 'report.json').exists():
                report = json.loads((work / 'report.json').read_text(encoding='utf-8'))
                if report['status'] == 'completed':
                    (work / 'stage.txt').write_text('completed', encoding='utf-8')
                    return
            (work / 'failure.log').unlink(missing_ok=True)
            try:
                (work / 'stage.txt').write_text('prepare', encoding='utf-8')
                for split in ('train', 'val'):
                    if (root / 'packed' / split / 'meta.json').exists():
                        continue
                    cmd = [*python, 'hstu_lite.prep', '--split', split]
                    if split == 'train':
                        cmd.extend(['--train-shards', str(args.train_shards)])
                    subprocess.run(cmd, cwd=project, env=env, stdout=log, stderr=log, check=True)
                subprocess.run([*python, 'hstu_lite.pilot', '--device', args.device,
                                '--steps', str(args.steps), '--batch-size', str(args.batch_size),
                                '--val-feeds', str(args.val_feeds)], cwd=project, env=env,
                               stdout=log, stderr=log, check=True)
            except Exception:
                error = traceback.format_exc()
                (work / 'failure.log').write_text(error, encoding='utf-8')
                write_json(work / 'error.json', dict(status='failed', traceback=error))
                raise


if __name__ == '__main__':
    main()
