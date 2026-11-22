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
    parser.add_argument('--packed', type=Path, help='Reuse immutable packed splits')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--sol-smoke', action='store_true',
                      help='Run sol.md section 6.1 using the full packed source')
    mode.add_argument('--sol-full', action='store_true', help='Run the three NN-1 seeds in section 6.2')
    parser.add_argument('--smoke-report', type=Path, help='Completed smoke report for section 6.2')
    parser.add_argument('--calibrate-smoke', action='store_true')
    parser.add_argument('--smoke-preset', default='position')
    parser.add_argument('--max-seconds', type=int, default=7200, help='Time limit per variant')
    args = parser.parse_args()
    if args.sol_full:
        if not args.smoke_report:
            parser.error('--sol-full requires a passing --smoke-report')
        report = json.loads(args.smoke_report.read_text(encoding='utf-8'))
        if report.get('status') != 'completed' or report.get('quality_gate') is not True:
            parser.error('The smoke has not passed the section 6.1 criteria')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from mlp_next.supervisor import guard_lock
    from mlp_trials.state import write_json, write_text

    root = args.output.resolve()
    work = root / 'work'
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, AVITO_ROOT=str(root), AVITO_RAW=str(args.data.resolve()),
               PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    packed = args.packed.resolve() if args.packed else root / 'packed'
    env['AVITO_PACK'] = str(packed)
    python = [sys.executable, '-X', 'utf8', '-u', '-m']
    with guard_lock(root / 'launcher.lock'):
        with (work / 'worker.log').open('a', encoding='utf-8') as log:
            if (work / 'report.json').exists():
                report = json.loads((work / 'report.json').read_text(encoding='utf-8'))
                if report['status'] == 'completed':
                    write_text(work / 'stage.txt', 'completed')
                    return
            (work / 'failure.log').unlink(missing_ok=True)
            try:
                try:
                    write_text(work / 'stage.txt', 'prepare')
                except OSError as error:
                    log.write(f'Monitor update deferred for prepare: {error}\n')
                    log.flush()
                use_full_source = args.sol_smoke or args.sol_full
                splits = ('val', 'test', 'train') if use_full_source else ('train', 'val')
                for split in splits:
                    if (packed / split / 'meta.json').exists():
                        continue
                    if args.packed:
                        raise FileNotFoundError(packed / split / 'meta.json')
                    cmd = [*python, 'hstu_lite.prep', '--split', split]
                    if split == 'train':
                        cmd.extend(['--train-shards', str(0 if use_full_source else args.train_shards)])
                    subprocess.run(cmd, cwd=project, env=env, stdout=log, stderr=log, check=True)
                cmd = [*python, 'hstu_lite.pilot', '--device', args.device,
                       '--steps', str(args.steps), '--batch-size', str(args.batch_size),
                       '--val-feeds', str(args.val_feeds), '--max-seconds', str(args.max_seconds)]
                if args.sol_smoke:
                    cmd.extend(['--sol-smoke', '--smoke-preset', args.smoke_preset])
                if args.sol_full:
                    cmd.append('--sol-full')
                if args.calibrate_smoke:
                    cmd.append('--calibrate-smoke')
                subprocess.run(cmd, cwd=project, env=env, stdout=log, stderr=log, check=True)
            except Exception:
                error = traceback.format_exc()
                (work / 'failure.log').write_text(error, encoding='utf-8')
                write_json(work / 'error.json', dict(status='failed', traceback=error))
                raise


if __name__ == '__main__':
    main()
