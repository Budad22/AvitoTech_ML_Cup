"""Screen every architecture on nested read-only samples, then train one full-data winner."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlp_trials.state import write_json
from scripts.gpu_suite import VARIANTS, process_identity, read, winner


def sample_indices(count):
    if count < 8:
        raise ValueError('Screening requires at least eight source shards')
    small = [round(i * (count - 1) / 3) for i in range(4)]
    large = sorted(set(small + [round(i * (count - 1) / 7) for i in range(8)]))
    # Retain the original four-file sample; add four evenly spread new shards.
    additions = [i for i in large if i not in small]
    large = sorted(small + [additions[round(i * (len(additions) - 1) / 3)] for i in range(4)])
    return small, large


def link_directory(target, source):
    if os.name == 'nt':
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(target), str(source)], check=True,
                       stdout=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        target.symlink_to(source, target_is_directory=True)


def sample_view(full, output, indices):
    """Only a new manifest and directory links are created; source Parquet is never rewritten."""
    import pyarrow.parquet as pq
    full, output = full.resolve(), output.resolve()
    parent = read(full / 'manifest.json')
    files = parent['splits']['train']['files']
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(files) for i in indices):
        raise ValueError('Invalid or duplicate sample indices')
    identity = dict(parent=str(full), parent_sha256=hashlib.sha256((full / 'manifest.json').read_bytes()).hexdigest(),
                    indices=indices)
    if output.exists():
        if read(output / 'sample.json') != identity or not (output / 'manifest.json').exists():
            raise ValueError('Existing sample differs or is incomplete; use a fresh directory')
        return read(output / 'manifest.json')
    output.mkdir(parents=True)
    for split in parent['splits']:
        link_directory(output / split, full / split)
    manifest = json.loads(json.dumps(parent))
    source = (full / parent['source_relative_path']).resolve()
    manifest['source_relative_path'] = os.path.relpath(source, output)
    train = manifest['splits']['train']
    train['files'] = [files[i] for i in indices]
    train['feeds'] = sum(file['feeds'] for file in train['files'])
    train['candidates'] = train['history_events'] = 0
    for file in train['files']:
        table = pq.read_table(full / 'train/features' / file['name'], columns=['candidate_count', 'history_length'])
        train['candidates'] += int(table['candidate_count'].to_numpy().sum())
        train['history_events'] += int(table['history_length'].to_numpy().sum())
    train['complete_impressions'] = False
    train['duplicate_candidate_occurrences'] = None  # Not re-audited for this view.
    manifest['screening'] = dict(identity, fraction_feeds=train['feeds'] / parent['splits']['train']['feeds'])
    write_json(output / 'sample.json', identity)
    write_json(output / 'manifest.json', manifest)
    return manifest


def promotion(cases):
    """Choose only from the larger sample after all 11 successful comparisons exist."""
    larger = [case for case in cases if case.get('stage') == '1p56']
    if {case['kind'] for case in larger if case.get('status') == 'completed'} != set(VARIANTS):
        raise ValueError('Cannot promote: all 11 larger-sample results must succeed')
    return winner(larger)


@contextmanager
def controller_lock(path):
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args):
    stages = getattr(args, 'stages', None) or ['0p78', '1p56', 'full']
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with controller_lock(directory / 'screen.lock'):
        config = dict(data=str(args.data.resolve()), code=str(ROOT), seed=42, batch_size=64,
                      epochs=5, screen_hours=args.screen_hours, full_hours=2,
                      parent_sha256=hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(), stages=stages,
                      run_prefix=getattr(args, 'run_prefix', 'EXP-026'))
        if read(directory / 'config.json') not in (None, config):
            raise ValueError('Screen configuration changed')
        write_json(directory / 'config.json', config)
        hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for folder in ('mlp', 'mlp_trials', 'recsys_data', 'training_logs', 'scripts')
                  for path in (ROOT / folder).rglob('*') if path.suffix in ('.py', '.ps1')}
        if read(directory / 'code-hashes.json') not in (None, hashes):
            raise ValueError('Frozen source changed')
        write_json(directory / 'code-hashes.json', hashes)
        state = read(directory / 'status.json') or dict(suite_id=directory.name, cases=[], total=sum(1 if s == 'full' else 11 for s in stages),
                                                       directory=str(directory), completed=0, failed=0)
        def update(**fields):
            from datetime import datetime, timezone
            state.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
            state['completed'] = sum(c['status'] == 'completed' for c in state['cases'])
            state['failed'] = sum(c['status'] == 'failed' for c in state['cases'])
            write_json(directory / 'status.json', state)
        orphan = state.get('child')
        if orphan and orphan['identity'] is not None:
            while process_identity(orphan['pid']) == orphan['identity']:
                time.sleep(5)
        update(status='running', phase='prepare', child=None)
        if args.active_root:
            write_json(args.active_root / 'active-suite.json', dict(suite_directory=str(directory),
                       suite_id=directory.name, task_name='AvitoMLPSuite'))
        try:
            indices = sample_indices(len(read(args.data / 'manifest.json')['splits']['train']['files']))
            for stage, selected in zip(('0p78', '1p56'), indices):
                if stage in stages:
                    sample_view(args.data, directory / 'data' / stage, selected)
            for stage_index, stage in enumerate(stages):
                offset = sum(1 if s == 'full' else 11 for s in stages[:stage_index])
                target = directory / stage
                data = args.data if stage == 'full' else directory / 'data' / stage
                arguments = ['scripts/gpu_suite.py', '--directory', str(target), '--data', str(data),
                             '--run-prefix', getattr(args, 'run_prefix', 'EXP-026') + '-' + stage, '--hours', str(2 if stage == 'full' else args.screen_hours)]
                if stage == 'full':
                    best = promotion(state['cases'])
                    write_json(directory / 'promotion.json', dict(kind=best['kind'], selection_ndcg=best['selection_ndcg'],
                               source_run=best['run_id'], criterion='larger sample selection NDCG', cases=state['cases']))
                    update(winner=best['kind'])
                    arguments += ['--variants', best['kind']]
                else:
                    arguments += ['--selection-only']
                if not (target / 'report.json').exists():
                    target.mkdir(exist_ok=True)
                    with (target / 'controller.log').open('a', encoding='utf-8') as log:
                        child = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT, stdout=log,
                                                 stderr=subprocess.STDOUT,
                                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                        update(child=dict(pid=child.pid, identity=process_identity(child.pid)))
                        while child.poll() is None:
                            current = read(target / 'status.json') or {}
                            cases = [dict(case, stage=stage) for case in current.get('cases', [])]
                            position = current.get('current')
                            if position:
                                position = dict(position, index=offset + position['index'], kind=stage + '/' + position['kind'])
                                if args.active_root:
                                    write_json(args.active_root / 'active-workspace.json', dict(workspace=str(args.active_root),
                                               run_id=position['run_id'], run_directory=position['output'],
                                               suite_directory=str(directory), task_name='AvitoMLPSuite'))
                            update(phase=stage + '/' + current.get('phase', 'initialize'), current=position,
                                   cases=[case for case in state['cases'] if case.get('stage') != stage] + cases)
                            if (directory / 'STOP_AFTER_CASE').exists():
                                (target / 'STOP_AFTER_CASE').touch()
                            time.sleep(5)
                        if child.returncode:
                            raise RuntimeError(f'{stage} controller failed; inspect {target}/controller.log')
                report = read(target / 'report.json')
                if not report:
                    update(status='paused', phase='paused', child=None)
                    return
                update(cases=[case for case in state['cases'] if case.get('stage') != stage] +
                             [dict(case, stage=stage) for case in report['cases']], child=None)
                if (directory / 'STOP_AFTER_CASE').exists():
                    update(status='paused', phase='paused')
                    return
            write_json(directory / 'report.json', dict(screening=state['cases'], promotion=read(directory / 'promotion.json'),
                                                       full=read(directory / 'full/report.json')))
            update(status='completed', phase='completed', current=None)
        except Exception as error:
            update(status='error', phase='error', error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--active-root', type=Path)
    parser.add_argument('--screen-hours', type=float, default=.2)
    parser.add_argument('--stages', nargs='+', choices=['0p78', '1p56', 'full'], default=['0p78', '1p56', 'full'])
    parser.add_argument('--run-prefix', default='EXP-026')
    options = parser.parse_args()
    if options.screen_hours <= 0:
        parser.error('screen-hours must be positive')
    if len(set(options.stages)) != len(options.stages) or ('full' in options.stages and ('1p56' not in options.stages or options.stages[-1] != 'full')):
        parser.error('Stages must be unique; full requires a preceding 1p56 stage')
    run(options)
