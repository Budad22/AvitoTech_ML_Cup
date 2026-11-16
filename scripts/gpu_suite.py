"""Sequential GPU experiments. One detached task, frozen code, independent run outputs."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlp_trials.state import write_json

VARIANTS = ('baseline', 'history_stats', 'history_pool', 'history_attention',
            'softmax', 'lambda', 'context_stats', 'context_pool', 'cross', 'raw_norm', 'low_lr')


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def winner(cases):
    valid = [case for case in cases if case.get('status') == 'completed']
    return max(valid, key=lambda case: case['selection_ndcg']) if valid else None


def process_identity(pid):
    """Windows process creation time prevents PID reuse from matching an orphan."""
    if os.name != 'nt':
        return None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 4
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    times = [ctypes.c_ulonglong() for _ in range(4)]
    try:
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
            return None
        return times[0].value
    finally:
        kernel.CloseHandle(handle)


def run(args):
    variants = getattr(args, 'variants', None) or VARIANTS
    selection_only = getattr(args, 'selection_only', False)
    run_prefix = getattr(args, 'run_prefix', 'EXP-024')
    device = getattr(args, 'device', 'cuda')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'controller.lock').open('a+b')
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        if lock.read(1) == b'':
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        # Keep the machine awake while this controller is running too.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(data=str(args.data.resolve()), code=str(ROOT), python=sys.executable,
                  variants=list(variants), seed=42, epochs=args.epochs, hours=args.hours, batch_size=args.batch_size,
                  selection_only=selection_only, run_prefix=run_prefix, device=device)
    existing = read(directory / 'suite-config.json')
    if existing and existing != config:
        raise ValueError('Immutable suite configuration changed; choose another directory')
    write_json(directory / 'suite-config.json', config)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in ('mlp', 'mlp_trials', 'recsys_data', 'training_logs', 'scripts')
              for path in (ROOT / folder).rglob('*') if path.suffix in ('.py', '.ps1')}
    previous_hashes = read(directory / 'code-hashes.json')
    if previous_hashes and hashes != previous_hashes:
        raise ValueError('Frozen code changed; refusing to mix versions')
    write_json(directory / 'code-hashes.json', hashes)
    state = read(directory / 'status.json') or dict(suite_id=directory.name, cases=[], created_at=datetime.now(timezone.utc).isoformat())
    state.update(controller_pid=os.getpid(), status='running', total=len(variants), directory=str(directory))

    def update(**fields):
        state.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
        state['completed'] = sum(c.get('status') == 'completed' for c in state['cases'])
        state['failed'] = sum(c.get('status') == 'failed' for c in state['cases'])
        write_json(directory / 'status.json', state)

    orphan = state.get('child')
    if orphan and process_identity(orphan['pid']) == orphan['identity'] and orphan['identity'] is not None:
        update(phase='recovering_child')
        while process_identity(orphan['pid']) == orphan['identity']:
            time.sleep(5)
    state.pop('child', None)
    update(phase='initialize')

    def command(arguments, log, phase):
        update(phase=phase)
        with log.open('a', encoding='utf-8') as stream:
            stream.write('COMMAND ' + subprocess.list2cmdline([sys.executable, '-u', *arguments]) + '\n')
            stream.flush()
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            child = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, creationflags=flags)
            update(child=dict(pid=child.pid, identity=process_identity(child.pid)))
            code = child.wait()
        state.pop('child', None)
        update()
        if code:
            raise RuntimeError(f'{phase} exited with {code}; inspect {log}')

    try:
        reference = directory / 'schema.json'
        if not reference.exists():
            command(['-m', 'mlp_trials', 'schema', '--data', str(args.data), '--output', str(reference)],
                    directory / 'schema.log', 'inspect')
        for index, kind in enumerate(variants):
            run_id = f'{run_prefix}-{kind.replace("_", "-")}-r01'
            output = directory / 'runs' / run_id
            output.mkdir(parents=True, exist_ok=True)
            prior = next((c for c in state['cases'] if c['kind'] == kind), None)
            if prior and prior.get('status') in ('completed', 'failed'):
                continue
            if (directory / 'STOP_AFTER_CASE').exists():
                update(status='paused', phase='paused')
                return
            update(current=dict(index=index + 1, kind=kind, run_id=run_id, output=str(output)))
            if args.active_root:
                write_json(args.active_root / 'active-suite.json', dict(suite_directory=str(directory), suite_id=directory.name,
                                                                      task_name='AvitoMLPSuite'))
                write_json(args.active_root / 'active-workspace.json', dict(workspace=str(args.active_root), run_id=run_id,
                              run_directory=str(output), task_name='AvitoMLPSuite', suite_directory=str(directory)))
            case = dict(kind=kind, run_id=run_id, output=str(output))
            try:
                result = read(output / 'result.json')
                if not result:
                    arguments = ['-m', 'mlp_trials', 'train', '--data', str(args.data), '--schema', str(reference),
                                 '--output', str(output), '--kind', kind, '--device', device,
                                 '--epochs', str(args.epochs), '--hours', str(args.hours),
                                 '--batch-size', str(args.batch_size), '--seed', '42',
                                 '--history-window', '128' if kind == 'history_stats' else '64',
                                 '--learning-rate', '.0003' if kind == 'low_lr' else '.001']
                    if (output / 'last.pt').exists():
                        arguments += ['--resume', str(output / 'last.pt')]
                    command(arguments, output / 'worker.log', 'train')
                    result = read(output / 'result.json')
                if not result:
                    update(status='paused', phase='paused')
                    return
                case.update(result)
                case['checkpoint_sha256'] = hashlib.sha256((output / 'best.pt').read_bytes()).hexdigest()
            except Exception as error:
                case.update(status='failed', error=str(error))
                (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            state['cases'] = [c for c in state['cases'] if c['kind'] != kind] + [case]
            update()
        best = winner(state['cases'])
        if best and selection_only:
            update(winner=best['run_id'])
            write_json(directory / 'report.json', dict(winner=best, cases=state['cases'],
                                                       data_fingerprint=read(reference)['fingerprint']))
        if best and not selection_only:
            update(winner=best['run_id'], phase='final_evaluation')
            checkpoint = Path(best['output']) / 'best.pt'
            common = ['--data', str(args.data), '--model', str(checkpoint), '--batch-size', str(args.batch_size), '--device', device]
            final = {}
            for part in ('selection', 'control', 'all'):
                target = directory / f'winner-{part}.json'
                if not target.exists():
                    command(['-m', 'mlp_trials', 'evaluate', *common, '--part', part, '--output', str(target)],
                            directory / 'final.log', 'final_evaluation')
                final[part] = read(target)
            prediction = directory / 'submission-winner.parquet'
            if not prediction.exists():
                command(['-m', 'mlp_trials', 'predict', *common, '--part', 'all', '--output', str(prediction)],
                        directory / 'final.log', 'predict')
            from scripts.validate_submission import validate
            manifest = read(args.data / 'manifest.json')
            source = (args.data / manifest['source_relative_path']).resolve()
            audit = validate(prediction, source / 'sample_submission.parquet')
            write_json(directory / 'submission-audit.json', audit)
            write_json(directory / 'report.json', dict(winner=best, final=final, cases=state['cases'], submission=audit,
                                                       data_fingerprint=read(reference)['fingerprint']))
        update(status='completed_with_errors' if state['failed'] else 'completed', phase='completed', current=None)
    except Exception as error:
        update(status='error', phase='error', error=str(error))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--active-root', type=Path)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hours', type=float, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS)
    parser.add_argument('--selection-only', action='store_true', help='Keep control/test sealed during screening')
    parser.add_argument('--run-prefix', default='EXP-024')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if min(args.epochs, args.hours, args.batch_size) <= 0:
        parser.error('Training limits must be positive')
    run(args)


if __name__ == '__main__':
    main()
