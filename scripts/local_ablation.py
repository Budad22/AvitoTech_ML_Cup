"""Autonomous local train/evaluate queue over frozen code trees and read-only data."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.03)


def now():
    return datetime.now(timezone.utc).isoformat()


def runner_alive(pid):
    if os.name != 'nt':
        return True
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        result = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(result))) and result.value == 259
    finally:
        kernel.CloseHandle(handle)


def status(directory):
    path = directory / 'status.json'
    if not path.exists():
        print('ОЖИДАНИЕ: задача ещё не создала файл состояния.')
        return
    state = json.loads(path.read_text(encoding='utf-8'))
    labels = {'running': 'РАБОТАЕТ', 'completed': 'ЗАВЕРШЕНО', 'failed': 'ОШИБКА'}
    print(f"{labels.get(state['status'], state['status'])} · {state['experiment_id']}")
    if state['status'] == 'completed':
        print(f"Оба варианта обучены и оценены. Результаты: {directory / 'report.json'}")
        print('Можно возвращаться к ассистенту для разбора результатов.')
        return
    if state['status'] == 'failed':
        print(state.get('error', 'См. worker-error.txt'))
        return
    if not runner_alive(state['runner_pid']):
        print('ПРОЦЕСС ОСТАНОВЛЕН: итоговая оценка не подтверждена. Обратись к ассистенту.')
        return
    variant = state.get('variant')
    print(f"Вариант {state.get('variant_index', 0)}/{state.get('total_variants', 0)}: {variant or 'инициализация'}")
    print('Этап: ' + {'train': 'обучение', 'evaluate': 'итоговая оценка', 'initialize': 'подготовка'}.get(state['phase'], state['phase']))
    progress_path = directory / variant / 'run/progress.json' if variant else None
    if state['phase'] == 'train' and progress_path and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding='utf-8')).get('progress') or {}
        if progress.get('total'):
            print(f"Эпоха {progress.get('epoch')}/{progress.get('epochs')} · батч "
                  f"{progress.get('current')}/{progress['total']}")
    print(f"Последнее обновление (UTC): {state['updated_at']}")


def run(directory, resume=False):
    config = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
    state = dict(experiment_id=config['experiment_id'], status='running', phase='initialize',
                 total_variants=len(config['variants']), started_at=now(), runner_pid=os.getpid())
    state_path = directory / 'status.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding='utf-8'))
        if previous['status'] == 'completed':
            return
        if not resume or runner_alive(previous['runner_pid']):
            raise FileExistsError('Experiment already started; use --resume only after its runner has stopped')
        write_json(directory / f'interrupted-status-{time.time_ns()}.json', previous)
        state['resumed_at'] = now()
    data = Path(config['data'])
    manifest = data / 'manifest.json'
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    python = str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable
    started = time.monotonic()
    results = {}

    def emit(**fields):
        state.update(fields, updated_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(state_path, state)

    def command(arguments, code, output, phase):
        emit(phase=phase, command=arguments)
        with (output / f'{phase}.log').open('a' if resume else 'w', encoding='utf-8') as log:
            child = subprocess.Popen(arguments, cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=(subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW)
                                     if os.name == 'nt' else 0)
            emit(child_pid=child.pid)
            while child.poll() is None:
                time.sleep(3)
                emit()
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, arguments)

    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        emit()
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != config['data_manifest_sha256']:
            raise ValueError('Prepared dataset manifest changed')
        for index, variant in enumerate(config['variants'], 1):
            name = variant['name']
            if not name or Path(name).name != name:
                raise ValueError('Variant name must be a directory name')
            code = Path(variant['code'])
            for relative, expected in variant.get('code_sha256', {}).items():
                if hashlib.sha256((code / relative).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'{name}: frozen code changed: {relative}')
            output = directory / name / 'run'
            output.mkdir(parents=True, exist_ok=resume)
            emit(variant=name, variant_index=index)
            training = [python, '-m', 'mlp', 'train', '--data', str(data), '--output', str(output),
                        '--device', 'cpu', '--epochs', '2', '--hours', '0.2', '--batch-size', '64', '--seed', '42',
                        '--cpu-threads', '4', '--arrow-threads', '2', '--prefetch', '2', '--embedding-cache', 'memmap']
            if resume and (output / 'last.pt').exists():
                training += ['--resume', str(output / 'last.pt')]
            command(training, code, output, 'train')
            if not (output / 'best.pt').exists():
                raise RuntimeError(f'{name}: no evaluated checkpoint')
            evaluation = [python, '-m', 'mlp', 'evaluate', '--data', str(data), '--model', str(output / 'best.pt'),
                          '--output', str(output / 'metrics.json'), '--device', 'cpu', '--cpu-threads', '4']
            command(evaluation, code, output, 'evaluate')
            results[name] = dict(variant=variant, metrics=json.loads((output / 'metrics.json').read_text()),
                                 history=json.loads((output / 'history.json').read_text()),
                                 training_config=json.loads((output / 'config.json').read_text()),
                                 train_command=training, evaluation_command=evaluation)
            checksums = {}
            for path in output.glob('*'):
                if path.suffix in ('.pt', '.json'):
                    checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(output / 'checksums.json', checksums)
            write_json(directory / 'partial-report.json', results)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != config['data_manifest_sha256']:
            raise ValueError('Prepared dataset manifest changed during experiment')
        write_json(directory / 'report.json', dict(config=config, results=results, finished_at=now(),
                                                 total_seconds=time.monotonic() - started, online_submissions=0))
        emit(status='completed', phase='completed', child_pid=None)
    except BaseException as error:
        (directory / 'worker-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        emit(status='failed', phase='error', error=str(error), child_pid=None)
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'status'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted queue from saved checkpoints')
    args = parser.parse_args()
    directory = args.directory.resolve()
    status(directory) if args.action == 'status' else run(directory, args.resume)


if __name__ == '__main__':
    main()
