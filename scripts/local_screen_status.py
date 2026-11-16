"""One-shot status of the detached laptop screening queue."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gpu_suite import read
from scripts.local_ablation import runner_alive
from scripts.suite_status import render_suite
from scripts.diagnostics import render_diagnostics


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--logs', nargs='?', type=int, const=30, default=0)
    args = parser.parse_args()
    state = read(args.directory / 'status.json')
    if not state:
        print('Очередь ещё не создала статус. Проверьте worker.err.log.')
        return
    current = state.get('current') or {}
    progress = read(Path(current['output']) / 'progress.json') if current else None
    alive = runner_alive(state['controller_pid'])
    print(render_suite(dict(suite=state, progress=progress, task='Running' if alive else 'Stopped', gpu=[])))
    print('Устройство: CPU ноутбука')
    if not alive and state['status'] == 'running':
        print('ОШИБКА: контроллер остановлен, завершение очереди не подтверждено.')
    if args.logs:
        paths = [args.directory / 'worker.err.log', args.directory / 'worker.log']
        if current:
            paths += [Path(current['output']) / name for name in ('failure.txt', 'worker.log')]
        logs = [dict(path=str(path), text='\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-args.logs:]))
                for path in paths if path.exists()]
        print(render_diagnostics(dict(status=state['status'], error=state.get('error'), logs=logs)))


if __name__ == '__main__':
    main()
