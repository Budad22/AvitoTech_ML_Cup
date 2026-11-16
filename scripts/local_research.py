"""Run or inspect the detached local MLP/PLR research campaign."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def status(directory, logs):
    from scripts.gpu_suite import process_identity
    state_path = directory / 'status.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    print('EXP-029 · CPU ноутбука · ' + state.get('status', 'подготовка/контроллер ещё не запущен'))
    if state.get('error'):
        print('ОШИБКА: ' + state['error'])
    ledger_path = directory / 'ledger.json'
    failed = []
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
        failed = [result for records in ledger.values() for result in records.values() if result['status'] == 'failed']
        for result in failed:
            print(f"ОШИБКА ОПЫТА {result['name']}, seed {result['seed']}: {result['error']}")
    if state.get('status') == 'running' and process_identity(state['controller_pid']) != state['controller_identity']:
        print('ОШИБКА: контроллер остановлен; статус running устарел. Нужна проверка перед возобновлением.')
    phases = dict(prepare='подготовка кеша', screen_200='быстрый отбор: 200 шагов',
                  screen_800='продолжение: 800 шагов', confirm_3200='подтверждение: 3200 шагов',
                  confirm_6400='проверка медленного обучения: 6400 шагов',
                  confirmation_evaluation='оценка отложенных лент', completed='завершено', error='ошибка', paused='пауза')
    print(f"Этап: {phases.get(state.get('phase', 'prepare'), state.get('phase'))} · гипотез: {state.get('hypotheses', 44)} · "
          f"закончено этапов: {state.get('completed', 0)} · ошибок: {state.get('failed', 0)}")
    if state.get('phase', 'prepare') == 'prepare':
        path = directory / 'cache-preparation/progress.json'
        if path.exists():
            prep = json.loads(path.read_text(encoding='utf-8'))
            progress = prep.get('progress', {})
            print(f"Кеш признаков: {prep.get('phase')} / {prep.get('split', '')} · "
                  f"{progress.get('current', 0)}/{progress.get('total', '?')} лент")
    current = state.get('current') or {}
    progress = state.get('progress') or {}
    if current:
        print(f"{current['name']} · seed {current['seed']} · шаг {progress.get('step', '?')}/{current.get('target', '?')}")
        if 'loss' in progress:
            print(f"Loss {progress['loss']:.6f} · {progress.get('steps_per_second', 0):.2f} шаг/с · "
                  f"ожидание кеша {progress.get('loader_wait_pct', 0):.1f}%")
        if 'fast_ndcg' in progress:
            print(f"Fast selection NDCG {progress['fast_ndcg']:.9f}")
    print(f"Работает {state.get('elapsed_seconds', 0) / 60:.1f} мин. Лимит безопасности — не ETA.")
    print('Таблица: ' + str(directory / 'leaderboard.md'))
    if (directory / 'report.json').exists():
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        print('ВЫВОД: ' + report['explanation'])
        print('Отчёт: ' + str(directory / 'report.md'))
    if logs:
        paths = [directory / name for name in ('failure.txt', 'worker.err.log', 'prepare.err.log', 'worker.log')]
        paths = [directory / 'runs' / f"{r['name']}-s{r['seed']}" / 'failure.txt' for r in failed] + paths
        if current:
            paths.insert(0, directory / 'runs' / f"{current['name']}-s{current['seed']}" / 'failure.txt')
        for path in paths:
            if path.exists() and path.stat().st_size:
                print('\nФайл: ' + str(path))
                for line in path.read_text(encoding='utf-8', errors='replace').splitlines()[-logs:]:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        print(line)
                        continue
                    if isinstance(event, dict) and 'event' in event:
                        labels = dict(batch_completed='Шаг обучения', validation_completed='Проверка на fast val',
                                      case_failed='ОШИБКА ОПЫТА', run_failed='ОШИБКА КОНТРОЛЛЕРА')
                        print(f"{event.get('timestamp', '')} · {labels.get(event['event'], event['event'])} · {event.get('name', '')} "
                              f"seed={event.get('seed', '')} step={event.get('step', '')}/{event.get('total', '')} "
                              f"loss={event.get('loss', '')} fast_NDCG={event.get('fast_ndcg', '')} {event.get('error', '')}")
                    else:
                        print(line)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--logs', type=int, nargs='?', const=30, default=0)
    parser.add_argument('--maximum-hours', type=float, default=8.)
    args = parser.parse_args()
    if not 0 <= args.logs <= 2000:
        parser.error('--logs must be between 0 and 2000')
    if args.command == 'status':
        status(args.directory, args.logs)
    else:
        if not args.cache or args.maximum_hours <= 0:
            parser.error('run requires --cache and positive --maximum-hours')
        from mlp_research.runner import run
        run(args)


if __name__ == '__main__':
    main()
