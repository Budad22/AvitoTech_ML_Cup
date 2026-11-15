"""Local viewer for any architecture: python -m training_logs --help."""
import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import sys

from rich.live import Live

from .dashboard import dashboard
from .view import build_view, plain_view
from .terminal import create_console


def demo_view():
    return dict(architecture='MLP', run_id='DEMO · вымышленные значения', host='preview',
                phase='training', label='Обучение', progress=dict(epoch=2, epochs=5, current=18420, total=49475, unit='батч'),
                metrics=dict(train_loss=.08421, val_ndcg=.73125, best_ndcg=.73681, learning_rate=.001),
                elapsed_seconds=6230, budget_seconds=43200, steps_per_second=7.9,
                gpu=dict(name='NVIDIA GeForce RTX 3060 Ti', utilization=81, used=2130, total=8192),
                freshness='Демонстрация оформления — не результат эксперимента.', source='demo', warnings=[])


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--demo', action='store_true', help='Preview with explicitly fictional values')
    parser.add_argument('--watch', type=float, nargs='?', const=10, default=0)
    parser.add_argument('--format', choices=['auto', 'rich', 'plain'], default='auto')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--logs', type=int, default=0)
    args = parser.parse_args()
    if not args.demo and not args.run_dir:
        parser.error('--run-dir is required unless --demo is used')
    if args.watch < 0 or args.logs < 0 or 0 < args.watch < 2:
        parser.error('--watch must be >=2 and --logs nonnegative')
    console = create_console(args.format)
    pretty = not args.json and (args.format == 'rich' or args.format == 'auto' and console.is_terminal)
    try:
        with ExitStack() as stack:
            live = stack.enter_context(Live(console=console, auto_refresh=False)) if pretty and args.watch else None
            while True:
                if args.demo:
                    view, state = demo_view(), {'demo': True}
                else:
                    path = args.run_dir / 'progress.json'
                    if not path.exists():
                        parser.exit(1, f'Пока нет событий: {path}\n')
                    state = json.loads(path.read_text(encoding='utf-8'))
                    phase = {'train': 'training', 'validate': 'evaluation', 'predict': 'prediction'}.get(state['phase'], state['phase'])
                    labels = dict(training='Обучение', evaluation='Валидация', prediction='Инференс',
                                  initialize='Инициализация', completed='Завершено', stopped='Остановлено', error='Ошибка')
                    view = build_view({'training_state': state}, host='local', phase=phase, label=labels.get(phase, phase))
                    view['gpu'] = (state.get('resources') or {}).get('gpu')
                details = None
                if args.logs and args.run_dir:
                    log = args.run_dir / 'events.jsonl'
                    if log.exists():
                        with log.open(encoding='utf-8') as source:
                            details = ''.join(deque(source, maxlen=args.logs))
                if args.json:
                    print(json.dumps(dict(state=state, view=view), ensure_ascii=False), flush=True)
                elif pretty:
                    panel = dashboard(view, width=console.width, height=console.height, details=details)
                    live.update(panel, refresh=True) if live else console.print(panel)
                else:
                    print(plain_view(view), flush=True)
                    if details:
                        print(details)
                if not args.watch:
                    break
                time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
