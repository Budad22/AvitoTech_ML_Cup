"""Rich presentation for any architecture's training view; no remote I/O."""
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from rich.padding import Padding


CYAN = '#56d4dd'
PURPLE = '#b39df3'
MUTED = '#8493a5'


class _PanelPair:
    """Match both panel dimensions using their rendered content at terminal width."""
    def __init__(self, left, right):
        self.panels = (left, right)

    def __rich_console__(self, console, options):
        side_by_side = options.max_width >= 90
        panel_width = (options.max_width - 2) // 2 if side_by_side else options.max_width
        measured = options.update(width=panel_width, height=None)
        for panel in self.panels:
            panel.width = panel_width
            panel.height = None
        height = max(len(console.render_lines(panel, measured, pad=False)) for panel in self.panels)
        for panel in self.panels:
            panel.height = height
        if side_by_side:
            columns = Table.grid(padding=0)
            columns.add_column(width=panel_width)
            columns.add_column(width=2)
            columns.add_column(width=panel_width)
            columns.add_row(self.panels[0], Text(''), self.panels[1])
            yield columns
        else:
            yield from self.panels


def duration(seconds):
    seconds = max(0, int(seconds or 0))
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def number(value):
    return f'{int(value):,}'.replace(',', ' ')


def dashboard(view, *, width=100, height=None, details=None):
    """A renderable independent of console/TTY, framework and architecture."""
    status_color = 'green' if view['phase'] == 'training' else 'red' if view['phase'] in ('stopped', 'error') else CYAN
    header = Text('TRAINING  ', style=f'bold {CYAN}')
    header.append(str(view.get('architecture', 'Model')), 'bold white')
    header.append('  /  ' + str(view.get('run_id', 'run')), MUTED)
    subtitle = Text(str(view.get('host', 'local')), style=MUTED)
    subtitle.append('   ● ' + view['label'], status_color)
    progress = view.get('progress') or {}
    compact = width < 90 and height is not None and height <= 30
    current, total = progress.get('current', 0), progress.get('total', 0)
    epoch = progress.get('epoch')
    title = Text(f"Эпоха {epoch} / {progress.get('epochs', '?')}" if epoch else 'Ожидание данных прогресса', style='bold white')
    if total:
        percent = min(100, current / total * 100)
        caption = Text(f"{number(current)} / {number(total)} {progress.get('unit', 'шагов')}  ·  {percent:.2f}%", style=CYAN)
        bar = ProgressBar(total=total, completed=min(total, current), complete_style=CYAN,
                          finished_style='green', pulse_style=CYAN)
    else:
        caption = Text('Точные значения появятся после первого события.', style=MUTED)
        bar = ProgressBar(total=100, completed=0, complete_style=CYAN)
    metrics = Table.grid(expand=True, padding=(0, 1))
    metrics.add_column(style=MUTED)
    metrics.add_column(justify='right', style='bold white')
    paired = compact and width >= 70
    if paired:
        metrics.add_column(style=MUTED)
        metrics.add_column(justify='right', style='bold white')
    labels = {'train_loss': 'Train loss', 'val_logloss': 'Val logloss', 'val_ndcg': 'Val NDCG',
              'best_ndcg': 'Лучший NDCG', 'learning_rate': 'Learning rate',
              'batches_per_second': 'Батчей/с, текущее окно', 'loader_wait_pct': 'Ожидание данных, %',
              'candidates_per_second': 'Кандидатов/с'}
    values = view.get('metrics') or {}
    if not values:
        metrics.add_row('Метрики', 'ожидаются')
    cells = []
    for key, value in values.items():
        if key == 'learning_rate' and isinstance(value, (int, float)):
            text = f'{value:.2e}'
        elif isinstance(value, (int, float)):
            text = f'{value:.6g}'
        else:
            text = str(value)
        cells.extend((Text(labels.get(key, key)), Text(text)))
        if not paired or len(cells) == 4:
            metrics.add_row(*cells)
            cells = []
    if cells:
        metrics.add_row(*cells)
    padding = (0, 1) if compact else (1, 2)
    work_items = [title, Padding(bar, 0), caption, metrics] if compact else [title, Text(''), Padding(bar, 0), caption, Text(''), metrics]
    work = Panel(Group(*work_items), title='[bold]ПРОГРЕСС[/bold]',
                 border_style=CYAN, box=box.ROUNDED, padding=padding)
    resources = Table.grid(expand=True, padding=(0, 1))
    resources.add_column(style=MUTED)
    resources.add_column(justify='right', style='white')
    resources.add_row('Прошло', duration(view.get('elapsed_seconds')))
    if view.get('budget_seconds'):
        resources.add_row('Бюджет', duration(view['budget_seconds']))
        resources.add_row('Осталось в бюджете', duration(view['budget_seconds'] - (view.get('elapsed_seconds') or 0)))
    if view.get('steps_per_second'):
        speed = view['steps_per_second']
        unit = progress.get('unit', 'шаг')
        label = 'Темп, текущее окно' if view.get('speed_source') == 'window' else 'Темп, средний'
        resources.add_row(label, f'{speed:.2f} {unit}/с' if speed >= 1 else f'{1 / speed:.1f} с/{unit}')
    gpu = view.get('gpu')
    resource_items = [resources]
    if gpu:
        resource_items += [Text(''), Text(gpu['name'], style=PURPLE),
            Text(f"GPU {gpu['utilization']}%  ·  VRAM {number(gpu['used'])} / {number(gpu['total'])} МиБ", style=MUTED),
            ProgressBar(total=100, completed=max(0, min(100, gpu['utilization'])), complete_style=PURPLE)]
    resources_panel = Panel(Group(*resource_items), title='[bold]ВРЕМЯ И РЕСУРСЫ[/bold]',
                            border_style='#48556b', box=box.ROUNDED, padding=padding)
    content = _PanelPair(work, resources_panel)
    foot = Text(view.get('freshness', ''), style=MUTED)
    if view.get('source') == 'checkpoint':
        foot.append('\nПоказан сохранённый прогресс; текущие шаги могут быть впереди.', 'yellow')
    warnings = [Text(message, style='yellow') for message in view.get('warnings', [])]
    extra = [Panel(Text(details), title='ПОДРОБНОСТИ', border_style='#48556b')] if details else []
    if compact:
        header.append('  ·  ' + view['label'], status_color)
    heading = header if compact else Group(header, subtitle)
    return Group(Panel(heading, border_style=CYAN, box=box.ROUNDED), content, foot, *warnings, *extra)
