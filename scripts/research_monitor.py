"""Expose a research worker through the existing read-only training monitor.

Run this wrapper as a scheduled task and register ``root/api`` in jobs.json.
It writes the monitor's standard stage/progress/report files. No API restart
or changes to another architecture's jobs are needed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from mlp_next.supervisor import guard_lock, read
from mlp_trials.state import write_json


TRAIN_STAGE = 'MLP: обучение (предел12эпох, лимит2ч)'


def completed_results(root):
    """Keep the dashboard compact; retain every run in the report details."""
    names = {'all_scalars_softmax': 'Все числовые', 'memory_cosine': 'Внимание к истории',
             'plr_softmax': 'Бейзлайн', 'experts_4': 'Четыре эксперта',
             'memory_multi4': 'Многоголовое внимание', 'position_nuisance': 'Поправка на позицию'}
    best, details = {}, []
    for path in sorted(Path(root).glob('*/runs/*/final.json')):
        if not path.parent.parent.parent.name.startswith('full_'):
            continue
        result = read(path)
        if result.get('selection_ndcg') is not None:
            details.append(dict(path=str(path), **result))
            name = result['name']
            if name not in best or result['selection_ndcg'] > best[name]['selection_ndcg']:
                best[name] = result
    selected = sorted(best, key=lambda n: (-best[n]['selection_ndcg'], n))[:2]
    if 'plr_softmax' in best and 'plr_softmax' not in selected:
        selected.append('plr_softmax')
    return dict(ndcg={names.get(n, n.replace('_', ' ')) + ' · selection': best[n]['selection_ndcg']
                      for n in selected}, details=details, selected={n: best[n] for n in selected},
                scope='Historical selection; not official full val or website score',
                aggregation='Best selection checkpoint/seed per architecture; top two plus baseline')


def publish(root, *, running):
    """Map native research state to the monitor's existing file contract."""
    root = Path(root)
    output = root / 'api'
    output.mkdir(exist_ok=True)
    state = read(root / 'status.json')
    status = state.get('status')
    phase = state.get('phase', '')
    progress = state.get('progress') or {}
    current = state.get('current') or {}
    if status == 'completed':
        stage = 'completed'
        write_json(output / 'comparison.json', completed_results(root))
    elif status == 'paused':
        stage = 'paused'
    elif status == 'error' or not running:
        stage = 'failed'
        (output / 'failure.log').write_text(state.get('error') or 'Worker exited before completion', encoding='utf-8')
    elif phase == 'final_evaluation':
        stage = 'evaluate'
    elif phase in ('preflight', 'prepare', 'waiting_for_EXP-031'):
        stage = 'prepare'
    else:
        stage = TRAIN_STAGE
    # Steps are the actual 12-epoch upper cap. Omit linear ETA because early
    # stopping and the independent two-hour budget can end training sooner.
    write_json(output / 'progress.json', dict(step=progress.get('step', 0),
               steps=current.get('target') if stage == TRAIN_STAGE else None,
               elapsed_seconds=None, model=current.get('name'), seed=current.get('seed'),
               updated_at=state.get('updated_at'), queue_elapsed_seconds=state.get('elapsed_seconds'),
               loss=progress.get('loss'), steps_per_second=progress.get('steps_per_second'),
               loader_wait_pct=progress.get('loader_wait_pct')))
    (output / 'stage.txt').write_text(stage, encoding='utf-8')
    return stage


def monitor(root, command, *, poll=5.):
    """Keep the scheduled task alive while the independent supervisor runs."""
    root = Path(root)
    with guard_lock(root / 'api-monitor.lock'):
        output = root / 'api'
        output.mkdir(exist_ok=True)
        (output / 'failure.log').unlink(missing_ok=True)
        (output / 'stage.txt').write_text('prepare', encoding='utf-8')
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        with (root / 'api-monitor-worker.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=root, stdout=log, stderr=log, creationflags=flags)
        while child.poll() is None:
            try:
                publish(root, running=True)
            except (OSError, ValueError, KeyError) as error:
                # Monitoring failures must never interrupt the training worker.
                print(json.dumps(dict(event='monitor_publish_failed', error=str(error))), file=sys.stderr, flush=True)
            time.sleep(poll)
        publish(root, running=False)
        return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--bootstrap', required=True)
    args = parser.parse_args()
    raise SystemExit(monitor(args.root, [args.python, '-X', 'utf8', args.bootstrap]))


if __name__ == '__main__':
    main()
