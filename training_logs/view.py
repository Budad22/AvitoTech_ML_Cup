"""Adapter from the existing Windows/MLP snapshot to a reusable dashboard view."""
from datetime import datetime, timezone


def build_view(snapshot, *, host, phase, label, legacy_events=()):
    config = snapshot.get('config') or {}
    state = snapshot.get('training_state') or {}
    if state.get('producer') == 'trainer' and phase == 'training':
        phases = {'initialize': ('initializing', 'Инициализация модели'),
                  'validate': ('evaluation', 'Валидация'), 'completed': ('transition', 'Обучение завершено'),
                  'error': ('error', 'Ошибка обучения')}
        phase, label = phases.get(state.get('phase'), (phase, label))
    view = dict(run_id=state.get('run_id', snapshot.get('run_id', 'EXP-013-r01')), architecture=state.get('architecture', 'MLP'),
                host=host, phase=phase, label=label, progress=state.get('progress', {}),
                metrics=dict(state.get('metrics') or {}), elapsed_seconds=state.get('elapsed_seconds', 0),
                budget_seconds=state.get('budget_seconds', config.get('hours', 12) * 3600),
                source=state.get('source', 'legacy log'), warnings=[])
    latest = next((e for e in reversed(list(legacy_events)) if 'batch' in e), None)
    stamp = (state.get('checkpoint_at') or state.get('timestamp')) if state.get('source') == 'checkpoint' else state.get('timestamp')
    if latest and latest.get('seconds', 0) >= view['elapsed_seconds']:
        view.update(progress=dict(epoch=latest['epoch'], epochs=config.get('epochs', 5),
                                  current=latest['batch'], total=snapshot.get('batches_per_epoch', 0), unit='батч'),
                    elapsed_seconds=latest.get('seconds', 0), source='worker.log')
        view['metrics']['train_loss'] = latest['loss']
        stamp = (snapshot.get('log') or {}).get('modified')
    history = [e for e in snapshot.get('history', []) if isinstance(e, dict) and 'ndcg' in e]
    suite = snapshot.get('suite') or {}
    if suite:
        current = suite.get('current') or {}
        view['warnings'].append(f"Очередь: {suite.get('completed', 0)}/{suite.get('total', '?')} завершено · ошибок {suite.get('failed', 0)} · {current.get('kind', suite.get('status', '?'))}")
        if state.get('phase') == 'completed' and suite.get('status') == 'running':
            view.update(phase='transition', label='Вариант завершён; очередь продолжает работу')
    error = suite.get('error') or state.get('error')
    if error:
        view['warnings'].insert(0, 'ОШИБКА: ' + str(error) + ' · Подробности: --logs')
    if history:
        view['metrics']['best_ndcg'] = max(e['ndcg'] for e in history)
        view['metrics']['val_ndcg'] = history[-1]['ndcg']
    final = snapshot.get('metrics') or {}
    if phase == 'complete' and final.get('selection'):
        view['metrics'].update(val_ndcg=final['selection']['ndcg'], val_logloss=final['selection']['logloss'])
        for part, key in (('all', 'full_val_ndcg'), ('control', 'control_ndcg')):
            if part in final:
                view['metrics'][key] = final[part]['ndcg']
    if view['metrics'].get('batches_per_second'):
        view['steps_per_second'] = view['metrics']['batches_per_second']
        view['speed_source'] = 'window'
    elif state.get('global_step') and view['elapsed_seconds']:
        view['steps_per_second'] = state['global_step'] / view['elapsed_seconds']
        if phase == 'training' and state['global_step'] >= 10 and view['steps_per_second'] < .1:
            view['warnings'].append('Низкий средний темп. Подробности процесса и ресурсов: --verbose.')
    migration = snapshot.get('migration') or {}
    if migration.get('stage') == 'copying':
        copied = migration.get('copied_bytes', 0) + migration.get('current_bytes', 0)
        total = migration.get('total_bytes', 0)
        view['warnings'].append(f'Перенос на SSD: {copied / 1024**3:.1f}/{total / 1024**3:.1f} ГиБ проверено/копируется.')
    for value in snapshot.get('gpu') or []:
        fields = value.split(',')
        if len(fields) == 4:
            view['gpu'] = dict(name=fields[0].strip(), used=float(fields[1]), total=float(fields[2]), utilization=float(fields[3]))
            break
    if stamp:
        age = max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(stamp.replace('Z', '+00:00'))).total_seconds()))
        view['freshness'] = f"Источник: {view['source']}  ·  прогресс сохранён {age // 60} мин {age % 60} с назад"
    else:
        view['freshness'] = 'Ожидается первое событие прогресса. Checkpoint сохраняется раз в 10 минут.'
    if snapshot.get('errors') or any('Traceback (most recent call last)' in s for s in snapshot.get('worker', [])):
        view['warnings'].append('В текущих логах есть диагностические сообщения: --logs 30')
    return view


def plain_view(view):
    lines = [f"{view['architecture']} / {view['run_id']} · {view['label']}"]
    progress = view.get('progress') or {}
    if progress.get('total'):
        lines.append(f"Эпоха {progress.get('epoch', '?')}/{progress.get('epochs', '?')} · "
                     f"{progress['current']}/{progress['total']} {progress.get('unit', 'шагов')} · "
                     f"{progress['current'] / progress['total']:.2%}")
    if view.get('metrics'):
        lines.append(' · '.join(f'{key}={value:.6g}' if isinstance(value, (int, float)) else f'{key}={value}'
                              for key, value in view['metrics'].items()))
    if view.get('elapsed_seconds'):
        lines.append(f"Измеренное время: {view['elapsed_seconds'] / 3600:.2f} ч · бюджет: {view.get('budget_seconds', 0) / 3600:.1f} ч")
    if view.get('gpu'):
        gpu = view['gpu']
        lines.append(f"GPU: {gpu['name']} · {gpu['utilization']}% · VRAM {gpu['used']:.0f}/{gpu['total']:.0f} МиБ")
    lines.append(view.get('freshness', ''))
    if view.get('source') == 'checkpoint':
        lines.append('Показан сохранённый прогресс; текущие шаги могут быть впереди.')
    lines.extend(view.get('warnings', []))
    return '\n'.join(lines)
