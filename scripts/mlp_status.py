"""Read-only stage detection and human-readable progress reports."""
import json
import gzip
from datetime import datetime, timezone
from pathlib import Path


def read_snapshot(client, root, *, fast=False, previous=None):
    import base64
    source = (Path(__file__).with_name('status_snapshot.ps1')).read_text(encoding='utf-8')
    script = ('$quick=$true;\n' if fast else '$quick=$false;\n') + '\n'.join(line.strip() for line in source.splitlines()
                       if line.strip() and not line.lstrip().startswith('#'))
    encoded = base64.b64encode(script.encode('utf-16-le')).decode()
    if len(encoded) + 70 >= 7800:
        payload = base64.b64encode(gzip.compress(script.encode('utf-8'))).decode()
        wrapper = (f"$m=[IO.MemoryStream]::new([Convert]::FromBase64String('{payload}')); "
                   "$g=[IO.Compression.GZipStream]::new($m,[IO.Compression.CompressionMode]::Decompress); "
                   "$r=[IO.StreamReader]::new($g,[Text.Encoding]::UTF8); "
                   "& ([ScriptBlock]::Create($r.ReadToEnd())); $r.Dispose()")
        encoded = base64.b64encode(wrapper.encode('utf-16-le')).decode()
    if len(encoded) + 70 >= 8191:
        raise ValueError('Status collector exceeds the Windows SSH command length limit')
    stdin, stdout, stderr = client.exec_command(
        'powershell -NoProfile -NonInteractive -EncodedCommand ' + encoded, timeout=90)
    stdin.close()
    payload, error = stdout.read(), stderr.read()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error.decode('utf-8', errors='replace'))
    snapshot = json.loads(payload.decode('utf-8-sig'))
    if fast and previous:
        changed = (snapshot['task']['state'] != previous['task']['state'] or
                   snapshot.get('run_id') != previous.get('run_id') or
                   (snapshot.get('training_state') or {}).get('phase') != (previous.get('training_state') or {}).get('phase') or
                   snapshot.get('last_command') != previous.get('last_command'))
        if changed:
            return read_snapshot(client, root)
        snapshot = {**previous, **snapshot}
    return snapshot


def stage(snapshot):
    """Use current processes first; a stale log never implies active training."""
    running = snapshot['task']['state'] == 'Running'
    if snapshot.get('complete'):
        return 'complete', 'Пайплайн завершён'
    if not running:
        code = snapshot['task'].get('result')
        return 'stopped', f'Задача не работает (код завершения: {code})'
    commands = ' '.join(p.get('CommandLine') or '' for p in snapshot.get('processes', [])).replace('-m mlp_trials ', '-m mlp ')
    for pattern, key, label in (
        ('-m mlp train', 'training', 'Обучение MLP'),
        ('-m mlp evaluate', 'evaluation', 'Оценка лучшей модели'),
        ('-m mlp predict', 'prediction', 'Предсказания для test'),
        ('-m recsys_data', 'preparation', 'Подготовка данных'),
    ):
        if pattern in commands:
            # Vocabulary inspection precedes the explicit training-start event.
            if key == 'training' and not any('training_started' in line for line in snapshot.get('worker', [])):
                if not snapshot.get('last_checkpoint'):
                    return 'initializing', 'Подготовка признаков и словаря MLP'
            return key, label
    if snapshot.get('partial') and not snapshot.get('archive'):
        return 'download', 'Загрузка датасета'
    if snapshot.get('archive') and not snapshot.get('extracted'):
        return ('extraction', 'Распаковка архива') if snapshot.get('extracted_bytes') else ('verification', 'Проверка SHA-256 архива')
    if snapshot.get('staging_files'):
        return 'preparation', 'Подготовка данных'
    if snapshot.get('prepared'):
        return 'transition', 'Переход между этапами (см. --logs)'
    if snapshot.get('extracted'):
        return 'checks', 'Проверка CUDA и тесты'
    return 'setup', 'Установка окружения'


def progress_events(snapshot):
    events = []
    for line in snapshot.get('worker', []):
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                events.append(item)
        except (ValueError, TypeError):
            pass
    return events


def age(value):
    if not value:
        return 'нет'
    seconds = max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(value.replace('Z', '+00:00'))).total_seconds()))
    return f'{seconds // 3600} ч {(seconds % 3600) // 60} мин назад' if seconds >= 3600 else f'{seconds // 60} мин {seconds % 60} с назад'


def render(snapshot, host, verbose=False, logs=0, previous=None):
    key, label = stage(snapshot)
    run_id = snapshot.get('run_id', 'EXP-013-r01')
    lines = [f'MLP · {host} · {run_id}', f'Стадия: {label}']
    current = snapshot.get('archive') or snapshot.get('partial')
    total = snapshot.get('total_bytes')
    if current:
        size = current['bytes']
        downloaded = f'{size / 1e9:.2f} ГБ'
        if total:
            downloaded += f' / {total / 1e9:.2f} ГБ ({size / total:.1%})'
        lines.append('Архив: ' + downloaded)
        if key == 'download' and previous and previous.get('partial'):
            elapsed = (datetime.fromisoformat(snapshot['checked_at']) - datetime.fromisoformat(previous['checked_at'])).total_seconds()
            delta = size - previous['partial']['bytes']
            if elapsed > 0 and delta > 0:
                speed = delta / elapsed
                detail = f'Скорость: {speed / 1e6:.1f} МБ/с'
                if total and total > size:
                    detail += f' · осталось примерно {(total - size) / speed / 60:.0f} мин'
                lines.append(detail)
    if key == 'preparation':
        lines.append(f"Промежуточных Parquet-файлов: {snapshot.get('staging_files', 0)}")
    if key == 'extraction':
        unpacked, expected = snapshot.get('extracted_bytes', 0), snapshot.get('extracted_total_bytes')
        detail = f'Распаковано не менее: {unpacked / 1e9:.2f} ГБ'
        if expected:
            detail += f' / {expected / 1e9:.2f} ГБ ({unpacked / expected:.1%})'
        lines.append(detail)
    if key in ('extraction', 'verification', 'preparation', 'initializing') and snapshot.get('io'):
        read = sum(float(p.get('IOReadBytesPersec') or 0) for p in snapshot['io'])
        written = sum(float(p.get('IOWriteBytesPersec') or 0) for p in snapshot['io'])
        lines.append(f'Ввод/вывод: чтение {read / 1e6:.1f} МБ/с · запись {written / 1e6:.1f} МБ/с')
    events = progress_events(snapshot)
    batch = next((e for e in reversed(events) if 'batch' in e), None)
    if key == 'training' and batch:
        config = snapshot.get('config') or {}
        lines.append(f"Эпоха: {batch['epoch']}/{config.get('epochs', 5)} · батч: {batch['batch']} · loss: {batch['loss']:.5f}")
        lines.append(f"Время обучения: {batch.get('seconds', 0) / 3600:.2f} ч / {config.get('hours', 12)} ч")
    elif key not in ('evaluation', 'prediction', 'complete'):
        lines.append('Обучение: ' + ('процесс остановлен; см. --logs' if key == 'stopped' else 'обновлений батчей пока нет' if key == 'training' else 'сейчас не выполняется'))
    history = [e for e in snapshot.get('history', []) if isinstance(e, dict) and 'ndcg' in e]
    if history:
        best = max(history, key=lambda e: e['ndcg'])
        lines.append(f"Лучший NDCG: {best['ndcg']:.6f} · эпоха: {best['epoch']}")
    for gpu in snapshot.get('gpu', []):
        fields = [field.strip() for field in gpu.split(',')]
        if len(fields) == 4:
            lines.append(f'GPU: {fields[0]} · загрузка {fields[3]}% · VRAM {fields[1]}/{fields[2]} МиБ')
    if snapshot.get('last_checkpoint'):
        lines.append('Checkpoint: ' + age(snapshot['last_checkpoint']['modified']))
    if snapshot.get('log'):
        lines.append('Лог обновлён: ' + age(snapshot['log']['modified']))
    errors = [s for s in snapshot.get('errors', []) if s.strip()]
    traceback = any('Traceback (most recent call last)' in s for s in snapshot.get('worker', []))
    if errors or traceback:
        lines.append('В логах есть сообщения об ошибках; подробности: --logs 30')
    if verbose:
        base = snapshot.get('workspace', 'E:\\AVITOCUP')
        config = snapshot.get('config') or {}
        lines += [f"Задача: {snapshot['task']['state']} · код: {snapshot['task'].get('result')}",
                  f"Защита от сна: {snapshot.get('awake')}",
                  f"Подготовленные данные: {config.get('data', 'не указаны')}",
                  f"Кеш: {config.get('embedding_cache', 'memmap')} · очередь: {config.get('prefetch', 0)} · batch_size: {config.get('batch_size', '?')}",
                  f'Каталог запуска: {base}\\AvitoTech_ML_Cup\\runs\\{run_id}',
                  'Процессы:']
        lines += [f"  PID {p['ProcessId']}: {p['CommandLine']}" for p in snapshot.get('processes', [])]
    if logs:
        for name, field in (('Пайплайн', 'worker'), ('Ошибки', 'errors'), ('Установка', 'bootstrap')):
            entries = snapshot.get(field) or []
            if entries:
                lines += [f'--- {name}: последние {min(logs, len(entries))} строк ---', *entries[-logs:]]
    return '\n'.join(lines)
