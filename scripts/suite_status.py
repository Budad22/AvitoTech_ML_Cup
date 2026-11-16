"""Read-only queue status over the existing SSH connection."""
import base64
import json


def read_suite(client):
    script = r"""+$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
$active=Get-Content 'Q:\AVITOCUP\active-suite.json' -Raw | ConvertFrom-Json
$root=[IO.Path]::GetFullPath($active.suite_directory)
if (!$root.StartsWith('Q:\AVITOCUP\experiments\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Invalid suite directory'}
$state=Get-Content "$root\status.json" -Raw | ConvertFrom-Json
$progress=if ($state.current.output -and (Test-Path "$($state.current.output)\progress.json")) {Get-Content "$($state.current.output)\progress.json" -Raw | ConvertFrom-Json}
@{suite=$state; progress=$progress; task=(Get-ScheduledTask -TaskName AvitoMLPSuite).State.ToString(); gpu=@(& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)} | ConvertTo-Json -Depth 15 -Compress
"""
    encoded = base64.b64encode(script.encode('utf-16-le')).decode()
    _, stdout, stderr = client.exec_command('powershell -NoProfile -NonInteractive -EncodedCommand ' + encoded, timeout=60)
    result, error = stdout.read(), stderr.read()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error.decode('utf-8', errors='replace'))
    return json.loads(result.decode('utf-8-sig'))


def render_suite(snapshot):
    suite = snapshot['suite']
    lines = [f"Очередь {suite['suite_id']}: {suite['status']} · задача {snapshot['task']}",
             f"Завершено {suite.get('completed', 0)}/{suite['total']} · ошибок {suite.get('failed', 0)}"]
    current = suite.get('current')
    if current:
        lines.append(f"Сейчас {current['index']}/{suite['total']}: {current['kind']} · этап {suite.get('phase', '?')}")
        progress = snapshot.get('progress') or {}
        position = progress.get('progress') or {}
        if position:
            lines.append(f"Эпоха {position.get('epoch', '?')}/{position.get('epochs', '?')} · батч {position.get('current', 0)}/{position.get('total', '?')}")
        metrics = progress.get('metrics') or {}
        if 'train_loss' in metrics:
            lines.append(f"Loss {metrics['train_loss']:.5f} · {metrics.get('batches_per_second', 0):.1f} батч/с · ожидание данных {metrics.get('loader_wait_pct', 0):.1f}%")
    for case in suite.get('cases', []):
        score = f"selection NDCG {case['selection_ndcg']:.6f}" if case.get('status') == 'completed' else case.get('error', '')
        lines.append(f"  {case['kind']}: {case['status']} · {score}")
    if suite.get('winner'):
        lines.append('Лидер: ' + suite['winner'])
    if suite.get('error'):
        lines.append('Ошибка очереди: ' + suite['error'])
    lines.append('GPU: нагрузка %, память МиБ / всего МиБ: ' + '; '.join(snapshot.get('gpu') or []))
    lines.append('Артефакты: ' + suite['directory'])
    return '\n'.join(lines)
