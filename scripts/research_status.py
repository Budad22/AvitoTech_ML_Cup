"""Read-only GPU research status; no terminal rendering runs on the server."""
import base64
import gzip
import json


def read_research(client, lines=0):
    script = r"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
if (!(Test-Path 'Q:\AVITOCUP\active-research.json')) {'null'; exit}
$active=Get-Content 'Q:\AVITOCUP\active-research.json' -Raw | ConvertFrom-Json
$root=[IO.Path]::GetFullPath($active.directory)
if (!$root.StartsWith('Q:\AVITOCUP\experiments\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Invalid research directory'}
$state=Get-Content "$root\status.json" -Raw | ConvertFrom-Json
$process=Get-Process -Id $state.controller_pid -ErrorAction SilentlyContinue
$alive=$false
if ($process) {$alive=$process.StartTime.ToUniversalTime().ToFileTimeUtc() -eq $state.controller_identity}
$stages=@{}
$ledger=if (Test-Path "$root\ledger.json") {Get-Content "$root\ledger.json" -Raw | ConvertFrom-Json}
foreach ($stage in $ledger.PSObject.Properties) {
 $records=@($stage.Value.PSObject.Properties | ForEach-Object {$_.Value})
 $stages[$stage.Name]=@{completed=@($records|Where-Object {$_.status -eq 'completed'}).Count;failed=@($records|Where-Object {$_.status -eq 'failed'}).Count}
}
function Tail($path) {
 if (!(Test-Path $path)) {return ''}
 $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]7)
 try {$n=[Math]::Min($stream.Length,262144);$null=$stream.Seek(-$n,[IO.SeekOrigin]::End);$reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8);$text=$reader.ReadToEnd();$reader.Dispose()} finally {$stream.Dispose()}
 (($text -split '\r?\n' | Select-Object -Last __LINES__) -join "`n")
}
$logs=@()
if (__LINES__ -gt 0) {
 $paths=@("$root\failure.txt","$root\worker.err.log","$root\worker.log","$root\prepare.err.log")
 foreach($stage in $ledger.PSObject.Properties){foreach($record in $stage.Value.PSObject.Properties){if($record.Value.status -eq 'failed'){$paths+= "$root\runs\$($record.Value.name)-s$($record.Value.seed)\failure.txt"}}}
 $logs=@(foreach($path in ($paths|Select-Object -Unique)){$content=Tail $path;if($content){@{path=$path;text=$content}}})
}
$report=if(Test-Path "$root\report.json"){Get-Content "$root\report.json" -Raw | ConvertFrom-Json}
@{research=$state;alive=$alive;stages=$stages;task=(Get-ScheduledTask $active.task_name).State.ToString();
gpu=@(& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits);
logs=$logs;decision=$report.decision;explanation=$report.explanation}|ConvertTo-Json -Depth 12 -Compress
""".replace('__LINES__', str(lines))
    payload = base64.b64encode(gzip.compress(script.encode())).decode()
    wrapper = (f"$m=[IO.MemoryStream]::new([Convert]::FromBase64String('{payload}'));"
               "$g=[IO.Compression.GZipStream]::new($m,[IO.Compression.CompressionMode]::Decompress);"
               "$r=[IO.StreamReader]::new($g,[Text.Encoding]::UTF8);"
               "& ([ScriptBlock]::Create($r.ReadToEnd()));$r.Dispose()")
    encoded = base64.b64encode(wrapper.encode('utf-16-le')).decode()
    stdin, stdout, stderr = client.exec_command('powershell -NoProfile -NonInteractive -EncodedCommand ' + encoded, timeout=60)
    stdin.close()
    result, error = stdout.read(), stderr.read()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error.decode('utf-8', errors='replace'))
    return json.loads(result.decode('utf-8-sig'))


def render_research(snapshot):
    state = snapshot['research']
    lines = [f"{state['experiment']} · {state.get('device', 'cuda').upper()} · {state['status']} · задача {snapshot['task']}"]
    if state.get('error'):
        lines.append('ОШИБКА: ' + state['error'])
    if state['status'] == 'running' and not snapshot['alive']:
        lines.append('ОШИБКА: процесс контроллера остановлен; сохранённый running устарел.')
    lines.append(f"Завершено этапов: {state['completed']} · ошибок: {state['failed']}")
    for stage, counts in snapshot.get('stages', {}).items():
        total = '/' + str(state['hypotheses'] * 3) if stage.startswith('confirm_') and 'hypotheses' in state else ''
        lines.append(f"  {stage}: готово {counts['completed']}{total}, ошибок {counts['failed']}")
    cache = state.get('device_cache') or {}
    if cache:
        lines.append(f"Кеш в GPU: train {cache['train_resident_bytes'] / 1024**2:.0f} МиБ, "
                     f"val {cache['val_resident_bytes'] / 1024**2:.0f} МиБ; 0 означает чтение с SSD/RAM")
    current, progress = state.get('current') or {}, state.get('progress') or {}
    if current:
        lines.append(f"Сейчас: {current['name']} · seed {current['seed']} · этап {state['phase']}")
        lines.append(f"Шаг {progress.get('step', 0)}/{current.get('target', '?')}")
        if 'loss' in progress:
            lines.append(f"Loss {progress['loss']:.6f} · {progress.get('steps_per_second', 0):.2f} шаг/с · ожидание данных {progress.get('loader_wait_pct', 0):.1f}%")
        if 'fast_ndcg' in progress:
            lines.append(f"Fast NDCG {progress['fast_ndcg']:.9f}")
    for row in snapshot.get('gpu', []):
        fields = [field.strip() for field in row.split(',')]
        if len(fields) == 4:
            lines.append(f"GPU: {fields[0]} · нагрузка {fields[1]}% · память {fields[2]}/{fields[3]} МиБ")
    if snapshot.get('explanation'):
        lines.append('ВЫВОД: ' + snapshot['explanation'])
    lines.append('Артефакты: ' + state['directory'])
    if snapshot.get('logs'):
        from scripts.diagnostics import render_diagnostics
        lines.append(render_diagnostics(dict(status=state['status'], error=state.get('error'),
                                             failed=state['failed'], logs=snapshot['logs'])))
    return '\n'.join(lines)
