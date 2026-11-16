"""Bounded error-first diagnostics for trainers and nested Windows controllers."""
import base64
import gzip
import json


def read_diagnostics(client, lines=30):
    if not 1 <= lines <= 2000:
        raise ValueError('Log line count must be between 1 and 2000')
    script = r"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
function Json($p){if(Test-Path $p){[IO.File]::ReadAllText($p)|ConvertFrom-Json}}
function Tail($p){
 if(!(Test-Path $p)){return ''}
 $s=[IO.File]::Open($p,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]7)
 try{$n=[Math]::Min($s.Length,262144); $null=$s.Seek(-$n,[IO.SeekOrigin]::End); $r=[IO.StreamReader]::new($s,[Text.Encoding]::UTF8); $t=$r.ReadToEnd(); $r.Dispose()}finally{$s.Dispose()}
 (($t -split '\r?\n' | Select-Object -Last __LINES__) -join "`n")
}
$a=Json 'Q:\AVITOCUP\active-suite.json'; $w=Json 'Q:\AVITOCUP\active-workspace.json'
$paths=[Collections.Generic.List[string]]::new()
if($a.suite_directory){
 $root=[IO.Path]::GetFullPath($a.suite_directory)
 if(!$root.StartsWith('Q:\AVITOCUP\experiments\',[StringComparison]::OrdinalIgnoreCase)){throw 'Invalid suite root'}
 $paths.Add("$root\controller.err.log"); $paths.Add("$root\controller.log")
 $state=Json "$root\status.json"
 foreach($stage in @('0p78','1p56','full')){
  $paths.Add("$root\$stage\controller.log")
  $child=Json "$root\$stage\status.json"
  foreach($c in $child.cases){if($c.status -eq 'failed'){$paths.Add("$($c.output)\failure.txt"); $paths.Add("$($c.output)\worker.log")}}
 }
}
$run=if($state.current.output){$state.current.output}else{$w.run_directory}
if($run){$paths.Add("$run\failure.txt"); $paths.Add("$run\worker.err.log"); $paths.Add("$run\worker.log")}
$logs=@(foreach($p in ($paths|Select-Object -Unique)){if(Test-Path $p){$content=Tail $p; if($content){@{path=$p;text=$content.ToString()}}}})
@{logs=$logs;error=$state.error;status=$state.status;current=$state.current;failed=$state.failed}|ConvertTo-Json -Depth 6 -Compress
""".replace('__LINES__', str(lines))
    # Compressed transport keeps the command below Windows OpenSSH's 8191-character limit.
    payload = base64.b64encode(gzip.compress(script.encode())).decode()
    wrapper = (f"$m=[IO.MemoryStream]::new([Convert]::FromBase64String('{payload}'));"
               "$g=[IO.Compression.GZipStream]::new($m,[IO.Compression.CompressionMode]::Decompress);"
               "$r=[IO.StreamReader]::new($g,[Text.Encoding]::UTF8);"
               "& ([ScriptBlock]::Create($r.ReadToEnd()));$r.Dispose()")
    encoded = base64.b64encode(wrapper.encode('utf-16-le')).decode()
    _, stdout, stderr = client.exec_command('powershell -NoProfile -NonInteractive -EncodedCommand ' + encoded, timeout=90)
    data, error = stdout.read(), stderr.read()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error.decode('utf-8', errors='replace'))
    return json.loads(data.decode('utf-8-sig'))


EVENTS = {'run_initializing': 'Подготовка модели', 'training_started': 'Обучение началось',
          'batch_completed': 'Обработан батч', 'validation_started': 'Проверка на selection',
          'epoch_completed': 'Эпоха завершена', 'training_completed': 'Обучение варианта завершено',
          'checkpoint_saved': 'Checkpoint сохранён', 'run_failed': 'ОШИБКА ОБУЧЕНИЯ',
          'validation_completed': 'Проверка fast val завершена', 'case_failed': 'ОШИБКА ОПЫТА',
          'device_cache_ready': 'Кеш GPU готов'}


def readable_line(line):
    try:
        value = json.loads(line)
    except ValueError:
        return line
    if not isinstance(value, dict) or 'event' not in value:
        return line
    metrics = value.get('metrics') or {key: value[source] for key, source in
              [('train_loss', 'loss'), ('val_ndcg', 'fast_ndcg')] if source in value}
    fields = [value.get('timestamp', ''), EVENTS.get(value['event'], value['event'])]
    if value.get('name'):
        fields.append(value['name'])
    if value.get('step'):
        fields.append(f"шаг {value['step']}/{value.get('total', '?')}")
    if value.get('epoch'):
        fields.append(f"эпоха {value['epoch']}")
    if value.get('batch'):
        fields.append(f"батч {value['batch']}")
    for key, label in (('train_loss', 'loss'), ('val_ndcg', 'selection NDCG'), ('best_ndcg', 'лучший NDCG')):
        if key in metrics:
            fields.append(f'{label}={metrics[key]:.6f}')
    if value.get('error'):
        fields.append(str(value['error']))
    return ' | '.join(filter(None, fields))


def render_diagnostics(snapshot):
    logs = snapshot.get('logs') or []
    errors = []
    if snapshot.get('error'):
        errors.append(str(snapshot['error']))
    for log in logs:
        text = log['text']
        for line in text.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and (event.get('event') == 'run_failed' or event.get('level') == 'ERROR'):
                errors.append(log['path'] + '\n' + readable_line(line))
        if 'Traceback (most recent call last)' in text:
            last = next((line for line in reversed(text.splitlines()) if line.strip()), '')
            errors.append(f"{log['path']}\n{readable_line(last)}")
    status = 'ОШИБКА: ' + '\n'.join(dict.fromkeys(errors)) if errors else (
        'Есть неудачные опыты — подробности ниже.' if snapshot.get('failed') else 'Ошибок в собранных журналах не обнаружено.')
    lines = ['ДИАГНОСТИКА', status, 'Статус очереди: ' + str(snapshot.get('status', 'неизвестен'))]
    for log in logs:
        lines += ['', 'Файл: ' + log['path'], *[readable_line(line) for line in log['text'].splitlines()]]
    return '\n'.join(lines)
