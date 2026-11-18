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
 if($stage.Value.status){
  $name=($stage.Name -split '/')[0]
  if(!$stages.ContainsKey($name)){$stages[$name]=@{completed=0;failed=0;partial=0}}
  if($stage.Value.status -eq 'completed'){$stages[$name].completed++}
  elseif($stage.Value.status -eq 'failed'){$stages[$name].failed++}
  else{$stages[$name].partial++}
  continue
 }
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
 foreach($stage in $ledger.PSObject.Properties){
  if($stage.Value.status){
   if($stage.Value.status -eq 'failed'){
    $phase=($stage.Name -split '/')[0]
    $paths+="$root/$phase/runs/$($stage.Value.name)-s$($stage.Value.seed)/failure.txt"
   }
  }else{foreach($record in $stage.Value.PSObject.Properties){if($record.Value.status -eq 'failed'){$paths+= "$root\runs\$($record.Value.name)-s$($record.Value.seed)\failure.txt"}}}
 }
 $logs=@(foreach($path in ($paths|Select-Object -Unique)){$content=Tail $path;if($content){@{path=$path;text=$content}}})
}
$report=if(Test-Path "$root\report.json"){Get-Content "$root\report.json" -Raw | ConvertFrom-Json}
$summary=@{}
if($report.final_runs){
 $best=@($report.final_runs|Where-Object {$_.name -eq $report.best_mlp})
 $reference=@($report.final_runs|Where-Object {$_.name -eq $report.baseline})
 $summary=@{name=$report.best_mlp;confirmation_ndcg=($best|Measure-Object confirmation_ndcg -Average).Average;
 reference_ndcg=($reference|Measure-Object confirmation_ndcg -Average).Average;
 delta=$report.differences.($report.best_mlp).delta;confirmation_feeds=$report.confirmation_feeds}
}
$config=Get-Content "$root\research-config.json" -Raw | ConvertFrom-Json
$task=if($active.task_name){(Get-ScheduledTask $active.task_name).State.ToString()}else{'detached'}
$supervisor=if(Test-Path "$root/supervisor.json"){Get-Content "$root/supervisor.json" -Raw|ConvertFrom-Json}else{$null}
$queued=$null
if(Test-Path 'Q:/AVITOCUP/active-research-queue.json'){
 $candidate=Get-Content 'Q:/AVITOCUP/active-research-queue.json' -Raw|ConvertFrom-Json
 if($candidate.experiment -ne $state.experiment){
  $queued=$candidate
  $queueRoot=[IO.Path]::GetFullPath($candidate.directory)
  if(!$queueRoot.StartsWith('Q:\AVITOCUP\experiments\',[StringComparison]::OrdinalIgnoreCase)){throw 'Invalid successor directory'}
  if(Test-Path "$queueRoot/status.json"){
   $qs=Get-Content "$queueRoot/status.json" -Raw|ConvertFrom-Json
   $qp=Get-Process -Id $qs.controller_pid -ErrorAction SilentlyContinue
   $qa=$false
   if($qp){$qa=$qp.StartTime.ToUniversalTime().ToFileTimeUtc() -eq $qs.controller_identity}
   $queued|Add-Member -NotePropertyName state -NotePropertyValue $qs -Force
   $queued|Add-Member -NotePropertyName alive -NotePropertyValue $qa -Force
  }
 }
}
@{research=$state;alive=$alive;stages=$stages;task=$task;
gpu=@(& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits);
logs=$logs;decision=$report.decision;explanation=$report.explanation;summary=$summary;queued=$queued;supervisor=$supervisor;
budget_seconds=$config.maximum_hours*3600}|ConvertTo-Json -Depth 12 -Compress
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
    if snapshot.get('supervisor'):
        guard=snapshot['supervisor']
        lines.append(f"Автовосстановление: {guard['event']} · перезапусков {guard['restarts']}")
    if snapshot.get('queued'):
        queued = snapshot['queued']
        lines.append(f"Следующая очередь: {queued['experiment']} · {queued['hypotheses']} новых гипотез · окно {queued['minimum_hours']} ч + резерв")
        if (queued.get('state') or {}).get('error'):
            lines.append('ОШИБКА СЛЕДУЮЩЕЙ ОЧЕРЕДИ: ' + queued['state']['error'])
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


def build_research_view(snapshot, host):
    """Translate research state into the shared Rich dashboard contract."""
    state = snapshot['research']
    current, measured = state.get('current') or {}, state.get('progress') or {}
    status = state['status']
    phase, label = {'completed': ('complete', 'Обучение завершено'),
                    'completed_with_errors': ('error', 'Завершено с ошибками'),
                    'paused': ('stopped', 'Приостановлено'), 'error': ('error', 'Ошибка обучения')}.get(
                        status, ('initializing', 'Подготовка данных') if state['phase'] == 'prepare' else
                        ('evaluation', 'Проверка отложенных лент') if state['phase'] == 'confirmation_evaluation' else
                        ('training', 'Обучение на GPU'))
    warnings = []
    if state.get('error'):
        warnings.append('ОШИБКА: ' + state['error'])
    if status == 'running' and not snapshot['alive']:
        phase, label = 'error', 'Контроллер остановлен'
        warnings.append('Сохранённый running устарел: живой процесс не обнаружен.')
    run_id = state['experiment']
    metrics = {}
    if current:
        run_id += '/' + current['name'] + '/s' + str(current['seed'])
        progress = dict(current=measured.get('step', 0), total=current.get('target', 0),
                        unit='шагов', title='Обучение · seed ' + str(current['seed']))
        for source, target in [('loss', 'train_loss'), ('fast_ndcg', 'val_ndcg'),
                               ('best_ndcg', 'best_ndcg'), ('loader_wait_pct', 'loader_wait_pct')]:
            if source in measured:
                metrics[target] = measured[source]
    else:
        total = sum(state.get('hypotheses', 0) * 3 if stage.startswith('confirm_') else count['completed']
                    for stage, count in snapshot.get('stages', {}).items())
        progress = dict(current=state['completed'], total=total, unit='этапов', title='Результаты всей очереди')
    summary = snapshot.get('summary') or {}
    if phase == 'complete' and summary:
        metrics.update(confirmation_ndcg=summary['confirmation_ndcg'], reference_ndcg=summary['reference_ndcg'],
                       confirmation_delta=summary['delta'])
        warnings.append('Лидер: ' + summary['name'] + ' · среднее по трём seed')
    warnings.append(f"Очередь: завершено этапов {state['completed']} · ошибок {state['failed']} · {state['phase']}")
    if snapshot.get('supervisor'):
        guard=snapshot['supervisor']
        warnings.append(f"Автовосстановление: {guard['event']} · перезапусков {guard['restarts']}")
    if snapshot.get('queued'):
        queued = snapshot['queued']
        warnings.append(f"Далее: {queued['experiment']} · {queued['hypotheses']} новых гипотез · {queued['minimum_hours']} ч + резерв")
        if (queued.get('state') or {}).get('error'):
            warnings.append('ОШИБКА СЛЕДУЮЩЕЙ ОЧЕРЕДИ: ' + queued['state']['error'])
    if snapshot.get('explanation'):
        warnings.append(snapshot['explanation'])
    gpu = None
    for row in snapshot.get('gpu', []):
        fields = [value.strip() for value in row.split(',')]
        if len(fields) == 4:
            gpu = dict(name=fields[0], utilization=float(fields[1]), used=float(fields[2]), total=float(fields[3]))
            break
    return dict(run_id=run_id, architecture='MLP research', host=host, phase=phase, label=label,
                progress=progress, metrics=metrics, gpu=gpu, warnings=warnings, source='research state',
                elapsed_seconds=state.get('elapsed_seconds', 0), budget_seconds=snapshot.get('budget_seconds', 0),
                steps_per_second=measured.get('steps_per_second'), speed_source='window',
                freshness='Состояние очереди: ' + state.get('updated_at', '?'),
                metric_labels={'val_ndcg': 'Fast NDCG (2 000 лент)', 'best_ndcg': 'Лучший fast NDCG',
                               'confirmation_ndcg': 'NDCG на отложенных лентах', 'reference_ndcg': 'Контрольная MLP',
                               'confirmation_delta': 'Δ к контрольной MLP'})


def display_research(client, snapshot, args, interval):
    """Reuse the existing panels, GPU gauge, and animations; logs remain scrollable."""
    import time
    from contextlib import ExitStack
    from rich.live import Live
    from training_logs.animation import AnimatedDashboard
    from training_logs.dashboard import dashboard
    from training_logs.terminal import create_console
    from scripts.diagnostics import render_diagnostics

    console = create_console(args.format)
    pretty = not args.json and (args.format == 'rich' or args.format == 'auto' and console.is_terminal)
    animated = AnimatedDashboard(enabled=not args.no_animation)
    try:
        with ExitStack() as stack:
            live = stack.enter_context(Live(animated, console=console, auto_refresh=True,
                                           refresh_per_second=12 if not args.no_animation else 1)) if pretty and interval and not args.logs else None
            while True:
                view = build_research_view(snapshot, args.host)
                if args.json:
                    print(json.dumps(snapshot, ensure_ascii=False), flush=True)
                elif pretty:
                    details = render_diagnostics(dict(status=snapshot['research']['status'],
                        error=snapshot['research'].get('error'), failed=snapshot['research']['failed'],
                        logs=snapshot.get('logs') or [])) if args.logs else None
                    if live:
                        animated.update(view)
                    else:
                        console.print(dashboard(view, width=console.width))
                        if details:
                            console.print(details, markup=False, highlight=False, soft_wrap=True)
                else:
                    print(render_research(snapshot), flush=True)
                pending = snapshot.get('queued') or {}
                handoff = pending.get('alive') and (pending.get('state') or {}).get('status') in ('waiting','running')
                if not interval or view['phase'] in ('error', 'stopped') or view['phase']=='complete' and not handoff:
                    break
                time.sleep(interval)
                snapshot = read_research(client, args.logs)
    except KeyboardInterrupt:
        print('\nПросмотр завершён. Серверное обучение продолжает работать.')
