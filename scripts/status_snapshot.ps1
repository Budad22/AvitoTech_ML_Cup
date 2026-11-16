# Read-only snapshot; UTF-8 JSON keeps presentation out of the SSH transport.
$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='SilentlyContinue'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
$base='E:\AVITOCUP'
$runId='EXP-013-r01'
if (Test-Path 'Q:\AVITOCUP\active-workspace.json') {
    $base='Q:\AVITOCUP'
    $active=Get-Content 'Q:\AVITOCUP\active-workspace.json' -Raw | ConvertFrom-Json
    if ($active.run_id -match '^[A-Za-z0-9-]+$') { $runId=$active.run_id }
}
$repo="$base\AvitoTech_ML_Cup"
$run="$repo\runs\$runId"
$taskName='AvitoMLP'
if ($active.task_name -eq 'AvitoMLPSuite') {
    $candidate=[IO.Path]::GetFullPath($active.run_directory)
    if ($candidate.StartsWith('Q:\AVITOCUP\experiments\',[StringComparison]::OrdinalIgnoreCase)) {
        $run=$candidate
        $taskName='AvitoMLPSuite'
    }
}
function Tail($path,$count) {
    if (!(Test-Path $path)) { return }
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
    $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true)
    $lines=[Collections.Generic.Queue[string]]::new()
    try {
        while (($line=$reader.ReadLine()) -ne $null) {
            $lines.Enqueue($line)
            if ($lines.Count -gt $count) { $null=$lines.Dequeue() }
        }
        $lines.ToArray()
    } finally { $reader.Dispose() }
}
function JsonFile($path) {
    if (!(Test-Path $path)) { return }
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]7)
    $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true)
    try { $content=$reader.ReadToEnd() } finally { $reader.Dispose() }
    $content | ConvertFrom-Json
}
function FileInfo($path) {
    $f=Get-Item $path
    if ($f) { @{bytes=$f.Length; modified=$f.LastWriteTimeUtc.ToString('o')} }
}
$suite=if ($taskName -eq 'AvitoMLPSuite') { JsonFile "$($active.suite_directory)\status.json" }
$done=if ($suite) { $suite.status -in @('completed','completed_with_errors') } else { Test-Path "$run\checksums.json" }
if ($quick) {
    $worker=@(Tail "$run\worker.log" 40)
    $snapshot=@{
        checked_at=[DateTime]::UtcNow.ToString('o')
        run_id=$runId
        run_directory=$run
        task=@{state=(Get-ScheduledTask $taskName).State.ToString()}
        training_state=JsonFile "$run\progress.json"
        history=@(JsonFile "$run\history.json")
        complete=$done
        suite=$suite
        worker=$worker
        last_command=@($worker | Where-Object { $_.StartsWith('COMMAND') } | Select-Object -Last 1)
        errors=@(Tail "$run\worker.err.log" 5)
        log=FileInfo "$run\worker.log"
        gpu=@(& nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null)
    }
    $snapshot | ConvertTo-Json -Depth 8 -Compress
    exit
}
$task=Get-ScheduledTask -TaskName $taskName
$info=Get-ScheduledTaskInfo -TaskName $taskName
$awake=Get-ScheduledTask -TaskName AvitoMLPAwake
$config=JsonFile "$run\config.json"
$data=if ($config.data) {JsonFile "$($config.data)\manifest.json"}
$batchSize=if ($config.batch_size) {$config.batch_size} else {64}
$batches=0
foreach ($file in $data.splits.train.files) { $batches += [Math]::Ceiling($file.feeds / $batchSize) }
$processes=@(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(python|uv|powershell)\.exe$' -and $_.CommandLine -match 'AvitoTech_ML_Cup|scripts.server_run.py|mlp_trials|gpu_suite.py'
} | Select-Object ProcessId,CommandLine)
$snapshot=@{
    workspace=$base
    run_id=$runId
    run_directory=$run
    migration=JsonFile 'Q:\AVITOCUP\migration-state.json'
    checked_at=[DateTime]::UtcNow.ToString('o')
    task=@{state=if ($task) {$task.State.ToString()} else {'Missing'}; result=$info.LastTaskResult; started=if ($info) {$info.LastRunTime.ToString('o')}}
    awake=if ($awake) {$awake.State.ToString()} else {'Missing'}
    prepared=((Test-Path "$repo\artifacts\prepared-full-v1\manifest.json") -or (Test-Path "$repo\artifacts\prepared-full-v2\manifest.json"))
    batches_per_epoch=$batches
    training_state=JsonFile "$run\progress.json"
    data_summary=@{feeds=$data.splits.train.feeds; candidates=$data.splits.train.candidates; path=$config.data}
    last_checkpoint=FileInfo "$run\last.pt"
    best_checkpoint=FileInfo "$run\best.pt"
    history=@(JsonFile "$run\history.json")
    config=$config
    metrics=JsonFile "$run\metrics.json"
    complete=$done
    suite=$suite
    processes=$processes
    io=@(Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | Where-Object { $_.IDProcess -in $processes.ProcessId } | Select-Object IOReadBytesPersec,IOWriteBytesPersec)
    worker=@(Tail "$run\worker.log" 250)
    last_command=@(Tail "$run\worker.log" 40 | Where-Object { $_.StartsWith('COMMAND') } | Select-Object -Last 1)
    errors=@(Tail "$run\worker.err.log" 30)
    bootstrap=@(Tail "$base\bootstrap.log" 30)
    log=FileInfo "$run\worker.log"
    gpu=@(& nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null)
}
$snapshot | ConvertTo-Json -Depth 8 -Compress
