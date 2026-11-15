# Read-only snapshot; UTF-8 JSON keeps presentation out of the SSH transport.
$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='SilentlyContinue'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
$base='E:\AVITOCUP'
$repo="$base\AvitoTech_ML_Cup"
$run="$repo\runs\EXP-013-r01"
function Tail($path,$count) {
    if (Test-Path $path) { @(Get-Content $path -Tail $count | ForEach-Object { "$_" }) }
}
function JsonFile($path) { if (Test-Path $path) { Get-Content $path -Raw | ConvertFrom-Json } }
function FileInfo($path) {
    $f=Get-Item $path
    if ($f) { @{bytes=$f.Length; modified=$f.LastWriteTimeUtc.ToString('o')} }
}
$task=Get-ScheduledTask -TaskName AvitoMLP
$info=Get-ScheduledTaskInfo -TaskName AvitoMLP
$awake=Get-ScheduledTask -TaskName AvitoMLPAwake
$processes=@(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(python|uv|powershell)\.exe$' -and $_.CommandLine -match 'AvitoTech_ML_Cup|scripts.server_run.py'
} | Select-Object ProcessId,CommandLine)
$staging=0
foreach ($directory in @(Get-ChildItem "$repo\artifacts" -Filter '.prepare-*' -Directory)) {
    $staging += @(Get-ChildItem -LiteralPath $directory.FullName -Recurse -Filter *.parquet -File).Count
}
$snapshot=@{
    checked_at=[DateTime]::UtcNow.ToString('o')
    task=@{state=if ($task) {$task.State.ToString()} else {'Missing'}; result=$info.LastTaskResult; started=if ($info) {$info.LastRunTime.ToString('o')}}
    awake=if ($awake) {$awake.State.ToString()} else {'Missing'}
    archive=FileInfo "$base\competition.zip"
    partial=FileInfo "$base\competition.partial"
    extracted=(Test-Path "$base\extraction.complete")
    extracted_bytes=(Get-ChildItem "$base\competition" -Recurse -File | Measure-Object Length -Sum).Sum
    prepared=(Test-Path "$repo\artifacts\prepared-full-v1\manifest.json")
    staging_files=$staging
    last_checkpoint=FileInfo "$run\last.pt"
    best_checkpoint=FileInfo "$run\best.pt"
    history=@(JsonFile "$run\history.json")
    config=JsonFile "$run\config.json"
    metrics=JsonFile "$run\metrics.json"
    complete=(Test-Path "$run\checksums.json")
    processes=$processes
    worker=@(Tail "$run\worker.log" 250)
    errors=@(Tail "$run\worker.err.log" 30)
    bootstrap=@(Tail "$base\bootstrap.log" 30)
    log=FileInfo "$run\worker.log"
    gpu=@(& nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null)
}
$snapshot | ConvertTo-Json -Depth 8 -Compress
