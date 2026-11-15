# One-time fresh launch on the verified SSD copy. Restart AvitoMLP to resume later.
param([ValidatePattern('^[A-Za-z0-9-]+$')][string]$RunId='EXP-013-r02')
$ErrorActionPreference='Stop'
$base='Q:\AVITOCUP'
$repository="$base\AvitoTech_ML_Cup"
do {
    $state=Get-Content "$base\migration-state.json" -Raw | ConvertFrom-Json
    if ($state.stage -eq 'failed') { throw "Migration failed: $($state.error)" }
    if ($state.stage -eq 'copying') { Start-Sleep 2 }
} while ($state.stage -eq 'copying')
if ($state.stage -ne 'verified_waiting_activation') { throw 'Verified SSD copy is required' }
$environment=Get-Content "$base\environment-ready.json" -Raw | ConvertFrom-Json
if (!$environment.ready) { throw 'SSD environment is not ready' }
if ((Get-ScheduledTask AvitoMLP).State -eq 'Running') { throw 'Another worker is running' }
$run="$repository\runs\$RunId"
if (Test-Path -LiteralPath $run) { throw 'Fresh run requires a new output directory' }
Expand-Archive -LiteralPath "$base\ssd-runtime.zip" -DestinationPath $repository -Force
$test=Start-Process -FilePath "$repository\.venv\Scripts\python.exe" -ArgumentList '-m unittest discover -s tests -v' -WorkingDirectory $repository -WindowStyle Hidden -Wait -PassThru -RedirectStandardError "$base\ssd-tests.log" -RedirectStandardOutput "$base\ssd-tests-stdout.log"
if ($test.ExitCode) { throw 'SSD tests failed' }
New-Item -ItemType Directory -Path $run | Out-Null
$launcher=@'
$ErrorActionPreference='Stop'
Set-Location Q:\AVITOCUP\AvitoTech_ML_Cup
$worker=Start-Process -FilePath Q:\AVITOCUP\AvitoTech_ML_Cup\.venv\Scripts\python.exe -ArgumentList '-u scripts\run_prepared.py --run-id __RUN_ID__' -WorkingDirectory Q:\AVITOCUP\AvitoTech_ML_Cup -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput runs\__RUN_ID__\worker.log -RedirectStandardError runs\__RUN_ID__\worker.err.log
if ($worker.ExitCode) { throw "Worker failed with exit code $($worker.ExitCode)" }
'@
[IO.File]::WriteAllText("$base\run-prepared.ps1",$launcher.Replace('__RUN_ID__',$RunId),[Text.UTF8Encoding]::new($false))
$active=@{workspace=$base;run_id=$RunId;started_at=[DateTime]::UtcNow.ToString('o');fresh_start=$true}
[IO.File]::WriteAllText("$base\active-workspace.json",($active | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
$action=New-ScheduledTaskAction -Execute powershell.exe -Argument "-NoProfile -ExecutionPolicy Bypass -File $base\run-prepared.ps1" -WorkingDirectory $repository
Set-ScheduledTask -TaskName AvitoMLP -Action $action | Out-Null
Start-ScheduledTask AvitoMLP
Start-ScheduledTask AvitoMLPAwake
$deadline=[DateTime]::UtcNow.AddMinutes(5)
do {
    Start-Sleep 2
    if ((Get-ScheduledTask AvitoMLP).State -ne 'Running') { throw 'New training stopped; inspect worker logs' }
    $progress=if (Test-Path "$run\progress.json") { Get-Content "$run\progress.json" -Raw | ConvertFrom-Json }
    if ($progress.producer -eq 'trainer' -and $progress.global_step -gt 0) { break }
    if ([DateTime]::UtcNow -gt $deadline) { throw 'First training step was not confirmed within five minutes' }
} while ($true)
[IO.File]::WriteAllText("$base\new-training-ready.json",($progress | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText("$base\migration-release.flag",$RunId)
