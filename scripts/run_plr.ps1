param(
    [string]$SuiteDirectory='Q:\AVITOCUP\experiments\MLP-PLR-028',
    [switch]$CheckOnly
)
$ErrorActionPreference='Stop'
$code=Split-Path $PSScriptRoot -Parent
$python='Q:\AVITOCUP\AvitoTech_ML_Cup\.venv\Scripts\python.exe'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
Set-Location $code
New-Item -ItemType Directory -Force $SuiteDirectory | Out-Null
if ($CheckOnly) {
    $worker=Start-Process -FilePath $python -ArgumentList '-m unittest discover -s tests -v' -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$SuiteDirectory\tests.log" -RedirectStandardError "$SuiteDirectory\tests.err.log"
    if ($worker.ExitCode) {throw 'PLR tests failed; inspect tests.err.log'}
    $worker=Start-Process -FilePath $python -ArgumentList "-u scripts\trial_preflight.py --kinds plr history_stats_plr --output $SuiteDirectory\preflight.json" -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$SuiteDirectory\preflight.log" -RedirectStandardError "$SuiteDirectory\preflight.err.log"
    if ($worker.ExitCode) {throw 'PLR CUDA preflight failed; inspect preflight.err.log'}
    exit 0
}
if (!(Test-Path "$SuiteDirectory\preflight.json")) {throw 'Run CheckOnly before PLR training'}
if ((Get-ScheduledTask AvitoMLPSuite -ErrorAction SilentlyContinue).State -eq 'Running') {
    throw 'Previous GPU queue is still running'
}
$arguments="-u scripts\mlp_screen.py --directory $SuiteDirectory --data Q:\AVITOCUP\AvitoTech_ML_Cup\artifacts\prepared-full-v2 --active-root Q:\AVITOCUP --run-prefix EXP-028 --variants plr history_stats_plr --task-name AvitoMLPPLR --promotion-control Q:\AVITOCUP\experiments\MLP-GPU-027\1p56\report.json"
$worker=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$SuiteDirectory\controller.log" -RedirectStandardError "$SuiteDirectory\controller.err.log"
exit $worker.ExitCode
