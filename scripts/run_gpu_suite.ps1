param(
    [string]$SuiteDirectory='Q:\AVITOCUP\experiments\MLP-SUITE-001',
    [string]$Data='Q:\AVITOCUP\AvitoTech_ML_Cup\artifacts\prepared-full-v2',
    [string]$Python='Q:\AVITOCUP\AvitoTech_ML_Cup\.venv\Scripts\python.exe',
    [switch]$CheckOnly
)
$ErrorActionPreference='Stop'
$code=Split-Path $PSScriptRoot -Parent
Set-Location $code
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
New-Item -ItemType Directory -Force $SuiteDirectory | Out-Null
if ($CheckOnly) {
    $checks=Start-Process -FilePath $Python -ArgumentList '-m unittest discover -s tests -v' -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$SuiteDirectory\tests.log" -RedirectStandardError "$SuiteDirectory\tests.err.log"
    if ($checks.ExitCode) { throw 'Synthetic unit tests failed; see tests.err.log' }
    $checks=Start-Process -FilePath $Python -ArgumentList "-u scripts\trial_preflight.py --output $SuiteDirectory\preflight.json" -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$SuiteDirectory\preflight.log" -RedirectStandardError "$SuiteDirectory\preflight.err.log"
    if ($checks.ExitCode) { throw 'CUDA preflight failed; see preflight.err.log' }
    exit 0
}
if (!(Test-Path "$SuiteDirectory\preflight.json")) { throw 'Run CheckOnly before the full queue' }
if ((Get-ScheduledTask -TaskName AvitoMLP -ErrorAction SilentlyContinue).State -eq 'Running') {
    throw 'Previous GPU training is still running'
}
$controller=Start-Process -FilePath $Python -ArgumentList "-u scripts\gpu_suite.py --directory $SuiteDirectory --data $Data --active-root Q:\AVITOCUP" -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$SuiteDirectory\controller.log" -RedirectStandardError "$SuiteDirectory\controller.err.log"
exit $controller.ExitCode
