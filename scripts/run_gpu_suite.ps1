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
    & $Python -m unittest discover -s tests -v *> "$SuiteDirectory\tests.log"
    if ($LASTEXITCODE) { throw 'Synthetic unit tests failed; see tests.log' }
    & $Python -u scripts\trial_preflight.py --output "$SuiteDirectory\preflight.json" *> "$SuiteDirectory\preflight.log"
    if ($LASTEXITCODE) { throw 'CUDA preflight failed; see preflight.log' }
    exit 0
}
if (!(Test-Path "$SuiteDirectory\preflight.json")) { throw 'Run CheckOnly before the full queue' }
if ((Get-ScheduledTask -TaskName AvitoMLP -ErrorAction SilentlyContinue).State -eq 'Running') {
    throw 'Previous GPU training is still running'
}
& $Python -u scripts\gpu_suite.py --directory $SuiteDirectory --data $Data --active-root 'Q:\AVITOCUP' *> "$SuiteDirectory\controller.log"
exit $LASTEXITCODE
