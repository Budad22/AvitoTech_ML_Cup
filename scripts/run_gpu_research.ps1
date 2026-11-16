param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$root='Q:\AVITOCUP\experiments\MLP-RESEARCH-GPU-030'
$code=Split-Path $PSScriptRoot -Parent
$python='Q:\AVITOCUP\AvitoTech_ML_Cup\.venv\Scripts\python.exe'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
if ($CheckOnly) {
    $worker=Start-Process -FilePath $python -ArgumentList '-X utf8 -m unittest discover -s tests -v' -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$root\tests.log" -RedirectStandardError "$root\tests.err.log"
    if ($worker.ExitCode) {throw 'GPU research tests failed'}
    $worker=Start-Process -FilePath $python -ArgumentList "-X utf8 -u scripts\research_preflight.py --finalists $root\migration.json --statistics $root\cpu-statistics.npz --output $root\preflight.json" -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$root\preflight.log" -RedirectStandardError "$root\preflight.err.log"
    if ($worker.ExitCode) {throw 'Frozen finalist CUDA checks failed'}
    exit 0
}
if (!(Test-Path "$root\preflight.json")) {throw 'CUDA preflight is required'}
if (!(Test-Path "$root\cache-verified.json")) {throw 'CPU/GPU data alignment audit is required'}
if ((Get-ScheduledTask AvitoMLPSuite -ErrorAction SilentlyContinue).State -eq 'Running') {throw 'Another GPU queue is running'}
$arguments="-X utf8 -u scripts\local_research.py run --directory $root --cache $root\cache --device cuda --experiment EXP-030 --finalists $root\migration.json --maximum-hours 2 --active-root Q:\AVITOCUP --task-name AvitoMLPResearchGPU"
$worker=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$root\worker.log" -RedirectStandardError "$root\worker.err.log"
exit $worker.ExitCode
