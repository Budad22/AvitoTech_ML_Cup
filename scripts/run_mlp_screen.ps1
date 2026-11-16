param([switch]$CheckOnly,[switch]$GpuOnly)
$ErrorActionPreference='Stop'
$root='Q:\AVITOCUP\experiments\MLP-SCREEN-001'
if ($GpuOnly) {$root='Q:\AVITOCUP\experiments\MLP-GPU-027'}
$code=Split-Path $PSScriptRoot -Parent
$python='Q:\AVITOCUP\AvitoTech_ML_Cup\.venv\Scripts\python.exe'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
if ($CheckOnly) {
    $worker=Start-Process -FilePath $python -ArgumentList '-m unittest discover -s tests -v' -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$root\tests.log" -RedirectStandardError "$root\tests.err.log"
    if ($worker.ExitCode) {throw 'Screen tests failed'}
    $worker=Start-Process -FilePath $python -ArgumentList "-u scripts\trial_preflight.py --output $root\preflight.json" -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$root\preflight.log" -RedirectStandardError "$root\preflight.err.log"
    if ($worker.ExitCode) {throw 'CUDA preflight failed'}
    exit 0
}
if (!(Test-Path "$root\preflight.json")) {throw 'Preflight missing'}
if ((Get-ScheduledTask AvitoMLPSuite -ErrorAction SilentlyContinue).State -ne 'Running') {throw 'Expected scheduled runner'}
$stageArgs=if ($GpuOnly) {' --stages 1p56 full --run-prefix EXP-027'} else {''}
$worker=Start-Process -FilePath $python -ArgumentList "-u scripts\mlp_screen.py --directory $root --data Q:\AVITOCUP\AvitoTech_ML_Cup\artifacts\prepared-full-v2 --active-root Q:\AVITOCUP$stageArgs" -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$root\controller.log" -RedirectStandardError "$root\controller.err.log"
exit $worker.ExitCode
