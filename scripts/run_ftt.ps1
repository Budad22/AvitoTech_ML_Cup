param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$root='Q:\AVITOCUP\experiments\FTT-NIGHT-001'
$code=Split-Path $PSScriptRoot -Parent
$python='Q:\AVITOCUP\AvitoTech_ML_Cup\.venv\Scripts\python.exe'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
if ($CheckOnly) {
    $arguments='-X utf8 -m unittest discover -s tests -v'
} else {
    $arguments="-X utf8 -u -m ftt.runner --data Q:\AVITOCUP\AvitoTech_ML_Cup\artifacts\prepared-full-v2 --output $root --device cuda --hours 8"
}
$worker=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$root\worker.log" -RedirectStandardError "$root\worker.err.log"
exit $worker.ExitCode
