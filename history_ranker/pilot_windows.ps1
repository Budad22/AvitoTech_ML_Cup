param(
    [string]$Base = 'C:/AvitoBaselines',
    [int]$Microbatch = 8
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$python = Join-Path $Base 'EXP-TRF-001-r01/.venv/Scripts/python.exe'
$env:PYTHONPATH = Join-Path $Base 'EXP-TRF-002-r01/dependencies'
$env:PYTHONIOENCODING = 'utf-8'
Add-Type -Namespace HistoryRanker -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
$null = [HistoryRanker.Power]::SetThreadExecutionState([uint32]2147483649)
function Invoke-Python {
    & $python -X utf8 -u @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
}
try {
    Invoke-Python -m unittest discover -s tests -v
    Invoke-Python -m history_ranker prepare --scope smoke --raw "$Base/datasets/full-v1/raw" --compact "$Base/datasets/full-v1/compact" --output cache-smoke
    Invoke-Python -m history_ranker pilot --cache cache-smoke --output runs/train4-a --microbatch $Microbatch
    Invoke-Python -m history_ranker evaluate-pilot --cache cache-smoke --pilot runs/train4-a --output runs/train4-early --microbatch $Microbatch
} finally {
    $null = [HistoryRanker.Power]::SetThreadExecutionState([uint32]2147483648)
}
