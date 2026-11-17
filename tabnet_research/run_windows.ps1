param(
    [Parameter(Mandatory=$true)][string]$Python,
    [string]$Cache,
    [string]$Output,
    [string]$Experiment = 'EXP-TABNET-001-r01',
    [int]$Steps = 6400,
    [int]$MaximumSeconds = 7200,
    [int]$EvaluationInterval = 0,
    [string]$ResumeFrom
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
if (-not $Cache) { $Cache = Join-Path $root 'artifacts\research-cache-train4-v1' }
if (-not $Output) { $Output = Join-Path $root "runs\tabnet\$Experiment" }
New-Item -ItemType Directory -Force $Output | Out-Null
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
Add-Type -Namespace AvitoTabNet -Name PowerState -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint state);'
$null = [AvitoTabNet.PowerState]::SetThreadExecutionState([uint32]2147483649)
try {
    & $Python -X utf8 -u -m unittest discover -s tabnet_research -p 'test_*.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'TabNet synthetic tests failed' }
    & $Python -X utf8 -u -m tabnet_research.cuda_preflight --mlp-source artifacts/mlp-source-d4f9edb --dependencies artifacts/tabnet-deps --cache $Cache --output (Join-Path $Output 'cuda-preflight.json')
    if ($LASTEXITCODE -ne 0) { throw 'Real-batch CUDA preflight failed' }
    $trainingArgs = @('-X','utf8','-u','-m','tabnet_research.run',
        '--mlp-source','artifacts/mlp-source-d4f9edb','--dependencies','artifacts/tabnet-deps',
        '--cache',$Cache,'--output',$Output,'--device','cuda','--steps',"$Steps",
        '--batch-size','16','--seeds','42','--maximum-seconds',"$MaximumSeconds",
        '--experiment',$Experiment,'--evaluation-interval',"$EvaluationInterval")
    if ($ResumeFrom) { $trainingArgs += @('--resume-from',$ResumeFrom) }
    & $Python @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw 'Training pair stopped: inspect status.json and worker logs' }
} finally {
    $null = [AvitoTabNet.PowerState]::SetThreadExecutionState([uint32]2147483648)
}
