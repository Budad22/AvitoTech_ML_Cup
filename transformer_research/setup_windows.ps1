$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root=Split-Path $PSScriptRoot -Parent
Set-Location $root
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:UV_NO_PROGRESS='1'
function Stage([string]$value){
    $json=@{stage=$value;pid=$PID;updated_at=(Get-Date).ToUniversalTime().ToString('o')}|ConvertTo-Json
    [IO.File]::WriteAllText("$root\setup-status.json",$json,(New-Object Text.UTF8Encoding($false)))
    Write-Output ((Get-Date).ToString('o')+' '+$value)
}
Add-Type -Namespace AvitoTransformer -Name PowerState -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint state);'
$null=[AvitoTransformer.PowerState]::SetThreadExecutionState([uint32]2147483649)
try{
    Stage 'creating_isolated_environment'
    $base='C:\AvitoTabNet\EXP-TABNET-001-r01\python\python.exe'
    if(!(Test-Path "$root\.bootstrap\Scripts\python.exe")){& $base -m venv "$root\.bootstrap";if($LASTEXITCODE){throw 'Bootstrap venv failed'}}
    & "$root\.bootstrap\Scripts\python.exe" -m pip install --disable-pip-version-check uv
    if($LASTEXITCODE){throw 'uv installation failed'}
    $uv="$root\.bootstrap\Scripts\uv.exe"
    & $uv --version
    $env:UV_PROJECT_ENVIRONMENT="$root\.venv"
    Stage 'installing_locked_dependencies'
    & $uv sync --project "$root\official" --python $base --locked --no-config --default-index https://pypi.org/simple --no-install-package torch
    if($LASTEXITCODE){throw 'Official locked dependencies installation failed'}
    Stage 'installing_cuda_torch'
    $python="$root\.venv\Scripts\python.exe"
    & $uv pip install --python $python --no-deps torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    if($LASTEXITCODE){throw 'CUDA Torch installation failed'}
    & $uv pip check --python $python
    if($LASTEXITCODE){throw 'Dependency consistency check failed'}
    Stage 'waiting_for_reviewed_source'
    $deadline=(Get-Date).AddMinutes(30)
    while(!(Test-Path "$root\SOURCE_READY")){
        if((Get-Date)-gt $deadline){throw 'Reviewed source was not delivered'}
        Start-Sleep -Seconds 3
    }
    Stage 'checking_training_code'
    & $python -X utf8 -m unittest discover -s transformer_research -p 'test_*.py' -v
    if($LASTEXITCODE){throw 'Transformer contracts failed'}
    Stage 'training'
    & $python -X utf8 -u -m transformer_research.train --source official/neural_baseline.py --cache C:/AvitoTabNet/EXP-TABNET-001-r01/artifacts/research-cache-train4-v1 --output runs/EXP-TRF-001-r01 --device cuda --epochs 5 --batch-size 32 --maximum-seconds 1800
    if($LASTEXITCODE){throw 'Transformer training stopped: inspect status/logs'}
    Stage 'completed'
}catch{Stage 'error';throw}
finally{$null=[AvitoTransformer.PowerState]::SetThreadExecutionState([uint32]2147483648)}
