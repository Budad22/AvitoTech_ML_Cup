$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root=Split-Path $PSScriptRoot -Parent;Set-Location $root
$env:PYTHONUNBUFFERED='1';$env:PYTHONIOENCODING='utf-8';$env:UV_NO_PROGRESS='1'
$env:PYTHONPATH="$root\dependencies"
$python='C:\AvitoBaselines\EXP-TRF-001-r01\.venv\Scripts\python.exe'
$uv='C:\AvitoBaselines\EXP-TRF-001-r01\.bootstrap\Scripts\uv.exe'
$raw='C:\AvitoBaselines\datasets\full-v1\raw';$compact='C:\AvitoBaselines\datasets\full-v1\compact'
$reference='C:\AvitoTabNet\EXP-TABNET-001-r01\artifacts\research-cache-train4-v1'
function Stage([string]$s){$j=@{stage=$s;pid=$PID;updated_at=(Get-Date).ToUniversalTime().ToString('o')}|ConvertTo-Json;[IO.File]::WriteAllText("$root\pipeline-status.json",$j,(New-Object Text.UTF8Encoding($false)));Write-Output ((Get-Date).ToString('o')+' '+$s)}
function WaitFile([string]$p,[int]$minutes){$until=(Get-Date).AddMinutes($minutes);while(!(Test-Path $p)){if((Get-Date)-gt $until){throw "Timed out waiting for $p"};Start-Sleep -Seconds 3}}
Add-Type -Namespace AvitoFullTransformer -Name PowerState -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint state);'
$null=[AvitoFullTransformer.PowerState]::SetThreadExecutionState([uint32]2147483649)
try{
    Stage 'installing_data_dependency'
    & $uv pip install --python $python --target "$root\dependencies" --no-deps pyarrow==21.0.0
    if($LASTEXITCODE){throw 'PyArrow target installation failed'}
    Stage 'downloading_full_train'
    & $python -X utf8 -u -m transformer_research.fetch --raw $raw --source-config full-source.json --transfer-manifest transfer_manifest.json --transfer-ready "$root\RAW_TRANSFER_READY" --workers 4
    if($LASTEXITCODE){throw 'Train download/transfer verification failed'}
    Stage 'waiting_for_reviewed_training_source';WaitFile "$root\SOURCE_READY" 60
    Stage 'testing_compact_loader'
    & $python -X utf8 -m unittest discover -s transformer_research -p 'test_*.py' -v
    if($LASTEXITCODE){throw 'Model/data/resume contracts failed'}
    Stage 'building_compact_full_train'
    & $python -X utf8 -u -m transformer_research.build_compact --raw $raw --output $compact --reference-cache $reference --source-config full-source.json
    if($LASTEXITCODE){throw 'Compact dataset preparation failed'}
    Stage 'auditing_inputs_and_speed'
    & $python -X utf8 -u -m transformer_research.audit_compact --cache $compact --reference-cache $reference --source official/neural_baseline.py --output loader-profile.json
    if($LASTEXITCODE){throw 'Compact input/profile verification failed'}
    Stage 'training'
    & $python -X utf8 -u -m transformer_research.train --source official/neural_baseline.py --cache $compact --dataset-format compact --experiment EXP-TRF-002-r01 --output runs/EXP-TRF-002-r01 --device cuda --epochs 5 --batch-size 32 --maximum-seconds 28800
    if($LASTEXITCODE){throw 'Full train stopped: inspect trainer status/checkpoint'}
    Stage 'completed'
}catch{Stage 'error';throw}
finally{$null=[AvitoFullTransformer.PowerState]::SetThreadExecutionState([uint32]2147483648)}
