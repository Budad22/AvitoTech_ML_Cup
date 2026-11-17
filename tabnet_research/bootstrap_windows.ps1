param(
    [string]$BasePython,
    [ValidateSet('cu126','cu128')][string]$CudaWheel = 'cu126'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$ProgressPreference = 'SilentlyContinue'
function Stage([string]$Name) {
    @{stage=$Name; pid=$PID; updated_at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'bootstrap-status.json')
    Write-Output ((Get-Date).ToString('o') + ' ' + $Name)
}
Add-Type -Namespace AvitoBootstrap -Name PowerState -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint state);'
$null = [AvitoBootstrap.PowerState]::SetThreadExecutionState([uint32]2147483649)
try {
    Stage 'preparing_python'
    if (-not $BasePython) {
        $BasePython = Join-Path $root 'python\python.exe'
        if (-not (Test-Path -LiteralPath $BasePython)) {
            $installer = Join-Path $root 'python-3.11.9-amd64.exe'
            Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $installer
            $target = Join-Path $root 'python'
            $process = Start-Process -FilePath $installer -ArgumentList "/quiet InstallAllUsers=0 TargetDir=`"$target`" PrependPath=0 Include_launcher=0 Include_test=0 Include_pip=1" -Wait -PassThru
            if ($process.ExitCode -ne 0) { throw "Python installer failed: $($process.ExitCode)" }
        }
    }
    $python = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & $BasePython -m venv (Join-Path $root '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Isolated venv creation failed' }
    }
    Stage 'installing_dependencies'
    & $python -m pip install --disable-pip-version-check numpy==2.0.2 pyarrow==21.0.0
    if ($LASTEXITCODE -ne 0) { throw 'Data dependencies installation failed' }
    & $python -m pip install --disable-pip-version-check torch==2.8.0 --index-url "https://download.pytorch.org/whl/$CudaWheel"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    Stage 'waiting_for_cache_transfer'
    $waitUntil = (Get-Date).AddMinutes(25)
    while (-not (Test-Path -LiteralPath (Join-Path $root 'transfer-complete.json'))) {
        if ((Get-Date) -gt $waitUntil) { throw 'Cache transfer did not complete; training not started' }
        Start-Sleep -Seconds 3
    }
    Stage 'extracting_cache'
    $cache = Join-Path $root 'artifacts\research-cache-train4-v1'
    if (-not (Test-Path -LiteralPath (Join-Path $cache 'metadata.json'))) {
        & $python -m zipfile -e (Join-Path $root 'research-cache-train4-v1.zip') $root
        if ($LASTEXITCODE -ne 0) { throw 'Cache extraction failed' }
    }
    Stage 'testing_and_training'
    & (Join-Path $PSScriptRoot 'run_windows.ps1') -Python $python -Cache $cache
    Stage 'completed'
} catch {
    Stage 'error'
    throw
} finally {
    $null = [AvitoBootstrap.PowerState]::SetThreadExecutionState([uint32]2147483648)
}
