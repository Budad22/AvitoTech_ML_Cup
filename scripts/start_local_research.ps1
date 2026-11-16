param(
    [Parameter(Mandatory=$true)][string]$Directory,
    [Parameter(Mandatory=$true)][string]$Code,
    [Parameter(Mandatory=$true)][string]$Module,
    [Parameter(Mandatory=$true)][string[]]$PythonArguments,
    [ValidatePattern('^[A-Za-z0-9-]+$')][string]$Label='worker'
)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($Directory)
$codeRoot=(Resolve-Path -LiteralPath $Code).Path
$python=(Resolve-Path -LiteralPath (Join-Path (Split-Path $PSScriptRoot -Parent) '.venv/Scripts/python.exe')).Path
New-Item -ItemType Directory -Force -Path $root | Out-Null
$configuration=Join-Path $root "launch-$Label.json"
if (Test-Path -LiteralPath $configuration) {throw 'Launch record already exists; inspect the existing process first'}
$config=@{code=$codeRoot;module=$Module;arguments=$PythonArguments;label=$Label}
[IO.File]::WriteAllText($configuration,($config|ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
$bootstrap=Join-Path $root "launch-$Label.py"
$source=@'
import json
import os
from pathlib import Path
import runpy
import sys

path = Path(__file__).resolve()
config = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
sys.stdout = (path.parent / (config['label'] + '.log')).open('a', encoding='utf-8', buffering=1)
sys.stderr = (path.parent / (config['label'] + '.err.log')).open('a', encoding='utf-8', buffering=1)
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
os.chdir(config['code'])
sys.path.insert(0, config['code'])
sys.argv = [config['module'], *config['arguments']]
runpy.run_module(config['module'], run_name='__main__')
'@
[IO.File]::WriteAllText($bootstrap,$source,[Text.UTF8Encoding]::new($false))
$startup=([wmiclass]'Win32_ProcessStartup').CreateInstance()
$startup.ShowWindow=0
$startup.CreateFlags=0x4208 # Detached, new process group, below-normal priority.
$result=([wmiclass]'Win32_Process').Create(('"'+$python+'" -u "'+$bootstrap+'"'),$codeRoot,$startup)
if ($result.ReturnValue -ne 0) {throw "Detached start failed: $($result.ReturnValue)"}
$result | Select-Object ProcessId,ReturnValue | ConvertTo-Json -Compress
