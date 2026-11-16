param(
    [string]$Directory='Q:\AVITOCUP\experiments\FTT-NIGHT-001',
    [string]$Python='Q:\AVITOCUP\AvitoTech_ML_Cup\.venv\Scripts\python.exe',
    [string]$Data='Q:\AVITOCUP\AvitoTech_ML_Cup\artifacts\prepared-full-v2'
)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($Directory)
$folder=if(Test-Path "$root/source-audit-v2.json") {'code-audit-v2'} else {'code'}
$code=(Resolve-Path (Join-Path $root $folder)).Path
if(Test-Path "$root/status.json") {
    $state=Get-Content "$root/status.json" -Raw | ConvertFrom-Json
    if($state.status -eq 'running' -and (Get-Process -Id $state.pid -ErrorAction SilentlyContinue)) {
        throw 'Existing controller is alive; inspect status instead of launching twice'
    }
}
$config=@{code=$code;data=$Data;output=$root}
[IO.File]::WriteAllText("$root/bootstrap.json",($config|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
$source=@'
import json,os,runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parent
config=json.loads((root/'bootstrap.json').read_text(encoding='utf-8'))
sys.stdout=(root/'worker.log').open('a',encoding='utf-8',buffering=1)
sys.stderr=(root/'worker.err.log').open('a',encoding='utf-8',buffering=1)
os.chdir(config['code'])
sys.path.insert(0,config['code'])
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
sys.argv=['ftt.runner','--data',config['data'],'--output',config['output'],'--device','cuda','--hours','8']
runpy.run_module('ftt.runner',run_name='__main__')
'@
[IO.File]::WriteAllText("$root/bootstrap.py",$source,[Text.UTF8Encoding]::new($false))
$startup=([wmiclass]'Win32_ProcessStartup').CreateInstance()
$startup.ShowWindow=0
$startup.CreateFlags=0x4208
$result=([wmiclass]'Win32_Process').Create(('"'+$Python+'" -X utf8 -u "'+$root+'\bootstrap.py"'),$code,$startup)
if($result.ReturnValue -ne 0){throw "Detached launch failed: $($result.ReturnValue)"}
$result | Select-Object ProcessId,ReturnValue | ConvertTo-Json -Compress
