"""One compact state snapshot for machines, Rich panels for people."""
import argparse
import base64
import getpass
import json
from pathlib import Path
import sys
import time

import paramiko
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def snapshot(client, lines=0):
    script = r"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
$root='Q:/AVITOCUP/experiments/FTT-NIGHT-001'
if (!(Test-Path "$root/status.json")) {throw 'FTT has no status yet'}
$s=Get-Content "$root/status.json" -Raw | ConvertFrom-Json
$p=Get-Process -Id $s.pid -ErrorAction SilentlyContinue
$s | Add-Member -NotePropertyName process_alive -NotePropertyValue ([bool]$p) -Force
$s | Add-Member -NotePropertyName gpu -NotePropertyValue (@(& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)) -Force
if (__LINES__ -gt 0) {
 $logs=@{};foreach($name in @('failure.txt','worker.err.log','events.jsonl')) {
  if(Test-Path "$root/$name") {$logs[$name]=(Get-Content "$root/$name" -Tail __LINES__) -join "`n"}
 };$s | Add-Member -NotePropertyName logs -NotePropertyValue $logs -Force
}
$s | ConvertTo-Json -Depth 10 -Compress
""".replace('__LINES__',str(lines))
    encoded = base64.b64encode(script.encode('utf-16-le')).decode()
    _, stdout, stderr = client.exec_command('powershell -NoProfile -NonInteractive -EncodedCommand '+encoded, timeout=30)
    output, error = stdout.read(), stderr.read()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error.decode('utf-8',errors='replace'))
    return json.loads(output.decode('utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='26.27.122.189')
    parser.add_argument('--user', default='bl1tz2200')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--logs', nargs='?', const=50, type=int, default=0)
    parser.add_argument('--watch', type=float, default=0)
    args = parser.parse_args()
    password = getpass.getpass('SSH password: ') if sys.stdin.isatty() else sys.stdin.readline().rstrip('\r\n')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known = Path.home()/'.ssh/known_hosts'
    if known.exists():
        client.load_host_keys(str(known))
    client.connect(args.host,username=args.user,password=password,look_for_keys=False,allow_agent=False,timeout=20)
    console = Console()
    try:
        while True:
            state = snapshot(client,args.logs)
            age = round(time.time()-state.get('heartbeat',0),1)
            keys = ('experiment','status','phase','model','seed','step','target','best_ndcg','completed','error')
            compact = {k:state[k] for k in keys if k in state}
            compact.update(heartbeat_age_seconds=age,process_alive=state['process_alive'],gpu=state['gpu'])
            if args.json:
                print(json.dumps(state if args.details else compact,ensure_ascii=False))
            else:
                if state.get('error'):
                    console.print(state['error'],style='bold red',markup=False)
                table = Table('Показатель','Состояние',expand=True)
                for key,value in (state if args.details else compact).items():
                    if key != 'logs':
                        table.add_row(key,str(value))
                console.print(Panel(table,title='FT-Transformer · автономная очередь'))
                for name,content in (state.get('logs') or {}).items():
                    console.print(name,style='bold yellow')
                    console.print(content,markup=False,highlight=False,soft_wrap=True)
            if not args.watch or state['status'] != 'running':
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        console.print('Просмотр закрыт. Обучение продолжает работать.')
    finally:
        client.close()


if __name__ == '__main__':
    main()
