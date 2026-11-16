"""Deploy/inspect the Windows GPU worker. Credentials stay in process memory.

Requires paramiko locally. Password is read interactively; never pass it as an argument.
"""
import argparse
import base64
import getpass
import hashlib
from pathlib import Path
import subprocess
import sys
import io
import json
import re
import gzip
import time
from types import SimpleNamespace

# Direct invocation adds scripts/, while shared renderers live at repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paramiko

ROOT = Path(__file__).resolve().parents[1]


def active_workspace(client):
    """Read the active runtime and run ID without modifying the server."""
    script = "if (Test-Path 'Q:\\AVITOCUP\\active-workspace.json') { Get-Content 'Q:\\AVITOCUP\\active-workspace.json' -Raw } else { '{\"workspace\":\"E:/AVITOCUP\",\"run_id\":\"EXP-013-r01\"}' }"
    encoded = base64.b64encode(script.encode('utf-16-le')).decode()
    _, stdout, stderr = client.exec_command('powershell -NoProfile -EncodedCommand ' + encoded)
    value = json.loads(stdout.read().decode('utf-8-sig'))
    workspace = value['workspace'].replace('\\', '/')
    run_id = value['run_id']
    if stdout.channel.recv_exit_status() or workspace not in ('E:/AVITOCUP', 'Q:/AVITOCUP') or not re.fullmatch(r'[A-Za-z0-9-]+', run_id):
        raise RuntimeError('Cannot determine the active server workspace')
    return workspace, run_id


class RemoteFiles:
    """Transfer compressed chunks when the server has no SFTP subsystem."""
    def __init__(self, client):
        self.client = client

    def stat(self, path):
        return None

    def upload(self, path, data):
        packed = gzip.compress(data)
        temporary = path + '.upload.gz'
        execute(self.client, f"New-Item -ItemType Directory -Force (Split-Path '{path}') | Out-Null; [IO.File]::WriteAllBytes('{temporary}',[byte[]]@())")
        for start in range(0, len(packed), 1600):
            payload = base64.b64encode(packed[start:start + 1600]).decode()
            execute(self.client, f"$s=[IO.File]::OpenWrite('{temporary}'); $null=$s.Seek(0,[IO.SeekOrigin]::End); $b=[Convert]::FromBase64String('{payload}'); $s.Write($b,0,$b.Length); $s.Dispose()")
        execute(self.client, f"$i=[IO.File]::OpenRead('{temporary}'); $g=[IO.Compression.GZipStream]::new($i,[IO.Compression.CompressionMode]::Decompress); $o=[IO.File]::Create('{path}'); $g.CopyTo($o); $o.Dispose(); $g.Dispose(); $i.Dispose(); Remove-Item -LiteralPath '{temporary}'")

    def put(self, local, remote):
        self.upload(remote, Path(local).read_bytes())

    def open(self, path, mode):
        owner = self
        class Writer(io.StringIO):
            def __exit__(self, *args):
                owner.upload(path, self.getvalue().encode())
                self.close()
        return Writer()

    def listdir_attr(self, path):
        script = f"Get-ChildItem '{path}' -File | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
        encoded = base64.b64encode(script.encode('utf-16-le')).decode()
        _, stdout, _ = self.client.exec_command('powershell -NoProfile -EncodedCommand ' + encoded)
        names = json.loads(stdout.read().decode())
        if isinstance(names, str):
            names = [names]
        return [SimpleNamespace(filename=name) for name in names]

    def get(self, remote, local):
        script = f"[Convert]::ToBase64String([IO.File]::ReadAllBytes('{remote}'))"
        encoded = base64.b64encode(script.encode('utf-16-le')).decode()
        _, stdout, _ = self.client.exec_command('powershell -NoProfile -EncodedCommand ' + encoded)
        Path(local).write_bytes(base64.b64decode(stdout.read()))


def execute(client, script):
    script = "$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8; " + script
    encoded = base64.b64encode(script.encode('utf-16-le')).decode()
    stdin, stdout, stderr = client.exec_command('powershell -NoProfile -NonInteractive -EncodedCommand ' + encoded, timeout=120)
    stdin.close()
    result = stdout.read().decode('utf-8', errors='replace').strip()
    if result:
        print(result, flush=True)
    errors = stderr.read().decode('utf-8', errors='replace')
    if errors:
        print(errors, flush=True)
    if stdout.channel.recv_exit_status():
        raise RuntimeError('Remote command failed')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['deploy', 'status', 'suite-status', 'fetch', 'sync'])
    parser.add_argument('--host', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Show task, paths and process details')
    parser.add_argument('--logs', type=int, nargs='?', const=20, default=0, metavar='N', help='Show the last N log lines (default: 20)')
    parser.add_argument('--json', action='store_true', help='Print the status snapshot as JSON')
    from training_logs.terminal import add_refresh_options, refresh_interval
    add_refresh_options(parser)
    parser.add_argument('--format', choices=['auto', 'rich', 'plain'], default='auto', help='Terminal dashboard or plain text; auto uses Rich only for a TTY')
    args = parser.parse_args()
    interval = refresh_interval(args)
    if args.logs < 0 or interval < 0 or 0 < interval < 1:
        parser.error('logs must be nonnegative; refresh interval must be at least 1 second')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(Path.home() / '.ssh' / 'known_hosts'))
    password = getpass.getpass('SSH password: ') if sys.stdin.isatty() else sys.stdin.readline().rstrip('\r\n')
    client.connect(args.host, username=args.user, password=password, timeout=20,
                   look_for_keys=False, allow_agent=False)
    client.get_transport().set_keepalive(15)
    try:
        if args.action == 'suite-status':
            from suite_status import read_suite, render_suite
            snapshot = read_suite(client)
            print(json.dumps(snapshot, ensure_ascii=False) if args.json else render_suite(snapshot), flush=True)
            return
        remote_base, run_id = active_workspace(client)
        if args.action == 'status':
            from mlp_status import read_snapshot, render, stage, progress_events
            from training_logs.view import build_view
            from training_logs.dashboard import dashboard
            from training_logs.animation import AnimatedDashboard
            from training_logs.terminal import create_console
            from rich.live import Live
            from contextlib import ExitStack
            console = create_console(args.format)
            pretty = not args.json and (args.format == 'rich' or args.format == 'auto' and console.is_terminal)
            previous = None
            full_at = 0.
            animated = AnimatedDashboard(enabled=not args.no_animation)
            try:
                with ExitStack() as stack:
                    live = stack.enter_context(Live(animated, console=console, auto_refresh=True,
                                                   refresh_per_second=12 if not args.no_animation else 1)) if pretty and interval else None
                    while True:
                        poll_started = time.monotonic()
                        fast = bool(pretty and interval and previous and not args.verbose and not args.logs and poll_started - full_at < 10)
                        snapshot = read_snapshot(client, ROOT, fast=fast, previous=previous)
                        if not fast:
                            full_at = poll_started
                        phase, label = stage(snapshot)
                        view = build_view(snapshot, host=args.host, phase=phase, label=label,
                                          legacy_events=progress_events(snapshot))
                        if args.json:
                            snapshot.update(stage=phase, stage_label=label, training_view=view)
                            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
                        else:
                            details = render(snapshot, args.host, args.verbose, args.logs, previous) if args.verbose or args.logs else None
                            panel = dashboard(view, width=console.width, height=console.height, details=details)
                            if live:
                                animated.update(view, details)
                            elif pretty:
                                console.print(panel)
                            else:
                                from training_logs.view import plain_view
                                print(plain_view(view), flush=True)
                                if details:
                                    print(details, flush=True)
                        previous = snapshot
                        if not interval:
                            break
                        time.sleep(max(0, interval - (time.monotonic() - poll_started)))
            except KeyboardInterrupt:
                if not args.json:
                    print('\nПросмотр завершён. Серверная задача продолжает работать.')
            return
        sftp = RemoteFiles(client)
        if args.action == 'fetch':
            local = ROOT / 'runs' / run_id
            local.mkdir(parents=True, exist_ok=True)
            from mlp_status import read_snapshot
            remote = read_snapshot(client, ROOT)['run_directory'].replace('\\', '/')
            for entry in sftp.listdir_attr(remote):
                if entry.filename.endswith(('.json', '.pt', '.log')):
                    sftp.get(remote + '/' + entry.filename, str(local / entry.filename))
            return
        if args.action == 'deploy' and remote_base.startswith('Q:'):
            raise RuntimeError('SSD workspace already deployed; use sync for source updates')
        execute(client, f"New-Item -ItemType Directory -Force '{remote_base}/AvitoTech_ML_Cup' | Out-Null")
        files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
        def mkdir(path):
            try:
                sftp.stat(path)
            except FileNotFoundError:
                parent = path.rsplit('/', 1)[0]
                if parent != path:
                    mkdir(parent)
                sftp.mkdir(path)
        for name in files:
            if args.action == 'sync' and not (name.startswith(('mlp/', 'recsys_data/', 'training_logs/', 'scripts/', 'tests/', 'docs/')) or name == 'experiments.md'):
                continue
            local = ROOT / name
            if not local.is_file():
                continue
            destination = remote_base + '/AvitoTech_ML_Cup/' + name
            mkdir(destination.rsplit('/', 1)[0])
            sftp.put(str(local), destination)
        if args.action == 'sync':
            return
        archive = ROOT.parent / 'competition.zip'
        print('Hashing local competition.zip', flush=True)
        with archive.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        with sftp.open(remote_base + '/competition.sha256', 'w') as target:
            target.write(digest)
        bootstrap = r'''$ErrorActionPreference = 'Stop'
Set-Location E:\AVITOCUP\AvitoTech_ML_Cup
Start-Transcript -Path E:\AVITOCUP\bootstrap.log -Append
if (!(Test-Path E:\AVITOCUP\uv.exe)) {
    $env:UV_INSTALL_DIR='E:\AVITOCUP'
    $env:UV_NO_MODIFY_PATH='1'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
& E:\AVITOCUP\uv.exe python install 3.11
if ($LASTEXITCODE) { throw 'Python installation failed' }
if (!(Test-Path .venv\Scripts\python.exe)) { & E:\AVITOCUP\uv.exe venv --python 3.11 .venv }
& E:\AVITOCUP\uv.exe pip install --python .venv\Scripts\python.exe numpy==2.0.2 pyarrow==21.0.0
if ($LASTEXITCODE) { throw 'Data dependency installation failed' }
& E:\AVITOCUP\uv.exe pip install --python .venv\Scripts\python.exe torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed' }
New-Item -ItemType Directory -Force runs\EXP-013-r01 | Out-Null
$worker = Start-Process -FilePath (Resolve-Path .venv\Scripts\python.exe) -ArgumentList '-u scripts\server_run.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput runs\EXP-013-r01\worker.log -RedirectStandardError runs\EXP-013-r01\worker.err.log
if ($worker.ExitCode) { throw 'MLP worker failed: inspect worker logs' }
Stop-Transcript
'''
        with sftp.open(remote_base + '/bootstrap.ps1', 'w') as target:
            target.write(bootstrap)
        # S4U needs no saved password and runs outside the SSH process tree.
        execute(client, r"$old=Get-ScheduledTask -TaskName AvitoMLP -ErrorAction SilentlyContinue; if ($old -and $old.State -eq 'Running') { throw 'Task already running' }; $action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -ExecutionPolicy Bypass -File E:\AVITOCUP\bootstrap.ps1'; $principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest; $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries; Register-ScheduledTask -TaskName AvitoMLP -Action $action -Principal $principal -Settings $settings -Force | Out-Null; Start-ScheduledTask -TaskName AvitoMLP; Get-ScheduledTask -TaskName AvitoMLP | Select-Object TaskName,State")
        execute(client, r"$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -ExecutionPolicy Bypass -File E:\AVITOCUP\AvitoTech_ML_Cup\scripts\keep_awake.ps1'; $principal=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest; $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew; Register-ScheduledTask -TaskName AvitoMLPAwake -Action $action -Principal $principal -Settings $settings -Force | Out-Null; Start-ScheduledTask -TaskName AvitoMLPAwake")
    except (OSError, paramiko.SSHException) as error:
        print(f'Не удалось получить ответ сервера: {error}. Серверная задача не остановлена.', file=sys.stderr)
        raise SystemExit(1)
    finally:
        client.close()


if __name__ == '__main__':
    main()
