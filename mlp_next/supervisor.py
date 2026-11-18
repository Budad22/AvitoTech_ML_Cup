"""Independent process guard: bounded restarts, identity checks and durable history."""
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from mlp_trials.state import write_json


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {}


def identity(pid):
    if os.name != 'nt':
        return None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] * 5
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    values = [ctypes.c_ulonglong() for _ in range(4)]
    try:
        if kernel.GetProcessTimes(handle, *[ctypes.byref(v) for v in values]):
            return values[0].value
    finally:
        kernel.CloseHandle(handle)


def worker_alive(state):
    pid, expected = state.get('controller_pid'), state.get('controller_identity')
    return bool(pid and expected and identity(pid) == expected)


@contextmanager
def guard_lock(path):
    with path.open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def supervise(root, command, *, poll=15., stale_seconds=900., retry_delay=15., max_retries=3):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with guard_lock(root / 'supervisor.lock'):
        previous = read(root / 'supervisor.json')
        restarts = previous.get('restarts', 0)
        failures = previous.get('failures_by_case', {})
        child = None
        launched = time.time()

        def record(event, **fields):
            value = dict(event=event, supervisor_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
                         restarts=restarts, failures_by_case=failures, **fields)
            write_json(root / 'supervisor.json', value)
            with (root / 'supervisor-events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')

        while True:
            state = read(root / 'status.json')
            alive = worker_alive(state)
            owned_alive = child is not None and child.poll() is None
            if state.get('status') in ('completed', 'completed_with_errors', 'budget_exhausted') and not alive and not owned_alive:
                record('finished', worker_status=state['status'])
                return state['status']
            if (root / 'SUPERVISOR_STOP').exists() or (root / 'STOP_AFTER_CASE').exists():
                record('owner_stop', worker_alive=alive or owned_alive)
                return 'paused'
            # A checkpoint pause must remain paused, including after the child exits.
            if not alive and not owned_alive and list(root.glob('*/STOP_AFTER_CHECKPOINT')):
                record('owner_checkpoint_pause')
                return 'paused'
            if alive or owned_alive:
                stamp = state.get('updated_at')
                heartbeat = datetime.fromisoformat(stamp).timestamp() if stamp else launched
                if owned_alive:
                    heartbeat=max(heartbeat,launched)
                if time.time() - heartbeat > stale_seconds:
                    record('stalled', worker_pid=state.get('controller_pid'), heartbeat_age=time.time()-heartbeat)
                    if owned_alive:
                        if os.name == 'nt':
                            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
                                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                           creationflags=subprocess.CREATE_NO_WINDOW,check=True)
                        else:
                            child.kill()
                        child.wait(timeout=30)
                    elif alive:
                        # Recheck identity immediately before terminating an adopted worker.
                        if worker_alive(state):
                            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                            kernel.OpenProcess.restype=ctypes.c_void_p
                            kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
                            kernel.CloseHandle.argtypes=[ctypes.c_void_p]
                            handle=kernel.OpenProcess(1,False,int(state['controller_pid']))
                            if not handle:raise OSError('Cannot terminate stalled worker')
                            try:
                                if not kernel.TerminateProcess(handle,1):raise OSError('Cannot terminate stalled worker')
                            finally:kernel.CloseHandle(handle)
                    time.sleep(poll)
                    continue
                record('monitoring', worker_pid=state.get('controller_pid'), worker_alive=True)
                time.sleep(poll)
                continue
            if child is not None:
                key=json.dumps([state.get('phase'),state.get('current')],sort_keys=True)
                failures[key]=failures.get(key,0)+1
                restarts+=1
                record('worker_exited',exit_code=child.returncode,worker_error=state.get('error'))
                if failures[key]>max_retries:
                    record('blocked_repeated_failure',worker_error=state.get('error'))
                    return 'blocked'
                time.sleep(retry_delay * min(failures[key],4))
            # An adopted process may have exited without a Popen handle; it still needs recovery.
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            with (root/'supervisor-worker.log').open('ab') as log:
                child=subprocess.Popen(command,cwd=root,stdout=log,stderr=log,creationflags=flags)
            launched=time.time()
            record('worker_started', launcher_pid=child.pid)
            # Old status can describe an old dead PID; give the new interpreter time to initialize.
            time.sleep(poll)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--bootstrap',required=True)
    args=parser.parse_args()
    result=supervise(args.root,[args.python,'-X','utf8',args.bootstrap])
    if result=='blocked':raise SystemExit(1)


if __name__=='__main__':
    main()
