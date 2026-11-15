"""Copy immutable runtime files to Q:, verify hashes, and keep the host awake.

Mutable run/checkpoint files are synchronized separately after the old task stops.
No source files are deleted. The release marker is written only after activation.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

SOURCE = Path('E:/AVITOCUP')
TARGET = Path('Q:/AVITOCUP')
PROJECT = 'AvitoTech_ML_Cup'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def main():
    if (TARGET / 'active-workspace.json').exists():
        raise RuntimeError('SSD workspace is active; refuse to overwrite runtime files')
    if SOURCE.resolve() != Path('E:/AVITOCUP').resolve() or TARGET.resolve() != Path('Q:/AVITOCUP').resolve():
        raise ValueError('Unexpected migration roots')
    marker = TARGET / 'migration-state.json'
    if TARGET.exists() and not marker.exists() and any(TARGET.iterdir()):
        raise FileExistsError('Destination is not an empty migration workspace')
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest_path = TARGET / 'migration-files.json'
    verified = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = [SOURCE / 'uv.exe']
    for directory in (SOURCE / PROJECT, SOURCE / 'competition'):
        for folder, directories, names in os.walk(directory):
            directories[:] = [name for name in directories if name not in
                              ('.venv', '.git', '__pycache__', 'runs') and not name.startswith('.prepare-')]
            files.extend(Path(folder) / name for name in names)
    total = sum(path.stat().st_size for path in files)
    state = dict(stage='copying', source=str(SOURCE), target=str(TARGET), total_bytes=total,
                 copied_bytes=0, files_done=0, total_files=len(files), started=time.time())
    write_json(marker, state)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for path in files:
            relative = path.relative_to(SOURCE)
            key = relative.as_posix()
            destination = TARGET / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            stat = path.stat()
            previous = verified.get(key)
            if previous and previous['bytes'] == stat.st_size and previous['mtime_ns'] == stat.st_mtime_ns and destination.exists():
                with destination.open('rb') as stream:
                    valid = hashlib.file_digest(stream, 'sha256').hexdigest() == previous['sha256']
                if valid:
                    state['copied_bytes'] += stat.st_size
                    state['files_done'] += 1
                    continue
            temporary = destination.with_name(destination.name + '.copying')
            digest = hashlib.sha256()
            copied = 0
            last_update = time.monotonic()
            with path.open('rb') as source, temporary.open('wb') as target:
                while chunk := source.read(16 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                    if time.monotonic() - last_update >= 2:
                        write_json(marker, dict(state, current_file=key, current_bytes=copied))
                        last_update = time.monotonic()
            with temporary.open('rb') as stream:
                target_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            if target_hash != digest.hexdigest() or path.stat().st_mtime_ns != stat.st_mtime_ns:
                raise RuntimeError(f'File changed or copy verification failed: {key}')
            os.replace(temporary, destination)
            os.utime(destination, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            verified[key] = dict(bytes=copied, mtime_ns=stat.st_mtime_ns, sha256=target_hash)
            write_json(manifest_path, verified)
            state['copied_bytes'] += copied
            state['files_done'] += 1
            write_json(marker, state)
        state.update(stage='verified_waiting_activation', finished=time.time())
        write_json(marker, state)
        while not (TARGET / 'migration-release.flag').exists():
            time.sleep(5)
        state['stage'] = 'activated'
        write_json(marker, state)
    except BaseException as error:
        write_json(marker, dict(state, stage='failed', error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
