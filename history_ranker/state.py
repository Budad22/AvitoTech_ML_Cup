import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity():
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_write(path, writer):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    temp = Path(name)
    try:
        writer(temp)
        for retry in range(10):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if retry == 9:
                    raise
                time.sleep(0.05)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda tmp: tmp.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    ))


class Lock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, *args):
        self.stream.close()
