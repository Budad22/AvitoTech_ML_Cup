"""Atomic outputs that tolerate short Windows reader sharing conflicts."""
import json
import os
from pathlib import Path
import tempfile
import time


def replace(source, destination):
    """Keep the old destination intact and retry bounded permission failures."""
    for attempt in range(21):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(min(.025 * (2 ** attempt), .25))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
        replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
