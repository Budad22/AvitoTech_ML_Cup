"""Atomic state replacement tolerating short-lived Windows monitoring readers."""
import json
import os
import time


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, allow_nan=False))


def write_text(path, text):
    """Preserve the last complete snapshot until the replacement succeeds."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    for attempt in range(21):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(min(.025 * (2 ** attempt), .25))
