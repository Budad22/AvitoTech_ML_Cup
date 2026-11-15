"""Bounded JSONL event history and an atomic latest-state file."""
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
import time
import uuid


def sanitize(value):
    if isinstance(value, dict):
        return {str(key): '[REDACTED]' if any(word in str(key).lower() for word in
                ('password', 'secret', 'token', 'authorization', 'api_key')) else sanitize(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class RunLogger:
    """One writer per run directory. No torch, terminal or GPU dependency.

    Metrics are scalar values with architecture-defined names. Progress uses
    current/total/unit plus optional epoch/epochs. Call after completed steps.
    """
    def __init__(self, output, run_id, architecture, *, producer='trainer', echo=True,
                 max_bytes=20 * 1024 * 1024, backups=3):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.context = dict(schema_version=1, run_id=run_id, architecture=architecture, producer=producer, source='events')
        self.echo = echo
        self.lock = threading.RLock()
        self.logger = logging.getLogger('training.' + uuid.uuid4().hex)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.handler = RotatingFileHandler(self.output / 'events.jsonl', maxBytes=max_bytes,
                                           backupCount=backups, encoding='utf-8')
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.handler)
        state_path = self.output / 'progress.json'
        self.state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}

    def emit(self, event, *, phase='train', level='INFO', metrics=None, progress=None, **fields):
        with self.lock:
            record = sanitize(dict(self.context, timestamp=datetime.now(timezone.utc).isoformat(),
                                   event=event, phase=phase, level=level.upper(), **fields))
            if metrics is not None:
                record['metrics'] = sanitize(metrics)
            if progress is not None:
                record['progress'] = sanitize(progress)
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            self.logger.log(getattr(logging, level.upper()), line)
            old_metrics = self.state.get('metrics', {})
            if self.state.get('phase') != phase and progress is None:
                self.state.pop('progress', None)
            self.state.update(record)
            # Metrics remain available during checkpoint/phase events.
            self.state['metrics'] = {**old_metrics, **(record.get('metrics') or {})}
            temporary = self.output / 'progress.json.tmp'
            temporary.write_text(json.dumps(self.state, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            for attempt in range(5):
                try:
                    os.replace(temporary, self.output / 'progress.json')
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(.02)
            if self.echo:
                print(line, flush=True)
            return record

    def close(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if isinstance(error, KeyboardInterrupt):
                self.emit('run_interrupted', phase='stopped', level='WARNING')
            elif error:
                self.emit('run_failed', phase='error', level='ERROR', error=str(error), error_type=kind.__name__)
        finally:
            self.close()
