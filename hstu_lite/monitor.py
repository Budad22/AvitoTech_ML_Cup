"""Publish HSTU progress without putting the training loop at risk."""
from datetime import datetime, timezone
import logging
import time

from mlp_trials.state import write_json, write_text


LOG = logging.getLogger(__name__)


class Progress:
    def __init__(self, root):
        self.root = root
        self.published_stage = None
        self.next_warning = 0.

    def publish(self, stage, *, step=0, steps=None, elapsed_seconds=None, **details):
        status = dict(step=step, steps=steps, elapsed_seconds=elapsed_seconds,
                      updated_at=datetime.now(timezone.utc).isoformat(), **details)
        try:
            write_json(self.root / 'status.json', status)
            if self.published_stage != stage:
                write_text(self.root / 'stage.txt', stage)
                self.published_stage = stage
        except OSError as error:
            if stage in ('completed', 'paused'):
                raise
            now = time.monotonic()
            if now >= self.next_warning:
                LOG.warning('Monitor update deferred at stage=%s step=%s: %s', stage, step, error)
                self.next_warning = now + 60.
