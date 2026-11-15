"""Bounded CPU prefetch and pinned host buffers; preserves feed/batch ordering."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch

from .model import features


def pack(batch, categories, pin=False):
    started = time.perf_counter()
    dense, category, target, _ = features(batch, categories)
    counts = batch.candidate_counts
    weights = np.repeat((1.0 / counts).astype(np.float32), counts)
    values = [torch.from_numpy(dense), torch.from_numpy(category),
              torch.from_numpy(target) if target is not None else None, torch.from_numpy(weights)]
    if pin:
        values = [value.pin_memory() if value is not None else None for value in values]
    return dict(values=values, feeds=len(counts), candidates=int(counts.sum()),
                feature_seconds=time.perf_counter() - started)


@contextmanager
def prepared_batches(source, categories, *, device='cpu', prefetch=2):
    """At most two completed/in-flight batches; one CPU producer, no RNG use."""
    iterator = iter(source)
    pin = device == 'cuda'
    def next_pack():
        try:
            return pack(next(iterator), categories, pin)
        except StopIteration:
            return None
    if not prefetch:
        def direct():
            while (value := next_pack()) is not None:
                yield value
        yield direct()
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='mlp-prefetch') as worker:
        pending = [worker.submit(next_pack) for _ in range(prefetch)]
        def buffered():
            while pending:
                value = pending.pop(0).result()
                if value is None:
                    return
                pending.append(worker.submit(next_pack))
                yield value
        try:
            yield buffered()
        finally:
            for future in pending:
                future.cancel()


def to_device(packed, device):
    return [value.to(device, non_blocking=device == 'cuda') if value is not None else None
            for value in packed['values']]


def feed_loss(logits, targets, weights, feeds):
    """Same equal-feed BCE objective, with one reduction instead of a Python loop."""
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (losses * weights).sum() / feeds
