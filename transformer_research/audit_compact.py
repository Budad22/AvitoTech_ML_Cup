"""Check common validation inputs and benchmark disposable CUDA training batches."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .compact import CompactFeeds, shuffled_loader
from .core import CachedFeeds, official_parts, write_json


def main():
    p = argparse.ArgumentParser()
    for name in ('cache', 'reference-cache', 'source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    ranker, collate, _ = official_parts(a.source)
    training = CompactFeeds(a.cache, 'train'); val = CompactFeeds(a.cache, 'val')
    ref = CachedFeeds(a.reference_cache, 'val')
    try:
        old_categories = json.loads((a.reference_cache / 'metadata.json').read_text())['categories']
        if val.metadata['categories'] != old_categories:
            raise ValueError('Category mapping changed; explicit semantic comparison required')
        maximum_error = 0.
        # Every selected feed: verify native inputs against previously audited normalized cache.
        for i in range(len(val)):
            actual, expected = val[i], ref[i]
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
                maximum_error = max(maximum_error, float((left - right).abs().max()))
        torch.manual_seed(42)
        model = ranker(len(training.metadata['categories'])).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        loader = shuffled_loader(training, 32, collate)
        torch.cuda.synchronize(); started = time.monotonic()
        for index, batch in enumerate(loader):
            features, category, users, padding, target = (x.cuda() for x in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, category, users, padding)[~padding]
            loss = F.binary_cross_entropy_with_logits(logits, target[~padding])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite profile objective')
            loss.backward(); optimizer.step()
            if index == 39:
                break
        torch.cuda.synchronize(); seconds = time.monotonic() - started
        report = dict(event='loader_verified', validation_feeds=len(val), max_input_error=maximum_error,
                      batches=40, seconds=seconds, steps_per_second=40 / seconds,
                      estimated_training_hours=5 * len(loader) * seconds / 40 / 3600,
                      gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                      train_feeds=len(training), train_rows=training.rows,
                      weights_discarded=True, test_read=False)
        write_json(a.output, report); print(json.dumps(report), flush=True)
    finally:
        training.close(); val.close(); ref.close()


if __name__ == '__main__':
    main()
