"""Measure a cloned model on SSD; never writes production checkpoints."""
import argparse
import copy
from itertools import islice
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import pyarrow as pa
from mlp.model import Ranker
from mlp.pipeline import prepared_batches, to_device, feed_loss
from mlp.training import batches


def measure(args, cache, prefetch):
    state = torch.load(args.model, map_location='cpu', weights_only=False)
    model = Ranker(len(state['categories'])).cuda().train()
    model.load_state_dict(state['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    source = islice(batches(args.data, 'train', state['config']['batch_size'], state['config']['seed'] + state['epoch'],
                            embedding_cache=cache, start_batch=state['cursor']), args.steps + 3)
    waits = features = transfers = compute = 0.
    examples = 0
    warm_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with prepared_batches(source, state['categories'], device='cuda', prefetch=prefetch) as prepared:
        iterator = iter(prepared)
        for i in range(args.steps + 3):
            waiting = time.perf_counter()
            packed = next(iterator)
            delay = time.perf_counter() - waiting
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
            dense, category, target, weights = to_device(packed, 'cuda')
            events[1].record()
            optimizer.zero_grad(set_to_none=True)
            loss = feed_loss(model(dense, category), target, weights, packed['feeds'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite benchmark loss')
            loss.backward()
            optimizer.step()
            events[2].record()
            torch.cuda.synchronize()
            if i == 2:
                warm_seconds = time.perf_counter() - warm_started
                started = time.perf_counter()
            elif i > 2:
                waits += delay
                features += packed['feature_seconds']
                transfers += events[0].elapsed_time(events[1]) / 1000
                compute += events[1].elapsed_time(events[2]) / 1000
                examples += packed['candidates']
        seconds = time.perf_counter() - started
    return dict(cache=cache, prefetch=prefetch, batches=args.steps, seconds=seconds,
                batches_per_second=args.steps / seconds, candidates_per_second=examples / seconds,
                warmup_seconds=warm_seconds, loader_wait_seconds=waits, cpu_feature_seconds=features,
                h2d_seconds=transfers, compute_seconds=compute,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(), last_loss=loss.item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    pa.set_cpu_count(2)
    results = []
    for cache, prefetch in [('memmap', 0), ('ram', 0), ('ram', 2)]:
        result = measure(args, cache, prefetch)
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
