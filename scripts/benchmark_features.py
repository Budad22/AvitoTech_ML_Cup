"""Compare CPU feature builders on identical batches; never trains or uses CUDA."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow as pa
import torch
from mlp.training import batches


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='Trusted checkpoint; categories/config only')
    parser.add_argument('--baseline', type=Path, default=ROOT)
    parser.add_argument('--candidate', type=Path, required=True, help='Tree containing mlp/model.py and recsys_data/core.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--concurrent-training', action='store_true')
    args = parser.parse_args()
    if min(args.batches, args.repeats) < 1:
        parser.error('batches and repeats must be positive')
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    state = torch.load(args.model, map_location='cpu', weights_only=False)
    categories = state['categories']
    source = iter(batches(args.data, 'train', state['config']['batch_size'], state['config']['seed'],
                          embedding_cache='ram'))
    raw = [next(source) for _ in range(args.batches)]
    source.close()
    cases = {}
    for name, directory in (('baseline', args.baseline), ('candidate', args.candidate)):
        model = module(f'benchmark_{name}_model', directory / 'mlp/model.py')
        core = module(f'benchmark_{name}_core', directory / 'recsys_data/core.py')
        cases[name] = (model.features, core.EmbeddingStore.lookup)
    stores = {id(batch.embeddings): batch.embeddings for batch in raw}.values()
    def configure(name):
        for store in stores:
            store.lookup = MethodType(cases[name][1], store)
        return cases[name][0]

    configure('baseline')(raw[0], categories)  # Exclude RAM-cache warmup.
    for batch in raw:
        expected = configure('baseline')(batch, categories)
        actual = configure('candidate')(batch, categories)
        for left, right in zip(expected[:3], actual[:3]):
            np.testing.assert_array_equal(left, right)
            if left.dtype != right.dtype or left.tobytes() != right.tobytes():
                raise AssertionError('Feature bytes changed, including dtype or signed zeros')
        if not expected[3].equals(actual[3]):
            raise AssertionError('Candidate table changed')
    timings = {name: [] for name in cases}
    for repeat in range(args.repeats):
        order = list(cases) if repeat % 2 == 0 else list(reversed(cases))
        for name in order:
            assemble = configure(name)
            started = time.perf_counter()
            for batch in raw:
                assemble(batch, categories)
            timings[name].append(time.perf_counter() - started)
    results = {}
    candidates = sum(int(batch.candidate_counts.sum()) for batch in raw)
    for name, seconds in timings.items():
        median = statistics.median(seconds)
        results[name] = dict(seconds=seconds, median_ms_per_batch=1000 * median / args.batches,
                             batches_per_second=args.batches / median, candidates_per_second=candidates / median)
    report = dict(batches=args.batches, candidates=candidates, repeats=args.repeats,
                  exact_feature_equality=True, cuda_used=False, concurrent_training=args.concurrent_training,
                  results=results, speedup=results['baseline']['median_ms_per_batch'] /
                                          results['candidate']['median_ms_per_batch'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
