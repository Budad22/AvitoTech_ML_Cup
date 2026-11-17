"""Check a real whole-feed CUDA step before starting the registered model pair."""
import argparse
import json
from pathlib import Path
import time

from tabnet_research.run import configure_source, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlp-source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure_source(args.mlp_source, args.dependencies)
    import numpy as np
    import torch
    from mlp_research.cache import CachedSplit
    from tabnet_research.model import make_model, training_loss
    from tabnet_research.run import BatchDevice

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; model training was not started')
    torch.set_num_threads(4)
    metadata = json.loads((args.cache / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['splits']['train'] != dict(feeds=24592, rows=1925512):
        raise ValueError('Expected the registered train4 cohort')
    with np.load(args.cache / 'statistics.npz') as saved:
        statistics = {name: saved[name].copy() for name in saved.files}
    data = BatchDevice(CachedSplit(args.cache, 'train'), 'cuda')
    measurements = []
    try:
        inputs = data.batch(np.arange(16), history=True, context=False)
        for kind in ('mlp', 'tabnet'):
            torch.manual_seed(42)
            model = make_model(kind, len(metadata['categories']), statistics).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                objective = training_loss(model, inputs, kind)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize()
            measurements.append(dict(kind=kind, steps=3, loss=float(objective.detach()),
                                     seconds=time.monotonic() - started,
                                     peak_bytes=torch.cuda.max_memory_allocated()))
            del optimizer, model, objective
            torch.cuda.empty_cache()
    finally:
        data.close()
    result = dict(status='passed', gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, cuda=torch.version.cuda, measurements=measurements,
                  note='Three diagnostic optimizer steps; weights discarded; no validation tuning',
                  test_read=False, submission_created=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
