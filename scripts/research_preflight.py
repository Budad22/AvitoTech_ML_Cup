"""Synthetic CUDA checks for the frozen finalists; no competition-data optimization."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from mlp_research.catalog import hypotheses
from mlp_research.models import ResearchRanker, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalists', type=Path, required=True)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    names = json.loads(args.finalists.read_text(encoding='utf-8'))['names']
    with np.load(args.statistics) as values:
        statistics = {key: values[key].copy() for key in values.files}
    results = []
    for config in hypotheses():
        if config['name'] not in names:
            continue
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        count = 16 * 300
        inputs = dict(dense=torch.rand(count, 519, device='cuda'),
                      category=torch.zeros(count, dtype=torch.long, device='cuda'),
                      history=torch.rand(count, 529, device='cuda'),
                      context=torch.rand(count, 17, device='cuda'), counts=torch.full((16,), 300),
                      weights=torch.full((count,), 1 / 300, device='cuda'),
                      target=(torch.arange(count, device='cuda') % 10 == 0).float())
        model = ResearchRanker(128, config, statistics).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        objective = loss(model(inputs), inputs, config['loss'])
        if not torch.isfinite(objective):
            raise ValueError('Nonfinite preflight loss: ' + config['name'])
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        results.append(dict(name=config['name'], device='cuda', loss=float(objective.detach()),
                            peak_mib=torch.cuda.max_memory_allocated() / 1024**2))
        del model, optimizer, inputs, objective
        torch.cuda.empty_cache()
    if {result['name'] for result in results} != set(names):
        raise ValueError('Incomplete CUDA checks')
    args.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), cases=results), indent=2), encoding='utf-8')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
