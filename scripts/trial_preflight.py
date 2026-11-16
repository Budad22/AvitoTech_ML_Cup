"""Synthetic forward/backward memory check; does not train on competition data."""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mlp_trials.models import KINDS, create_model, score
from mlp_trials.losses import objective


def check(batch_size=64, candidates=300, device='cuda'):
    torch.set_num_threads(4)
    results = []
    for kind in KINDS:
        torch.manual_seed(42)
        n, h = batch_size * candidates, 128 if kind == 'history_stats' else 64
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=device)
        inputs = dict(dense=zeros(n, 519), category=zeros(n, dtype=torch.long),
                      counts=torch.full((batch_size,), candidates, device=device, dtype=torch.long),
                      feed=torch.arange(batch_size, device=device).repeat_interleave(candidates),
                      row=torch.arange(candidates, device=device).repeat(batch_size),
                      target=zeros(n), weights=torch.full((n,), 1 / candidates, device=device),
                      context=zeros(n, 17), norms=zeros(n, 2), h_vectors=zeros(batch_size, h, 128),
                      h_present=torch.ones((batch_size, h), device=device, dtype=torch.bool),
                      h_mask=torch.ones((batch_size, h), device=device, dtype=torch.bool),
                      h_lengths=torch.full((batch_size,), h, device=device, dtype=torch.long),
                      h_category=zeros(batch_size, h, dtype=torch.long), h_type=zeros(batch_size, h, dtype=torch.long),
                      h_source=zeros(batch_size, h, dtype=torch.long),
                      h_position=torch.arange(h, device=device).repeat(batch_size, 1))
        inputs['target'][::5] = 1
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model = create_model(128, kind, h).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = objective(score(model, inputs, kind), inputs, kind)
        if not torch.isfinite(loss):
            raise ValueError(f'{kind}: nonfinite loss')
        loss.backward()
        optimizer.step()
        peak = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
        results.append(dict(kind=kind, loss=loss.item(), peak_cuda_mib=round(peak / 1024**2, 1)))
        print(json.dumps(results[-1]), flush=True)
        del inputs, model, optimizer, loss
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--candidates', type=int, default=300)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.batch_size, args.candidates, args.device)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
