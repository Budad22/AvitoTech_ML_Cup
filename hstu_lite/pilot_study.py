"""Compare matched HSTU-lite smoke checkpoints on all official validation feeds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mlp_trials.state import write_json
from .common import Split, ndcg_full, val_holdout_mask
from .nn_train import Ranker, predict
from .seed_study import paired_result, read_report


def evaluate_checkpoint(work, val, device):
    report = read_report(work / 'report.json')
    if report['status'] != 'completed':
        raise ValueError(f'Pilot incomplete: {work}')
    result = report['results']['warmup_ema']
    path = work / 'warmup_ema' / 'best.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['step'] != result['best_step'] or state['val'] != result['ndcg']:
        raise ValueError(f'Checkpoint does not match report: {path}')
    model = Ranker(**state['cfg']).to(device)
    model.load_state_dict(state['sd'])
    scores = predict(model, 'val', np.arange(val.F), device, workers=0)
    _, per_feed = ndcg_full(val.c_off, scores, val.target)
    with path.open('rb') as stream:
        sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    return per_feed, dict(model=state['cfg'], step=state['step'],
                          permutation_error=result['permutation_error'], sha256=sha256)


def run(reference, candidate, device):
    prior = read_report(reference / 'plan.json')
    current = read_report(candidate / 'plan.json')
    if prior != {key: value for key, value in current.items() if key != 'model_parameters'}:
        raise ValueError('Plans must match outside model configuration')
    val = Split('val')
    before, baseline = evaluate_checkpoint(reference, val, device)
    after, wide = evaluate_checkpoint(candidate, val, device)
    if before.shape != after.shape or baseline['model'] == wide['model']:
        raise ValueError('Expected different models on the same feeds')
    late = val_holdout_mask(val)
    slices = dict(early=~late, late=late, full=np.ones(val.F, dtype=bool))
    metrics = {name: paired_result(val.user_id[mask], before[mask], after[mask])
               for name, mask in slices.items()}
    promote = (metrics['early']['ci95'][0] > 0
               and wide['permutation_error'] < 1e-3)
    return dict(reference=baseline, candidate=wide, comparisons=metrics,
                decision='promote' if promote else 'keep_reference',
                selection_hash=prior['selection_hash'], train_feed_hash=prior['train_feed_hash'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.output.name == 'report.json':
        parser.error('Choose a separate filename for the paired comparison')
    torch.set_num_threads(4 if args.device == 'cuda' else 2)
    result = run(args.reference, args.candidate, args.device)
    write_json(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
