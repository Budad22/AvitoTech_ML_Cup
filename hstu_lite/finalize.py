"""Validate the last sol.md experiments and prepare the two final blends."""
import argparse
import json
from pathlib import Path

import numpy as np

from mlp_trials.state import write_json
from scripts.validate_submission import validate
from .blend import load_models, mix, rank_models, write_submission
from .common import RAW, Split, ndcg_full, paired_result, val_holdout_mask


def completed(work):
    result = json.loads((work / 'report.json').read_bytes())
    if result['status'] != 'completed':
        raise ValueError(f'Experiment incomplete: {work}')
    return result


def compare(val, masks, before, after):
    return {key: paired_result(val.user_id[mask], before[mask], after[mask])
            for key, mask in masks.items()}


def improves(comparison):
    return comparison['early']['ci95'][0] > 0


def run(reference, nn1, nn2, recent, output):
    original = completed(reference)
    pretrain, finetuned, recency = (completed(path) for path in (nn1, nn2, recent))
    if pretrain['train_feeds'] != 3_151_208 or set(pretrain['results']) != {'nn1_s3', 'nn1_s4'}:
        raise ValueError('Expected NN-1 seeds 3 and 4 on the full train')
    if finetuned['seeds'] != [[3, 3], [4, 4]] or finetuned['epochs'] != 3:
        raise ValueError('Unexpected NN-2 seed schedule')
    if recency['train_feeds'] != 2_000_000 or set(recency['results']) != {'nn1_s0'}:
        raise ValueError('Expected NN-1 seed 0 on 2 million feeds')
    folds = [json.loads(path.read_bytes()) for path in nn2.glob('seed*-fold*/fold.json')]
    expected = {(seed, fold) for seed in (3, 4) for fold in range(5)}
    if len(folds) != len(expected) or {(row['seed'], row['fold']) for row in folds} != expected:
        raise ValueError('Expected ten completed user folds')
    if output.resolve() in {path.resolve() for path in (reference, nn1, nn2, recent)}:
        raise ValueError('Final export needs a separate directory')

    val, test = Split('val'), Split('test')
    for seed in (3, 4):
        if sum(row['heldout_feeds'] for row in folds if row['seed'] == seed) != val.F:
            raise ValueError(f'Invalid held-out feed count for seed {seed}')
    late = val_holdout_mask(val)
    masks = dict(early=~late, late=late, full=np.ones(val.F, dtype=bool))

    def per_feed(scores):
        return ndcg_full(val.c_off, scores, val.target)[1]

    def summarize(scores):
        values = per_feed(scores)
        return {name: float(values[mask].mean()) for name, mask in masks.items()}

    weights = original['blend']['weights']
    top = original['blend']['tiebreak']
    base = load_models(reference)
    more_nn1, more_nn2 = load_models(nn1), load_models(nn2)
    if set(more_nn1) != {'nn1_s3', 'nn1_s4'} or set(more_nn2) != {'nn2_s3', 'nn2_s4'}:
        raise ValueError('Missing paired seed predictions')
    if set(base) & (set(more_nn1) | set(more_nn2)):
        raise ValueError('New seed identifiers collide with the reference')
    expanded = base | more_nn1 | more_nn2
    base_rank, _ = rank_models(base, val, test)
    expanded_rank, expanded_metrics = rank_models(expanded, val, test)
    before = per_feed(mix(base_rank, weights, top, 0))
    if not np.isclose(before.mean(), original['blend']['blended_val_ndcg'], atol=1e-12, rtol=0):
        raise ValueError('Reference blend does not reproduce the reported NDCG')
    after = per_feed(mix(expanded_rank, weights, top, 0))
    seeds = compare(val, masks, before, after)
    if improves(seeds):
        models, ranked, value, selected = expanded, expanded_rank, after, 'five_seeds'
    else:
        models, ranked, value, selected = base, base_rank, before, 'reference'

    recent_model = load_models(recent)
    if set(recent_model) != {'nn1_s0'}:
        raise ValueError('Recency export must contain NN-1 seed 0')
    rank_models(recent_model, val, test)
    recent_pair = compare(val, masks, per_feed(base['nn1_s0'][0]), per_feed(recent_model['nn1_s0'][0]))
    recent_blend = None
    if improves(recent_pair):
        recent_rank, _ = rank_models(models | recent_model, val, test)
        candidate = per_feed(mix(recent_rank, weights, top, 0))
        recent_blend = compare(val, masks, value, candidate)
        if improves(recent_blend):
            ranked, value, selected = recent_rank, candidate, selected + '_recent'

    output.mkdir(parents=True, exist_ok=True)
    if selected == 'reference':
        path = reference / 'submission.parquet'
        expected = original['blend']['audit']['sha256']
        primary_audit = validate(path, Path(RAW, 'sample_submission.parquet'), expected)
    else:
        primary_audit = write_submission(test, mix(ranked, weights, top, 1), output / 'primary.parquet')
    primary = {name: float(value[mask].mean()) for name, mask in masks.items()}

    backup_models = {name: ranked[name] for name in ('nn2', 'meta')}
    backup_weights = dict.fromkeys(backup_models, .5)
    backup_top = max(backup_models, key=lambda name: per_feed(backup_models[name][0])[~late].mean())
    backup = summarize(mix(backup_models, backup_weights, backup_top, 0))
    backup_audit = write_submission(test, mix(backup_models, backup_weights, backup_top, 1),
                                    output / 'backup.parquet')
    result = dict(experiment=output.parent.name, status='completed', selected=selected,
                  folds=len(folds), weights=weights,
                  comparisons=dict(five_seeds=seeds, recent_nn1=recent_pair, recent_blend=recent_blend),
                  expanded_component_ndcg=expanded_metrics,
                  training=dict(nn1=pretrain, nn2=finetuned, recency=recency),
                  primary=dict(ndcg=primary, audit=primary_audit),
                  backup=dict(ndcg=backup, weights=backup_weights, tiebreak=backup_top, audit=backup_audit),
                  ndcg={'NDCG · основной · вся val': primary['full'],
                        'NDCG · резерв · вся val': backup['full']})
    write_json(output / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('reference', 'nn1', 'nn2', 'recent', 'output'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.reference, args.nn1, args.nn2, args.recent, args.output)))


if __name__ == '__main__':
    main()
