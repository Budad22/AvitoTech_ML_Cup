"""Compare additional NN-2 seeds under the frozen ensemble weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from mlp_trials.state import write_json
from .blend import write_submission
from .common import Split, feed_rank_pct, ndcg_full, val_holdout_mask


def read_report(path):
    return json.loads(path.read_bytes())


def read_scores(path, split):
    scores = np.load(path, mmap_mode='r', allow_pickle=False)
    if scores.shape != (split.N,) or scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise ValueError(f'Invalid predictions: {path}')
    return scores


def rank_scores(path, split):
    return feed_rank_pct(split.c_off, read_scores(path, split))


def blended_scores(split, reference, extra, weights, tiebreak):
    nn2_suffix = 'oof' if split.has_target else 'test'
    nn1_suffix = split.name
    nn2 = sorted(reference.glob(f'nn2_s*_{nn2_suffix}.npy'))
    more = sorted(extra.glob(f'nn2_s*_{nn2_suffix}.npy'))
    if not nn2 or not more or {path.name for path in nn2} & {path.name for path in more}:
        raise ValueError('Expected original NN-2 seeds and distinct additional seeds')
    original_ranks = [rank_scores(path, split) for path in nn2]
    extra_ranks = [rank_scores(path, split) for path in more]
    components = {}
    for name in weights:
        if name == 'nn1':
            paths = sorted(reference.glob(f'nn1_s*_{nn1_suffix}.npy'))
            if not paths:
                raise ValueError('Missing reference NN-1 predictions')
            components[name] = np.mean([rank_scores(path, split) for path in paths], axis=0)
        elif name == 'nn2':
            components[name] = np.mean(original_ranks, axis=0)
        else:
            suffix = nn1_suffix if name == 'gbdt_t' else nn2_suffix
            components[name] = rank_scores(reference / f'{name}_{suffix}.npy', split)

    def blend():
        return sum(weights[name] * ranks for name, ranks in components.items()) + 1e-4 * components[tiebreak]

    baseline = blend()
    original_nn2 = components['nn2']
    components['nn2'] = np.mean(original_ranks + extra_ranks, axis=0)
    return baseline, blend(), original_nn2, components['nn2']


def paired_result(users, before, after):
    """Bootstrap users and average per feed, preserving repeated-user clusters."""
    _, cluster = np.unique(users, return_inverse=True)
    delta = after - before
    sums = np.bincount(cluster, weights=delta)
    counts = np.bincount(cluster)
    rng = np.random.default_rng(20261123)
    draws = np.empty(2000)
    for index in range(len(draws)):
        sample = rng.integers(len(sums), size=len(sums))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return dict(reference_ndcg=float(before.mean()), candidate_ndcg=float(after.mean()),
                delta=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                feeds=len(before), users=len(sums))


def run(reference, extra, output):
    if output.resolve() in (reference.resolve(), extra.resolve()):
        raise ValueError('Use a separate comparison output directory')
    training = read_report(extra / 'report.json')
    if training['status'] != 'completed':
        raise ValueError('NN-2 cross-validation has not completed')
    seeds = [seed for seed, _ in training['seeds']]
    folds = [read_report(path) for path in extra.glob('seed*-fold*/fold.json')]
    expected_folds = {(seed, fold) for seed in seeds for fold in range(5)}
    if {(row['seed'], row['fold']) for row in folds} != expected_folds or len(folds) != len(expected_folds):
        raise ValueError('Incomplete user folds')
    for suffix in ('oof', 'test'):
        if {path.name for path in extra.glob(f'nn2_s*_{suffix}.npy')} != {f'nn2_s{seed}_{suffix}.npy' for seed in seeds}:
            raise ValueError(f'Unexpected new seed predictions for {suffix}')

    baseline = read_report(reference / 'blend.json')
    weights, tiebreak = baseline['weights'], baseline['tiebreak']
    val = Split('val')
    scores = blended_scores(val, reference, extra, weights, tiebreak)
    ranked = [ndcg_full(val.c_off, score, val.target) for score in scores]
    if not np.isclose(ranked[0][0], baseline['blended_val_ndcg'], atol=1e-12, rtol=0):
        raise ValueError('Frozen blend does not reproduce the reference NDCG')
    late = val_holdout_mask(val)
    slices = {'early': ~late, 'late': late, 'full': np.ones(val.F, dtype=bool)}
    comparisons = {
        key: paired_result(val.user_id[mask], ranked[0][1][mask], ranked[1][1][mask])
        for key, mask in slices.items()
    }
    promote = comparisons['early']['ci95'][0] > 0
    test = Split('test')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(experiment=training['experiment'], status='completed', training=training,
                  decision='promote' if promote else 'keep_reference', selection='early',
                  weights=weights, tiebreak=tiebreak, comparisons=comparisons,
                  nn2_full_val={'reference': ranked[2][0], 'candidate': ranked[3][0]},
                  ndcg={'NDCG · reference blend · full val': ranked[0][0],
                        'NDCG · five NN-2 blend · full val': ranked[1][0]})
    if promote:
        _, candidate, _, _ = blended_scores(test, reference, extra, weights, tiebreak)
        result['audit'] = write_submission(test, candidate, output / 'submission.parquet')
    else:
        for seed in seeds:
            read_scores(extra / f'nn2_s{seed}_test.npy', test)
    write_json(output / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Completed ensemble work directory')
    parser.add_argument('--extra', type=Path, required=True, help='Additional NN-2 work directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.reference, args.extra, args.output)))


if __name__ == '__main__':
    main()
