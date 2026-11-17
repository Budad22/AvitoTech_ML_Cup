"""Pure experiment selection and paired evaluation rules shared by both queues."""
import numpy as np


def rank_result(result):
    return -result['best'], result['seconds'], result['number']


def growth_rank(result):
    scores = {row['step']: row['ndcg'] for row in result['history']}
    return -(scores[800] - scores[200]), result['number']


def select_candidates(results, control, top_count):
    """Keep the control, top candidates, and one remaining slow learner."""
    candidates = sorted((r for r in results if r['number'] != control), key=rank_result)
    selected = [control, *[r['number'] for r in candidates[:top_count]]]
    remaining = [r for r in candidates if r['number'] not in selected]
    if remaining:
        selected.append(min(remaining, key=growth_rank)['number'])
    return selected


def validation_partition(timestamps, boundary, rng):
    """Consume the same RNG draws for the fixed fast-selection sample in every queue."""
    selection = np.flatnonzero(timestamps < boundary)
    fast = np.sort(rng.permutation(selection)[:min(2000, len(selection) // 2)])
    return selection, fast, np.setdiff1d(selection, fast)


def paired_summary(candidate, control, users, rng, repetitions=20000):
    """Resample whole users and seeds, preserving the original sampling order."""
    delta = np.asarray(candidate) - np.asarray(control)
    _, groups, counts = np.unique(users, return_inverse=True, return_counts=True)
    sums = np.stack([np.bincount(groups, weights=row) for row in delta])
    draws = []
    for _ in range(repetitions):
        seeds = rng.integers(0, 3, 3)
        sampled = rng.integers(0, len(counts), len(counts))
        draws.append(sums[seeds].mean(0)[sampled].sum() / counts[sampled].sum())
    lower, upper = np.quantile(draws, [.025, .975])
    per_seed = delta.mean(1)
    return dict(delta=float(delta.mean()), interval=[float(lower), float(upper)],
                per_seed=per_seed.tolist(),
                qualifies=bool(delta.mean() > 0 and (per_seed > 0).sum() >= 2 and lower > 0))
