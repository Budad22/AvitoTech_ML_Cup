"""Binary NDCG with pessimistic ties, matching the official evaluator."""
import numpy as np


def ndcg(target, score, cutoff=None):
    target, score = np.asarray(target), np.asarray(score)
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite prediction')
    ranked = target[np.lexsort((target, -score))]
    ideal = np.sort(target)[::-1]
    discount = 1 / np.log2(np.arange(len(target)) + 2)
    denominator = np.dot(ideal[:cutoff], discount[:cutoff])
    return float(np.dot(ranked[:cutoff], discount[:cutoff]) / denominator) if denominator else 1.
