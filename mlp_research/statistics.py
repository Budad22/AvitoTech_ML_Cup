"""Paired seed/user bootstrap and explicitly bounded architecture conclusions."""
import numpy as np

from mlp.metrics import ndcg


def paired_interval(candidate, control, users, comparisons=1, repetitions=10000, seed=2026):
    """Resample complete users and seeds; Bonferroni covers the declared pair family."""
    delta = np.atleast_2d(candidate) - np.atleast_2d(control)
    if delta.shape[1] != len(users) or not np.isfinite(delta).all():
        raise ValueError('Paired metric alignment or finiteness error')
    _, groups, counts = np.unique(users, return_inverse=True, return_counts=True)
    sums = np.stack([np.bincount(groups, weights=row, minlength=len(counts)) for row in delta])
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repetitions, 64):
        size = min(64, repetitions - start)
        sampled_seeds = rng.integers(0, len(delta), size=(size, len(delta)))
        sampled_groups = rng.integers(0, len(counts), size=(size, len(counts)))
        average = sums[sampled_seeds].mean(axis=1)
        numerator = np.take_along_axis(average, sampled_groups, axis=1).sum(axis=1)
        denominator = counts[sampled_groups].sum(axis=1)
        draws.extend((numerator / denominator).tolist())
    tail = .05 / (2 * max(1, comparisons))
    lower, upper = np.quantile(draws, [tail, 1 - tail])
    per_seed = delta.mean(axis=1)
    return dict(delta=float(delta.mean()), lower=float(lower), upper=float(upper),
                per_seed=per_seed.tolist(), positive_seeds=int((per_seed > 0).sum()),
                comparisons=comparisons, bootstrap_repetitions=repetitions,
                evidence=bool(lower > 0 and (per_seed > 0).sum() >= 2))


def blend_metrics(left, right, labels, offsets):
    """A predeclared 50/50 blend after within-feed scaling; no label-fitted weights."""
    result = []
    for l_scores, r_scores in zip(left, right):
        per_feed = []
        for a, b in zip(offsets[:-1], offsets[1:]):
            l_values, r_values = l_scores[a:b], r_scores[a:b]
            l_values = (l_values - l_values.mean()) / max(float(l_values.std()), 1e-6)
            r_values = (r_values - r_values.mean()) / max(float(r_values.std()), 1e-6)
            per_feed.append(ndcg(labels[a:b], .5 * l_values + .5 * r_values))
        result.append(per_feed)
    return np.asarray(result)


def conclusion(metrics, users, families, fast_scores, *, incomplete=False):
    names = list(metrics)
    # Reserve the predeclared fixed blend as one additional comparison candidate.
    comparisons = max(1, (len(names) + 1) * len(names) // 2)
    reference = metrics['history_control']
    differences = {name: paired_interval(values, reference, users, comparisons)
                   for name, values in metrics.items() if name != 'history_control'}
    mlps = [name for name in names if families[name] in ('mlp', 'plr', 'encoding', 'control')]
    alternatives = [name for name in names if families[name] == 'alternative' and name != 'linear_ranker']
    best_mlp = max(mlps, key=lambda name: fast_scores[name])
    best_alternative = max(alternatives, key=lambda name: fast_scores[name]) if alternatives else None
    comparison = paired_interval(metrics[best_alternative], metrics[best_mlp], users, comparisons) if best_alternative else None
    gain = any(value['evidence'] for name, value in differences.items() if families[name] in ('mlp', 'plr', 'encoding'))
    if incomplete:
        decision = 'inconclusive_incomplete'
        explanation = 'Не все подтверждающие опыты завершены; отрицательный вывод об архитектуре запрещён.'
    elif comparison and comparison['evidence']:
        decision = 'prioritize_alternative'
        explanation = 'Альтернатива устойчивее выбранной MLP на этом стенде; приоритет — проверка на большем train/GPU.'
    elif gain:
        decision = 'continue_mlp'
        explanation = 'Изменение MLP подтвердило прирост к контролю; направление заслуживает большей выборки/GPU.'
    else:
        decision = 'inconclusive_no_confirmed_gain'
        explanation = 'Нового устойчивого прироста нет. Остановить этот перебор и проверить объём данных или новые сигналы; предел всех MLP не доказан.'
    return dict(decision=decision, explanation=explanation, baseline='history_control', differences=differences,
                best_mlp=best_mlp, best_alternative=best_alternative, alternative_vs_mlp=comparison,
                scope='Only frozen 0.78% train, tested models and budgets. Confirmation is held out within EXP-029; this selection was used by historical experiments. Larger-data replication is required.')
