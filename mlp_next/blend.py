"""Bounded val-only checks of existing predictions; never creates a submission."""
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from recsys_data import PreparedSplit
from mlp.metrics import ndcg


def keys(feed, item):
    return (np.asarray(feed).astype('uint64') << np.uint64(32)) | np.asarray(item).astype('uint32').astype('uint64')


def aligned_scores(path, canonical):
    table = pq.read_table(path, columns=['feed_id', 'item_id', 'score'])
    source = keys(table['feed_id'].to_numpy(), table['item_id'].to_numpy())
    order = np.argsort(source)
    if len(source) != len(canonical) or len(np.unique(source)) != len(source):
        raise ValueError('Prediction key count/uniqueness mismatch')
    if not np.array_equal(source[order], np.sort(canonical)):
        raise ValueError('Prediction pair coverage mismatch')
    score = table['score'].to_numpy()[order][np.searchsorted(source[order], canonical)]
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite blend input')
    return score


def check_blends(data, candidates):
    available = {k: Path(p) for k, p in candidates.items() if p and Path(p).exists()}
    missing = [k for k in candidates if k not in available]
    if 'history_stats' not in available:
        return dict(status='unavailable', missing=missing, reason='No saved full history_stats val predictions')
    ds = PreparedSplit(data, 'val')
    feeds, items, labels, times, counts = [], [], [], [], []
    for batch in ds.iter_batches(128, include_labels=True):
        table = batch.candidate_table(include_target=True)
        feeds.extend(table['feed_id'].to_numpy())
        items.extend(table['item_id'].to_numpy())
        labels.extend(table['target'].to_numpy())
        times.extend(batch.features['timestamp'].to_numpy())
        counts.extend(batch.candidate_counts)
    canonical = keys(feeds, items)
    offsets = np.r_[0, np.cumsum(counts)]
    labels = np.asarray(labels)
    selection = np.asarray(times) < 1790249984

    def normalize(score):
        out = np.empty_like(score)
        for a, b in zip(offsets[:-1], offsets[1:]):
            part = score[a:b]
            out[a:b] = (part - part.mean()) / max(float(part.std()), 1e-6)
        return out

    def metrics(score):
        score = score.astype('float32')
        values = np.array([ndcg(labels[a:b], score[a:b]) for a, b in zip(offsets[:-1], offsets[1:])])
        return dict(selection=float(values[selection].mean()), full_val=float(values.mean()),
                    late_control=float(values[~selection].mean()))

    scores = {name: normalize(aligned_scores(path, canonical)) for name, path in available.items()}
    baseline = scores['history_stats']
    rows = [dict(expert='history_stats', weight=0, metrics=metrics(baseline))]
    # One small fixed grid against the same baseline; no control-based retuning.
    for name, values in scores.items():
        if name != 'history_stats':
            for weight in (.1, .25, .5, 1.):
                rows.append(dict(expert=name, weight=weight,
                                 metrics=metrics((1-weight)*baseline + weight*values)))
    chosen = max(rows, key=lambda r: r['metrics']['selection'])
    return dict(status='completed', rows=rows, chosen=chosen, missing=missing,
                rows_count=len(canonical), feeds=len(counts), test_read=False,
                limitation='Compared with full MLP parent. Existing CatBoost ensemble predictions unavailable; no claim of beating it. Historical val was used before.')
