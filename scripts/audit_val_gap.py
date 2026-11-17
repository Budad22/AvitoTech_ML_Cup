"""Audit saved val scores and time cohorts; never train or read test."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import polars as pl
from catboost.utils import eval_metric
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys_data import PreparedSplit
import pyarrow.parquet as pq


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def official_metrics(path):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == 'metrics') or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'METRICS' for t in n.targets))]
    # Python 3.9 compatibility for the official function's two zip(strict=True)
    # calls: both operate on sized sequences, so validate equal lengths first.
    def sized_zip(*sequences, strict=False):
        if strict and len({len(s) for s in sequences}) > 1:
            raise ValueError('zip lengths differ')
        return zip(*sequences)
    scope = dict(np=np, pl=pl, eval_metric=eval_metric, zip=sized_zip)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['metrics']


def keys(feed, item):
    return (feed.astype(np.uint64) << np.uint64(32)) | item.astype(np.uint64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--prediction', action='append', required=True, help='NAME=PATH')
    p.add_argument('--official', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); cohort = []; ids = []; items = []; labels = []
    for batch in PreparedSplit(a.data, 'val').iter_batches(512, include_labels=True):
        cohort.append(batch.features.select(['feed_id', 'timestamp', 'candidate_count', 'history_length']).to_pydict())
        table = batch.candidate_table(include_target=True)
        ids.append(table['feed_id'].to_numpy()); items.append(table['item_id'].to_numpy()); labels.append(table['target'].to_numpy())
    feeds, item, target = map(np.concatenate, (ids, items, labels))
    meta = {name: np.concatenate([np.asarray(b[name]) for b in cohort]) for name in cohort[0]}
    offsets = np.r_[0, meta['candidate_count'].cumsum(dtype=np.int64)]
    counts = meta['candidate_count']; n = len(counts)
    positive = np.add.reduceat(target.astype(np.int64), offsets[:-1])
    random_expected = np.array([
        (p / length) * (1 / np.log2(np.arange(length) + 2)).sum() /
        (1 / np.log2(np.arange(p) + 2)).sum() if p else 1.
        for length, p in zip(counts, positive)])
    early = meta['timestamp'] < 1790249984
    partition = dict(selection=early, late=~early, full=np.ones(n, dtype=bool))
    report = dict(audit='AUDIT-VAL-GAP-001', boundary=1790249984, test_read=False, training_performed=False,
                  official_source_sha256=sha256(a.official), cohorts={}, models={})
    for name, keep in partition.items():
        report['cohorts'][name] = dict(feeds=int(keep.sum()), rows=int(counts[keep].sum()),
             zero_positive_feeds=int((positive[keep] == 0).sum()),
             zero_positive_fraction=float((positive[keep] == 0).mean()),
             candidate_mean=float(counts[keep].mean()), positive_rate=float(positive[keep].sum() / counts[keep].sum()),
             mean_history_length=float(meta['history_length'][keep].mean()),
             random_ranking_expected_ndcg=float(random_expected[keep].mean()))
    expected = keys(feeds, item)
    official = official_metrics(a.official)
    for spec in a.prediction:
        name, path = spec.split('=', 1); path = Path(path)
        pred = pq.read_table(path, columns=['feed_id', 'item_id', 'score'])
        actual = keys(pred['feed_id'].to_numpy(), pred['item_id'].to_numpy())
        order = np.argsort(actual); sorted_keys = actual[order]
        if len(actual) != len(expected) or len(np.unique(actual)) != len(actual):
            raise ValueError('Prediction coverage/uniqueness differs: ' + name)
        where = np.searchsorted(sorted_keys, expected)
        if (where >= len(actual)).any() or not np.array_equal(sorted_keys[where], expected):
            raise ValueError('Prediction keys differ: ' + name)
        scores = pred['score'].to_numpy()[order[where]]
        if not np.isfinite(scores).all(): raise ValueError('Nonfinite scores')
        per_feed = np.empty(n)
        for i, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
            y = target[left:right]; s = scores[left:right]
            d = 1 / np.log2(np.arange(right - left) + 2)
            denominator = d[:int(positive[i])].sum()
            per_feed[i] = y[np.lexsort((y, -s))] @ d / denominator if denominator else 1.
        measured = official(pl.DataFrame(dict(feed_id=feeds, target=target, score=scores)))
        local = float(per_feed.mean())
        if abs(local - measured['ndcg']) > 1e-10:
            raise ValueError('Official metric mismatch')
        days = meta['timestamp'].astype(np.int64) // 86400
        result = dict(prediction_sha256=sha256(path), official=measured,
            official_difference=local-measured['ndcg'],
            ndcg={k: float(per_feed[v].mean()) for k,v in partition.items()},
            row_weighted_ndcg=float(np.average(per_feed, weights=counts)),
            daily=[dict(day_utc=datetime.fromtimestamp(int(day)*86400,timezone.utc).date().isoformat(),
                feeds=int((days==day).sum()), ndcg=float(per_feed[days==day].mean())) for day in np.unique(days)])
        result['early_minus_late'] = result['ndcg']['selection'] - result['ndcg']['late']
        report['models'][name] = result
        print(name, json.dumps(result['ndcg']), 'early_minus_late', result['early_minus_late'], flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('COHORTS', json.dumps(report['cohorts']), flush=True)


if __name__ == '__main__':
    main()
