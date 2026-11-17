"""Descriptive exposure diagnostics; observed CTR is not a propensity estimate."""
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from recsys_data import PreparedSplit


def exposure(data):
    ds = PreparedSplit(data, 'train')
    boundaries = np.array([0, 1, 5, 20, 100, 301])
    counts = np.zeros(5, dtype=np.int64)
    positive = counts.copy()
    for file in ds.info['files']:
        name = file['name']
        positions = pc.list_flatten(pq.read_table(Path(data) / 'train/diagnostics' / name)['pos']).to_numpy()
        target = pc.list_flatten(pq.read_table(Path(data) / 'train/labels' / name)['target']).to_numpy()
        if len(positions) != len(target):
            raise ValueError('Exposure label/position alignment differs')
        bins = np.searchsorted(boundaries[1:], positions, side='right')
        counts += np.bincount(bins, minlength=5)
        positive += np.bincount(bins, weights=target, minlength=5).astype('int64')
    return dict(boundaries=boundaries.tolist(), counts=counts.tolist(), positives=positive.tolist(),
                observed_rate=(positive / np.maximum(counts, 1)).tolist(),
                limitation='Quality and exposure are confounded. Do not use inverse CTR as propensity weights.')
