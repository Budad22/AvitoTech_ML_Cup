"""Read-only cached candidate features; histories are not repeated for every item."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from recsys_data import PreparedSplit
from mlp.training import inspect, select
from mlp_trials.data import pack, context_features
from mlp_trials.models import history_stats
from mlp_trials.state import write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(root, split, boundary):
    dataset = PreparedSplit(root, split)
    columns = ['feed_id', 'user_id', 'timestamp', 'candidate_count', 'history_length']
    tables = [pq.read_table(Path(root) / split / 'features' / file['name'], columns=columns)
              for file in dataset.info['files']]
    table = pa.concat_tables(tables)
    if split == 'val':
        table = table.filter(pc.less(table['timestamp'], boundary))
    arrays = {name: table[name].to_numpy() for name in columns}
    arrays['offsets'] = np.r_[0, np.cumsum(arrays['candidate_count'], dtype=np.int64)]
    return arrays


def fast_partition(meta, size=2000, seed=2026):
    """Freeze stratified whole feeds using metadata only, never labels or predictions."""
    n = len(meta['feed_id'])
    if n < 2:
        raise ValueError('Selection needs at least two feeds')
    size = min(size, max(1, n // 2))
    rng = np.random.default_rng(seed)
    history = np.searchsorted(np.quantile(meta['history_length'], [.33, .67]), meta['history_length'])
    counts = np.searchsorted(np.quantile(meta['candidate_count'], [.33, .67]), meta['candidate_count'])
    groups = (meta['timestamp'].astype(np.int64) // 86400) * 9 + history * 3 + counts
    chosen = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        take = min(len(indices), int(size * len(indices) / n))
        chosen.extend(rng.permutation(indices)[:take].tolist())
    remaining = np.setdiff1d(np.arange(n), chosen)
    chosen.extend(rng.permutation(remaining)[:size - len(chosen)].tolist())
    fast = np.sort(chosen)
    return fast, np.setdiff1d(np.arange(n), fast)


def build(root, output, emit):
    """Commit metadata only after every cache array is flushed and validated."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(manifest=sha256(root / 'manifest.json'), version=1,
                     source={name: sha256(Path(__file__).parents[1] / name) for name in
                             ('mlp_research/cache.py', 'mlp/model.py', 'mlp_trials/data.py', 'mlp_trials/models.py')})
    if (output / 'metadata.json').exists():
        existing = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
        if existing['signature'] != signature:
            raise ValueError('Cache signature changed; use another cache directory')
        return existing
    prior = output / 'building.json'
    if prior.exists() and json.loads(prior.read_text()) != signature:
        raise ValueError('Incomplete cache belongs to other inputs/code')
    write_json(prior, signature)
    categories, boundary, sizes = inspect(root, 64)
    info = dict(signature=signature, categories=categories, boundary=boundary, partitions=sizes, splits={})
    for split in ('train', 'val'):
        meta = metadata(root, split, boundary)
        feeds, rows = len(meta['feed_id']), int(meta['offsets'][-1])
        folder = output / split
        folder.mkdir(exist_ok=True)
        for name, values in meta.items():
            np.save(folder / (name + '.npy'), values, allow_pickle=False)
        shapes = {'dense': ((rows, 519), 'float32'), 'history_candidate': ((rows, 15), 'float32'),
                  'history_feed': ((feeds, 514), 'float32'), 'context': ((rows, 17), 'float32'),
                  'category': ((rows,), 'int64'), 'target': ((rows,), 'uint8'), 'item_id': ((rows,), 'int32')}
        arrays = {name: np.lib.format.open_memmap(folder / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
                  for name, (shape, dtype) in shapes.items()}
        dataset = PreparedSplit(root, split, embedding_cache='memmap')
        row = feed = 0
        for batch in dataset.iter_batches(64, include_labels=True):
            if split == 'val':
                keep = np.flatnonzero(batch.features['timestamp'].to_numpy() < boundary)
                if not len(keep):
                    continue
                batch = select(batch, keep)
            packed = pack(batch, categories, kind='history_stats', history_window=128)
            values = packed['tensors']
            with torch.no_grad():
                stats = history_stats(values).numpy()
            n, f = packed['candidates'], packed['feeds']
            if not np.array_equal(meta['feed_id'][feed:feed + f], batch.features['feed_id'].to_numpy()):
                raise ValueError('Cache feed alignment changed')
            arrays['dense'][row:row + n] = values['dense'].numpy()
            arrays['history_candidate'][row:row + n] = stats[:, :15]
            offsets = np.r_[0, values['counts'].numpy().cumsum()]
            arrays['history_feed'][feed:feed + f] = stats[offsets[:-1], 15:]
            arrays['context'][row:row + n] = context_features(values['dense'].numpy(), values['category'].numpy(),
                                                             values['counts'].numpy())
            arrays['category'][row:row + n] = values['category'].numpy()
            arrays['target'][row:row + n] = values['target'].numpy()
            arrays['item_id'][row:row + n] = packed['table']['item_id'].to_numpy()
            row += n
            feed += f
            emit('cache_progress', split=split, current=feed, total=feeds, rows=row)
        if (row, feed) != (rows, feeds):
            raise ValueError('Incomplete cache coverage')
        for array in arrays.values():
            array.flush()
        del arrays
        info['splits'][split] = dict(feeds=feeds, rows=rows)
    train = CachedSplit(output, 'train')
    scalars = np.column_stack((train.arrays['dense'][:, :5], train.arrays['history_candidate'],
                               train.arrays['history_feed'][train.feed_for_rows(), :2]))
    if not np.isfinite(scalars).all():
        raise ValueError('Nonfinite cached scalars')
    np.savez(output / 'statistics.npz', mean=scalars.mean(axis=0, dtype=np.float64).astype('float32'),
             scale=np.maximum(scalars.std(axis=0, dtype=np.float64), 1e-6).astype('float32'),
             quantiles=np.quantile(scalars[:, :5], np.linspace(0, 1, 257), axis=0).T.astype('float32'))
    validation = CachedSplit(output, 'val')
    fast, confirmation = fast_partition(validation.arrays)
    np.save(output / 'fast.npy', fast)
    np.save(output / 'confirmation.npy', confirmation)
    info.update(fast_feeds=len(fast), confirmation_feeds=len(confirmation))
    train.close()
    validation.close()
    write_json(output / 'metadata.json', info)
    return info


class CachedSplit:
    def __init__(self, root, split):
        self.root = Path(root) / split
        self.arrays = {path.stem: np.load(path, mmap_mode='r', allow_pickle=False) for path in self.root.glob('*.npy')}
        self.feeds = len(self.arrays['feed_id'])

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()

    def feed_for_rows(self):
        return np.repeat(np.arange(self.feeds), self.arrays['candidate_count'])

    def batch(self, feeds, *, history=True, context=False):
        feeds = np.asarray(feeds, dtype=np.int64)
        counts = self.arrays['candidate_count'][feeds].astype(np.int64)
        rows = np.concatenate([np.arange(self.arrays['offsets'][i], self.arrays['offsets'][i + 1]) for i in feeds])
        inputs = {name: torch.from_numpy(np.array(self.arrays[name][rows], copy=True))
                  for name in ('dense', 'category', 'target')}
        inputs['target'] = inputs['target'].float()
        inputs['counts'] = torch.from_numpy(counts)
        inputs['weights'] = torch.from_numpy(np.repeat((1 / counts).astype('float32'), counts))
        if history:
            hist = np.concatenate((self.arrays['history_candidate'][rows],
                                   np.repeat(self.arrays['history_feed'][feeds], counts, axis=0)), axis=1)
            inputs['history'] = torch.from_numpy(hist)
        if context:
            inputs['context'] = torch.from_numpy(np.array(self.arrays['context'][rows], copy=True))
        return inputs
