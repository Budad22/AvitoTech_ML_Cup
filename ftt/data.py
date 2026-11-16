"""Immutable whole-feed caches built from the shared prepared data contract."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from recsys_data import PreparedSplit
from recsys_data.core import FeedBatch, _flat


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def pack(batch):
    table = batch.candidate_table(include_target=True)
    counts = batch.candidate_counts.astype('int64')
    item, present = batch.candidate_vectors()
    user = batch.user_vectors()
    lengths = batch.features['history_length'].to_numpy().astype('int64')
    position = np.arange(50)[None]
    mask = position < lengths[:, None]
    indices = batch.history_offsets[:-1, None] + position
    def padded(name, fill):
        flat = _flat(batch.features, name).to_numpy()
        out = np.full(mask.shape, fill, dtype=flat.dtype)
        out[mask] = flat[indices[mask]]
        return out
    h, hp = batch.embeddings.lookup(padded('history__item_id__embedding_idx', -1))
    h = normalize(h)
    hmean = (h * hp[..., None]).sum(1) / np.maximum(hp.sum(1, keepdims=True), 1)
    return dict(counts=counts, numeric=np.column_stack([table[f'f{i}'].to_numpy() for i in range(1, 6)]),
                item=normalize(item), category=table['category_id'].to_numpy().astype('int64') + 128,
                target=table['target'].to_numpy().astype('float32'), user=normalize(user),
                history_mean=hmean.astype('float32'), history_vectors=h,
                history_mask=mask, history_category=padded('history__category_id', 0).astype('int64') + 128,
                history_event=padded('history__event_type', 0).astype('int64') + 128,
                history_source=padded('history__event_source', 0).astype('int64') + 128,
                flags=np.column_stack([present, np.repeat(np.linalg.norm(user, axis=1) == 0, counts),
                                       np.repeat(lengths == 0, counts), np.repeat(np.log1p(lengths), counts)]).astype('float32'),
                feed_id=batch.features['feed_id'].to_numpy(), user_id=batch.features['user_id'].to_numpy(),
                timestamp=batch.features['timestamp'].to_numpy(), item_id=table['item_id'].to_numpy())


FEED_KEYS = {'counts', 'user', 'history_mean', 'history_vectors', 'history_mask',
             'history_category', 'history_event', 'history_source', 'feed_id', 'user_id', 'timestamp'}


def prepare(data, output, emit):
    data, output = Path(data), Path(output)
    signature = dict(manifest=digest(data / 'manifest.json'), code=digest(__file__), version=1)
    if (output / 'metadata.json').exists():
        info = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
        if info['signature'] != signature:
            raise ValueError('Cache fingerprint mismatch: choose a fresh output')
        return info
    val = PreparedSplit(data, 'val')
    times = np.concatenate([pq.read_table(data / 'val/features' / f['name'], columns=['timestamp'])['timestamp'].to_numpy()
                            for f in val.info['files']])
    boundary = int(np.sort(times)[int(len(times) * .8)])
    info = dict(signature=signature, boundary=boundary, splits={})
    for split in ('train', 'val'):
        ds = PreparedSplit(data, split)
        files = ds.info['files']
        small = set(np.linspace(0, len(files)-1, max(1, round(len(files) * .0078))).round().astype(int))
        medium = set(np.linspace(0, len(files)-1, max(1, round(len(files) * .05))).round().astype(int)) | small
        selected = [int(i) for i in sorted(medium)] if split == 'train' else list(range(len(files)))
        chunks, total = [], 0
        for shard_index in selected:
            ds.info = dict(ds.info, files=[files[shard_index]])
            for batch in ds.iter_batches(128, include_labels=True):
                arrays = pack(batch)
                folder = output / split / str(len(chunks))
                folder.mkdir(parents=True, exist_ok=True)
                for key, array in arrays.items():
                    np.save(folder / (key + '.npy'), array, allow_pickle=False)
                n = len(arrays['counts'])
                chunks.append(dict(folder=str(folder.relative_to(output)), feeds=n,
                                   small=shard_index in small, shard=shard_index))
                total += n
                emit('prepare', split=split, feeds=total, shard=shard_index, shards=len(selected))
        info['splits'][split] = chunks
        if ds.embeddings._vectors is not None:
            ds.embeddings._vectors._mmap.close()
    write_json(output / 'metadata.json', info)
    return info


class Cache:
    def __init__(self, root, split, size='small'):
        self.root = Path(root)
        self.info = json.loads((self.root / 'metadata.json').read_text(encoding='utf-8'))
        self.chunks, self.index = [], []
        for record in self.info['splits'][split]:
            if split == 'train' and size == 'small' and not record['small']:
                continue
            arrays = {p.stem: np.load(p, mmap_mode='r') for p in (self.root / record['folder']).glob('*.npy')}
            arrays['offsets'] = np.r_[0, np.cumsum(arrays['counts'])]
            chunk = len(self.chunks)
            self.chunks.append(arrays)
            self.index.extend((chunk, i) for i in range(record['feeds']))
        self.feeds = len(self.index)
        self.users = np.array([self.chunks[c]['user_id'][i] for c, i in self.index])
        self.timestamps = np.array([self.chunks[c]['timestamp'][i] for c, i in self.index])

    def batch(self, chosen, device):
        values = {}
        counts = []
        for index in chosen:
            c, i = self.index[int(index)]
            arrays = self.chunks[c]
            a, b = arrays['offsets'][i:i+2]
            counts.append(int(b-a))
            for key, array in arrays.items():
                if key == 'offsets':
                    continue
                piece = array[i:i+1] if key in FEED_KEYS else array[a:b]
                values.setdefault(key, []).append(piece)
        result = {key: torch.as_tensor(np.concatenate(parts), device=device) for key, parts in values.items()}
        result['counts'] = counts
        for key in ('user', 'history_mean'):
            result[key] = torch.repeat_interleave(result[key], torch.tensor(counts, device=device), dim=0)
        return result

    def statistics(self):
        # Bounded deterministic sample; transforms fit on this training cache only.
        pieces = [c['numeric'][::max(1, len(c['numeric']) // 2048)] for c in self.chunks]
        x = np.concatenate(pieces).astype('float32')
        return x.mean(0), np.maximum(x.std(0), 1e-6), np.quantile(x, np.linspace(0, 1, 17), axis=0).T.astype('float32')

    def close(self):
        for chunk in self.chunks:
            for value in chunk.values():
                if isinstance(value, np.memmap):
                    value._mmap.close()
        self.chunks.clear()


class Full:
    """Sequential full-train reader: bounded RAM, no duplicated full feature cache."""
    def __init__(self, data, statistics):
        self.dataset = PreparedSplit(data, 'train')
        self.files = self.dataset.info['files']
        self.offsets = np.r_[0, np.cumsum([f['feeds'] for f in self.files])]
        self.feeds = int(self.offsets[-1])
        self.fitted = statistics
        self.loaded = {}

    def statistics(self):
        return self.fitted

    def batch(self, chosen, device):
        features, labels = [], []
        for index in chosen:
            shard = int(np.searchsorted(self.offsets, index, side='right')-1)
            row = int(index-self.offsets[shard])
            if shard not in self.loaded:
                name = self.files[shard]['name']
                self.loaded = {shard: (pq.read_table(self.dataset.root / 'train/features' / name),
                                       pq.read_table(self.dataset.root / 'train/labels' / name))}
            f, y = self.loaded[shard]
            features.append(f.slice(row, 1)); labels.append(y.slice(row, 1))
        arrays = pack(FeedBatch(pa.concat_tables(features), pa.concat_tables(labels), None, self.dataset.embeddings))
        result = {k:torch.as_tensor(v, device=device) for k,v in arrays.items()}
        result['counts'] = arrays['counts'].tolist()
        for key in ('user', 'history_mean'):
            result[key] = torch.repeat_interleave(result[key], torch.tensor(result['counts'], device=device), dim=0)
        return result
