"""Two-pass projected Parquet conversion, bounded RAM, atomic ready manifest."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from .core import sha256, write_json

FLOATS = ['f1', 'f2', 'f3', 'f4', 'f5']
SMALL = ['feed_id', 'timestamp', 'user_id', 'item_id', 'category_id']
COLUMNS = SMALL + FLOATS + ['item_id__embedding_idx', 'user_embedding', 'target']


def batches(raw, split, columns, boundary):
    for path in sorted((raw / split / 'impressions').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
            if split == 'val':
                batch = batch.filter(pc.less(batch.column('timestamp'), boundary))
            if batch.num_rows:
                yield batch


def column(batch, name):
    return batch.column(batch.schema.get_field_index(name))


def flatten(batch, name):
    array = column(batch, name)
    if array.null_count:
        raise ValueError('Null list: ' + name)
    values = pc.list_flatten(array)
    if values.null_count:
        raise ValueError('Null values: ' + name)
    return values.to_numpy(zero_copy_only=False)


def lengths(batch, name):
    return pc.list_value_length(column(batch, name)).to_numpy(zero_copy_only=False)


def count(raw, split, boundary):
    feeds = rows = 0
    categories = set()
    for batch in batches(raw, split, SMALL, boundary):
        counts = lengths(batch, 'item_id')
        if (counts <= 0).any() or not np.array_equal(counts, lengths(batch, 'category_id')):
            raise ValueError('Candidate lengths invalid')
        feeds += len(counts); rows += int(counts.sum())
        categories.update(np.unique(flatten(batch, 'category_id')).tolist())
    return dict(feeds=feeds, rows=rows), categories


def convert(raw, output, split, counts, mapping, boundary, item_count, diagnostics=None):
    folder = output / split; folder.mkdir(exist_ok=True)
    n, r = counts['feeds'], counts['rows']
    schema = dict(floats=('float32', (r, 5)), category=('uint8', (r,)), target=('uint8', (r,)),
                  item_index=('int32', (r,)), item_id=('int32', (r,)), users=('int8', (n, 128)),
                  offsets=('int64', (n + 1,)), feed_id=('int32', (n,)),
                  timestamp=('int32', (n,)), user_id=('int32', (n,)))
    arrays = {key: np.lib.format.open_memmap(folder / (key + '.npy'), mode='w+', dtype=dtype, shape=shape)
              for key, (dtype, shape) in schema.items()}
    arrays['offsets'][0] = 0
    f = rpos = 0
    duplicate_candidate_rows = 0
    for batch in batches(raw, split, COLUMNS, boundary):
        sizes = lengths(batch, 'item_id'); b = len(sizes); rows = int(sizes.sum())
        for key in FLOATS + ['category_id', 'target', 'item_id__embedding_idx']:
            if not np.array_equal(sizes, lengths(batch, key)):
                raise ValueError('Unaligned list: ' + key)
        fs, rs = slice(f, f + b), slice(rpos, rpos + rows)
        for key in ('feed_id', 'timestamp', 'user_id'):
            value = column(batch, key).to_numpy(zero_copy_only=False)
            if value.min() < np.iinfo(np.int32).min or value.max() > np.iinfo(np.int32).max:
                raise ValueError('Int32 key overflow')
            arrays[key][fs] = value
        users = np.array(column(batch, 'user_embedding').to_pylist())
        if users.shape != (b, 128) or not np.isfinite(users).all() or (users < -128).any() or (users > 127).any() or not np.equal(users, users.astype(np.int8)).all():
            raise ValueError('User vectors are not int8[128]')
        arrays['users'][fs] = users
        floats = np.column_stack([flatten(batch, key) for key in FLOATS]).astype(np.float32)
        if not np.isfinite(floats).all():
            raise ValueError('Nonfinite float features')
        arrays['floats'][rs] = floats
        targets = flatten(batch, 'target')
        if not np.isin(targets, [0, 1]).all():
            raise ValueError('Nonbinary targets')
        arrays['target'][rs] = targets
        indices = flatten(batch, 'item_id__embedding_idx')
        if (indices < -1).any() or (indices >= item_count).any():
            raise ValueError('Invalid item embedding indices')
        arrays['item_index'][rs] = indices
        item_ids = flatten(batch, 'item_id')
        if (item_ids < 0).any() or (item_ids > np.iinfo(np.int32).max).any():
            raise ValueError('Invalid item IDs')
        arrays['item_id'][rs] = item_ids
        cats = flatten(batch, 'category_id')
        encoded = np.zeros(rows, dtype=np.uint8)
        for value in np.unique(cats):
            encoded[cats == value] = mapping.get(int(value), 0)
        arrays['category'][rs] = encoded
        arrays['offsets'][f + 1:f + b + 1] = rpos + sizes.cumsum()
        # Pair uniqueness inside each complete feed, in a bounded batch.
        keys = (np.repeat(arrays['feed_id'][fs].astype(np.uint64), sizes) << np.uint64(32)) | item_ids.astype(np.uint64)
        duplicates = rows - len(np.unique(keys))
        if duplicates and split != 'train':
            raise ValueError('Duplicate feed/item pair in validation')
        # Native CandidateData retains repeated train candidates. Preserve every
        # candidate, its position and target; report duplicates rather than dedup.
        duplicate_candidate_rows += duplicates
        f += b; rpos += rows
    if f != n or rpos != r or len(np.unique(arrays['feed_id'])) != n:
        raise ValueError('Split coverage or feed uniqueness differs')
    for array in arrays.values():
        array.flush(); array._mmap.close()
    if diagnostics is not None:
        diagnostics['duplicate_candidate_rows_preserved'] = duplicate_candidate_rows
    return {str(path.relative_to(output)).replace('\\', '/'): sha256(path) for path in sorted(folder.glob('*.npy'))}


def build(raw, output, reference, source):
    source_cfg = json.loads(source.read_text())
    ready = raw / 'raw-ready.json'
    if not ready.exists():
        raise ValueError('Verified raw data marker missing')
    identity = dict(source_sha256=sha256(source), raw_ready_sha256=sha256(ready),
                    reference_sha256=sha256(reference / 'metadata.json'), schema_version=2,
                    builder_sha256=sha256(Path(__file__)))
    if (output / 'metadata.json').exists():
        metadata = json.loads((output / 'metadata.json').read_text())
        if metadata['identity'] != identity:
            raise ValueError('Completed dataset identity differs')
        for name, digest in metadata['files_sha256'].items():
            if sha256(output / name) != digest:
                raise ValueError('Completed dataset checksum changed: ' + name)
        return
    output.mkdir(parents=True, exist_ok=True)
    ref = json.loads((reference / 'metadata.json').read_text())
    boundary = ref['boundary']
    splits = {}; cats = None; items = {}
    for split in ('train', 'val'):
        splits[split], observed = count(raw, split, boundary)
        if split == 'train':
            cats = {int(value): index + 1 for index, value in enumerate(sorted(observed))}
        meta = json.loads((raw / split / 'item_embeddings/data.meta').read_text())
        matrix = raw / split / 'item_embeddings/data.memmap'
        if meta['dtype'] != 'int8' or meta['shape'][1] != 128 or meta['order'] != 'C' or matrix.stat().st_size != int(np.prod(meta['shape'], dtype=np.int64)):
            raise ValueError('Item matrix schema differs')
        items[split] = dict(meta, path=os.path.relpath(matrix, output).replace('\\', '/'))
        print(json.dumps(dict(event='counted', split=split, **splits[split])), flush=True)
    if splits['train']['feeds'] != source_cfg['expected_train_feeds'] or splits['val'] != ref['splits']['val'] or len(cats) > 255:
        raise ValueError('Expected full train/selection coverage differs')
    hashes = {}; diagnostics = {}
    for split in ('train', 'val'):
        diagnostics[split] = {}
        hashes.update(convert(raw, output, split, splits[split], cats, boundary, items[split]['shape'][0], diagnostics[split]))
        print(json.dumps(dict(event='converted', split=split)), flush=True)
    for name in ('fast.npy', 'confirmation.npy'):
        shutil.copyfile(reference / name, output / name); hashes[name] = sha256(output / name)
    for name in ('offsets', 'feed_id', 'item_id', 'target'):
        actual = np.load(output / 'val' / (name + '.npy'), mmap_mode='r')
        expected = np.load(reference / 'val' / (name + '.npy'), mmap_mode='r')
        if not np.array_equal(actual, expected):
            raise ValueError('Selection parity failed: ' + name)
        actual._mmap.close(); expected._mmap.close()
    write_json(output / 'metadata.json', dict(identity=identity, splits=splits, categories=cats, boundary=boundary,
               items=items, files_sha256=hashes, diagnostics=diagnostics, train_shards=len(source_cfg['members']), test_read=False))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--reference-cache', type=Path, required=True)
    p.add_argument('--source-config', type=Path, required=True)
    a = p.parse_args(); started = time.monotonic(); build(a.raw, a.output, a.reference_cache, a.source_config)
    print(json.dumps(dict(event='compact_ready', seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
