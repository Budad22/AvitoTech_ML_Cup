"""Read-only trial inputs; technical row indices are used only for alignment."""
import random

import numpy as np
import torch

from recsys_data import PreparedSplit
from recsys_data.core import CANDIDATES, _flat
from mlp.model import features, normalize
from mlp.training import select

HISTORY_KINDS = {'history_stats', 'history_pool', 'history_attention'}
HISTORY_COLUMNS = ['history_length', 'history__category_id', 'history__event_type',
                   'history__event_source', 'history__item_id__embedding_idx']


def batches(root, split, batch_size, *, seed=None, start_batch=0, history=False, cache='memmap'):
    dataset = PreparedSplit(root, split, embedding_cache=cache)
    dataset.info = dict(dataset.info, files=list(dataset.info['files']))
    if seed is not None:
        random.Random(seed).shuffle(dataset.info['files'])
    # Skip complete shards without decoding earlier batches on resume.
    skipped = 0
    while dataset.info['files']:
        count = (dataset.info['files'][0]['feeds'] + batch_size - 1) // batch_size
        if skipped + count > start_batch:
            break
        skipped += count
        dataset.info['files'].pop(0)
    columns = ['feed_id', 'timestamp', 'user_embedding', 'candidate_count', 'candidate_embedding_present', *CANDIDATES]
    if history:
        columns += HISTORY_COLUMNS
    for index, batch in enumerate(dataset.iter_batches(batch_size, include_labels=split != 'test',
                                                      feature_columns=columns), skipped):
        if index < start_batch:
            continue
        if seed is not None:
            order = np.random.default_rng(np.random.SeedSequence([seed, index])).permutation(len(batch.features))
            batch = select(batch, order)
        yield batch


def category_indices(codes, categories):
    lookup = np.zeros(256, dtype=np.int64)
    for code, index in categories.items():
        lookup[code] = index
    return lookup[codes]


def context_features(dense, category, counts):
    output = np.empty((len(dense), 17), dtype=np.float32)
    offset = 0
    for count in counts:
        end = offset + count
        numeric = dense[offset:end, :5].astype(np.float64)
        mean, std = numeric.mean(axis=0), numeric.std(axis=0)
        output[offset:end, 5:10] = numeric - mean
        output[offset:end, 10:15] = (numeric - mean) / np.maximum(std, 1e-6)
        for column in range(5):
            _, inverse, frequencies = np.unique(numeric[:, column], return_inverse=True, return_counts=True)
            average_rank = (np.cumsum(frequencies) - frequencies + (frequencies - 1) / 2) / max(1, count - 1)
            output[offset:end, column] = average_rank[inverse]
        _, inverse, frequencies = np.unique(category[offset:end], return_inverse=True, return_counts=True)
        output[offset:end, 15] = frequencies[inverse] / count
        output[offset:end, 16] = np.log1p(count)
        offset = end
    return output


def pack(batch, categories, pin=False, *, kind='baseline', history_window=64):
    dense, category, target, table = features(batch, categories)
    counts = batch.candidate_counts.astype(np.int64)
    offsets = np.r_[0, counts.cumsum()]
    feeds = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    rows = np.arange(len(dense), dtype=np.int64) - np.repeat(offsets[:-1], counts)
    arrays = dict(dense=dense, category=category, counts=counts, feed=feeds, row=rows,
                  weights=np.repeat((1 / counts).astype(np.float32), counts))
    if target is not None:
        arrays['target'] = target
    if kind in ('context_stats', 'context_pool'):
        arrays['context'] = context_features(dense, category, counts)
    if kind == 'raw_norm':
        item, _ = batch.candidate_vectors()
        arrays['norms'] = np.column_stack((np.repeat(np.log1p(np.linalg.norm(batch.user_vectors(), axis=1)), counts),
                                           np.log1p(np.linalg.norm(item, axis=1)))).astype(np.float32)
    if kind in HISTORY_KINDS:
        lengths = batch.features['history_length'].to_numpy().astype(np.int64)
        positions = np.broadcast_to(np.arange(history_window), (len(counts), history_window))
        mask = positions < lengths[:, None]
        history_offsets = np.r_[0, lengths.cumsum()]
        indices = history_offsets[:-1, None] + positions
        def padded(column, fill):
            values = _flat(batch.features, column).to_numpy()
            result = np.full(mask.shape, fill, dtype=values.dtype)
            result[mask] = values[indices[mask]]
            return result
        vectors, present = batch.embeddings.lookup(padded('history__item_id__embedding_idx', -1))
        arrays.update(h_vectors=normalize(vectors), h_present=present, h_mask=mask, h_lengths=lengths,
                      h_category=category_indices(padded('history__category_id', 0), categories),
                      h_type=np.where(mask, padded('history__event_type', 0).astype(np.int64) + 129, 0),
                      h_source=np.where(mask, padded('history__event_source', 0).astype(np.int64) + 129, 0),
                      h_position=positions.copy())
    tensors = {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()}
    if pin:
        tensors = {name: value.pin_memory() for name, value in tensors.items()}
    return dict(tensors=tensors, table=table, feeds=len(counts), candidates=len(dense))


def move(packed, device):
    return {name: value.to(device, non_blocking=device == 'cuda') for name, value in packed['tensors'].items()}


def pad(values, inputs):
    shape = (len(inputs['counts']), int(inputs['counts'].max()), *values.shape[1:])
    output = values.new_zeros(shape)
    output[inputs['feed'], inputs['row']] = values
    return output
