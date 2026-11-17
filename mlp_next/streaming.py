"""Bounded full-train streaming with unchanged statistics and complete feeds."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from recsys_data import PreparedSplit
from recsys_data.core import FeedBatch
from mlp_trials.data import pack, context_features
from mlp_trials.models import history_stats


def sequential_order(feeds, seed, step, batch_size):
    batches = (feeds + batch_size - 1) // batch_size
    start = (step % batches) * batch_size
    return np.arange(start, min(start + batch_size, feeds))


class StreamingSplit:
    def __init__(self, data, categories, config, device):
        self.dataset = PreparedSplit(data, 'train')
        self.files = self.dataset.info['files']
        self.offsets = np.r_[0, np.cumsum([f['feeds'] for f in self.files])]
        self.feeds = int(self.offsets[-1])
        self.categories, self.config, self.device = categories, config, device
        self.loaded = {}

    def batch(self, feeds, *, history=True, context=False):
        features, labels, positions = [], [], []
        for index in feeds:
            shard = int(np.searchsorted(self.offsets, index, side='right') - 1)
            if shard not in self.loaded:
                name = self.files[shard]['name']
                kinds = ('features', 'labels', 'diagnostics') if self.config.get('position_nuisance') else ('features', 'labels')
                self.loaded = {shard: tuple(pq.read_table(self.dataset.root / 'train' / kind / name) for kind in kinds)}
            f, y = self.loaded[shard][:2]
            row = int(index - self.offsets[shard])
            features.append(f.slice(row, 1))
            labels.append(y.slice(row, 1))
            if self.config.get('position_nuisance'):
                positions.extend(self.loaded[shard][2]['pos'][row].as_py())
        batch = FeedBatch(pa.concat_tables(features), pa.concat_tables(labels), None, self.dataset.embeddings)
        window = 256 if self.config.get('sequence') else 128
        tensors = pack(batch, self.categories, kind='history_stats', history_window=window)['tensors']
        # Statistics must use the same original window128 as the research cache.
        statistics_inputs = {k: v[:, :128] if k.startswith('h_') and v.ndim >= 2 else v
                             for k, v in tensors.items()}
        stats = history_stats(statistics_inputs)
        result = {k: tensors[k] for k in ('dense', 'category', 'target', 'counts', 'weights')}
        result['history'] = stats
        if self.config.get('position_nuisance'):
            result['position_bucket'] = torch.from_numpy(np.searchsorted([1, 5, 20, 50, 100], positions, side='right').astype('int64'))
        if context:
            result['context'] = torch.from_numpy(context_features(tensors['dense'].numpy(),
                                                    tensors['category'].numpy(), tensors['counts'].numpy()))
        if self.config.get('sequence'):
            result.update({k: tensors[k] for k in ('h_vectors', 'h_mask', 'h_present', 'h_category', 'h_type', 'h_source')})
            result['h_vectors'] = result['h_vectors'].half()
        return {k: v if k == 'counts' else v.to(self.device) for k, v in result.items()}

    def close(self):
        self.loaded.clear()
        vectors = self.dataset.embeddings._vectors
        if isinstance(vectors, np.memmap):
            vectors._mmap.close()
