"""Read-only supplemental positions and bounded CPU prefetch for full feeds."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from recsys_data import PreparedSplit
from mlp_next.data import SequenceSplit
from mlp_next.streaming import StreamingSplit
from mlp_trials.state import write_json
from mlp_research.cache import sha256


def prepare_positions(view, cache, output):
    view, cache, output = Path(view), Path(cache), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(view=sha256(view / 'manifest.json'), cache=sha256(cache / 'metadata.json'), version=2)
    if (output / 'metadata.json').exists():
        import json
        if __import__('json').loads((output / 'metadata.json').read_text()) != identity:
            raise ValueError('Position cache identity changed')
        return
    dataset = PreparedSplit(view, 'train')
    ids = np.load(cache / 'train/feed_id.npy', mmap_mode='r')
    total = int(np.load(cache / 'train/offsets.npy', mmap_mode='r')[-1])
    array = np.lib.format.open_memmap(output / 'buckets.npy', mode='w+', dtype='int64', shape=(total,))
    row = feed = 0
    for record in dataset.info['files']:
        table = pq.read_table(view / 'train/diagnostics' / record['name'])
        if not np.array_equal(ids[feed:feed+len(table)], table['feed_id'].to_numpy()):
            raise ValueError('Position feed keys differ from training cache')
        positions = pc.list_flatten(table['pos']).to_numpy()
        array[row:row+len(positions)] = np.searchsorted([1, 5, 20, 50, 100], positions, side='right')
        row += len(positions)
        feed += len(table)
    if row != total or feed != len(ids):
        raise ValueError('Incomplete position coverage')
    array.flush()
    write_json(output / 'metadata.json', identity)


class ExtendedSplit(SequenceSplit):
    def __init__(self, cache, split, device='cpu', *, positions):
        super().__init__(cache, split, device)
        self.positions = np.load(Path(positions) / 'buckets.npy', mmap_mode='r') if split == 'train' else None

    def batch(self, feeds, **kwargs):
        values = super().batch(feeds, **kwargs)
        if self.positions is not None:
            rows = np.concatenate([np.arange(self.arrays['offsets'][i], self.arrays['offsets'][i+1]) for i in feeds])
            values['position_bucket'] = torch.from_numpy(np.array(self.positions[rows], copy=True)).to(self.device)
        return values

    def close(self):
        if self.positions is not None:
            self.positions._mmap.close()
        super().close()


class PrefetchedStream:
    """Only the producer thread touches the source reader; CUDA stays on its owner thread."""
    def __init__(self, data, categories, config, device, batch_size=64):
        self.source = StreamingSplit(data, categories, config, 'cpu')
        self.feeds, self.device, self.batch_size = self.source.feeds, device, batch_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = {}

    def batch(self, feeds, *, history=True, context=False):
        feeds = np.asarray(feeds)
        key = tuple(feeds.tolist())
        future = self.pending.pop(key, None)
        if future is None:
            future = self.executor.submit(self.source.batch, feeds, history=history, context=context)
        result = future.result()
        start = int(feeds[-1]) + 1
        for _ in range(2):
            if start >= self.feeds:
                start = 0
            following = np.arange(start, min(start + self.batch_size, self.feeds))
            next_key = tuple(following.tolist())
            if next_key not in self.pending:
                self.pending[next_key] = self.executor.submit(self.source.batch, following, history=history, context=context)
            start += len(following)
        return {k: v if k == 'counts' else v.to(self.device) for k, v in result.items()}

    def close(self):
        for future in self.pending.values():
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.source.close()
