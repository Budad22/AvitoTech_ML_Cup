"""Immutable sequence supplements aligned to the existing research cache."""
from pathlib import Path
import json

import numpy as np
import torch

from recsys_data import PreparedSplit
from mlp.training import select
from mlp_trials.data import pack
from mlp_trials.state import write_json
from mlp_research.cache import sha256
from mlp_research.device_data import DeviceSplit


def build_sequences(data, cache, output, emit):
    data, cache, output = Path(data), Path(cache), Path(output)
    info = json.loads((cache / 'metadata.json').read_text())
    signature = dict(manifest=sha256(data / 'manifest.json'), cache=sha256(cache / 'metadata.json'),
                     code=sha256(__file__), window=256)
    if (output / 'metadata.json').exists():
        if json.loads((output / 'metadata.json').read_text()) != signature:
            raise ValueError('Sequence cache identity changed')
        return
    categories = {int(k): v for k, v in info['categories'].items()}
    for split in ('train', 'val'):
        ids = np.load(cache / split / 'feed_id.npy', mmap_mode='r')
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)
        shapes = dict(h_vectors=((len(ids), 256, 128), 'float16'),
                      **{k: ((len(ids), 256), 'bool' if k in ('h_mask', 'h_present') else 'int64')
                         for k in ('h_mask', 'h_present', 'h_category', 'h_type', 'h_source')})
        arrays = {k: np.lib.format.open_memmap(folder / (k + '.npy'), mode='w+', dtype=d, shape=s)
                  for k, (s, d) in shapes.items()}
        position = 0
        dataset = PreparedSplit(data, split)
        for batch in dataset.iter_batches(32, include_labels=True):
            if split == 'val':
                keep = np.flatnonzero(batch.features['timestamp'].to_numpy() < info['boundary'])
                if not len(keep):
                    continue
                batch = select(batch, keep)
            packed = pack(batch, categories, kind='history_attention', history_window=256)
            n = packed['feeds']
            if not np.array_equal(ids[position:position + n], batch.features['feed_id'].to_numpy()):
                raise ValueError('Sequence/candidate feed keys differ')
            for key, array in arrays.items():
                array[position:position + n] = packed['tensors'][key].numpy()
            position += n
            emit('sequence_cache_progress', split=split, current=position, total=len(ids))
        if position != len(ids):
            raise ValueError('Incomplete sequence coverage')
        for array in arrays.values():
            array.flush()
        del arrays
    write_json(output / 'metadata.json', signature)


class SequenceSplit(DeviceSplit):
    def __init__(self, cache, split, device='cpu'):
        super().__init__(cache, split, device, reserve_bytes=3 * 1024**3)
        self.sequence = {p.stem: np.load(p, mmap_mode='r')
                         for p in (Path(cache).parent / 'sequences' / split).glob('*.npy')}
        if not self.sequence:
            raise ValueError('Sequence cache is missing')
        self.need_sequence = True

    def batch(self, feeds, *, history=True, context=False):
        result = super().batch(feeds, history=history, context=context)
        if self.need_sequence:
            for name, array in self.sequence.items():
                result[name] = torch.from_numpy(np.array(array[feeds], copy=True)).to(self.device)
        return result

    def close(self):
        for array in self.sequence.values():
            array._mmap.close()
        self.sequence.clear()
        super().close()
