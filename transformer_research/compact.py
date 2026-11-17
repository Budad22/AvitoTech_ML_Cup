"""Disk-backed official inputs; no dense embedding expansion on disk or in RAM."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler


class CompactFeeds(Dataset):
    def __init__(self, root, split):
        if split not in ('train', 'val'):
            raise ValueError('Only train/val supported')
        root = Path(root)
        self.metadata = json.loads((root / 'metadata.json').read_text())
        self.arrays = {name: np.load(root / split / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                       for name in ('floats', 'category', 'target', 'item_index', 'item_id', 'users',
                                    'offsets', 'feed_id', 'timestamp', 'user_id')}
        spec = self.metadata['items'][split]
        self.items = np.memmap(root / spec['path'], mode='r', dtype=spec['dtype'],
                               shape=tuple(spec['shape']), order=spec['order'])
        self.rows = len(self.arrays['target'])

    def __len__(self):
        return len(self.arrays['feed_id'])

    def __getitem__(self, feed):
        left, right = self.arrays['offsets'][feed:feed + 2]
        rows = slice(left, right)
        indices = self.arrays['item_index'][rows]
        missing = indices == -1
        items = self.items[np.maximum(indices, 0)].astype(np.float32)
        items[missing] = 0
        features = torch.cat((torch.from_numpy(self.arrays['floats'][rows].copy()),
                              F.normalize(torch.from_numpy(items), dim=1),
                              torch.from_numpy(missing)[:, None]), dim=1)
        user = F.normalize(torch.from_numpy(self.arrays['users'][feed].astype(np.float32)), dim=0)
        return (features, torch.from_numpy(self.arrays['category'][rows].astype(np.int64)), user,
                torch.from_numpy(self.arrays['target'][rows].astype(np.float32)))

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()
        self.items._mmap.close()


class SkippableBatchSampler:
    """Advance native random batches without rereading already trained feeds."""
    def __init__(self, base):
        self.base = base
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.base)
        return self.iterator

    def __len__(self):
        return len(self.base)

    def skip(self, count):
        for _ in range(count):
            next(self.iterator)


def shuffled_loader(dataset, batch_size, collate):
    batches = SkippableBatchSampler(BatchSampler(RandomSampler(dataset), batch_size, drop_last=False))
    return DataLoader(dataset, batch_sampler=batches, collate_fn=collate)
