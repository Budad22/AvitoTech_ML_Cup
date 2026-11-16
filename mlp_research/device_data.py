"""Optional resident CUDA inputs; the immutable feature cache stays on SSD."""
import numpy as np
import torch

from .cache import CachedSplit


class DeviceSplit(CachedSplit):
    def __init__(self, root, split, device='cpu', *, reserve_bytes=1536 * 1024**2):
        super().__init__(root, split)
        self.device = device
        self.resident = {}
        names = ('dense', 'category', 'target', 'history_candidate', 'history_feed', 'context')
        self.required_bytes = sum(self.arrays[name].nbytes for name in names)
        if device == 'cuda':
            available, _ = torch.cuda.mem_get_info()
            if self.required_bytes + reserve_bytes <= available:
                try:
                    for name in names:
                        # Own the host buffer: the read-only memmap is never exposed as writable tensor storage.
                        self.resident[name] = torch.from_numpy(np.array(self.arrays[name], copy=True)).to(device)
                except torch.OutOfMemoryError:
                    self.resident.clear()
                    torch.cuda.empty_cache()
        self.resident_bytes = self.required_bytes if self.resident else 0

    def close(self):
        self.resident.clear()
        super().close()

    def batch(self, feeds, *, history=True, context=False):
        if not self.resident:
            inputs = super().batch(feeds, history=history, context=context)
            return {name: value if name == 'counts' else value.to(self.device) for name, value in inputs.items()}
        feeds = np.asarray(feeds, dtype=np.int64)
        counts = self.arrays['candidate_count'][feeds].astype(np.int64)
        rows = np.concatenate([np.arange(self.arrays['offsets'][i], self.arrays['offsets'][i + 1]) for i in feeds])
        index = torch.as_tensor(rows, device=self.device)
        inputs = {name: self.resident[name].index_select(0, index) for name in ('dense', 'category', 'target')}
        inputs['target'] = inputs['target'].float()
        # Counts stay on the host for offsets and feed metadata; model features and losses are CUDA tensors.
        inputs['counts'] = torch.from_numpy(counts)
        inputs['weights'] = torch.as_tensor(np.repeat((1 / counts).astype('float32'), counts), device=self.device)
        if history:
            repeated = torch.as_tensor(np.repeat(feeds, counts), device=self.device)
            inputs['history'] = torch.cat((self.resident['history_candidate'].index_select(0, index),
                                           self.resident['history_feed'].index_select(0, repeated)), dim=1)
        if context:
            inputs['context'] = self.resident['context'].index_select(0, index)
        return inputs
