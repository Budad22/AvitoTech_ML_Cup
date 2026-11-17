"""Contracts for official padding, input isolation and exact shuffled-loader resume."""
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from transformer_research.compact import shuffled_loader
from transformer_research.core import CachedFeeds, fit, ndcg, official_parts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'official/neural_baseline.py'
if not SOURCE.exists():
    SOURCE = ROOT / 'artifacts/official-baseline-review/neural_baseline.py'


def synthetic(root):
    folder = root / 'train'; folder.mkdir()
    counts = np.array([5, 7, 6, 8], dtype=np.int64)
    offsets = np.r_[0, counts.cumsum()]
    rng = np.random.default_rng(123)
    dense = rng.normal(size=(int(counts.sum()), 519)).astype(np.float32)
    category = rng.integers(1, 4, size=len(dense), dtype=np.int64)
    target = np.zeros(len(dense), dtype=np.uint8)
    for index, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
        user = rng.normal(size=128).astype(np.float32)
        user /= np.linalg.norm(user)
        if index == 1:
            user[:] = 0
        dense[left:right, 5:133] = user
        items = dense[left:right, 133:261]
        items /= np.maximum(np.linalg.norm(items, axis=1, keepdims=True), 1e-12)
        dense[left:right, 517] = 0
        dense[left, 133:261] = 0
        dense[left, 517] = 1
        target[left + 1] = 1
    arrays = dict(dense=dense, category=category, target=target, offsets=offsets,
                  feed_id=np.arange(4, dtype=np.int32), item_id=np.arange(len(dense), dtype=np.int32))
    for name, values in arrays.items():
        np.save(folder / (name + '.npy'), values)
    return CachedFeeds(root, 'train')


class MetricContracts(unittest.TestCase):
    def test_full_ndcg_ties_and_no_positives(self):
        self.assertAlmostEqual(ndcg(np.array([1, 0]), np.array([0., 0.])), 1 / np.log2(3))
        self.assertEqual(ndcg(np.zeros(2), np.zeros(2)), 1.)
        self.assertAlmostEqual(ndcg(np.r_[1, np.zeros(49)], np.arange(50)), 1 / np.log2(51))


@unittest.skipIf(sys.version_info < (3, 10), 'Pinned official collate requires Python >=3.10')
class TransformerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ranker, cls.collate, cls.config = official_parts(SOURCE)
        cls.collate = staticmethod(cls.collate)

    def test_official_input_schema_padding_and_permutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = synthetic(Path(temporary))
            try:
                batch = self.collate([data[0], data[1]])
                features, category, users, padding, targets = batch
                self.assertEqual(features.shape, (2, 7, 134))
                self.assertEqual(padding.sum().item(), 2)
                self.assertEqual(features[0, 0, 133].item(), 1)
                torch.testing.assert_close(features[0, 0, 5:133], torch.zeros(128))
                torch.testing.assert_close(users[1], torch.zeros(128))
                self.assertEqual(targets[0, 1].item(), 1)
                model = self.ranker(3).eval()
                self.assertEqual(len(model.encoder.layers), 2)
                self.assertEqual(model.encoder.layers[0].self_attn.num_heads, 4)
                with torch.no_grad():
                    expected = model(features, category, users, padding)
                    order = torch.arange(6, -1, -1)
                    actual = model(features[:, order], category[:, order], users, padding[:, order])
                    torch.testing.assert_close(actual, expected[:, order], atol=2e-5, rtol=2e-5)
            finally:
                data.close()

    def test_resume_preserves_native_shuffle_optimizer_rng_and_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = synthetic(root)
            try:
                def invoke(folder, emit):
                    torch.manual_seed(42)
                    model = self.ranker(3)
                    loader = (shuffled_loader(data, 2, self.collate) if folder.name == 'resumed' else
                              DataLoader(data, batch_size=2, shuffle=True, collate_fn=self.collate))
                    validation = DataLoader(data, batch_size=2, collate_fn=self.collate)
                    return fit(model, loader, validation, folder, {'case': 'synthetic'}, 'cpu', 2,
                               emit, time.monotonic() + 60)
                full, resumed = root / 'full', root / 'resumed'
                invoke(full, lambda *a, **k: None)
                def interrupt(event, **fields):
                    if event == 'batch_completed' and fields['step'] == 1:
                        (resumed / 'STOP_AFTER_CHECKPOINT').touch()
                with self.assertRaises(InterruptedError):
                    invoke(resumed, interrupt)
                (resumed / 'STOP_AFTER_CHECKPOINT').unlink()
                invoke(resumed, lambda *a, **k: None)
                left = torch.load(full / 'last.pt', weights_only=False)
                right = torch.load(resumed / 'last.pt', weights_only=False)
                self.assertEqual(left['step'], right['step'])
                self.assertEqual(left['history'], right['history'])
                self.assertTrue(torch.equal(left['rng'], right['rng']))
                for name, value in left['model'].items():
                    self.assertTrue(torch.equal(value, right['model'][name]), name)
                for index, values in left['optimizer']['state'].items():
                    for name, value in values.items():
                        self.assertTrue(torch.equal(value, right['optimizer']['state'][index][name]), name)
            finally:
                data.close()

    def test_cuda_backward_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            data = synthetic(Path(temporary))
            try:
                features, category, users, padding, target = (x.cuda() for x in self.collate([data[0], data[1]]))
                model = self.ranker(3).cuda().train()
                loss = F.binary_cross_entropy_with_logits(model(features, category, users, padding)[~padding], target[~padding])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            finally:
                data.close()


if __name__ == '__main__':
    unittest.main()
