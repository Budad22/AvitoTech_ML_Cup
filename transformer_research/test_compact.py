"""Compact conversion preserves native input semantics and rejects corrupt feeds."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformer_research.build_compact import count, convert
from transformer_research.compact import CompactFeeds, shuffled_loader


class CompactContracts(unittest.TestCase):
    def test_roundtrip_native_features_and_reject_bad_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / 'raw'; output = root / 'compact'
            folder = raw / 'train/impressions'; folder.mkdir(parents=True); output.mkdir()
            values = dict(feed_id=[7, 8], timestamp=[10, 11], user_id=[2, 3],
                          item_id=[[10, 11], [12]], category_id=[[2, 3], [2]],
                          target=[[1, 0], [0]], item_id__embedding_idx=[[0, -1], [1]],
                          user_embedding=[[0] * 128, [1] * 128])
            values.update({f'f{i}': [[float(i), float(i + 1)], [float(i + 2)]] for i in range(1, 6)})
            path = folder / 'one.parquet'; pq.write_table(pa.table(values), path)
            counts, cats = count(raw, 'train', 20)
            self.assertEqual(counts, dict(feeds=2, rows=3)); self.assertEqual(cats, {2, 3})
            convert(raw, output, 'train', counts, {2: 1, 3: 2}, 20, 2)
            matrix = root / 'items.bin'; vectors = np.ones((2, 128), dtype=np.int8); vectors[1] = 0
            matrix.write_bytes(vectors.tobytes())
            (output / 'metadata.json').write_text(json.dumps(dict(items=dict(train=dict(
                path='../items.bin', dtype='int8', shape=[2, 128], order='C')))))
            data = CompactFeeds(output, 'train')
            try:
                features, cats, user, target = data[0]
                torch.testing.assert_close(features[:, :5], torch.tensor([[1,2,3,4,5],[2,3,4,5,6]], dtype=torch.float32))
                torch.testing.assert_close(features[0, 5:133], torch.ones(128) / np.sqrt(128))
                self.assertEqual(float(features[1, 133]), 1.)
                self.assertTrue(torch.equal(features[1, 5:133], torch.zeros(128)))
                self.assertTrue(torch.equal(user, torch.zeros(128)))
                self.assertEqual(cats.tolist(), [1, 2]); self.assertEqual(target.tolist(), [1., 0.])
                self.assertTrue(torch.equal(data[1][0][0, 5:133], torch.zeros(128)))
            finally:
                data.close()
            # Actual full train contains repeated items within a feed. Match
            # native CandidateData: retain both rows and their separate labels.
            values['item_id'] = [[10, 10], [12]]
            pq.write_table(pa.table(values), path)
            diagnostics = {}
            convert(raw, output, 'train', counts, {2: 1, 3: 2}, 20, 2, diagnostics)
            self.assertEqual(diagnostics['duplicate_candidate_rows_preserved'], 1)
            self.assertEqual(np.load(output / 'train/item_id.npy').tolist(), [10, 10, 12])
            self.assertEqual(np.load(output / 'train/target.npy').tolist(), [1, 0, 0])
            (raw / 'val/impressions').mkdir(parents=True)
            pq.write_table(pa.table(values), raw / 'val/impressions/one.parquet')
            with self.assertRaisesRegex(ValueError, 'validation'):
                convert(raw, output, 'val', counts, {2: 1, 3: 2}, 20, 2)
            values['item_id__embedding_idx'] = [[0, -2], [1]]
            pq.write_table(pa.table(values), path)
            with self.assertRaisesRegex(ValueError, 'indices'):
                convert(raw, output, 'train', counts, {2: 1, 3: 2}, 20, 2)

    def test_skip_does_not_load_skipped_feeds_and_matches_native_shuffle(self):
        from torch.utils.data import DataLoader, Dataset
        class Counting(Dataset):
            calls = 0
            def __len__(self): return 23
            def __getitem__(self, i): self.calls += 1; return i
        data = Counting()
        torch.manual_seed(42)
        native = list(DataLoader(data, batch_size=4, shuffle=True))
        torch.manual_seed(42); data.calls = 0
        loader = shuffled_loader(data, 4, None); iterator = iter(loader)
        loader.batch_sampler.skip(3)
        self.assertEqual(data.calls, 0)
        actual = list(iterator)
        for left, right in zip(actual, native[3:]):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(data.calls, 11)
