"""Behavioral contracts for the ranker, metric and checkpoint."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mlp.metrics import ndcg
from mlp.model import Ranker, features
from mlp.training import save_checkpoint
from recsys_data import PreparedSplit, prepare_dataset
import test_data


class MLPContracts(unittest.TestCase):
    def test_feature_optimization_matches_original_values_exactly(self):
        fixture = test_data.DataContracts()
        fixture.setUp()
        batch = None
        try:
            rng = np.random.default_rng(19)
            vectors = rng.integers(-128, 128, (3, 128), dtype=np.int8)
            vectors[2] = 0
            vectors.tofile(fixture.source / 'train/item_embeddings/data.memmap')
            users = rng.integers(-128, 128, (3, 128), dtype=np.int8)
            users[1] = 0
            raw = fixture.table.set_column(fixture.table.schema.get_field_index('user_embedding'),
                                           'user_embedding', pa.array(users.tolist(), type=pa.list_(pa.int8())))
            pq.write_table(raw, fixture.source / 'train/impressions/part-00000.parquet')
            output = fixture.root / 'prepared'
            prepare_dataset(fixture.source, output, ('train',))
            batch = next(PreparedSplit(output, 'train').iter_batches(include_labels=True))
            dense, category, target, table = features(batch, {2: 1, 5: 2})
            def original_normalize(values):
                return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
            user = np.repeat(original_normalize(batch.user_vectors()), batch.candidate_counts, axis=0)
            item, present = batch.candidate_vectors()
            item = original_normalize(item)
            numeric = np.column_stack([table[f'f{i}'].to_numpy() for i in range(1, 6)])
            expected = np.column_stack((numeric, user, item, user * item, abs(user - item),
                                        ~present, np.linalg.norm(user, axis=1) == 0)).astype(np.float32)
            np.testing.assert_array_equal(dense, expected)
            self.assertEqual(dense.tobytes(), expected.tobytes())
            np.testing.assert_array_equal(category, [0, 1, 0, 0, 2] * 3)
            np.testing.assert_array_equal(target, table['target'].to_numpy())
            self.assertTrue(dense.flags.c_contiguous)
        finally:
            batch = None
            fixture.tearDown()

    def test_metric_order_and_pessimistic_ties(self):
        self.assertEqual(ndcg([1, 0], [2., 1.]), 1.)
        self.assertAlmostEqual(ndcg([1, 0], [1., 1.]), 1 / np.log2(3))
        self.assertEqual(ndcg([0, 0], [0., 0.]), 1.)
        self.assertLess(ndcg([1, 0, 0], [0., 1., 2.]), 1.)

    def test_missing_vectors_and_unknown_categories(self):
        fixture = test_data.DataContracts()
        fixture.setUp()
        try:
            output = fixture.root / 'prepared'
            prepare_dataset(fixture.source, output, ('train',))
            batch = next(PreparedSplit(output, 'train').iter_batches(include_labels=True))
            dense, category, target, _ = features(batch, {})
            self.assertTrue(np.isfinite(dense).all())
            self.assertEqual(dense.shape, (15, 519))
            self.assertTrue((category == 0).all())
            np.testing.assert_array_equal(dense[1, 133:261], 0)
            self.assertEqual(dense[1, -2], 1)
            np.testing.assert_array_equal(target[:5], [0, 1, 0, 0, 0])
        finally:
            del batch
            fixture.tearDown()

    def test_learning_permutation_and_checkpoint(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = Ranker(2)
        x, c = torch.randn(24, 519), torch.zeros(24, dtype=torch.long)
        y = (x[:, 0] > 0).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model.eval()
        initial = torch.nn.functional.binary_cross_entropy_with_logits(model(x, c), y).item()
        for _ in range(25):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, c), y)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * .5)
        order = torch.randperm(len(x))
        torch.testing.assert_close(model(x[order], c[order]), model(x, c)[order])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'last.pt'
            save_checkpoint(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
            state = torch.load(path, weights_only=False)
            restored = Ranker(2).eval()
            restored.load_state_dict(state['model'])
            torch.testing.assert_close(restored(x, c), model(x, c))
