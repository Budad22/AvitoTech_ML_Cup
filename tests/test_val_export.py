"""A blending export must align exact val keys and preserve logits."""
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import test_mlp_trials
from scripts.export_mlp_val import validate_val


class ValExportTests(unittest.TestCase):
    def test_shuffled_raw_logits_and_exact_val_alignment(self):
        fixture = test_mlp_trials.TrialContracts()
        fixture.setUp()
        try:
            feeds = np.repeat(np.array([100, 101, 102], dtype=np.int32), 5)
            items = np.tile(np.array([101, 102, 103, 104, 105], dtype=np.int32), 3)
            score = np.tile(np.array([-5., 2., -.5, .001, 0.], dtype=np.float32), 3)
            order = np.random.default_rng(42).permutation(15)
            table = pa.table({'feed_id': feeds[order], 'item_id': items[order], 'score': score[order]})
            path = fixture.root / 'val.parquet'
            pq.write_table(table, path)
            result = validate_val(path, fixture.data, expected_rows=15, expected_feeds=3)
            self.assertEqual(result['full_val_ndcg'], 1.)
            np.testing.assert_array_equal(pq.read_table(path)['score'].to_numpy(), score[order])
            for values in ([100] * 15, list(feeds[:-1]) + [999]):
                wrong = table.set_column(0, 'feed_id', pa.array(values, type=pa.int32()))
                pq.write_table(wrong, path)
                with self.assertRaises(ValueError):
                    validate_val(path, fixture.data, expected_rows=15, expected_feeds=3)
            pq.write_table(table.set_column(2, 'score', pa.array([float('nan')] * 15, type=pa.float32())), path)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                validate_val(path, fixture.data, expected_rows=15, expected_feeds=3)
        finally:
            fixture.tearDown()
