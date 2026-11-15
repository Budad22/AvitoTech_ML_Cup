from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa

from feed_context import CONTEXT_FEATURES, FIELDS, add_feed_percentiles


class FeedContextContracts(unittest.TestCase):
    def setUp(self):
        self.batch = SimpleNamespace(candidate_offsets=np.array([0, 4, 7]))
        self.table = pa.table({name: pa.array([1, 1, 3, 0, 7, 7, 7], type=pa.float32()) for name in FIELDS})

    def test_midrank_ties_constant_group_and_no_cross_feed_influence(self):
        result = add_feed_percentiles(self.batch, self.table)
        for name in CONTEXT_FEATURES:
            np.testing.assert_array_equal(result[name].to_numpy(), [.5, .5, .875, .125, .5, .5, .5])
            self.assertEqual(result[name].type, pa.float32())
        changed = pa.table({name: pa.array([1, 1, 3, 0, -99, 0, 999], type=pa.float32()) for name in FIELDS})
        np.testing.assert_array_equal(add_feed_percentiles(self.batch, changed)['f1_feed_percentile'].to_numpy()[:4],
                                      result['f1_feed_percentile'].to_numpy()[:4])

    def test_candidate_permutation_and_no_implicit_fields(self):
        order = [3, 1, 0, 2, 6, 4, 5]
        expected = add_feed_percentiles(self.batch, self.table).take(order)
        actual = add_feed_percentiles(self.batch, self.table.take(order))
        self.assertTrue(actual.equals(expected))
        self.assertEqual(actual.column_names, FIELDS + CONTEXT_FEATURES)


if __name__ == "__main__":
    unittest.main()
