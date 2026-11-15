import unittest
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

from history_categories import add_history_categories, feature_names


class HistoryCategoryContracts(unittest.TestCase):
    def batch(self):
        # Slice to exercise nonzero Arrow offsets. Repeats and category 0 remain events.
        features = pa.table({"history__category_id": [[9], [2, 2, 0, 3], [], [3]],
                             "history_is_empty": [False, False, True, False]}).slice(1)
        return SimpleNamespace(features=features, candidate_counts=np.array([3, 2, 2]),
                               candidate_offsets=np.array([0, 3, 5, 7]))

    def test_recency_denominator_empty_history_and_repeated_events(self):
        candidates = pa.table({"category_id": [2, 0, 3, 2, 3, 3, 2]})
        result = add_history_categories(self.batch(), candidates, [2, 16, 64, 256])
        np.testing.assert_allclose(result["history_category_share_2"].to_numpy(), [1, 0, 0, 0, 0, 1, 0])
        for window in (16, 64, 256):
            np.testing.assert_allclose(result[f"history_category_share_{window}"].to_numpy(),
                                       [.5, .25, .25, 0, 0, 1, 0])
        np.testing.assert_array_equal(result["history_is_empty"].to_numpy(), [0, 0, 0, 1, 1, 0, 0])
        self.assertEqual(result["history_category_share_16"].type, pa.float32())

    def test_candidate_permutation_and_no_auxiliary_inputs(self):
        candidates = pa.table({"category_id": [2, 0, 3, 2, 3, 3, 2]})
        order = [2, 0, 1, 4, 3, 6, 5]
        expected = add_history_categories(self.batch(), candidates, [16]).take(order)
        actual = add_history_categories(self.batch(), candidates.take(order), [16])
        self.assertTrue(expected.equals(actual))
        self.assertEqual(actual.column_names, ["category_id", *feature_names([16])])
        self.assertIs(add_history_categories(self.batch(), candidates, []), candidates)


if __name__ == "__main__":
    unittest.main()
