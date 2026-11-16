from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa

from history_signals import add_history_centroid, add_history_event_types, CENTROID_FEATURES, EVENT_FEATURES


class Vectors:
    def lookup(self, indices):
        matrix = np.zeros((4, 128), dtype=np.float32)
        matrix[0, 0], matrix[1, 1], matrix[2, 0] = 127., 2., -127.
        present = indices >= 0
        result = np.zeros((len(indices), 128), dtype=np.float32)
        result[present] = matrix[indices[present]]
        return result, present


class HistorySignals(unittest.TestCase):
    def batch(self):
        features = pa.table({
            "history__item_id__embedding_idx": [[0], [0, 0, 1, -1, 3], [], [0, 2]],
            "history__category_id": [[9], [2, 2, 3, 0, 3], [], [0, 2]],
            "history__event_type": [[9], [1, 1, 1, 0, 2], [], [0, 0]],
        }).slice(1)
        return SimpleNamespace(features=features, embeddings=Vectors(),
                               candidate_counts=np.array([4, 2, 2]),
                               candidate_offsets=np.array([0, 4, 6, 8]))

    def candidates(self):
        return pa.table({"item_id__embedding_idx": [0, 1, -1, 3, 0, 1, 0, 1],
                         "category_id": [2, 3, 0, 4, 2, 3, 0, 2]})

    def test_centroid_normalizes_before_averaging_keeps_repeats_and_masks(self):
        out = add_history_centroid(self.batch(), self.candidates())
        np.testing.assert_allclose(out[CENTROID_FEATURES[0]],
                                   [2/np.sqrt(5), 1/np.sqrt(5), 0, 0, 0, 0, 0, 0], atol=1e-6)
        np.testing.assert_array_equal(out[CENTROID_FEATURES[1]], [1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(out[CENTROID_FEATURES[2]], [3, 3, 3, 3, 0, 0, 2, 2])

    def test_event_types_are_nominal_conditional_shares_with_counts(self):
        out = add_history_event_types(self.batch(), self.candidates())
        np.testing.assert_allclose(out[EVENT_FEATURES[1]], [2/3, 1/3, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(out[EVENT_FEATURES[12]], [3, 3, 3, 3, 0, 0, 0, 0])
        np.testing.assert_allclose(out[EVENT_FEATURES[0]], [0, 0, 1, 0, 0, 0, .5, .5])
        np.testing.assert_array_equal(out[EVENT_FEATURES[10]], np.zeros(8))

    def test_cutoff_is_by_event_order_and_candidates_are_permutation_invariant(self):
        batch = self.batch()
        for transform in (add_history_centroid, add_history_event_types):
            order = [2, 0, 3, 1, 5, 4, 7, 6]
            self.assertTrue(transform(batch, self.candidates()).take(order).equals(
                transform(batch, self.candidates().take(order))))
        single = SimpleNamespace(features=pa.table({
            "history__item_id__embedding_idx": [[0]*64 + [1]],
            "history__category_id": [[2]*64 + [3]],
            "history__event_type": [[1]*64 + [2]]}), embeddings=Vectors(),
            candidate_counts=np.array([1]), candidate_offsets=np.array([0, 1]))
        candidate = self.candidates().slice(0, 1)
        out = add_history_centroid(single, candidate)
        self.assertEqual(out[CENTROID_FEATURES[0]][0].as_py(), 1.)
        self.assertEqual(out[CENTROID_FEATURES[2]][0].as_py(), 64.)
        out = add_history_event_types(single, candidate)
        self.assertEqual(out[EVENT_FEATURES[1]][0].as_py(), 1.)
        self.assertEqual(out[EVENT_FEATURES[13]][0].as_py(), 0.)


if __name__ == "__main__":
    unittest.main()
