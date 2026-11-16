from types import SimpleNamespace
import unittest
import numpy as np
import pyarrow as pa

from history_neighbors import add_history_neighbors, NEIGHBOR_FEATURES


class Vectors:
    def lookup(self, indices):
        matrix = np.zeros((4, 128), dtype=np.float32)
        matrix[0, 0], matrix[1, 1], matrix[2, 0] = 127, 2, -127
        values = np.zeros((len(indices), 128), dtype=np.float32)
        present = indices >= 0
        values[present] = matrix[indices[present]]
        return values, present


class NearestHistoryContracts(unittest.TestCase):
    def test_negative_values_missing_and_small_histories(self):
        batch = SimpleNamespace(features=pa.table({"history__item_id__embedding_idx":
            [[0], [0, 0, 1, -1, 3], [2, -1], []]}).slice(1), embeddings=Vectors(),
            candidate_offsets=np.array([0, 3, 5, 6]))
        candidates = pa.table({"item_id__embedding_idx": [0, 1, -1, 0, 3, 0]})
        result = add_history_neighbors(batch, candidates)
        np.testing.assert_allclose(result[NEIGHBOR_FEATURES[0]], [1, 1, 0, -1, 0, 0])
        np.testing.assert_allclose(result[NEIGHBOR_FEATURES[1]], [2/3, 1/3, 0, -1, 0, 0])
        np.testing.assert_array_equal(result[NEIGHBOR_FEATURES[2]], [1, 1, 0, 1, 0, 0])
        order = [2, 0, 1, 4, 3, 5]
        self.assertTrue(result.take(order).equals(add_history_neighbors(batch, candidates.take(order))))

    def test_cutoff_is_first_64_events_not_first_64_valid_vectors(self):
        batch = SimpleNamespace(features=pa.table({"history__item_id__embedding_idx": [[-1]*64+[0]]}),
                                embeddings=Vectors(), candidate_offsets=np.array([0, 1]))
        out = add_history_neighbors(batch, pa.table({"item_id__embedding_idx": [0]}))
        for name in NEIGHBOR_FEATURES:
            self.assertEqual(out[name][0].as_py(), 0.)


if __name__ == "__main__":
    unittest.main()
