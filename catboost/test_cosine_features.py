from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa

from cosine_features import add_user_item_cosine


class Vectors:
    def __init__(self):
        self.items = np.zeros((4, 128), dtype=np.int8)
        self.items[0] = 127
        self.items[1] = -127
        self.items[2, :64], self.items[2, 64:] = 127, -127

    def lookup(self, indices):
        result = np.zeros((len(indices), 128), dtype=np.float32)
        valid = indices >= 0
        result[valid] = self.items[indices[valid]].astype(np.float32)
        return result, valid


class CosineContracts(unittest.TestCase):
    def batch(self):
        users = np.zeros((2, 128), dtype=np.int8)
        users[0] = 127
        return SimpleNamespace(user_vectors=lambda: users, embeddings=Vectors(),
                               candidate_counts=np.array([5, 2]))

    def test_int8_arithmetic_orthogonal_missing_and_zero_vectors(self):
        table = pa.table({"item_id__embedding_idx": [0, 1, 2, -1, 3, 0, -1]})
        result = add_user_item_cosine(self.batch(), table)
        np.testing.assert_allclose(result["user_item_cosine"].to_numpy(), [1, -1, 0, 0, 0, 0, 0], atol=1e-6)
        np.testing.assert_array_equal(result["user_item_cosine_valid"].to_numpy(), [1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(result["user_item_cosine"].type, pa.float32())

    def test_lookup_follows_permuted_candidate_indices(self):
        table = pa.table({"item_id__embedding_idx": [0, 1, 2, -1, 3, 0, -1]})
        order = [4, 2, 0, 3, 1, 6, 5]
        expected = add_user_item_cosine(self.batch(), table).take(order)
        actual = add_user_item_cosine(self.batch(), table.take(order))
        self.assertTrue(actual.equals(expected))


if __name__ == "__main__":
    unittest.main()
