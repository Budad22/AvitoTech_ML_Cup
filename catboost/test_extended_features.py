from types import SimpleNamespace
import unittest
import numpy as np
import pyarrow as pa

from extended_features import add_extended_features, LONG_FEATURES, PROJECTED_FEATURES, PROJECTION


class Vectors:
    def lookup(self, indices):
        matrix = np.zeros((4,128),dtype=np.float32)
        matrix[0,0],matrix[1,1],matrix[2,0]=127,2,-127
        values=np.zeros((len(indices),128),dtype=np.float32)
        present=indices>=0
        values[present]=matrix[indices[present]]
        return values,present


class ExtendedContracts(unittest.TestCase):
    def test_long_history_includes_older_events_and_handles_negative_and_missing(self):
        users=np.zeros((3,128),dtype=np.float32);users[0,0]=127
        batch=SimpleNamespace(features=pa.table({'history__item_id__embedding_idx':
            [[2]*64+[0],[-1,2],[]]}),embeddings=Vectors(),candidate_offsets=np.array([0,3,5,6]),
            candidate_counts=np.array([3,2,1]),user_vectors=lambda:users)
        candidates=pa.table({'item_id__embedding_idx':[0,1,-1,0,3,0]})
        result=add_extended_features(batch,candidates)
        np.testing.assert_allclose(result[LONG_FEATURES[0]],[1,0,0,-1,0,0])
        np.testing.assert_allclose(result[LONG_FEATURES[1]],[-1/3,0,0,-1,0,0])
        np.testing.assert_array_equal(result[LONG_FEATURES[2]],[1,1,0,1,0,0])
        products=np.column_stack([result[name].to_numpy() for name in PROJECTED_FEATURES])
        np.testing.assert_allclose(products[0],PROJECTION[0]**2,atol=1e-8)
        np.testing.assert_allclose(products[1],PROJECTION[0]*PROJECTION[1],atol=1e-8)
        np.testing.assert_array_equal(products[2:],np.zeros((4,32)))
        order=[2,0,1,4,3,5]
        self.assertTrue(result.take(order).equals(add_extended_features(batch,candidates.take(order))))

    def test_projection_is_orthonormal_and_fixed(self):
        np.testing.assert_allclose(np.einsum('ik,ij->kj',PROJECTION,PROJECTION),np.eye(32),atol=3e-7)


if __name__ == '__main__':
    unittest.main()
