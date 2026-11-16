import unittest
import numpy as np
import pyarrow as pa
from blend import align,normalize,mixture,per_feed


class BlendContracts(unittest.TestCase):
    def table(self):
        return pa.table({'feed_id':pa.array([1,1,2],type=pa.int32()),
                         'item_id':pa.array([10,20,30],type=pa.int32()),
                         'score':pa.array([-.3,2.,1.],type=pa.float32())})

    def test_key_alignment_and_duplicate_extra_and_nonfinite_rejection(self):
        truth=self.table().select(['feed_id','item_id'])
        np.testing.assert_array_equal(align(truth,self.table().take([2,0,1])),self.table()['score'].to_numpy())
        with self.assertRaises(ValueError):align(truth,self.table().take([0,0,2]))
        bad=self.table().set_column(1,'item_id',pa.array([10,20,31],type=pa.int32()))
        with self.assertRaises(ValueError):align(truth,bad)
        bad=self.table().set_column(2,'score',pa.array([0,float('nan'),1],type=pa.float32()))
        with self.assertRaises(ValueError):align(truth,bad)

    def test_rank_ties_constant_feeds_and_permutation(self):
        scores=np.array([1,1,3,0,7,7,7],dtype=np.float32);counts=np.array([4,3])
        np.testing.assert_array_equal(normalize(scores,counts,'rank'),[.5,.5,.875,.125,.5,.5,.5])
        for method in ['rank','zscore']:
            order=[3,1,0,2,6,4,5]
            np.testing.assert_array_equal(normalize(scores,counts,method)[order],normalize(scores[order],counts,method))
        np.testing.assert_array_equal(normalize(scores,counts,'zscore')[4:],np.zeros(3))

    def test_endpoints_preserve_raw_float32_and_rounding_before_tie_metric(self):
        cat=np.array([.3,.2,.1],dtype=np.float32);mlp=-cat
        normalized_cat=normalize(cat,[3],'rank');normalized_mlp=normalize(mlp,[3],'rank')
        np.testing.assert_array_equal(mixture(cat,mlp,normalized_cat,normalized_mlp,0),mlp)
        np.testing.assert_array_equal(mixture(cat,mlp,normalized_cat,normalized_mlp,1),cat)
        blend=mixture(cat,mlp,normalized_cat,normalized_mlp,.5)
        np.testing.assert_array_equal(blend,np.full(3,.5,dtype=np.float32))
        truth=pa.table({'target':[1,0,0]})
        self.assertAlmostEqual(per_feed(truth,blend,np.array([3]))[0],.5)


if __name__=='__main__':unittest.main()
