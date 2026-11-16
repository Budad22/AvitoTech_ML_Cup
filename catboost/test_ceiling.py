import unittest
import pyarrow as pa

from ceiling import BASE_FLAGS, changes, attach
from extended_features import LONG_FEATURES, PROJECTED_FEATURES
from snowball import features_for


class SearchContracts(unittest.TestCase):
    def test_changes_are_isolated_and_do_not_mutate_control(self):
        config={"depth":6,"learning_rate":.1,"iterations":200,"random_seed":42}
        flags=dict(BASE_FLAGS)
        out,new=changes(config,flags,"long_history")
        self.assertEqual(out,config)
        self.assertFalse(flags['long_history'])
        self.assertTrue(new['long_history'])
        self.assertEqual(features_for(new),features_for(flags)+LONG_FEATURES)
        out,new=changes(config,flags,"projected_products")
        self.assertEqual(features_for(new),features_for(flags)+PROJECTED_FEATURES)
        for kind,key,value in [('depth8','depth',8),('depth4','depth',4),('l2','l2_leaf_reg',10)]:
            out,new=changes(config,flags,kind)
            self.assertEqual(out,{**config,key:value})
            self.assertEqual(new,flags)
        self.assertEqual(config['depth'],6)
        self.assertEqual(flags,BASE_FLAGS)

    def test_horizontal_cache_attachment_requires_exact_keys_and_order(self):
        base=pa.table({'feed_id':[1,1],'item_id':[10,20]})
        extra=base.append_column('feature',pa.array([.1,.2]))
        self.assertEqual(attach(base,extra,['feature'])['feature'].to_pylist(),[.1,.2])
        with self.assertRaises(ValueError):attach(base,extra.take([1,0]),['feature'])


if __name__=='__main__':unittest.main()
