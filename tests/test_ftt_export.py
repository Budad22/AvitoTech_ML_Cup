import tempfile
from pathlib import Path
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import test_data as fixtures
from recsys_data import PreparedSplit,prepare_dataset
from recsys_data.core import FeedBatch
from ftt.data import pack
from ftt.inference import pack_features,model_inputs
from ftt.export import SCHEMA,pair_keys,validate


class ExportContracts(unittest.TestCase):
    def test_label_free_features_match_frozen_training_pipeline(self):
        fixture=fixtures.DataContracts();fixture.setUp()
        dataset=None
        try:
            prepared=fixture.root/'prepared'
            prepare_dataset(fixture.source,prepared,splits=('train',))
            dataset=PreparedSplit(prepared,'train')
            for batch in dataset.iter_batches(3,include_labels=True):
                trained=pack(batch)
                unlabelled=FeedBatch(batch.features,None,None,batch.embeddings)
                inferred=pack_features(unlabelled)
                self.assertNotIn('target',inferred)
                for key,value in inferred.items():
                    np.testing.assert_array_equal(value,trained[key])
                inputs=model_inputs(unlabelled,'cpu')
                self.assertEqual(inputs['counts'],[5,5,5])
                self.assertEqual(tuple(inputs['user'].shape),(15,128))
        finally:
            if dataset is not None and dataset.embeddings._vectors is not None:
                dataset.embeddings._vectors._mmap.close()
            fixture.tearDown()

    def test_predictions_cover_pairs_with_raw_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'predictions.parquet'
            table=pa.table(dict(feed_id=[2,1],item_id=[4,3],score=[-2.,7.]),schema=SCHEMA)
            pq.write_table(table,path)
            expected=pair_keys([1,2],[3,4])
            self.assertEqual(validate(path,expected)['rows'],2)
            for bad in (table.set_column(2,'score',pa.array([float('nan'),1.],type=pa.float32())),
                        pa.concat_tables([table,table]),table.slice(0,1)):
                pq.write_table(bad,path)
                with self.assertRaises(ValueError):validate(path,expected)
