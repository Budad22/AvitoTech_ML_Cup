"""Repeated train exposures stay distinct without weakening evaluation keys."""
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

import test_data
from recsys_data import PreparedSplit, prepare_dataset
from recsys_data.core import BASE_SCHEMA, _prepare_table


class DuplicatePolicyTests(unittest.TestCase):
    def test_repeated_exposures_require_explicit_train_policy(self):
        table = test_data.feeds().slice(0, 1)
        column = table.schema.get_field_index('item_id')
        table = table.set_column(column, 'item_id', pa.array([[101, 101, 103, 104, 105]], type=pa.list_(pa.int32())))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            _prepare_table(table, 'train', 3)
        features, labels, positions = _prepare_table(table, 'train', 3, allow_train_duplicate_items=True)
        self.assertEqual(features['item_id'].to_pylist(), table['item_id'].to_pylist())
        self.assertTrue(labels['target'].equals(table['target']))
        self.assertTrue(positions['pos'].equals(table['pos']))
        validation = table.drop(['pos'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            _prepare_table(validation, 'val', 3, allow_train_duplicate_items=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            _prepare_table(validation.drop(['target']), 'test', 3, allow_train_duplicate_items=True)

    def test_roundtrip_manifest_and_projected_reader(self):
        fixture = test_data.DataContracts()
        fixture.setUp()
        try:
            table = fixture.table
            column = table.schema.get_field_index('item_id')
            table = table.set_column(column, 'item_id', pa.array([[101, 101, 103, 104, 105]] * 3, type=pa.list_(pa.int32())))
            pq.write_table(table, fixture.source / 'train/impressions/part-00000.parquet')
            output = fixture.root / 'prepared'
            manifest = prepare_dataset(fixture.source, output, ('train',), allow_train_duplicate_items=True)
            self.assertEqual(manifest['candidate_key_policy'], 'train-occurrences-v1')
            self.assertEqual(manifest['splits']['train']['duplicate_candidate_occurrences'], 3)
            dataset = PreparedSplit(output, 'train')
            batch = next(dataset.iter_batches(include_labels=True, include_diagnostics=True))
            self.assertTrue(batch.features.select(list(BASE_SCHEMA)).equals(table.select(list(BASE_SCHEMA))))
            self.assertEqual(batch.candidate_table(include_target=True)['target'].to_pylist(), [0, 1, 0, 0, 0] * 3)
            projected = next(dataset.iter_batches(feature_columns=['feed_id', 'timestamp'], include_labels=True))
            self.assertEqual(projected.features.column_names, ['feed_id', 'timestamp'])
            self.assertTrue(projected.labels.equals(batch.labels))
            pq.write_table(table, fixture.source / 'train/impressions/part-extra.parquet')
            with self.assertRaisesRegex(ValueError, 'Duplicate feed_id'):
                prepare_dataset(fixture.source, fixture.root / 'bad', ('train',), allow_train_duplicate_items=True)
            self.assertFalse((fixture.root / 'bad').exists())
        finally:
            fixture.tearDown()
