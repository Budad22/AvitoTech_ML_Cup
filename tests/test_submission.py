"""Submission contracts use tiny synthetic tables, never competition data."""
import tempfile
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.validate_submission import validate


class SubmissionTests(unittest.TestCase):
    def test_exact_coverage_is_independent_of_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = pa.table({'feed_id': pa.array([1, 1, 2], type=pa.int32()),
                              'item_id': pa.array([10, 20, 30], type=pa.int32()),
                              'score': pa.array([.1, .2, .3], type=pa.float32())})
            pq.write_table(table, root / 'prediction.parquet')
            pq.write_table(table.take([2, 0, 1]).select(['feed_id', 'item_id']), root / 'template.parquet')
            report = validate(root / 'prediction.parquet', root / 'template.parquet')
            self.assertEqual(report['rows'], 3)
            self.assertEqual(report['feeds'], 2)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                validate(root / 'prediction.parquet', root / 'template.parquet', '0' * 64)

    def test_invalid_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def table(items, scores, dtype=pa.float32()):
                return pa.table({'feed_id': pa.array([1, 1], type=pa.int32()),
                                 'item_id': pa.array(items, type=pa.int32()),
                                 'score': pa.array(scores, type=dtype)})
            pq.write_table(table([10, 20], [.1, .2]), root / 'template.parquet')
            for value, reason in ((table([10, 10], [.1, .2]), 'Duplicate'),
                                  (table([10, 30], [.1, .2]), 'pairs differ'),
                                  (table([10, 20], [.1, float('nan')]), 'Nonfinite'),
                                  (table([10, 20], [.1, None]), 'null'),
                                  (table([10, 20], [.1, .2], pa.float64()), 'schema')):
                with self.subTest(reason=reason):
                    pq.write_table(value, root / 'prediction.parquet')
                    with self.assertRaisesRegex(ValueError, reason):
                        validate(root / 'prediction.parquet', root / 'template.parquet')
