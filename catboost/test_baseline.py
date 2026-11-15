import tempfile
from pathlib import Path
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from baseline import KEY_SCHEMA, prediction_table, ranking_metrics, validate_submission


class BaselineContracts(unittest.TestCase):
    def test_official_ties_empty_groups_and_equal_feed_weight(self):
        result = ranking_metrics([1, 1, 2, 2, 2, 2, 3, 3], [1, 0, 1, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 1, 2, 3, 0, 1])
        expected = (1 / np.log2(3) + 1 / np.log2(5) + 1) / 3
        self.assertAlmostEqual(result["ndcg"], expected)
        self.assertTrue(result["independent_catboost_check"])

    def test_full_metric_is_not_cutoff_30(self):
        result = ranking_metrics([1] * 50, [1] + [0] * 49, np.arange(50))
        self.assertAlmostEqual(result["ndcg"], 1 / np.log2(51))
        self.assertEqual(result["ndcg_at_10"], 0.)
        self.assertEqual(result["ndcg_at_30"], 0.)
        with self.assertRaises(ValueError):
            ranking_metrics([1, 2, 1], [1, 1, 0], [0, 1, 2])

    def test_submission_requires_exact_unique_keys_and_finite_scores(self):
        keys = pa.table({"feed_id": pa.array([1, 1], type=pa.int32()),
                         "item_id": pa.array([10, 20], type=pa.int32())})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, prediction = root / "template.parquet", root / "submission.parquet"
            pq.write_table(prediction_table(keys, [0, 0]), template)
            valid = prediction_table(keys, [.2, .1]).take([1, 0])
            self.assertEqual(valid.schema, KEY_SCHEMA)
            pq.write_table(valid, prediction)
            self.assertTrue(validate_submission(prediction, template)["exact_template_coverage"])
            duplicate = pa.concat_tables([valid.slice(0, 1), valid.slice(0, 1)])
            pq.write_table(duplicate, prediction)
            with self.assertRaises(ValueError):
                validate_submission(prediction, template)
            with self.assertRaises(ValueError):
                prediction_table(keys, [0., float("nan")])


if __name__ == "__main__":
    unittest.main()
