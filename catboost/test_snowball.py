import unittest

import numpy as np
import pyarrow as pa

from snowball import features_for, temporal_partition
from history_signals import CENTROID_FEATURES, EVENT_FEATURES


class SnowballContracts(unittest.TestCase):
    def test_temporal_partition_keeps_equal_timestamps_together(self):
        metadata = pa.table({"timestamp": [9, 1, 8, 2, 8, 3, 8, 4, 7, 5]})
        mask, report = temporal_partition(metadata)
        timestamps = metadata["timestamp"].to_numpy()
        self.assertLess(timestamps[mask].max(), timestamps[~mask].min())
        for timestamp in np.unique(timestamps):
            self.assertEqual(len(np.unique(mask[timestamps == timestamp])), 1)
        self.assertEqual(report["selection_feeds"] + report["control_feeds"], 10)
        self.assertTrue(mask.any() and (~mask).any())

    def test_rejected_groups_do_not_enter_model_features(self):
        baseline = features_for({"centroid": False, "event_types": False})
        centroid = features_for({"centroid": True, "event_types": False})
        types_only = features_for({"centroid": False, "event_types": True})
        combined = features_for({"centroid": True, "event_types": True})
        self.assertEqual(centroid, baseline + CENTROID_FEATURES)
        self.assertEqual(types_only, baseline + EVENT_FEATURES)
        self.assertEqual(combined, baseline + CENTROID_FEATURES + EVENT_FEATURES)


if __name__ == "__main__":
    unittest.main()
