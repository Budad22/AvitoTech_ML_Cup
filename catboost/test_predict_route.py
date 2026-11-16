import copy
import unittest

from predict_route import compatible_manifests, feature_names


class RoutePredictionContracts(unittest.TestCase):
    def manifest(self):
        return {"format_version": 1, "archive_etag": "verified-archive", "splits": {
            "train": {"files": ["a"]}, "val": {"files": [{"name": "v", "source_sha256": "val-hash"}]},
            "test": {"files": [{"name": "t", "source_sha256": "test-hash"}]}}}

    def test_different_train_volume_requires_identical_val_test_and_archive(self):
        original = self.manifest()
        inference = copy.deepcopy(original)
        inference["splits"]["train"]["files"] = ["a", "b"]
        compatible_manifests(original, inference)
        for split in ("val", "test"):
            changed = copy.deepcopy(inference)
            changed["splits"][split]["files"][0]["source_sha256"] = "changed"
            with self.assertRaises(ValueError):
                compatible_manifests(original, changed)
        inference["archive_etag"] = "another-archive"
        with self.assertRaises(ValueError):
            compatible_manifests(original, inference)

    def test_feature_order_and_rejected_groups(self):
        config = {"history_category_windows": [16, 64, 256], "user_item_cosine": True,
                  "feed_percentiles": True, "centroid": True, "event_types": False, "neighbors": True}
        names = feature_names(config)
        self.assertEqual(names[-3:], ["history_max_cosine_64", "history_top3_mean_cosine_64", "history_neighbor_valid_64"])
        self.assertFalse(any(name.startswith("history_type_") for name in names))
        with self.assertRaises(ValueError):
            feature_names({**config, "neighbors": 1})
        with self.assertRaises(ValueError):
            feature_names({**config, "unknown": True})


if __name__ == "__main__":
    unittest.main()
