"""Safety contracts: no silent leakage, key corruption, or missing-vector indexing."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from recsys_data import EmbeddingStore, PreparedSplit, prepare_dataset
from recsys_data.core import BASE_SCHEMA, _prepare_table


def feeds(split="train"):
    schema = dict(BASE_SCHEMA)
    if split != "test":
        schema["target"] = pa.list_(pa.int8())
    if split == "train":
        schema["pos"] = pa.list_(pa.int16())
    records = []
    for i in range(3):
        history = [] if i == 1 else [70, 70, 71]
        row = {"feed_id": 10 + i, "timestamp": 1000 + i, "user_id": i,
               "item_id": [101, 102, 103, 104, 105], "category_id": [0, 2, 3, 4, 5],
               **{f"f{j}": [0., .2, .4, .6, 1.] for j in range(1, 6)},
               "item_id__embedding_idx": [0, -1, 1, 2, 0],
               "user_embedding": [0 if i == 1 else 127] * 128,
               "history__item_id": history, "history__category_id": [0, 2, 2] if history else [],
               "history__event_type": [1, 2, 3] if history else [],
               "history__event_source": [0, 1, 2] if history else [],
               "history__item_id__embedding_idx": [1, -1, 0] if history else []}
        if split != "test":
            row["target"] = [0, 1, 0, 0, 0]
        if split == "train":
            row["pos"] = [0, 2, 3, 8, 300]
        records.append(row)
    return pa.Table.from_pylist(records, schema=pa.schema(list(schema.items())))


class DataContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw"
        embedding_dir = self.source / "train/item_embeddings"
        embedding_dir.mkdir(parents=True)
        (embedding_dir / "data.meta").write_text(json.dumps({"dtype": "int8", "shape": [3, 128], "order": "C"}))
        self.vectors = np.array([[127] * 128, [2] * 128, [-4] * 128], dtype=np.int8)
        self.vectors.tofile(embedding_dir / "data.memmap")
        self.table = feeds()
        impressions = self.source / "train/impressions"
        impressions.mkdir()
        pq.write_table(self.table, impressions / "part-00000.parquet", row_group_size=1)

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_vector_and_float_arithmetic(self):
        store = EmbeddingStore(self.source / "train/item_embeddings")
        vectors, present = store.lookup(np.array([0, -1, 2], dtype=np.int32))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_array_equal(present, [True, False, True])
        np.testing.assert_array_equal(vectors[1], np.zeros(128))
        self.assertEqual(np.einsum("i,i->", vectors[0], vectors[0]), 127 * 127 * 128)
        for invalid in (np.array([-2]), np.array([3]), np.array([.5])):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                store.lookup(invalid)

    def test_sliced_lists_preserve_alignment_and_history(self):
        # Nonzero Arrow offsets expose accidental use of the whole underlying .values buffer.
        raw = self.table.slice(1, 2)
        features, labels, diagnostics = _prepare_table(raw, "train", 3)
        self.assertNotIn("target", features.column_names)
        self.assertNotIn("pos", features.column_names)
        self.assertEqual(features["history__item_id"].to_pylist(), [[], [70, 70, 71]])
        self.assertEqual(features["history_is_empty"].to_pylist(), [True, False])
        self.assertEqual(features["user_embedding_is_zero"].to_pylist(), [True, False])
        self.assertEqual(features["candidate_embedding_present"].to_pylist(), [[True, False, True, True, True]] * 2)
        for name in BASE_SCHEMA:
            self.assertTrue(features[name].equals(raw[name]), name)
        self.assertTrue(labels["target"].equals(raw["target"]))
        self.assertTrue(diagnostics["pos"].equals(raw["pos"]))

    def test_roundtrip_whole_feeds_and_explicit_labels(self):
        output = self.root / "prepared"
        prepare_dataset(self.source, output, ("train",), batch_size=1)
        dataset = PreparedSplit(output, "train")
        batches = list(dataset.iter_batches(batch_size=2, include_labels=True, include_diagnostics=True))
        restored = pa.concat_tables([b.features.select(list(BASE_SCHEMA)) for b in batches])
        self.assertTrue(restored.equals(self.table.select(list(BASE_SCHEMA))))
        candidates = pa.concat_tables([b.candidate_table(include_target=True) for b in batches])
        self.assertEqual(candidates["item_id"].to_pylist(), [101, 102, 103, 104, 105] * 3)
        self.assertEqual(candidates["target"].to_pylist(), [0, 1, 0, 0, 0] * 3)
        np.testing.assert_array_equal(batches[0].candidate_offsets, [0, 5, 10])
        history = pa.concat_tables([b.history_table() for b in batches])
        self.assertEqual(history["history__item_id"].to_pylist(), [70, 70, 71] * 2)
        self.assertEqual(history["history_index"].to_pylist(), [0, 1, 2] * 2)
        vectors, present = batches[0].candidate_vectors()
        self.assertEqual(vectors.shape, (10, 128))
        self.assertFalse(present[1])
        self.assertEqual(batches[0].user_vectors().dtype, np.float32)
        batch = next(dataset.iter_batches())
        self.assertIsNone(batch.labels)
        self.assertIsNone(batch.diagnostics)
        self.assertNotIn("target", batch.candidate_table().column_names)
        with self.assertRaises(ValueError):
            batch.candidate_table(include_target=True)
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.source, output, ("train",))

    def test_bad_inputs_fail_without_changing_raw_data(self):
        changes = {
            "f1": [[float("nan"), 0., 0., 0., 0.]],
            "category_id": [[1, 2]],
            "item_id__embedding_idx": [[0, -2, 0, 0, 0]],
            "item_id": [[101, 101, 103, 104, 105]],
            "target": [[0, 2, 0, 0, 0]],
            "user_embedding": [[1] * 127],
            "history__event_type": [[1, 2]],
        }
        original = self.table.slice(0, 1)
        for name, values in changes.items():
            with self.subTest(column=name):
                index = original.schema.get_field_index(name)
                bad = original.set_column(index, name, pa.array(values, type=original[name].type))
                with self.assertRaises(ValueError):
                    _prepare_table(bad, "train", 3)
        field = original.schema.get_field_index("f1")
        null = original.set_column(field, "f1", pa.array([[None] * 5], type=pa.list_(pa.float32())))
        with self.assertRaises(ValueError):
            _prepare_table(null, "train", 3)

    def test_duplicate_feeds_across_shards_abort_atomic_output(self):
        pq.write_table(self.table, self.source / "train/impressions/part-00001.parquet")
        output = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "Duplicate feed_id"):
            prepare_dataset(self.source, output, ("train",))
        self.assertFalse(output.exists())

    def test_test_split_and_missing_matrix_are_explicit(self):
        test = self.source / "test"
        (test / "impressions").mkdir(parents=True)
        (test / "item_embeddings").mkdir()
        (test / "item_embeddings/data.meta").write_text(json.dumps({"dtype": "int8", "shape": [3, 128], "order": "C"}))
        pq.write_table(feeds("test"), test / "impressions/part-00000.parquet")
        output = self.root / "prepared"
        with self.assertRaises(FileNotFoundError):
            prepare_dataset(self.source, output, ("test",))
        self.assertFalse(output.exists())
        manifest = prepare_dataset(self.source, output, ("test",), allow_missing_embeddings=True)
        self.assertFalse(manifest["splits"]["test"]["embedding_available"])
        self.assertIsNone(manifest["splits"]["test"]["complete_impressions"])
        dataset = PreparedSplit(output, "test")
        with self.assertRaisesRegex(ValueError, "no labels"):
            next(dataset.iter_batches(include_labels=True))
        batch = next(dataset.iter_batches())
        with self.assertRaises(FileNotFoundError):
            batch.candidate_vectors()


if __name__ == "__main__":
    unittest.main()
