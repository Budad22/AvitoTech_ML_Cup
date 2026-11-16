import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from compact_embeddings import write_compact
from recsys_data import EmbeddingStore


class CompactContracts(unittest.TestCase):
    def test_exact_int8_values_shapes_missing_repeats_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.meta").write_text(json.dumps({"dtype":"int8", "shape":[5,128], "order":"C"}))
            matrix = np.random.default_rng(42).integers(-128,128,size=(5,128),dtype=np.int16).astype(np.int8)
            matrix[3] = 0
            matrix.tofile(root / "data.memmap")
            original = EmbeddingStore(root)
            compact = write_compact(original, [4,-1,1,1,3], root / "compact", {"scope":"test"})
            indices = np.array([[4,-1],[1,3],[1,4]],dtype=np.int32)
            a, am = original.lookup(indices)
            b, bm = compact.lookup(indices)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(am,bm)
            for invalid in (np.array([0]),np.array([5]),np.array([-2]),np.array([1.5])):
                with self.assertRaises((ValueError,TypeError)):
                    compact.lookup(invalid)
            with self.assertRaises(FileExistsError):
                write_compact(original, [1], root / "compact", {})


if __name__ == "__main__":
    unittest.main()
