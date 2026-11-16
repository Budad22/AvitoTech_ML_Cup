"""Lossless, scoped int8 lookup cache; original datasets and vectors stay unchanged."""
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from baseline import sha256
from recsys_data import PreparedSplit


class CompactStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.meta = self.manifest["source"]["metadata"]
        self.shape = tuple(self.meta["shape"])
        self.path = self.directory / "vectors.int8"
        self.available = self.path.is_file()
        self.indices = np.load(self.directory / "indices.npy", mmap_mode="r")
        self.vectors = np.memmap(self.path, dtype=np.int8, mode="r", shape=(len(self.indices), 128))

    def lookup(self, indices):
        indices = np.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Embedding indices must be integers")
        if np.any((indices < -1) | (indices >= self.shape[0])):
            raise ValueError("Embedding index outside original source matrix")
        present = indices >= 0
        values = np.zeros((*indices.shape, 128), dtype=np.float32)
        positions = np.searchsorted(self.indices, indices[present])
        if np.any(positions >= len(self.indices)):
            raise ValueError("Requested vector is outside the compact cache scope")
        if not np.array_equal(self.indices[positions], indices[present]):
            raise ValueError("Requested vector is outside the compact cache scope")
        values[present] = self.vectors[positions].astype(np.float32)
        return values, present


def write_compact(source, requested, output, identity):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    unique = np.unique(np.asarray(requested, dtype=np.int32))
    unique = unique[unique >= 0]
    if not len(unique) or unique[-1] >= source.shape[0]:
        raise ValueError("Invalid compact cache scope")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compact-", dir=output.parent) as temporary:
        staging = Path(temporary) / "cache"
        staging.mkdir()
        np.save(staging / "indices.npy", unique)
        raw = np.memmap(source.path, dtype=np.int8, mode="r", shape=source.shape)
        with (staging / "vectors.int8").open("xb") as destination:
            for start in range(0, len(unique), 65536):
                raw[unique[start:start+65536]].tofile(destination)
        del raw
        manifest = {"identity": identity,
                    "source": {"path": str(source.path), "metadata": source.meta,
                               "bytes": source.path.stat().st_size, "mtime_ns": source.path.stat().st_mtime_ns},
                    "unique_vectors": len(unique), "matrix_bytes": len(unique)*128,
                    "seconds": time.perf_counter()-started,
                    "files": {name: sha256(staging / name) for name in ("indices.npy", "vectors.int8")}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        staging.rename(output)
    return CompactStore(output)


def prepare_train_cache(data16, original_data, output):
    data16, original_data, output = map(Path, (data16, original_data, output))
    dataset = PreparedSplit(data16, "train")
    source_dataset = PreparedSplit(original_data, "train")
    source = source_dataset.embeddings
    if dataset.manifest["archive_etag"] != source_dataset.manifest["archive_etag"] or dataset.embeddings.meta != source.meta:
        raise ValueError("Train matrix identity differs between datasets")
    identity = {"scope_manifest_sha256": sha256(data16 / "manifest.json"),
                "source_archive_etag": dataset.manifest["archive_etag"]}
    if output.exists():
        saved = json.loads((output / "manifest.json").read_text())
        stat = source.path.stat()
        if (saved["identity"] != identity or saved["source"]["metadata"] != source.meta
                or saved["source"]["bytes"] != stat.st_size or saved["source"]["mtime_ns"] != stat.st_mtime_ns):
            raise ValueError("Compact cache source identity changed")
        for name, expected in saved["files"].items():
            if sha256(output / name) != expected:
                raise ValueError(f"Compact cache checksum differs: {name}")
        return CompactStore(output)
    requested = []
    for shard in dataset.info["files"]:
        path = data16 / "train/features" / shard["name"]
        for batch in pq.ParquetFile(path).iter_batches(columns=["item_id__embedding_idx", "history__item_id__embedding_idx"]):
            for column in batch.columns:
                requested.append(pc.list_flatten(column).to_numpy(zero_copy_only=False))
    values = np.concatenate(requested)
    print(f"Compact cache: {len(values):,} requested candidate/history rows", flush=True)
    result = write_compact(source, values, output, identity)
    print(f"Compact cache ready: {len(result.indices):,} unique vectors, {result.manifest['matrix_bytes']/2**20:.1f} MiB", flush=True)
    return result
