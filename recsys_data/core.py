"""Lossless feed preparation. No fitted transforms or model features live here."""
from __future__ import annotations

from contextlib import ExitStack, closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FORMAT_VERSION = 1
FLOATS = ("f1", "f2", "f3", "f4", "f5")
CANDIDATES = {"item_id": pa.int32(), "category_id": pa.int8(),
              **{name: pa.float32() for name in FLOATS},
              "item_id__embedding_idx": pa.int32()}
HISTORY = {"history__item_id": pa.int32(), "history__category_id": pa.int8(),
           "history__event_type": pa.int8(), "history__event_source": pa.int8(),
           "history__item_id__embedding_idx": pa.int32()}
BASE_SCHEMA = {"feed_id": pa.int32(), "timestamp": pa.int32(), "user_id": pa.int32(),
               **{name: pa.list_(dtype) for name, dtype in CANDIDATES.items()},
               "user_embedding": pa.list_(pa.int8()),
               **{name: pa.list_(dtype) for name, dtype in HISTORY.items()}}


def _np(array):
    return array.to_numpy(zero_copy_only=False)


def _flat(table, name):
    return pc.list_flatten(table[name]).combine_chunks()


def _lengths(table, name):
    return _np(pc.list_value_length(table[name]).combine_chunks())


def _list_mask(table, name):
    lengths = _lengths(table, name)
    offsets = pa.array(np.r_[0, np.cumsum(lengths, dtype=np.int64)], type=pa.int32())
    return pa.ListArray.from_arrays(offsets, pc.greater_equal(_flat(table, name), 0))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EmbeddingStore:
    """Keep int8 on disk; return float32 vectors and explicit presence masks."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "data.meta").read_text())
        shape = self.meta.get("shape")
        if (self.meta.get("dtype") != "int8" or self.meta.get("order") != "C"
                or not isinstance(shape, list) or len(shape) != 2
                or any(type(x) is not int for x in shape) or shape[0] < 1 or shape[1] != 128):
            raise ValueError(f"Invalid embedding metadata: {self.directory}")
        self.shape = tuple(shape)
        self.path = self.directory / "data.memmap"
        self.available = self.path.is_file()
        if self.available and self.path.stat().st_size != self.shape[0] * self.shape[1]:
            raise ValueError(f"Embedding file size disagrees with metadata: {self.path}")
        self._vectors = None

    def lookup(self, indices):
        indices = np.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Embedding indices must be integers")
        if np.any((indices < -1) | (indices >= self.shape[0])):
            raise ValueError("Embedding index outside [-1, number_of_vectors)")
        present = indices >= 0
        vectors = np.zeros((*indices.shape, 128), dtype=np.float32)
        if present.any():
            if not self.available:
                raise FileNotFoundError(f"Embedding matrix is not downloaded: {self.path}")
            if self._vectors is None:
                self._vectors = np.memmap(self.path, dtype=np.int8, mode="r", shape=self.shape)
            vectors[present] = self._vectors[indices[present]].astype(np.float32)
        return vectors, present


def _validate(table, split, embedding_rows):
    expected = dict(BASE_SCHEMA)
    if split in ("train", "val"):
        expected["target"] = pa.list_(pa.int8())
    if split == "train":
        expected["pos"] = pa.list_(pa.int16())
    if set(table.column_names) != set(expected):
        raise ValueError(f"Unexpected {split} columns; missing={set(expected)-set(table.column_names)}, "
                         f"extra={set(table.column_names)-set(expected)}")
    for name, dtype in expected.items():
        column = table[name]
        if column.type != dtype:
            raise ValueError(f"{name}: expected {dtype}, got {column.type}")
        if column.null_count or (pa.types.is_list(dtype) and _flat(table, name).null_count):
            raise ValueError(f"Null values in {name}")
    counts, history_lengths = _lengths(table, "item_id"), _lengths(table, "history__item_id")
    if np.any(counts < 5) or np.any(history_lengths > 256):
        raise ValueError("Feed has fewer than 5 candidates or more than 256 history events")
    for name in CANDIDATES:
        if not np.array_equal(_lengths(table, name), counts):
            raise ValueError(f"Misaligned candidate list: {name}")
    for name in HISTORY:
        if not np.array_equal(_lengths(table, name), history_lengths):
            raise ValueError(f"Misaligned history list: {name}")
    if np.any(_lengths(table, "user_embedding") != 128):
        raise ValueError("User embedding must have 128 components")
    for name in FLOATS:
        if not np.isfinite(_np(_flat(table, name))).all():
            raise ValueError(f"Nonfinite values in {name}")
    for name in ("item_id__embedding_idx", "history__item_id__embedding_idx"):
        idx = _np(_flat(table, name))
        if np.any((idx < -1) | (idx >= embedding_rows)):
            raise ValueError(f"Invalid embedding index in {name}")
    for name, maximum in (("category_id", 53), ("history__category_id", 53),
                          ("history__event_type", 10), ("history__event_source", 30)):
        values = _np(_flat(table, name))
        if np.any((values < 0) | (values > maximum)):
            raise ValueError(f"Unknown categorical code range: {name}")
    for name in ("feed_id", "user_id", "item_id", "history__item_id"):
        values = _np(_flat(table, name)) if pa.types.is_list(table[name].type) else _np(table[name].combine_chunks())
        if np.any(values < 0):
            raise ValueError(f"Negative identifier in {name}")
    feed_ids = _np(table["feed_id"].combine_chunks()).astype(np.int64)
    keys = (np.repeat(feed_ids, counts) << 32) | _np(_flat(table, "item_id")).astype(np.int64)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Duplicate (feed_id, item_id) candidate keys")
    if "target" in expected:
        if not np.array_equal(_lengths(table, "target"), counts):
            raise ValueError("Misaligned target list")
        labels = _np(_flat(table, "target"))
        if np.any((labels != 0) & (labels != 1)):
            raise ValueError("Targets must be binary")
        offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        cumulative = np.r_[0, np.cumsum(labels, dtype=np.int64)]
        if np.any(cumulative[offsets[1:]] - cumulative[offsets[:-1]] < 1):
            raise ValueError("Feed has no positive targets")
    if "pos" in expected:
        if not np.array_equal(_lengths(table, "pos"), counts):
            raise ValueError("Misaligned position list")
        positions = _np(_flat(table, "pos"))
        if np.any((positions < 0) | (positions > 300)):
            raise ValueError("Train positions must be in [0, 300]")
    return counts, history_lengths


def _prepare_table(table, split, embedding_rows):
    counts, history_lengths = _validate(table, split, embedding_rows)
    labels = table.select(["feed_id", "target"]).replace_schema_metadata(None) if "target" in table.column_names else None
    diagnostics = table.select(["feed_id", "pos"]).replace_schema_metadata(None) if "pos" in table.column_names else None
    features = table.select(list(BASE_SCHEMA)).replace_schema_metadata(None)
    additions = {
        "candidate_count": pa.array(counts, type=pa.int32()),
        "history_length": pa.array(history_lengths, type=pa.int32()),
        "candidate_embedding_present": _list_mask(table, "item_id__embedding_idx"),
        "history_embedding_present": _list_mask(table, "history__item_id__embedding_idx"),
        "user_embedding_is_zero": pa.array(np.all(_np(_flat(table, "user_embedding")).reshape(-1, 128) == 0, axis=1)),
        "history_is_empty": pa.array(history_lengths == 0),
    }
    for name, values in additions.items():
        features = features.append_column(name, values)
    return features, labels, diagnostics


def prepare_dataset(source, output, splits=("train", "val", "test"), *,
                    batch_size=512, allow_missing_embeddings=False, archive_manifest=None):
    """Validate and atomically write a new prepared directory; never overwrite inputs."""
    source, output = Path(source).resolve(), Path(output).resolve()
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "val", "test"}:
        raise ValueError("Select distinct train/val/test splits")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Prepared output must be separate from the raw dataset")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new version: {output}")
    archive_path = Path(archive_manifest) if archive_manifest else source.parent / "archive_manifest.json"
    archive = json.loads(archive_path.read_text()) if archive_path.exists() else None
    manifest = {"format_version": FORMAT_VERSION, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_relative_path": os.path.relpath(source, output), "batch_size": batch_size,
                "versions": {"numpy": np.__version__, "pyarrow": pa.__version__},
                "code_sha256": _sha256(__file__), "splits": {},
                "archive_etag": archive.get("etag") if archive else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        with closing(sqlite3.connect(str(Path(temporary) / "keys.sqlite"))) as registry:
            registry.execute("CREATE TABLE feed_keys (feed_id INTEGER PRIMARY KEY)")
            for split in splits:
                store = EmbeddingStore(source / split / "item_embeddings")
                if not store.available and not allow_missing_embeddings:
                    raise FileNotFoundError(f"Missing {store.path}; explicitly allow metadata-only preparation")
                paths = sorted((source / split / "impressions").glob("*.parquet"))
                if not paths:
                    raise FileNotFoundError(f"No impressions files for {split}")
                expected_names = {m["name"].rsplit("/", 1)[-1] for m in archive["members"]
                                  if f"/{split}/impressions/" in m["name"] and m["name"].endswith(".parquet")} if archive else None
                if expected_names is not None and set(p.name for p in paths) - expected_names:
                    raise ValueError(f"Local {split} has files absent from archive manifest")
                info = {"feeds": 0, "candidates": 0, "history_events": 0, "files": [],
                        "embedding_meta": store.meta, "embedding_available": store.available,
                        "embedding_meta_sha256": _sha256(store.directory / "data.meta"),
                        "complete_impressions": set(p.name for p in paths) == expected_names if expected_names is not None else None,
                        "archive_files": len(expected_names) if expected_names is not None else None,
                        "has_labels": split != "test", "has_diagnostics": split == "train"}
                for path in paths:
                    parquet = pq.ParquetFile(path)
                    file_info = {"name": path.name, "source_sha256": _sha256(path), "feeds": 0}
                    with ExitStack() as stack:
                        writers = {}
                        for batch in parquet.iter_batches(batch_size=batch_size):
                            raw = pa.Table.from_batches([batch])
                            features, labels, diagnostics = _prepare_table(raw, split, store.shape[0])
                            try:
                                registry.executemany("INSERT INTO feed_keys VALUES (?)", ((int(x),) for x in _np(features["feed_id"].combine_chunks())))
                            except sqlite3.IntegrityError as error:
                                raise ValueError("Duplicate feed_id within or between splits") from error
                            for kind, table in (("features", features), ("labels", labels), ("diagnostics", diagnostics)):
                                if table is None:
                                    continue
                                if kind not in writers:
                                    destination = staging / split / kind / path.name
                                    destination.parent.mkdir(parents=True, exist_ok=True)
                                    writers[kind] = stack.enter_context(pq.ParquetWriter(destination, table.schema, compression="zstd"))
                                writers[kind].write_table(table, row_group_size=batch_size)
                            info["feeds"] += len(features)
                            file_info["feeds"] += len(features)
                            info["candidates"] += int(pc.sum(features["candidate_count"]).as_py())
                            info["history_events"] += int(pc.sum(features["history_length"]).as_py())
                    if file_info["feeds"] == 0:
                        raise ValueError(f"Empty impressions shard: {path}")
                    info["files"].append(file_info)
                registry.commit()
                manifest["splits"][split] = info
                print(f"{split}: {info['feeds']} feeds, {info['candidates']} candidates; complete={info['complete_impressions']}", flush=True)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(output)
    return manifest


@dataclass
class FeedBatch:
    """Whole feeds; labels and original positions are separate from features."""
    features: pa.Table
    labels: pa.Table | None
    diagnostics: pa.Table | None
    embeddings: EmbeddingStore

    @property
    def candidate_counts(self):
        return _np(self.features["candidate_count"].combine_chunks())

    @property
    def candidate_offsets(self):
        return np.r_[0, np.cumsum(self.candidate_counts, dtype=np.int64)]

    @property
    def history_offsets(self):
        return np.r_[0, np.cumsum(_np(self.features["history_length"].combine_chunks()), dtype=np.int64)]

    def user_vectors(self):
        return _np(_flat(self.features, "user_embedding")).astype(np.float32).reshape(-1, 128)

    def candidate_vectors(self):
        return self.embeddings.lookup(_np(_flat(self.features, "item_id__embedding_idx")))

    def history_vectors(self):
        return self.embeddings.lookup(_np(_flat(self.features, "history__item_id__embedding_idx")))

    def candidate_table(self, *, include_target=False):
        """Flatten candidates together; preserve contiguous groups and key metadata."""
        counts = self.candidate_counts
        offsets = self.candidate_offsets
        columns = {"feed_id": pa.array(np.repeat(_np(self.features["feed_id"].combine_chunks()), counts), type=pa.int32()),
                   "candidate_index": pa.array(np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts), type=pa.int32()),
                   **{name: _flat(self.features, name) for name in CANDIDATES},
                   "candidate_embedding_present": _flat(self.features, "candidate_embedding_present")}
        if include_target:
            if self.labels is None:
                raise ValueError("Labels were not requested or are unavailable for this split")
            columns["target"] = _flat(self.labels, "target")
        return pa.table(columns)

    def history_table(self):
        """History once per feed; history_index=0 is the newest event."""
        lengths = _np(self.features["history_length"].combine_chunks())
        offsets = self.history_offsets
        return pa.table({"feed_id": pa.array(np.repeat(_np(self.features["feed_id"].combine_chunks()), lengths), type=pa.int32()),
                         "history_index": pa.array(np.arange(offsets[-1]) - np.repeat(offsets[:-1], lengths), type=pa.int32()),
                         **{name: _flat(self.features, name) for name in HISTORY},
                         "history_embedding_present": _flat(self.features, "history_embedding_present")})


class PreparedSplit:
    def __init__(self, root, split):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported prepared format version")
        self.split = split
        self.info = self.manifest["splits"][split]
        source = (self.root / self.manifest["source_relative_path"]).resolve()
        self.embeddings = EmbeddingStore(source / split / "item_embeddings")
        if self.embeddings.meta != self.info["embedding_meta"]:
            raise ValueError("Source embedding metadata has changed since preparation")

    def iter_batches(self, batch_size=512, *, include_labels=False, include_diagnostics=False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if include_labels and not self.info["has_labels"]:
            raise ValueError("Test has no labels")
        if include_diagnostics and not self.info["has_diagnostics"]:
            raise ValueError("Original positions are only available for train")
        kinds = ["features"] + (["labels"] if include_labels else []) + (["diagnostics"] if include_diagnostics else [])
        for shard in self.info["files"]:
            iterators = [pq.ParquetFile(self.root / self.split / kind / shard["name"]).iter_batches(batch_size=batch_size)
                         for kind in kinds]
            for batches in zip_longest(*iterators):
                if any(batch is None for batch in batches):
                    raise ValueError("Prepared features and auxiliary tables have different row counts")
                tables = {kind: pa.Table.from_batches([batch]) for kind, batch in zip(kinds, batches)}
                for kind in kinds[1:]:
                    if not tables[kind]["feed_id"].equals(tables["features"]["feed_id"]):
                        raise ValueError("Prepared auxiliary table is misaligned with features")
                yield FeedBatch(tables["features"], tables.get("labels"), tables.get("diagnostics"), self.embeddings)
