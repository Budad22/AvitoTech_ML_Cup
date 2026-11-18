"""Build offsets and sidecar arrays while preserving the raw order of shows."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.lib.format import open_memmap
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from recsys_data.core import BASE_SCHEMA, FLOATS, HISTORY, EmbeddingStore
from .state import Lock, read_json, sha256, write_json

SMOKE_SHARDS = (0, 170, 341, 511)
DAY = {
    day: int(datetime(2026, 9, day, tzinfo=timezone.utc).timestamp())
    for day in (22, 24, 25)
}
WINDOWS = {"early": (DAY[22], DAY[24], 7581), "late": (DAY[24], DAY[25], 6488)}
HISTORY_NAMES = {
    "history__item_id": "history_item_id", "history__category_id": "history_category",
    "history__event_type": "history_type", "history__event_source": "history_source",
    "history__item_id__embedding_idx": "history_index",
}
CANDIDATE_NAMES = {
    "item_id": "item_id", "category_id": "category",
    "item_id__embedding_idx": "item_index", "target": "target",
}
TRAIN_FEEDS = 3151208
TRAIN_CANDIDATES = 246745836
FULL_DUPLICATES = 4


def lengths(table, key):
    return pc.list_value_length(table[key]).to_numpy()


def shards(raw, split, scope):
    files = sorted((raw / split / "impressions").glob("part-*.parquet"))
    expected = 512 if split == "train" else 64
    if len(files) != expected:
        raise ValueError(f"Expected {expected} {split} shards, found {len(files)}")
    if split == "train" and scope == "smoke":
        return [files[i] for i in SMOKE_SHARDS]
    return files


def batches(files, window=None, counts_only=False):
    columns = ["timestamp", "item_id", "history__item_id"] if counts_only else None
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
            if window is not None:
                time = batch["timestamp"].to_numpy()
                batch = batch.filter(pa.array((window[0] <= time) & (time < window[1])))
            if batch.num_rows:
                yield batch


def checked_arrays(batch, train, embedding_rows):
    schema = {**BASE_SCHEMA, "target": pa.list_(pa.int8())}
    if train:
        schema["pos"] = pa.list_(pa.int16())
    candidate_counts = lengths(batch, "item_id")
    history_counts = lengths(batch, "history__item_id")
    if (candidate_counts < 1).any() or (history_counts > 256).any():
        raise ValueError("Invalid feed or history length")
    ranges = {
        "category_id": (0, 53), "history__category_id": (0, 53),
        "history__event_type": (0, 10), "history__event_source": (0, 30),
        "target": (0, 1), "pos": (0, np.iinfo(np.int16).max),
    }
    values = {}
    for name, dtype in schema.items():
        col = batch[name]
        if col.type != dtype or col.null_count:
            raise ValueError(f"Invalid type or null in {name}")
        if pa.types.is_list(dtype):
            expected = history_counts if name in HISTORY else candidate_counts
            if name == "user_embedding":
                expected = np.full(batch.num_rows, 128)
            children = pc.list_flatten(col)
            if children.null_count or not np.array_equal(lengths(batch, name), expected):
                raise ValueError(f"Invalid or misaligned list {name}")
            value = children.to_numpy()
        else:
            value = col.to_numpy()
        if name in ranges:
            low, high = ranges[name]
            if ((value < low) | (value > high)).any():
                raise ValueError(f"Out of range: {name}")
        elif name.endswith("__embedding_idx"):
            if ((value < -1) | (value >= embedding_rows)).any():
                raise ValueError(f"Out of range: {name}")
        elif name in FLOATS and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite values: {name}")
        elif name in ("feed_id", "user_id", "item_id", "history__item_id") and (value < 0).any():
            raise ValueError(f"Negative ID: {name}")
        values[name] = value
    return values, candidate_counts, history_counts


def create_arrays(path, feeds, candidates, events, train):
    spec = {
        "offsets": ("int64", feeds + 1), "history_offsets": ("int64", feeds + 1),
        "feed_id": ("int32", feeds), "history_item_id": ("int32", events),
        "history_index": ("int32", events), "history_category": ("int8", events),
        "history_type": ("int8", events), "history_source": ("int8", events),
    }
    if train:
        spec.update(compact_feed=("int64", feeds), pos=("int16", candidates))
    else:
        spec.update(timestamp=("int32", feeds), user_id=("int32", feeds),
                    users=("int8", (feeds, 128)), category=("int8", candidates),
                    item_id=("int32", candidates), item_index=("int32", candidates),
                    target=("int8", candidates), floats=("float32", (candidates, 5)))
    return {key: open_memmap(path / f"{key}.npy", mode="w+", dtype=dtype,
                             shape=(shape,) if isinstance(shape, int) else shape)
            for key, (dtype, shape) in spec.items()}


def build_split(raw, compact, path, name, scope, compact_meta):
    train = name == "train"
    split = "train" if train else "val"
    files = shards(raw, split, scope)
    window = None if train else WINDOWS[name]
    embedding_store = EmbeddingStore(raw / split / "item_embeddings")
    if not embedding_store.available:
        raise FileNotFoundError(embedding_store.path)
    feeds = candidates = events = 0
    for batch in batches(files, window, counts_only=True):
        feeds += batch.num_rows
        candidates += int(lengths(batch, "item_id").sum())
        events += int(lengths(batch, "history__item_id").sum())
    if window and feeds != window[2]:
        raise ValueError(f"Unexpected {name} cohort size: {feeds}")
    if train and scope == "full" and (feeds, candidates) != (TRAIN_FEEDS, TRAIN_CANDIDATES):
        raise ValueError("Full train counts differ from verified source")

    path.mkdir()
    arrays = create_arrays(path, feeds, candidates, events, train)
    f = c = h = duplicates = max_pos = 0
    arrays["offsets"][0] = arrays["history_offsets"][0] = 0
    if train:
        source = {key: np.load(compact / "train" / f"{key}.npy", mmap_mode="r")
                  for key in ("feed_id", "offsets", "item_id", "item_index", "category", "target")}
        compact_ids = source["feed_id"]
        sorter = np.argsort(compact_ids)
        ranked_ids = compact_ids[sorter]
        old_to_raw = np.zeros(len(compact_meta["categories"]) + 1, dtype=np.int8)
        for original, encoded in compact_meta["categories"].items():
            old_to_raw[encoded] = int(original)
    try:
        for batch in batches(files, window):
            values, cc, hc = checked_arrays(batch, train, embedding_store.shape[0])
            fn, cn, hn = f + len(cc), c + int(cc.sum()), h + int(hc.sum())
            arrays["feed_id"][f:fn] = values["feed_id"]
            arrays["offsets"][f + 1:fn + 1] = c + np.cumsum(cc, dtype=np.int64)
            arrays["history_offsets"][f + 1:fn + 1] = h + np.cumsum(hc, dtype=np.int64)
            for key, dest in HISTORY_NAMES.items():
                arrays[dest][h:hn] = values[key]
            keys = (np.repeat(values["feed_id"].astype(np.int64), cc) << 32) | values["item_id"]
            duplicates += len(keys) - len(np.unique(keys))
            if train:
                found = np.searchsorted(ranked_ids, values["feed_id"])
                if (found >= len(ranked_ids)).any() or not np.array_equal(ranked_ids[found], values["feed_id"]):
                    raise ValueError("Missing train feed in compact")
                feed_indices = sorter[found]
                starts = source["offsets"][feed_indices]
                if not np.array_equal(source["offsets"][feed_indices + 1] - starts, cc):
                    raise ValueError("Raw and compact train offsets differ")
                raw_offsets = np.cumsum(cc, dtype=np.int64) - cc
                rows = np.arange(cn - c) + np.repeat(starts - raw_offsets, cc)
                for key, dest in CANDIDATE_NAMES.items():
                    observed = source[dest][rows]
                    if dest == "category":
                        observed = old_to_raw[observed]
                    if not np.array_equal(values[key], observed):
                        raise ValueError(f"Raw and compact train {key} differ")
                arrays["compact_feed"][f:fn] = feed_indices
                arrays["pos"][c:cn] = values["pos"]
                max_pos = max(max_pos, int(values["pos"].max()))
            else:
                arrays["timestamp"][f:fn] = values["timestamp"]
                arrays["user_id"][f:fn] = values["user_id"]
                arrays["users"][f:fn] = values["user_embedding"].reshape(-1, 128)
                for key, dest in CANDIDATE_NAMES.items():
                    arrays[dest][c:cn] = values[key]
                arrays["floats"][c:cn] = np.column_stack([values[key] for key in FLOATS])
            f, c, h = fn, cn, hn
        if (f, c, h) != (feeds, candidates, events) or np.unique(arrays["feed_id"]).size != feeds:
            raise ValueError("Split coverage or feed uniqueness differs")
        if (not train and duplicates) or (train and scope == "full" and duplicates != FULL_DUPLICATES):
            raise ValueError(f"Unexpected count of duplicate candidate keys in {name}: {duplicates}")
    finally:
        for array in arrays.values():
            array.flush()
            array._mmap.close()
    return dict(feeds=feeds, candidates=candidates, events=events, duplicate_candidates=duplicates,
                max_pos=max_pos if train else None,
                files={p.name: sha256(p) for p in sorted(path.glob("*.npy"))},
                files_read=[p.name for p in files])


def prepare(raw, compact, output, scope):
    raw, compact, output = (Path(p).resolve() for p in (raw, compact, output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with Lock(output.with_suffix(".lock")):
        source_metadata = compact / "metadata.json"
        raw_ready = sha256(raw / "raw-ready.json")
        compact_meta = read_json(source_metadata)
        if compact_meta["identity"]["raw_ready_sha256"] != raw_ready:
            raise ValueError("Compact cache and raw archive identities differ")
        identity = dict(raw=str(raw), compact=str(compact), scope=scope,
                        raw_sha256=raw_ready, compact_sha256=sha256(source_metadata),
                        builder_sha256=sha256(__file__))
        if output.exists():
            metadata = read_json(output / "metadata.json")
            if metadata["identity"] != identity:
                raise ValueError("Cache exists with different identity")
            for split, info in metadata["splits"].items():
                for name, digest in info["files"].items():
                    if sha256(output / split / name) != digest:
                        raise ValueError(f"Cache checksum mismatch: {split}/{name}")
            return metadata
        with TemporaryDirectory(dir=output.parent, prefix=output.name + "-") as staging:
            dest = Path(staging) / "cache"
            dest.mkdir()
            splits = {}
            for name in ("train", "early", "late"):
                splits[name] = build_split(raw, compact, dest / name, name, scope, compact_meta)
                print(name, {k: splits[name][k] for k in ("feeds", "candidates", "events")}, flush=True)
            early, late = [np.load(dest / name / "feed_id.npy") for name in ("early", "late")]
            if np.intersect1d(early, late).size:
                raise ValueError("Validation cohorts overlap")
            metadata = dict(identity=identity, splits=splits, utc_windows=WINDOWS)
            write_json(dest / "metadata.json", metadata)
            dest.rename(output)
        return metadata
