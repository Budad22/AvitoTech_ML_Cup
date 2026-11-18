"""Disk-backed feeds and whole-feed collation."""
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from recsys_data.core import EmbeddingStore
from .state import read_json, sha256


def percentiles(floats):
    values = np.empty_like(floats)
    for j, column in enumerate(floats.T):
        _, inverse, counts = np.unique(column, return_inverse=True, return_counts=True)
        values[:, j] = ((np.cumsum(counts) - 0.5 * counts) / len(column))[inverse]
    return values


class Feeds:
    def __init__(self, cache, split):
        if split not in ("train", "early", "late"):
            raise ValueError(f"Invalid split: {split}")
        self.cache = Path(cache)
        self.split = split
        self.metadata = read_json(self.cache / "metadata.json")
        self.info = self.metadata["splits"][split]
        identity = self.metadata["identity"]
        train = split == "train"
        self.sidecar = {
            Path(name).stem: np.load(self.cache / split / name, mmap_mode="r")
            for name in self.info["files"]
        }
        self.items = EmbeddingStore(Path(identity["raw"]) / ("train" if train else "val") / "item_embeddings")
        if train:
            compact = Path(identity["compact"])
            manifest = compact / "metadata.json"
            if sha256(manifest) != identity["compact_sha256"]:
                raise ValueError("Compact cache metadata changed")
            self.base = {key: np.load(compact / "train" / f"{key}.npy", mmap_mode="r")
                         for key in ("offsets", "item_index", "item_id", "category", "target",
                                     "floats", "users", "user_id")}
            mapping = read_json(manifest)["categories"]
            self.raw_category = np.zeros(len(mapping) + 1, dtype=np.int8)
            for original, encoded in mapping.items():
                self.raw_category[encoded] = int(original)
        else:
            self.base = self.sidecar

    def __len__(self):
        return self.info["feeds"]

    def __getitem__(self, index):
        side = self.sidecar
        base = self.base
        train = self.split == "train"
        feed = side["compact_feed"][index] if train else index
        row = slice(*base["offsets"][feed:feed + 2])
        event = slice(*side["history_offsets"][index:index + 2])
        item, item_present = self.items.lookup(base["item_index"][row])
        history, history_present = self.items.lookup(side["history_index"][event])
        category = base["category"][row]
        if train:
            category = self.raw_category[category]
        floats = np.array(base["floats"][row])
        example = dict(
            user=np.array(base["users"][feed]), item=item, missing=~item_present,
            category=category.astype(np.int64), item_id=base["item_id"][row].astype(np.int64),
            floats=floats, percentile=percentiles(floats),
            history_item=history, history_missing=~history_present,
            labels=base["target"][row].astype(np.float32),
        )
        for key in ("history_item_id", "history_category", "history_type", "history_source"):
            example[key] = side[key][event].astype(np.int64)
        if train:
            example["pos"] = side["pos"][slice(*side["offsets"][index:index + 2])].astype(np.int64)
        return {key: torch.from_numpy(value) for key, value in example.items()}


def collate(rows):
    features = {}
    target = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        padded = torch.stack(values) if key == "user" else pad_sequence(values, batch_first=True)
        (target if key in ("labels", "pos") else features)[key] = padded
    for mask, key in (("valid", "item_id"), ("history_valid", "history_item_id")):
        sizes = torch.tensor([len(row[key]) for row in rows])
        features[mask] = torch.arange(features[key].shape[1])[None, :] < sizes[:, None]
    return features, target


def load_batch(dataset, indices, device):
    features, target = collate([dataset[int(index)] for index in indices])
    return ({k: v.to(device) for k, v in features.items()},
            {k: v.to(device) for k, v in target.items()})
