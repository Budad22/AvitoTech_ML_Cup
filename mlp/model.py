"""Feature assembly and a small candidate-wise network."""
import numpy as np
import torch
from torch import nn


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def features(batch, categories):
    table = batch.candidate_table(include_target=batch.labels is not None)
    counts = batch.candidate_counts
    users = normalize(batch.user_vectors())
    user = np.repeat(users, counts, axis=0)
    item, present = batch.candidate_vectors()
    # Candidate vectors are owned float32 buffers; normalizing them cannot alter
    # the read-only embedding cache. Write derived features into one allocation.
    item /= np.maximum(np.linalg.norm(item, axis=-1, keepdims=True), 1e-12)
    dense = np.empty((len(item), 519), dtype=np.float32)
    for i in range(5):
        dense[:, i] = table[f'f{i + 1}'].to_numpy()
    dense[:, 5:133] = user
    dense[:, 133:261] = item
    np.multiply(user, item, out=dense[:, 261:389])
    difference = dense[:, 389:517]
    np.subtract(user, item, out=difference)
    np.abs(difference, out=difference)
    dense[:, 517] = ~present
    dense[:, 518] = np.repeat(np.linalg.norm(users, axis=1) == 0, counts)
    lookup = np.zeros(256, dtype=np.int64)
    for code, index in categories.items():
        if -128 <= code <= 127:
            lookup[code] = index
    codes = table['category_id'].to_numpy()
    category = lookup[codes]
    target = table['target'].to_numpy().astype(np.float32) if batch.labels is not None else None
    return dense, category, target, table


class Ranker(nn.Module):
    def __init__(self, category_count):
        super().__init__()
        self.category = nn.Embedding(category_count + 1, 16, padding_idx=0)
        self.network = nn.Sequential(nn.Linear(535, 256), nn.ReLU(), nn.Dropout(.1),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.1),
                                     nn.Linear(128, 1))

    def forward(self, dense, category):
        return self.network(torch.cat((dense, self.category(category)), dim=-1)).squeeze(-1)
