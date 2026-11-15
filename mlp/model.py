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
    user = np.repeat(normalize(batch.user_vectors()), counts, axis=0)
    item, present = batch.candidate_vectors()
    item = normalize(item)
    numeric = np.column_stack([table[f'f{i}'].to_numpy() for i in range(1, 6)])
    dense = np.column_stack((numeric, user, item, user * item, abs(user - item),
                             ~present, np.linalg.norm(user, axis=1) == 0)).astype(np.float32)
    category = np.array([categories.get(int(c), 0) for c in table['category_id'].to_pylist()])
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
