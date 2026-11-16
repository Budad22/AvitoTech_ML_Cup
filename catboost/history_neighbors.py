"""Candidate-specific nearest-event similarities, without labels or original positions."""
import numpy as np
import pyarrow as pa

NEIGHBOR_FEATURES = ["history_max_cosine_64", "history_top3_mean_cosine_64", "history_neighbor_valid_64"]


def add_history_neighbors(batch, candidates):
    items, item_present = batch.embeddings.lookup(candidates["item_id__embedding_idx"].to_numpy())
    item_norms = np.linalg.norm(items, axis=1)
    item_valid = item_present & (item_norms > 0)
    items = np.divide(items, item_norms[:, None], out=np.zeros_like(items), where=item_valid[:, None])
    histories = batch.features["history__item_id__embedding_idx"].to_pylist()
    lengths = np.array([min(64, len(history)) for history in histories])
    indices = np.array([idx for history in histories for idx in history[:64]], dtype=np.int32)
    vectors, present = batch.embeddings.lookup(indices)
    norms = np.linalg.norm(vectors, axis=1)
    valid = present & (norms > 0)
    vectors = np.divide(vectors, norms[:, None], out=np.zeros_like(vectors), where=valid[:, None])
    offsets = np.r_[0, np.cumsum(lengths)]
    values = np.zeros((len(candidates), 3), dtype=np.float32)
    for row, (start, end) in enumerate(zip(batch.candidate_offsets[:-1], batch.candidate_offsets[1:])):
        left, right = offsets[row:row+2]
        history = vectors[left:right][valid[left:right]]
        if not len(history):
            continue
        # Avoid int8 multiplication and a padded zero winning over negative similarities.
        similarities = np.einsum("id,jd->ij", items[start:end], history)
        similarities = np.clip(similarities, -1., 1.)
        count = min(3, len(history))
        largest = np.partition(similarities, len(history)-count, axis=1)[:, -count:]
        values[start:end, 0] = similarities.max(axis=1)
        values[start:end, 1] = largest.mean(axis=1)
        values[start:end, 2] = item_valid[start:end]
        values[start:end, :2][~item_valid[start:end]] = 0.
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite nearest-history features")
    for column, name in enumerate(NEIGHBOR_FEATURES):
        candidates = candidates.append_column(name, pa.array(values[:, column]))
    return candidates
