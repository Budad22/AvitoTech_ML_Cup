"""Fixed long-history and projected user/item interactions; no label-derived transforms."""
import hashlib
import numpy as np
import pyarrow as pa

LONG_FEATURES = ["history_max_cosine_256", "history_top3_mean_cosine_256", "history_neighbor_valid_256"]
PROJECTED_FEATURES = [f"user_item_projected_product_{i}" for i in range(32)]
# Fixed rotation, independent of labels, dataset and training RNG.
PROJECTION = np.linalg.qr(np.random.default_rng(1701).normal(size=(128, 32)))[0].astype(np.float32)
PROJECTION_SHA256 = hashlib.sha256(PROJECTION.tobytes()).hexdigest()


def add_extended_features(batch, candidates):
    items, present = batch.embeddings.lookup(candidates["item_id__embedding_idx"].to_numpy())
    norms = np.linalg.norm(items, axis=1)
    item_valid = present & (norms > 0)
    items = np.divide(items, norms[:, None], out=np.zeros_like(items), where=item_valid[:, None])
    histories = batch.features["history__item_id__embedding_idx"].to_pylist()
    lengths = np.array([min(256, len(history)) for history in histories])
    indices = np.array([idx for history in histories for idx in history[:256]], dtype=np.int32)
    vectors, history_present = batch.embeddings.lookup(indices)
    norms = np.linalg.norm(vectors, axis=1)
    history_valid = history_present & (norms > 0)
    vectors = np.divide(vectors, norms[:, None], out=np.zeros_like(vectors), where=history_valid[:, None])
    offsets = np.r_[0, np.cumsum(lengths)]
    values = np.zeros((len(candidates), 3), dtype=np.float32)
    for row, (start, end) in enumerate(zip(batch.candidate_offsets[:-1], batch.candidate_offsets[1:])):
        left, right = offsets[row:row+2]
        history = vectors[left:right][history_valid[left:right]]
        if not len(history):
            continue
        similarities = np.clip(np.einsum("id,jd->ij", items[start:end], history), -1., 1.)
        count = min(3, len(history))
        values[start:end, 0] = similarities.max(axis=1)
        values[start:end, 1] = np.partition(similarities, len(history)-count, axis=1)[:, -count:].mean(axis=1)
        values[start:end, 2] = item_valid[start:end]
        values[start:end, :2][~item_valid[start:end]] = 0.
    users = batch.user_vectors().astype(np.float32, copy=False)
    norms = np.linalg.norm(users, axis=1)
    users = np.divide(users, norms[:, None], out=np.zeros_like(users), where=norms[:, None] > 0)
    owners = np.repeat(np.arange(len(users)), batch.candidate_counts)
    item_projection = np.einsum("ij,jk->ik", items, PROJECTION)
    user_projection = np.einsum("ij,jk->ik", users, PROJECTION)
    products = item_projection * user_projection[owners]
    if not np.isfinite(values).all() or not np.isfinite(products).all():
        raise ValueError("Nonfinite extended feature")
    for j, name in enumerate(LONG_FEATURES):
        candidates = candidates.append_column(name, pa.array(values[:, j]))
    for j, name in enumerate(PROJECTED_FEATURES):
        candidates = candidates.append_column(name, pa.array(products[:, j]))
    return candidates
