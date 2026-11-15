"""User/item cosine from each split's own vectors; no history or target lookup."""
import numpy as np
import pyarrow as pa

COSINE_FEATURES = ["user_item_cosine", "user_item_cosine_valid"]


def add_user_item_cosine(batch, candidates):
    users = batch.user_vectors().astype(np.float32, copy=False)
    items, present = batch.embeddings.lookup(candidates["item_id__embedding_idx"].to_numpy())
    items = items.astype(np.float32, copy=False)
    owners = np.repeat(np.arange(len(users)), batch.candidate_counts)
    denominator = np.linalg.norm(items, axis=1) * np.linalg.norm(users, axis=1)[owners]
    valid = present & (denominator > 0)
    dot = np.einsum("ij,ij->i", items, users[owners])
    scores = np.divide(dot, denominator, out=np.zeros_like(dot), where=valid)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite user/item cosine")
    scores = np.clip(scores, -1., 1.)
    return candidates.append_column(COSINE_FEATURES[0], pa.array(scores)).append_column(
        COSINE_FEATURES[1], pa.array(valid.astype(np.float32)))
