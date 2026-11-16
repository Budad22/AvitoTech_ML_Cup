"""Label-free recent-history features for controlled CatBoost experiments."""
import numpy as np
import pyarrow as pa

CENTROID_FEATURES = ["history_centroid_cosine_64", "history_centroid_cosine_valid_64",
                     "history_vector_count_64"]
EVENT_FEATURES = [f"history_type_{code}_category_share_64" for code in range(11)] + [
    f"history_type_{code}_count_64" for code in range(11)]


def add_history_centroid(batch, candidates):
    """Average valid normalized vectors from the first 64 events, retaining repeats."""
    centroids = np.zeros((len(batch.features), 128), dtype=np.float32)
    counts = np.zeros(len(batch.features), dtype=np.float32)
    histories = batch.features["history__item_id__embedding_idx"].to_pylist()
    lengths = np.array([min(64, len(h)) for h in histories])
    indices = np.array([i for h in histories for i in h[:64]], dtype=np.int32)
    vectors, present = batch.embeddings.lookup(indices)
    norms = np.linalg.norm(vectors, axis=1)
    valid = present & (norms > 0)
    normalized = np.divide(vectors, norms[:, None], out=np.zeros_like(vectors),
                           where=valid[:, None])
    offsets = np.r_[0, np.cumsum(lengths)]
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        counts[i] = valid[start:end].sum()
        if counts[i]:
            centroids[i] = normalized[start:end].sum(axis=0) / counts[i]
    items, item_present = batch.embeddings.lookup(candidates["item_id__embedding_idx"].to_numpy())
    owners = np.repeat(np.arange(len(centroids)), batch.candidate_counts)
    denominator = np.linalg.norm(items, axis=1) * np.linalg.norm(centroids, axis=1)[owners]
    defined = item_present & (denominator > 0)
    dot = np.einsum("ij,ij->i", items, centroids[owners])
    scores = np.divide(dot, denominator, out=np.zeros_like(dot), where=defined)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite history centroid cosine")
    values = [np.clip(scores, -1., 1.), defined.astype(np.float32), counts[owners]]
    for name, value in zip(CENTROID_FEATURES, values):
        candidates = candidates.append_column(name, pa.array(value, type=pa.float32()))
    return candidates


def add_history_event_types(batch, candidates):
    """Category share conditional on event type; counts distinguish absent types."""
    shares = np.zeros((len(candidates), 11), dtype=np.float32)
    counts = np.zeros_like(shares)
    histories = batch.features.select(["history__category_id", "history__event_type"]).to_pylist()
    categories = candidates["category_id"].to_numpy()
    for history, start, end in zip(histories, batch.candidate_offsets[:-1], batch.candidate_offsets[1:]):
        cats = np.asarray(history["history__category_id"][:64], dtype=np.int32)
        types = np.asarray(history["history__event_type"][:64], dtype=np.int32)
        if cats.shape != types.shape:
            raise ValueError("Misaligned history categories and event types")
        for code in range(11):
            selected = cats[types == code]
            counts[start:end, code] = len(selected)
            if len(selected):
                frequency = np.bincount(selected, minlength=54) / len(selected)
                shares[start:end, code] = frequency[categories[start:end]]
    for column, name in enumerate(EVENT_FEATURES):
        values = shares[:, column] if column < 11 else counts[:, column - 11]
        candidates = candidates.append_column(name, pa.array(values))
    return candidates
