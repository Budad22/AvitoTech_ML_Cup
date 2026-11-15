"""Per-feed category frequencies; newest history event comes first."""
import numpy as np
import pyarrow as pa


def feature_names(windows):
    return [f"history_category_share_{window}" for window in windows] + (["history_is_empty"] if windows else [])


def add_history_categories(batch, candidates, windows):
    if not windows:
        return candidates
    categories = candidates["category_id"].to_numpy()
    shares = np.zeros((len(candidates), len(windows)), dtype=np.float32)
    for history, start, end in zip(batch.features["history__category_id"].to_pylist(),
                                   batch.candidate_offsets[:-1], batch.candidate_offsets[1:]):
        for column, window in enumerate(windows):
            selected = history[:window]
            if selected:
                frequencies = np.bincount(selected, minlength=54) / len(selected)
                shares[start:end, column] = frequencies[categories[start:end]]
    for column, name in enumerate(feature_names(windows)[:-1]):
        candidates = candidates.append_column(name, pa.array(shares[:, column]))
    empty = np.repeat(batch.features["history_is_empty"].to_numpy(), batch.candidate_counts)
    return candidates.append_column("history_is_empty", pa.array(empty.astype(np.float32)))
