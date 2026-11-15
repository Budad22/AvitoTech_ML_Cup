"""Midrank percentiles within a feed; equal values always get equal percentiles."""
import numpy as np
import pyarrow as pa

FIELDS = [f"f{i}" for i in range(1, 6)]
CONTEXT_FEATURES = [f"{name}_feed_percentile" for name in FIELDS]


def add_feed_percentiles(batch, candidates):
    for name, output_name in zip(FIELDS, CONTEXT_FEATURES):
        source = candidates[name].to_numpy()
        percentiles = np.empty(len(candidates), dtype=np.float32)
        for start, end in zip(batch.candidate_offsets[:-1], batch.candidate_offsets[1:]):
            _, inverse, counts = np.unique(source[start:end], return_inverse=True, return_counts=True)
            midranks = (np.cumsum(counts) - counts * .5) / (end - start)
            percentiles[start:end] = midranks[inverse]
        candidates = candidates.append_column(output_name, pa.array(percentiles))
    return candidates
