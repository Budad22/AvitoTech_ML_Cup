"""Label-free inference features, kept bit-compatible with the frozen training cache.

The parity test guards this adapter without invalidating existing data.py cache fingerprints.
"""
import numpy as np
import torch
from recsys_data.core import _flat
from ftt.data import normalize


def pack_features(batch):
    table = batch.candidate_table(include_target=False)
    counts = batch.candidate_counts.astype('int64')
    item, present = batch.candidate_vectors()
    user = batch.user_vectors()
    lengths = batch.features['history_length'].to_numpy().astype('int64')
    position = np.arange(50)[None]
    mask = position < lengths[:, None]
    indices = batch.history_offsets[:-1, None] + position
    def padded(name, fill):
        flat = _flat(batch.features, name).to_numpy()
        out = np.full(mask.shape, fill, dtype=flat.dtype)
        out[mask] = flat[indices[mask]]
        return out
    h, hp = batch.embeddings.lookup(padded('history__item_id__embedding_idx', -1))
    h = normalize(h)
    hmean = (h * hp[..., None]).sum(1) / np.maximum(hp.sum(1, keepdims=True), 1)
    return dict(counts=counts, numeric=np.column_stack([table[f'f{i}'].to_numpy() for i in range(1, 6)]),
                item=normalize(item), category=table['category_id'].to_numpy().astype('int64') + 128,
                user=normalize(user),
                history_mean=hmean.astype('float32'), history_vectors=h,
                history_mask=mask, history_category=padded('history__category_id', 0).astype('int64') + 128,
                history_event=padded('history__event_type', 0).astype('int64') + 128,
                history_source=padded('history__event_source', 0).astype('int64') + 128,
                flags=np.column_stack([present, np.repeat(np.linalg.norm(user, axis=1) == 0, counts),
                                       np.repeat(lengths == 0, counts), np.repeat(np.log1p(lengths), counts)]).astype('float32'),
                feed_id=batch.features['feed_id'].to_numpy(), user_id=batch.features['user_id'].to_numpy(),
                timestamp=batch.features['timestamp'].to_numpy(), item_id=table['item_id'].to_numpy())


def model_inputs(batch, device):
    arrays = pack_features(batch)
    counts = arrays["counts"].tolist()
    result = {key: torch.as_tensor(value, device=device) for key,value in arrays.items()}
    result["counts"] = counts
    for key in ("user", "history_mean"):
        result[key] = torch.repeat_interleave(result[key], torch.tensor(counts, device=device), dim=0)
    return result
