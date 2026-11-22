import os, json, numpy as np

ROOT = os.environ.get("AVITO_ROOT", os.path.expanduser("~/avito"))
RAW  = os.environ.get("AVITO_RAW", os.path.join(ROOT, "data", "competition"))   # train/ val/ test/ sample_submission.parquet
PACK = os.path.join(ROOT, "packed")
WORK = os.path.join(ROOT, "work")
D_EMB, H_MAX = 128, 256

# name -> (dtype, width)   width>1 => 2D
FEED_COLS = {"feed_id": ("int32", 1), "user_id": ("int32", 1), "ts": ("int32", 1), "uemb": ("int8", D_EMB)}
CAND_COLS = {"c_item": ("int32", 1), "c_eidx": ("int32", 1), "c_cat": ("int8", 1), "c_f": ("float32", 5),
             "target": ("int8", 1), "pos": ("int16", 1)}
HIST_COLS = {"h_item": ("int32", 1), "h_eidx": ("int32", 1), "h_cat": ("int8", 1), "h_et": ("int8", 1), "h_es": ("int8", 1)}
ALL_COLS = {**FEED_COLS, **CAND_COLS, **HIST_COLS}


def _mm(path, dtype, width):
    n = os.path.getsize(path) // (np.dtype(dtype).itemsize * width)
    if n == 0:
        return np.zeros((0, width) if width > 1 else 0, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(n, width) if width > 1 else (n,))


class Split:
    """Packed split: flat arrays + offsets. Feeds i: candidates c_off[i]:c_off[i+1], history h_off[i]:h_off[i+1]."""
    def __init__(self, name, pack=None):
        p = os.path.join(pack or PACK, name)
        self.name, self.path = name, p
        for col, (dt, w) in ALL_COLS.items():
            f = os.path.join(p, col + ".bin")
            setattr(self, col, _mm(f, dt, w) if os.path.exists(f) else None)
        self.meta = json.loads(open(os.path.join(p, "meta.json"), encoding="utf-8").read())
        self.c_off = np.load(os.path.join(p, "c_off.npy"))
        self.h_off = np.load(os.path.join(p, "h_off.npy"))
        self.F = len(self.c_off) - 1
        self.N = int(self.c_off[-1])
        self.has_target = self.target is not None and len(self.target) == self.N
        self.has_pos = self.pos is not None and len(self.pos) == self.N

    def cand_feed(self):
        """feed index for every candidate row"""
        return np.repeat(np.arange(self.F, dtype=np.int32), np.diff(self.c_off))


def load_item_emb(split, in_ram=False):
    d = os.path.join(RAW, split, "item_embeddings")
    path = os.path.join(d, "data.memmap")
    size = os.path.getsize(path)
    assert size % D_EMB == 0, f"{path}: size {size} not divisible by {D_EMB}"
    meta = os.path.join(d, "data.meta")
    if os.path.exists(meta):
        print(f"[{split}] data.meta:", open(meta, "rb").read()[:200])
    E = np.memmap(path, dtype=np.int8, mode="r", shape=(size // D_EMB, D_EMB))
    return np.array(E) if in_ram else E


def gather_emb(E, idx):
    """idx int array, -1 => zero vector. Returns int8 [n,128] and present-mask."""
    out = np.zeros((len(idx), D_EMB), np.int8)
    if ((idx < -1) | (idx >= len(E))).any():
        raise ValueError("Embedding index out of bounds")
    m = idx >= 0
    if m.any():
        ii = idx[m]
        o = np.argsort(ii, kind="stable")          # sorted reads are much faster on memmap
        tmp = np.asarray(E[ii[o]])
        r = np.empty_like(tmp); r[o] = tmp
        out[m] = r
    return out, m


def ndcg_full(c_off, score, y):
    """Official-style: NDCG over full feed, equal feed weights, ties -> negatives first, feeds w/o positives = 1."""
    F = len(c_off) - 1
    sizes = np.diff(c_off)
    gid = np.repeat(np.arange(F), sizes)
    score = np.asarray(score, np.float64); y = np.asarray(y, np.float64)
    if not len(sizes) or np.any(sizes <= 0) or len(score) != c_off[-1] or len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError("Invalid feed offsets or predictions")
    order = np.lexsort((y, -score, gid))                 # by feed, score desc, negatives first on ties
    rank = np.arange(len(order)) - c_off[gid[order]]     # 0-based rank in feed
    disc = 1.0 / np.log2(rank + 2.0)
    dcg = np.bincount(gid[order], weights=y[order] * disc, minlength=F)
    npos = np.bincount(gid, weights=y, minlength=F).astype(np.int64)
    cum = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(sizes.max() if F else 1) + 2.0))])
    idcg = cum[npos]
    res = np.where(npos > 0, dcg / np.maximum(idcg, 1e-12), 1.0)
    return float(res.mean()), res


def average_ranks(values):
    """Zero-based ranks with equal values sharing the same rank."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(values)]
    counts = np.diff(boundaries)
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.repeat((boundaries[:-1] + boundaries[1:] - 1) / 2, counts)
    return result


def feed_rank_pct(c_off, score):
    """per-feed percentile rank in [0,1] (1 = best)"""
    if len(score) != int(c_off[-1]) or not np.isfinite(score).all():
        raise ValueError("Invalid rank input")
    ranks = np.empty(len(score), dtype=np.float32)
    for left, right in zip(c_off[:-1], c_off[1:]):
        ranks[left:right] = average_ranks(score[left:right]) / max(right - left - 1, 1)
    return ranks


def val_folds(S, k=5):
    """deterministic user-grouped folds for val feeds (shared by NN and GBDT)"""
    u = np.asarray(S.user_id).astype(np.uint64)
    return ((u * np.uint64(2654435761)) % np.uint64(2 ** 32) % np.uint64(k)).astype(np.int64)


def val_holdout_mask(S):
    """True for later val days (24-25 Sep, Moscow time): used as honest holdout"""
    day = (np.asarray(S.ts).astype(np.int64) + 3 * 3600) // 86400
    return day > day.min()


def feed_rows(c_off, feeds):
    """flat row indices of the given feeds (in order) and their sizes"""
    feeds = np.asarray(feeds)
    sizes = (c_off[feeds + 1] - c_off[feeds]).astype(np.int64)
    start = np.repeat(np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes)
    return np.repeat(c_off[feeds], sizes) + (np.arange(sizes.sum()) - start), sizes
