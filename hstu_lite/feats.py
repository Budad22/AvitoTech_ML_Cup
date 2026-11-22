"""Hand-crafted ranking features for GBDT.  python feats.py --split val|test|train [--max-feeds 150000]"""
import os, argparse, numpy as np, pandas as pd
from .common import Split, load_item_emb, feed_rows, WORK, H_MAX
try:
    from numba import njit, prange
except ImportError:                       # slow pure-python fallback (only for tests)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
    prange = range

NET = 10
NAMES = ([f"f{i}" for i in range(1, 6)] + [f"f{i}_rk" for i in range(1, 6)] + [f"f{i}_z" for i in range(1, 6)] +
         ["log_ncand", "hist_len", "cos_uc", "dot_uc", "cos_uc_rk", "cos_uc_dm",
          "h_max", "h_mean", "h_max16", "h_mean16", "h_decay"] + [f"h_max_et{i}" for i in range(1, NET + 1)] +
         ["m_cnt", "m_first"] + [f"m_et{i}" for i in range(1, NET + 1)] +
         ["cat_share", "cat_cnt16", "cat_first"] + [f"cat_et{i}" for i in range(1, NET + 1)] +
         ["feed_cat_share", "cos_centroid", "c_miss", "u_zero", "h_empty", "cat"])
K = len(NAMES)


@njit(cache=True)
def _rankpct(x):
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, np.float32)
    left = 0
    while left < n:
        right = left + 1
        while right < n and x[order[right]] == x[order[left]]:
            right += 1
        ranks[order[left:right]] = (left + right - 1) / (2 * max(n - 1, 1))
        left = right
    return ranks



@njit(parallel=True, cache=True)
def kernel(feeds, out_off, c_off, h_off, c_item, c_eidx, c_cat, c_f, h_item, h_eidx, h_cat, h_et, uemb, E, out):
    for t in prange(len(feeds)):
        f = feeds[t]; c0 = c_off[f]; C = c_off[f + 1] - c0
        h0 = h_off[f]; H = min(h_off[f + 1] - h0, H_MAX); o0 = out_off[t]
        D = E.shape[1]
        u = uemb[f].astype(np.float32); un = np.sqrt((u * u).sum())
        hv = np.zeros((H, D), np.float32); hp = np.zeros(H, np.bool_)
        for j in range(H):
            e = h_eidx[h0 + j]
            if e >= 0:
                v = E[e].astype(np.float32); nv = np.sqrt((v * v).sum())
                if nv > 0: hv[j] = v / nv; hp[j] = True
        cv = np.zeros((C, D), np.float32); cp = np.zeros(C, np.bool_)
        for i in range(C):
            e = c_eidx[c0 + i]
            if e >= 0:
                v = E[e].astype(np.float32); nv = np.sqrt((v * v).sum())
                if nv > 0: cv[i] = v / nv; cp[i] = True
        cent = np.zeros(D, np.float32)
        for i in range(C): cent += cv[i]
        cn = np.sqrt((cent * cent).sum())
        if cn > 0: cent /= cn
        # f stats
        for k in range(5):
            x = c_f[c0:c0 + C, k].astype(np.float32)
            r = _rankpct(x); mu = x.mean(); sd = x.std() + 1e-6
            for i in range(C):
                out[o0 + i, k] = x[i]; out[o0 + i, 5 + k] = r[i]; out[o0 + i, 10 + k] = (x[i] - mu) / sd
        cos_uc = np.zeros(C, np.float32)
        for i in range(C):
            d = 0.0
            for q in range(D): d += cv[i, q] * u[q]
            cos_uc[i] = d / un if (un > 0 and cp[i]) else 0.0
            out[o0 + i, 18] = d
        r = _rankpct(cos_uc); m = cos_uc.mean()
        cat_cnt = np.zeros(64, np.float32)
        for i in range(C): cat_cnt[c_cat[c0 + i]] += 1
        for i in range(C):
            row = o0 + i
            out[row, 15] = np.log1p(C); out[row, 16] = H
            out[row, 17] = cos_uc[i]; out[row, 19] = r[i]; out[row, 20] = cos_uc[i] - m
            hmax = -2.0; hsum = 0.0; hn = 0; hmax16 = -2.0; hsum16 = 0.0; hn16 = 0; dsum = 0.0; dw = 0.0
            for q in range(NET): out[row, 26 + q] = -2.0
            mcnt = 0.0; mfirst = H_MAX + 1.0; ccnt = 0.0; ccnt16 = 0.0; cfirst = H_MAX + 1.0
            it = c_item[c0 + i]; ct = c_cat[c0 + i]
            for j in range(H):
                et = h_et[h0 + j] - 1
                if et < 0 or et >= NET: et = 0
                if cp[i] and hp[j]:
                    s = 0.0
                    for q in range(D): s += cv[i, q] * hv[j, q]
                    if s > hmax: hmax = s
                    hsum += s; hn += 1
                    w = np.exp(-j / 32.0); dsum += w * s; dw += w
                    if j < 16:
                        if s > hmax16: hmax16 = s
                        hsum16 += s; hn16 += 1
                    if s > out[row, 26 + et]: out[row, 26 + et] = s
                if h_item[h0 + j] == it:
                    mcnt += 1; out[row, 38 + et] += 1
                    if j < mfirst: mfirst = j
                if h_cat[h0 + j] == ct:
                    ccnt += 1; out[row, 51 + et] += 1
                    if j < 16: ccnt16 += 1
                    if j < cfirst: cfirst = j
            out[row, 21] = hmax; out[row, 22] = hsum / hn if hn > 0 else -2.0
            out[row, 23] = hmax16; out[row, 24] = hsum16 / hn16 if hn16 > 0 else -2.0
            out[row, 25] = dsum / dw if dw > 0 else -2.0
            out[row, 36] = mcnt; out[row, 37] = mfirst
            out[row, 48] = ccnt / H if H > 0 else -1.0; out[row, 49] = ccnt16; out[row, 50] = cfirst
            out[row, 61] = cat_cnt[ct] / C
            s = 0.0
            for q in range(D): s += cv[i, q] * cent[q]
            out[row, 62] = s
            out[row, 63] = 0.0 if cp[i] else 1.0
            out[row, 64] = 1.0 if un == 0 else 0.0
            out[row, 65] = 1.0 if H == 0 else 0.0
            out[row, 66] = ct


def cross_feed_feats(S, feeds, future):
    """transductive features inside the given feed set (no labels): item popularity, prior/later exposure of the same user."""
    rows, sizes = feed_rows(S.c_off, feeds)
    cf = np.repeat(feeds, sizes)
    uid, ts = np.asarray(S.user_id), np.asarray(S.ts)
    df = pd.DataFrame({"u": uid[cf], "it": np.asarray(S.c_item)[rows], "ts": ts[cf], "f": cf})
    days = max((ts[feeds].max() - ts[feeds].min()) / 86400.0, 1.0)
    per_day = len(feeds) / days
    pop_c = df.groupby("it")["f"].transform("size").to_numpy() / per_day
    hrows, hsizes = feed_rows(S.h_off, feeds)
    h_it = np.asarray(S.h_item)[hrows]
    hcnt = pd.Series(h_it).value_counts()
    pop_h = df["it"].map(hcnt).fillna(0).to_numpy() / per_day
    by_time = df.groupby(["u", "it", "ts"], sort=True)["f"].nunique().rename("exposures").reset_index()
    by_time["prior"] = by_time.groupby(["u", "it"])["exposures"].cumsum() - by_time["exposures"]
    prior = df.merge(by_time[["u", "it", "ts", "prior"]], on=["u", "it", "ts"], how="left", validate="many_to_one")["prior"].to_numpy(np.float32)
    res = {"pop_cand": np.log1p(pop_c).astype(np.float32), "pop_hist": np.log1p(pop_h).astype(np.float32), "prior_imp": prior}
    if future:
        raise ValueError("Future histories are disabled for this protocol")
    return res


def build(split, feeds=None, future=False, E=None):
    S = Split(split)
    if E is None: E = load_item_emb(split)
    feeds = np.arange(S.F) if feeds is None else np.asarray(feeds)
    sizes = (S.c_off[feeds + 1] - S.c_off[feeds]).astype(np.int64)
    out_off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    X = np.zeros((out_off[-1], K), np.float32)
    E_arr = E.view(np.ndarray) if isinstance(E, np.memmap) else E
    kernel(feeds, out_off, S.c_off, S.h_off, np.asarray(S.c_item), np.asarray(S.c_eidx), np.asarray(S.c_cat),
           np.asarray(S.c_f), np.asarray(S.h_item), np.asarray(S.h_eidx), np.asarray(S.h_cat), np.asarray(S.h_et),
           np.asarray(S.uemb), E_arr, X)
    df = pd.DataFrame(X, columns=NAMES)
    for k, v in cross_feed_feats(S, feeds, future).items(): df[k] = v
    rows, _ = feed_rows(S.c_off, feeds)
    df["_row"] = rows; df["_feed"] = np.repeat(feeds, sizes)
    df["_y"] = np.asarray(S.target)[rows] if S.has_target else -1
    if S.has_pos: df["pos"] = np.asarray(S.pos)[rows].astype(np.float32)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-feeds", type=int, default=0, help="train: use N most recent feeds")
    ap.add_argument("--future", type=int, default=0)
    a = ap.parse_args()
    feeds = None
    if a.max_feeds:
        S = Split(a.split); feeds = np.sort(np.argsort(np.asarray(S.ts), kind="stable")[-a.max_feeds:])
    df = build(a.split, feeds, bool(a.future))
    os.makedirs(WORK, exist_ok=True)
    df.to_parquet(os.path.join(WORK, f"feats_{a.split}.parquet"), index=False)
    print(df.shape, "->", os.path.join(WORK, f"feats_{a.split}.parquet"))
