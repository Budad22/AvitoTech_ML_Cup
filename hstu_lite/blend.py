"""Blend OOF predictions on val (per-feed rank), choose weights, write submission.  python blend.py"""
import os, glob, itertools, numpy as np, pandas as pd
from .common import Split, WORK, RAW, ndcg_full, feed_rank_pct, feed_rows, val_holdout_mask
from scripts.validate_submission import validate

def load_models():
    M = {}
    for f in sorted(glob.glob(os.path.join(WORK, "*_oof.npy"))):
        name = os.path.basename(f)[:-8]
        t = f.replace("_oof.npy", "_test.npy")
        if os.path.exists(t): M[name] = (np.load(f), np.load(t))
    for f in sorted(glob.glob(os.path.join(WORK, "nn1_s*_val.npy"))):     # NN-1 never saw val labels -> honest "OOF"
        M[os.path.basename(f)[:-8]] = (np.load(f), np.load(f.replace("_val.npy", "_test.npy")))
    return M

def main():
    Sv, Ste = Split("val"), Split("test"); y = np.asarray(Sv.target)
    M = load_models(); names = list(M)
    assert names, "no *_oof.npy in work/"
    R = {}
    for k, (o, t) in M.items():
        assert not np.isnan(o).any() and not np.isnan(t).any(), k
        R[k] = (feed_rank_pct(Sv.c_off, o), feed_rank_pct(Ste.c_off, t))
        print(f"{k:12s} OOF NDCG {ndcg_full(Sv.c_off, o, y)[0]:.5f}")
    for pre in ("nn1", "nn2"):                               # NN seeds -> one model (mean per-feed rank)
        nn = [k for k in names if k.startswith(pre + "_")]
        if len(nn) > 1:
            R[pre] = tuple(np.mean([R[k][i] for k in nn], 0) for i in (0, 1))
            M[pre] = R[pre]
            for k in nn: R.pop(k); M.pop(k)
            print(f"{pre} (seeds avg) OOF NDCG {ndcg_full(Sv.c_off, R[pre][0], y)[0]:.5f}")
    names = list(R)
    feeds = np.flatnonzero(~val_holdout_mask(Sv))
    rows, sizes = feed_rows(Sv.c_off, feeds)
    offsets = np.r_[0, sizes.cumsum()]
    labels = y[rows]
    models = np.stack([R[k][0][rows] for k in names])
    # Deterministic coordinate search on the simplex, starting from each
    # singleton. Avoid an exponential Cartesian grid over millions of rows.
    def score(weights):
        return ndcg_full(offsets, np.einsum("i,ij->j", weights, models), labels)[0]

    weights = np.eye(len(names), dtype=np.float32)
    initial = [score(w) for w in weights]
    w = weights[int(np.argmax(initial))].copy()
    sc = max(initial)
    for size in (.25, .10, .05):
        for _ in range(3):
            improved = False
            for i, j in itertools.permutations(range(len(names)), 2):
                if w[i] + 1e-6 < size:
                    continue
                proposal = w.copy()
                proposal[i] -= size
                proposal[j] += size
                value = score(proposal)
                if value > sc + 1e-9:
                    w, sc, improved = proposal, value, True
            if not improved:
                break
    best = (sc, w.tolist())
    sc, w = best; print("weights", dict(zip(names, w)), "blend selection NDCG (weights fitted on this partition)", round(sc, 5))
    # tie-break with the strongest single model (tiny epsilon) so equal blended ranks never collapse
    top = max(names, key=lambda k: ndcg_full(Sv.c_off, M[k][0], y)[0])
    score = sum(wi * R[k][1] for wi, k in zip(w, names)) + 1e-4 * R[top][1]
    feed = np.repeat(np.asarray(Ste.feed_id), np.diff(Ste.c_off))
    sub = pd.DataFrame({"feed_id": feed.astype(np.int32), "item_id": np.asarray(Ste.c_item).astype(np.int32),
                        "score": score.astype(np.float32)})
    tmpl = pd.read_parquet(os.path.join(RAW, "sample_submission.parquet"), columns=["feed_id", "item_id"])
    out = tmpl.merge(sub, on=["feed_id", "item_id"], how="left", validate="one_to_one")
    assert len(out) == len(tmpl) == len(sub), (len(out), len(tmpl), len(sub))
    assert out["score"].notna().all() and np.isfinite(out["score"]).all()
    out["feed_id"] = out["feed_id"].astype(np.int32); out["item_id"] = out["item_id"].astype(np.int32)
    out["score"] = out["score"].astype(np.float32)
    path = os.path.join(WORK, "submission.parquet"); out.to_parquet(path, index=False)
    print("saved", path, out.shape, out.dtypes.to_dict())

if __name__ == "__main__":
    main()
