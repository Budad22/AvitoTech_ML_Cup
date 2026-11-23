"""LightGBM LambdaMART.
  python gbdt.py t      -> GBDT-T on train subset (pos as feature, set to constant at inference)
  python gbdt.py meta   -> meta-GBDT on randomized val, 5-fold user-grouped, uses NN-1 + GBDT-T scores as features
"""
import os, sys, glob, json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from .common import Split, WORK, ndcg_full, feed_rank_pct, val_folds, val_holdout_mask, feed_rows
from mlp_trials.state import replace_atomic

BASE = dict(objective="lambdarank", metric="None", lambdarank_truncation_level=100000,
            learning_rate=0.03, num_leaves=63, min_data_in_leaf=200, min_sum_hessian_in_leaf=1e-3,
            feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255,
            num_threads=min(os.cpu_count() or 1, 6), verbose=-1, seed=0)
DROP = ["_row", "_feed", "_y"]


def groups(df):
    return df.groupby("_feed", sort=False).size().to_numpy()


def full_ndcg(prediction, dataset):
    offsets = np.r_[0, np.cumsum(dataset.get_group())]
    return "full_ndcg", ndcg_full(offsets, prediction, dataset.get_label())[0], True


def ds(df, cols):
    return lgb.Dataset(df[cols].to_numpy(np.float32), label=df["_y"].clip(lower=0).to_numpy(), group=groups(df),
                       feature_name=cols, categorical_feature=["cat"], free_raw_data=True)


def save_scores(name, scores):
    if not np.isfinite(scores).all():
        raise ValueError(f'Incomplete predictions for {name}')
    path = Path(WORK) / (name + '.npy')
    tmp = path.with_suffix('.tmp.npy')
    np.save(tmp, scores, allow_pickle=False)
    replace_atomic(tmp, path)


def nn1_feature(split, rows):
    fs = sorted(glob.glob(os.path.join(WORK, f"nn1_s*_{split}.npy")))
    if not fs: return None
    S = Split(split); acc = 0
    for f in fs:
        p = np.load(f).astype(np.float64)
        acc = acc + (p - np.nanmean(p)) / (np.nanstd(p) + 1e-9)
    return (acc / len(fs)).astype(np.float32)[rows]


def add_model_feats(df, split):
    S = Split(split)
    v = nn1_feature(split, df["_row"].to_numpy())
    if v is not None:
        df["nn1"] = v
        sub_off = np.concatenate([[0], np.cumsum(groups(df))])
        df["nn1_rk"] = feed_rank_pct(sub_off, v)
    p = os.path.join(WORK, f"gbdt_t_{split}.npy")
    if os.path.exists(p):
        g = np.load(p)[df["_row"].to_numpy()]
        df["gbdt_t"] = g
        df["gbdt_t_rk"] = feed_rank_pct(np.concatenate([[0], np.cumsum(groups(df))]), g)
    return df


def run_t(progress_callbacks=lambda _: []):
    tr = pd.read_parquet(os.path.join(WORK, "feats_train.parquet"))
    va = pd.read_parquet(os.path.join(WORK, "feats_val.parquet"))
    te = pd.read_parquet(os.path.join(WORK, "feats_test.parquet"))
    cols = [c for c in tr.columns if c not in DROP]                    # includes "pos"
    Sv = Split("val"); hold = val_holdout_mask(Sv)[va["_feed"].to_numpy()]
    params = dict(BASE, learning_rate=0.05, num_leaves=127, min_data_in_leaf=500)
    va["pos"] = 5.0
    m = lgb.train(params, ds(tr, cols), 3000, valid_sets=[ds(va[~hold], cols)], feval=full_ndcg,
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
                  + progress_callbacks("gbdt_t"))
    vh = va[hold].copy(); off = np.concatenate([[0], np.cumsum(groups(vh))]); best = None
    for P in (0.0, 5.0, 20.0, 50.0):                                    # constant position at inference
        vh["pos"] = P
        sc = ndcg_full(off, m.predict(vh[cols].to_numpy(np.float32), num_iteration=m.best_iteration), vh["_y"].to_numpy())[0]
        print(f"pos={P}: late val tuning NDCG {sc:.5f}")
        if best is None or sc > best[0]: best = (sc, P)
    sc, P = best; print("chosen pos", P, sc, "iters", m.best_iteration)
    m.save_model(os.path.join(WORK, 'gbdt_t.txt'), num_iteration=m.best_iteration)
    val_score = None
    for split, df in (("val", va), ("test", te)):
        df["pos"] = P; out = np.full(Split(split).N, np.nan, np.float32)
        out[df["_row"].to_numpy()] = m.predict(df[cols].to_numpy(np.float32), num_iteration=m.best_iteration)
        save_scores(f'gbdt_t_{split}', out)
        if split == 'val':
            val_score = ndcg_full(Sv.c_off, out, Sv.target)[0]
    return dict(position=P, best_iteration=m.best_iteration, late_position_tuning_ndcg=sc,
                best_early_ndcg=m.best_score["valid_0"]["full_ndcg"], full_val_ndcg=val_score)


def run_meta(obj="lambdarank", tag="meta", progress_callbacks=lambda _: []):
    va = add_model_feats(pd.read_parquet(os.path.join(WORK, "feats_val.parquet")), "val")
    te = add_model_feats(pd.read_parquet(os.path.join(WORK, "feats_test.parquet")), "test")
    cols = [c for c in va.columns if c not in DROP + ["pos"]]
    Sv, Ste = Split("val"), Split("test")
    folds = val_folds(Sv)[va["_feed"].to_numpy()]
    params = dict(BASE, objective=obj)
    # Tune the number of trees by user folds, then freeze it for all OOF refits.
    iters = []
    for k in range(5):
        learn = folds != k
        check = folds == k
        model = lgb.train(params, ds(va[learn], cols), 5000, valid_sets=[ds(va[check], cols)],
                          feval=full_ndcg, callbacks=[lgb.early_stopping(200, verbose=False)]
                          + progress_callbacks(f"{tag} ? tune fold {k + 1}/5"))
        iters.append(model.best_iteration)
        print(f"{tag} tuning fold {k + 1}/5: best iteration {model.best_iteration}, "
              f"NDCG {model.best_score['valid_0']['full_ndcg']:.7f}", flush=True)
    n_iter = max(1, int(np.median(iters) * 1.1))
    print(f"{tag}: selected {n_iter} rounds from {iters}", flush=True)
    # pass 2: fixed rounds, honest OOF + test average
    oof = np.full(Sv.N, np.nan, np.float32); tp = np.zeros(len(te))
    for k in range(5):
        m = lgb.train(params, ds(va[folds != k], cols), n_iter,
                      callbacks=progress_callbacks(f"{tag} ? OOF fold {k + 1}/5"))
        idx = folds == k
        local_off = np.r_[0, np.cumsum(groups(va.loc[idx]))]
        oof[va["_row"].to_numpy()[idx]] = feed_rank_pct(local_off, m.predict(va.loc[idx, cols].to_numpy(np.float32)))
        p = m.predict(te[cols].to_numpy(np.float32)); tp += feed_rank_pct(np.r_[0, np.cumsum(groups(te))], p) / 5
        if k == 0:
            imp = pd.Series(m.feature_importance("gain"), cols).sort_values(ascending=False)
            print(imp.head(25).to_string())
    oof_score = ndcg_full(Sv.c_off, oof, np.asarray(Sv.target))[0]
    print(f"{tag} OOF NDCG:", oof_score)
    out = np.full(Ste.N, np.nan, np.float32); out[te["_row"].to_numpy()] = tp
    save_scores(f'{tag}_oof', oof)
    save_scores(f'{tag}_test', out)
    return dict(objective=obj, iterations=n_iter, tuning_iterations=iters, oof_ndcg=oof_score)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "t": run_t()
    elif cmd == "meta": run_meta("lambdarank", "meta")
    elif cmd == "meta_xe": run_meta("rank_xendcg", "meta_xe")
