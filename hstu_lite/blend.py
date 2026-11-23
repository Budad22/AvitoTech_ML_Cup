"""Blend OOF predictions on val (per-feed rank), choose weights, write submission.  python blend.py"""
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from pathlib import Path
from .common import Split, WORK, RAW, ndcg_full, feed_rank_pct, feed_rows, paired_result, val_holdout_mask
from scripts.validate_submission import validate
from mlp_trials.state import replace_atomic, write_json

def load_models(work=WORK):
    M = {}
    for f in sorted(glob.glob(os.path.join(work, "*_oof.npy"))):
        name = os.path.basename(f)[:-8]
        t = f.replace("_oof.npy", "_test.npy")
        if os.path.exists(t): M[name] = (np.load(f), np.load(t))
    for f in sorted(glob.glob(os.path.join(work, "nn1_s*_val.npy"))):     # NN-1 never saw val labels -> honest "OOF"
        M[os.path.basename(f)[:-8]] = (np.load(f), np.load(f.replace("_val.npy", "_test.npy")))
    if Path(work, 'gbdt_t_val.npy').exists() and Path(work, 'gbdt_t_test.npy').exists():
        M['gbdt_t'] = (np.load(Path(work, 'gbdt_t_val.npy')), np.load(Path(work, 'gbdt_t_test.npy')))
    return M

def write_submission(split, scores, path):
    feed = np.repeat(np.asarray(split.feed_id), np.diff(split.c_off))
    sub = pd.DataFrame({"feed_id": feed.astype(np.int32), "item_id": np.asarray(split.c_item).astype(np.int32),
                        "score": scores.astype(np.float32)})
    tmpl = pd.read_parquet(os.path.join(RAW, "sample_submission.parquet"), columns=["feed_id", "item_id"])
    out = tmpl.merge(sub, on=["feed_id", "item_id"], how="left", validate="one_to_one")
    assert len(out) == len(tmpl) == len(sub), (len(out), len(tmpl), len(sub))
    assert out["score"].notna().all() and np.isfinite(out["score"]).all()
    out["feed_id"] = out["feed_id"].astype(np.int32); out["item_id"] = out["item_id"].astype(np.int32)
    out["score"] = out["score"].astype(np.float32)
    temporary = path.with_suffix('.tmp.parquet')
    out.to_parquet(temporary, index=False)
    audit = validate(temporary, Path(RAW, 'sample_submission.parquet'))
    replace_atomic(temporary, path)
    audit['file'] = str(path.resolve())
    return audit


def rank_models(models, val, test):
    if not models:
        raise ValueError('No model predictions')
    ranked, metrics = {}, {}
    for name, (before, after) in models.items():
        if before.shape != (val.N,) or after.shape != (test.N,) or not np.isfinite(after).all():
            raise ValueError(f'Invalid predictions for {name}')
        ranked[name] = (feed_rank_pct(val.c_off, before), feed_rank_pct(test.c_off, after))
        metrics[name] = ndcg_full(val.c_off, before, val.target)[0]
        print(f'{name:12s} full val NDCG {metrics[name]:.7f}')
    for prefix in ('nn1', 'nn2'):
        seeds = [name for name in ranked if name.startswith(prefix + '_')]
        if len(seeds) > 1:
            ranked[prefix] = tuple(np.mean([ranked[name][i] for name in seeds], axis=0) for i in (0, 1))
            for name in seeds:
                del ranked[name]
            metrics[prefix] = ndcg_full(val.c_off, ranked[prefix][0], val.target)[0]
            print(f'{prefix} (seeds avg) full val NDCG {metrics[prefix]:.7f}')
    return ranked, metrics


def fit_weights(val, ranked, early):
    rows, sizes = feed_rows(val.c_off, np.flatnonzero(early))
    offsets = np.r_[0, sizes.cumsum()]
    labels = val.target[rows]
    names = list(ranked)
    scores = np.stack([ranked[name][0][rows] for name in names])

    def score(weights):
        return ndcg_full(offsets, np.einsum('i,ij->j', weights, scores), labels)[0]

    weights = np.eye(len(names), dtype=np.float32)
    initial = [score(w) for w in weights]
    top = int(np.argmax(initial))
    w = weights[top].copy()
    best = initial[top]
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
                if value > best + 1e-9:
                    w, best, improved = proposal, value, True
            if not improved:
                break
    return dict(zip(names, w.tolist())), names[top]


def mix(ranked, weights, tiebreak, index):
    if set(ranked) != set(weights):
        raise ValueError('Ensemble components do not match the weights')
    return sum(weight * ranked[name][index] for name, weight in weights.items()) + 1e-4 * ranked[tiebreak][index]


def main(reference=None):
    val, test = Split('val'), Split('test')
    ranked, components = rank_models(load_models(WORK), val, test)
    late = val_holdout_mask(val)
    scope = ('OOF for NN-2/meta, val for NN-1/GBDT-T. '
             'NN-2 epochs and GBDT-T position use late feeds, and meta rounds use val folds.')
    comparison = None
    if reference is None:
        weights, top = fit_weights(val, ranked, ~late)
        scope += ' Early feeds fit blend weights.'
    else:
        baseline = json.loads((reference / 'blend.json').read_bytes())
        weights, top = baseline['weights'], baseline['tiebreak']
        before, _ = rank_models(load_models(reference), val, test)
        before_ndcg, before_feeds = ndcg_full(val.c_off, mix(before, weights, top, 0), val.target)
        if not np.isclose(before_ndcg, baseline['blended_val_ndcg'], atol=1e-12, rtol=0):
            raise ValueError('Frozen weights do not reproduce the reference NDCG')
        scope += ' Ensemble weights and tiebreak reuse the reference.'

    full, per_feed = ndcg_full(val.c_off, mix(ranked, weights, top, 0), val.target)
    if reference is not None:
        masks = dict(early=~late, late=late, full=np.ones(val.F, dtype=bool))
        comparison = {name: paired_result(val.user_id[mask], before_feeds[mask], per_feed[mask])
                      for name, mask in masks.items()}
    promote = comparison is None or comparison['early']['ci95'][0] > 0
    audit = None
    if promote:
        path = Path(WORK, 'submission.parquet')
        audit = write_submission(test, mix(ranked, weights, top, 1), path)
        print('saved', path, audit['rows'], 'rows')
    result = dict(weights=weights, tiebreak=top, component_val_ndcg=components,
                  selection_ndcg=float(per_feed[~late].mean()), late_oof_ndcg=float(per_feed[late].mean()),
                  blended_val_ndcg=full, audit=audit, scope=scope)
    if comparison is not None:
        result.update(reference=str(reference), comparisons=comparison,
                      decision='promote' if promote else 'keep_reference')
    write_json(Path(WORK, 'blend.json'), result)
    print(f'Blend full val NDCG: {full:.8f}')
    return result


if __name__ == '__main__':
    main()
