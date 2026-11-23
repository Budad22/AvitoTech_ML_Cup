"""Run feature generation, LightGBM stages and the rank blend (§6.4–6.6)."""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numba
import numpy as np

from mlp_next.supervisor import guard_lock
from mlp_trials.state import replace_atomic, write_json, write_text
from . import blend, feats, gbdt
from .common import Split, WORK
from .monitor import Progress


def generate_features(work, progress=None, train_feeds=150_000):
    for step, split in enumerate(('val', 'test', 'train'), 1):
        output = work / f'feats_{split}.parquet'
        if output.exists():
            continue
        if progress is not None:
            progress.publish('prepare', step=step - 1, steps=3, split=split)
        chosen = None
        if split == 'train':
            source = Split(split)
            chosen = np.sort(np.argsort(source.ts, kind='stable')[-train_feeds:])
        frame = feats.build(split, chosen)
        temporary = output.with_suffix('.tmp.parquet')
        frame.to_parquet(temporary, index=False)
        replace_atomic(temporary, output)
        print(f'{split}: {len(frame)} rows, {frame.shape[1]} columns', flush=True)


def copy_inputs(work, nn1, nn2):
    for seed in range(3):
        for root, stem, suffixes in ((nn1, f'nn1_s{seed}', ('val', 'test')),
                                     (nn2, f'nn2_s{seed}', ('oof', 'test'))):
            for suffix in suffixes:
                name = f'{stem}_{suffix}.npy'
                source = root / name
                destination = work / name
                if destination.exists():
                    continue
                temporary = destination.with_suffix('.tmp.npy')
                shutil.copyfile(source, temporary)
                replace_atomic(temporary, destination)


def training_progress(progress, variant):
    started = time.monotonic()
    next_update = started
    progress.publish('train', variant=variant)

    def report(env):
        nonlocal next_update
        now = time.monotonic()
        if now < next_update:
            return
        next_update = now + 30
        step = env.iteration + 1
        metrics = {name: value for _, name, value, *_ in (env.evaluation_result_list or ())}
        progress.publish('train', variant=variant, step=step, steps=env.end_iteration,
                         elapsed_seconds=now - started, **metrics)
        score = metrics.get('full_ndcg')
        detail = '' if score is None else f', validation NDCG {score:.7f}'
        print(f'{variant}: boosting round {step}/{env.end_iteration}{detail}', flush=True)

    return [report]


def run(work, nn1, nn2, progress, train_feeds=150_000, reference=None):
    for root in (nn1, nn2):
        report = json.loads((root / 'report.json').read_bytes())
        if report['status'] != 'completed':
            raise ValueError(f'Prerequisite has not completed: {root}')
    copy_inputs(work, nn1, nn2)
    generate_features(work, progress, train_feeds)
    results = {}

    def callbacks(variant):
        return training_progress(progress, variant)

    stages = [('gbdt_t', lambda: gbdt.run_t(callbacks)),
              ('meta', lambda: gbdt.run_meta(progress_callbacks=callbacks)),
              ('meta_xe', lambda: gbdt.run_meta('rank_xendcg', 'meta_xe', callbacks))]
    for name, train in stages:
        path = work / f'{name}-result.json'
        if not path.exists():
            write_json(path, train())
        results[name] = json.loads(path.read_bytes())
    progress.publish('evaluate', variant='blend')
    combined = blend.main(reference)
    return dict(ndcg={'NDCG · ансамбль · ранняя val': combined['selection_ndcg'],
                      'NDCG · ансамбль · поздняя OOF': combined['late_oof_ndcg'],
                      'NDCG · ансамбль · вся val': combined['blended_val_ndcg']},
                trees=results, blend=combined, train_feeds=train_feeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nn1', type=Path, required=True)
    parser.add_argument('--nn2', type=Path, required=True)
    parser.add_argument('--train-feeds', type=int, default=150_000)
    parser.add_argument('--reference-blend', type=Path, help='Reuse a completed blend and compare paired NDCG')
    args = parser.parse_args()
    if args.train_feeds < 1:
        parser.error('--train-feeds must be positive')
    numba.set_num_threads(min(4, numba.get_num_threads()))
    work = Path(WORK)
    if args.reference_blend and args.reference_blend.resolve() == work.resolve():
        parser.error('--reference-blend must be in a separate experiment directory')
    work.mkdir(parents=True, exist_ok=True)
    progress = Progress(work)
    with guard_lock(work / 'trees.lock'):
        configuration = work / 'trees-config.json'
        identity = dict(train_feeds=args.train_feeds, nn1=str(args.nn1.resolve()),
                        nn2=str(args.nn2.resolve()),
                        reference_blend=str(args.reference_blend.resolve()) if args.reference_blend else None)
        if configuration.exists():
            if json.loads(configuration.read_bytes()) != identity:
                raise ValueError('Tree configuration changed. Use a new experiment directory.')
        else:
            if args.train_feeds != 150_000 and (work / 'feats_train.parquet').exists():
                raise ValueError('Custom train size requires a fresh train feature output')
            write_json(configuration, identity)
        report = work / 'report.json'
        if report.exists():
            write_text(work / 'stage.txt', 'completed')
            return
        (work / 'failure.log').unlink(missing_ok=True)
        try:
            progress.publish('prepare')
            result = run(work, args.nn1, args.nn2, progress, args.train_feeds, args.reference_blend)
            write_json(report, dict(experiment=work.parent.name, status='completed', **result))
            progress.publish('completed')
        except Exception:
            write_text(work / 'failure.log', traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
