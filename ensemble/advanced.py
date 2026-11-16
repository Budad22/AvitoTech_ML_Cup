"""Sequential val-only checks of gating, seed bagging and a meta correction."""
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blend import (ROOT, WEIGHTS, MIN_DELTA, align, normalize, mixture, per_feed,
                   bootstrap, dump, ranking_metrics, prediction_table, sha256)
from catboost import CatBoostRanker, Pool, __version__ as catboost_version

SOURCE = ROOT.parent / 'AvitoTech_ML_Cup'
OUTPUT = ROOT / 'runs/ensemble/EXP-ENS-003-005'
BOUNDARY = 1790249984
META_BOUNDARY = 1790216704
EXTRA_FEATURES = ['category_id', 'history_category_share_16',
                  'history_category_share_64', 'history_category_share_256',
                  'user_item_cosine', 'user_item_cosine_valid',
                  'history_centroid_cosine_64', 'history_centroid_cosine_valid_64']
PARAMS = dict(iterations=150, depth=3, learning_rate=.03, l2_leaf_reg=20,
              loss_function='YetiRank', eval_metric='NDCG:type=Base',
              task_type='CPU', boosting_type='Plain', random_seed=42,
              thread_count=8, early_stopping_rounds=20, one_hot_max_size=64,
              allow_writing_files=False)


def add_correction(baseline, correction):
    baseline = np.asarray(baseline, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if baseline.shape != correction.shape or not np.isfinite(correction).all():
        raise ValueError('Invalid meta correction')
    return np.asarray(baseline + correction, dtype=np.float32)


def gated_scores(cat, mlp, zcat, zmlp, counts, short, weights):
    """Weights are per feed; endpoints preserve raw parent predictions."""
    short_rows = np.repeat(short, counts)
    a = mixture(cat, mlp, zcat, zmlp, weights[0])
    b = mixture(cat, mlp, zcat, zmlp, weights[1])
    return np.where(short_rows, a, b).astype(np.float32)


def pool(frame, truth, baseline=None):
    return Pool(frame, label=truth['target'].to_numpy(),
                group_id=truth['feed_id'].to_numpy(),
                cat_features=['category_id'], baseline=baseline)


def metric(truth, scores, row_mask):
    return ranking_metrics(truth['feed_id'].to_numpy()[row_mask],
                           truth['target'].to_numpy()[row_mask], scores[row_mask])


def validate_inputs(inputs, cache):
    provenance = json.loads((inputs / 'provenance.json').read_text())
    for name, digest in provenance['files'].items():
        assert sha256(inputs / name) == digest, name
    cache_manifest = json.loads((cache / 'manifest.json').read_text())
    for name in ['val.parquet', 'val_metadata.parquet']:
        assert sha256(cache / name) == cache_manifest['files'][name], name
    reports = [json.loads((inputs / name).read_text()) for name in
               ['catboost_result.json', 'catboost_seed17_result.json']]
    a, b = reports
    assert {k: v for k, v in a['config'].items() if k != 'random_seed'} == {
        k: v for k, v in b['config'].items() if k != 'random_seed'}
    assert [a['config']['random_seed'], b['config']['random_seed']] == [42, 17]
    for key in ['features', 'train', 'prepared_manifest_sha256']:
        assert a[key] == b[key], key
    modules = ['catboost/baseline.py', 'catboost/history_categories.py',
               'catboost/cosine_features.py', 'catboost/feed_context.py',
               'catboost/history_signals.py', 'catboost/history_neighbors.py',
               'recsys_data/core.py']
    archives = []
    for report in reports:
        archive = SOURCE / report['source_snapshot']
        assert sha256(archive) == report['source_snapshot_sha256']
        archives.append(zipfile.ZipFile(archive))
    try:
        for name in modules:
            assert archives[0].read(name) == archives[1].read(name), name
    finally:
        for archive in archives:
            archive.close()
    for name, report in zip(['catboost_val_predictions.parquet',
                             'catboost_seed17_val_predictions.parquet'], reports):
        assert sha256(inputs / name) == report['artifacts']['val_predictions.parquet']
    return {'provenance': provenance, 'seed17_report_sha256': sha256(inputs / 'catboost_seed17_result.json'),
            'seed17_predictions_sha256': b['artifacts']['val_predictions.parquet'],
            'seed_feature_sources_equal': modules,
            'cache_manifest_sha256': sha256(cache / 'manifest.json')}


def make_features(cached, meta, counts, cat, mlp, current):
    zcat, zmlp = (normalize(s, counts, 'zscore') for s in (cat, mlp))
    rcat, rmlp = (normalize(s, counts, 'rank') for s in (cat, mlp))
    frame = cached.select(EXTRA_FEATURES).to_pandas()
    frame['category_id'] = frame['category_id'].astype(np.int32)
    frame['cat_z'] = zcat.astype(np.float32)
    frame['mlp_z'] = zmlp.astype(np.float32)
    frame['score_difference'] = (zcat - zmlp).astype(np.float32)
    frame['absolute_score_difference'] = np.abs(zcat - zmlp).astype(np.float32)
    frame['cat_rank'] = rcat.astype(np.float32)
    frame['mlp_rank'] = rmlp.astype(np.float32)
    frame['rank_difference'] = (rcat - rmlp).astype(np.float32)
    frame['current_score'] = current
    frame['log_history_length'] = np.repeat(np.log1p(meta['history_length'].to_numpy()), counts).astype(np.float32)
    frame['log_candidate_count'] = np.repeat(np.log1p(counts), counts).astype(np.float32)
    frame['user_embedding_is_zero'] = np.repeat(meta['user_embedding_is_zero'].to_numpy(), counts).astype(np.float32)
    assert not set(frame) & {'target', 'feed_id', 'item_id', 'timestamp', 'pos'}
    assert np.isfinite(frame.to_numpy()).all()
    return frame


def run():
    if OUTPUT.exists():
        raise FileExistsError('Output already exists; reserve new run IDs before rerunning')
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    inputs = ROOT / 'runs/ensemble/inputs'
    cache = SOURCE / 'artifacts/catboost-snowball-v1'
    provenance = validate_inputs(inputs, cache)
    cached = pq.read_table(cache / 'val.parquet', columns=['feed_id', 'item_id', 'target', *EXTRA_FEATURES])
    truth = cached.select(['feed_id', 'item_id', 'target'])
    meta = pq.read_table(cache / 'val_metadata.parquet')
    counts = meta['candidate_count'].to_numpy()
    assert np.array_equal(truth['feed_id'].to_numpy(), np.repeat(meta['feed_id'].to_numpy(), counts))
    ts = meta['timestamp'].to_numpy()
    selection, fit, tune = ts < BOUNDARY, ts < META_BOUNDARY, (ts >= META_BOUNDARY) & (ts < BOUNDARY)
    assert (int(selection.sum()), int(fit.sum()), int(tune.sum())) == (11534, 8076, 3458)
    masks = {'selection': selection, 'meta_fit': fit, 'meta_tune': tune, 'late_control': ~selection,
             'full_diagnostic': np.ones(len(counts), dtype=bool)}
    rows = {key: np.repeat(value, counts) for key, value in masks.items()}
    cat42, cat17, mlp = [align(truth, pq.read_table(inputs / name)) for name in
                        ['catboost_val_predictions.parquet', 'catboost_seed17_val_predictions.parquet',
                         'mlp_val_predictions.parquet']]
    z42, z17, zm = [normalize(s, counts, 'zscore') for s in (cat42, cat17, mlp)]
    short = meta['history_length'].to_numpy() < 64
    weights = [.25, .25]
    current = gated_scores(cat42, mlp, z42, zm, counts, short, weights)
    initial_report = json.loads((ROOT / 'ensemble/results/EXP-ENS-001-002.json').read_text())
    initial_path = ROOT / 'runs/ensemble/EXP-ENS-001-002/val_predictions.parquet'
    assert sha256(initial_path) == initial_report['artifacts']['val_predictions.parquet']
    np.testing.assert_array_equal(current, align(truth, pq.read_table(initial_path)))
    candidates = []

    # EXP-ENS-003: select only within the previously fixed two selection groups.
    grid = []
    chosen_weights = []
    for group, group_mask in [('history_lt64', short), ('history_ge64', ~short)]:
        group_rows = np.repeat(selection & group_mask, counts)
        group_grid = []
        for weight in WEIGHTS:
            scores = mixture(cat42, mlp, z42, zm, weight)
            value = metric(truth, scores, group_rows)['ndcg']
            group_grid.append({'group': group, 'weight': weight, 'ndcg': value})
        winner = max(group_grid, key=lambda row: (row['ndcg'], row['weight'] == .25, -row['weight']))
        chosen_weights.append(winner['weight'])
        grid.extend(group_grid)
    candidate = gated_scores(cat42, mlp, z42, zm, counts, short, chosen_weights)
    before = metric(truth, current, rows['selection'])['ndcg']
    after = metric(truth, candidate, rows['selection'])['ndcg']
    accepted = after - before >= MIN_DELTA
    decision = {'experiment': 'EXP-ENS-003-r01', 'weights_short_long': chosen_weights,
                'threshold_history_length': 64, 'grid': grid, 'baseline_selection': before,
                'candidate_selection': after, 'delta': after - before, 'accepted': accepted,
                'minimum_delta': MIN_DELTA}
    dump(OUTPUT / 'EXP-ENS-003-frozen.json', decision)
    print(json.dumps(decision), flush=True)
    candidates.append((decision, current.copy(), candidate))
    if accepted:
        current, weights = candidate, chosen_weights

    # EXP-ENS-004: a fixed seed bag replaces the CatBoost expert, with no weight tuning.
    bag = (z42 + z17) / 2
    zbag = normalize(bag, counts, 'zscore')
    candidate = gated_scores(bag, mlp, zbag, zm, counts, short, weights)
    before = metric(truth, current, rows['selection'])['ndcg']
    after = metric(truth, candidate, rows['selection'])['ndcg']
    accepted = after - before >= MIN_DELTA
    decision = {'experiment': 'EXP-ENS-004-r01', 'weights_short_long': weights,
                'cat_seed_weights': {'42': .5, '17': .5}, 'baseline_selection': before,
                'candidate_selection': after, 'delta': after - before, 'accepted': accepted,
                'minimum_delta': MIN_DELTA}
    dump(OUTPUT / 'EXP-ENS-004-frozen.json', decision)
    print(json.dumps(decision), flush=True)
    candidates.append((decision, current.copy(), candidate))
    cat_expert = bag if accepted else cat42
    expert_name = 'seed42_seed17_standardized_average' if accepted else 'seed42'
    if accepted:
        current = candidate
    base_config = {'cat_expert': expert_name, 'weights_short_long': weights,
                   'threshold_history_length': 64, 'normalization': 'per_feed_zscore'}

    # EXP-ENS-005: baseline is passed only to fit/eval pools, prediction is correction.
    frame = make_features(cached, meta, counts, cat_expert, mlp, current)
    fit_rows, tune_rows = rows['meta_fit'], rows['meta_tune']
    fit_truth, tune_truth = [truth.filter(pa.array(mask)) for mask in [fit_rows, tune_rows]]
    fit_pool = pool(frame.loc[fit_rows], fit_truth, current[fit_rows])
    tune_pool = pool(frame.loc[tune_rows], tune_truth, current[tune_rows])
    model = CatBoostRanker(**PARAMS)
    model.fit(fit_pool, eval_set=tune_pool, verbose=10)
    model.save_model(str(OUTPUT / 'meta.cbm'))
    # Model.predict without a baseline pool MUST be the additive correction.
    correction = model.predict(Pool(frame, cat_features=['category_id']))
    candidate = add_correction(current, correction)
    best_iteration = int(model.get_best_iteration())
    logged = float(model.get_evals_result()['validation']['NDCG:type=Base'][best_iteration])
    unrounded = current[tune_rows].astype(np.float64) + correction[tune_rows]
    independent = ranking_metrics(tune_truth['feed_id'].to_numpy(), tune_truth['target'].to_numpy(), unrounded)['ndcg']
    assert abs(logged - independent) < 1e-10, (logged, independent)
    # CatBoost predict on a Pool carrying baseline adds it, independently checking the contract.
    with_baseline = model.predict(tune_pool)
    np.testing.assert_allclose(with_baseline, unrounded, atol=1e-12, rtol=0)
    before = metric(truth, current, tune_rows)['ndcg']
    after = metric(truth, candidate, tune_rows)['ndcg']
    accepted = after - before >= MIN_DELTA
    decision = {'experiment': 'EXP-ENS-005-r01', 'base_config': base_config,
                'params': PARAMS, 'features': list(frame), 'meta_fit_boundary': META_BOUNDARY,
                'selection_boundary': BOUNDARY, 'fit_feeds': int(fit.sum()), 'tune_feeds': int(tune.sum()),
                'best_iteration': best_iteration, 'trees': model.tree_count_,
                'baseline_meta_tune': before, 'candidate_meta_tune': after, 'delta': after - before,
                'accepted': accepted, 'minimum_delta': MIN_DELTA,
                'prediction_contract': 'predict without baseline = correction; add baseline exactly once',
                'logged_tune_ndcg': logged, 'independent_unrounded_tune_ndcg': independent,
                'model_sha256': sha256(OUTPUT / 'meta.cbm')}
    dump(OUTPUT / 'EXP-ENS-005-frozen.json', decision)
    dump(OUTPUT / 'meta_training_history.json', model.get_evals_result())
    print(json.dumps(decision), flush=True)
    candidates.append((decision, current.copy(), candidate))
    if accepted:
        current = candidate

    # Decisions are all frozen. Only now inspect late/full diagnostics and intervals.
    reports = []
    users = meta['user_id'].to_numpy()
    for decision, baseline, candidate in candidates:
        run_id = decision['experiment']
        output = OUTPUT / run_id
        output.mkdir()
        base_values, values = [per_feed(truth, s, counts) for s in (baseline, candidate)]
        delta = values - base_values
        metrics = {name: {'baseline_ndcg': float(base_values[mask].mean()),
                          'candidate_ndcg': float(values[mask].mean()),
                          'delta': float(delta[mask].mean()), 'feeds': int(mask.sum())}
                   for name, mask in masks.items()}
        intervals = {name: bootstrap(delta[masks[name]], users[masks[name]])
                     for name in ['meta_tune', 'late_control']}
        report = {'decision': decision, 'metrics': metrics, 'bootstrap': intervals,
                  'full_val_contains_meta_training': run_id.startswith('EXP-ENS-005'),
                  'test_read': False, 'submission_created': False}
        pq.write_table(prediction_table(truth, candidate), output / 'candidate_val_predictions.parquet', compression='zstd')
        report['candidate_predictions_sha256'] = sha256(output / 'candidate_val_predictions.parquet')
        dump(output / 'result.json', report)
        dump(ROOT / f'ensemble/results/{run_id}.json', report)
        reports.append(report)
    pq.write_table(prediction_table(truth, current), OUTPUT / 'val_predictions.parquet', compression='zstd')
    final = {'base_config': base_config, 'use_meta_correction': candidates[-1][0]['accepted'],
             'parent_provenance': provenance, 'decisions': [r['decision'] for r in reports],
             'meta_model_sha256': sha256(OUTPUT / 'meta.cbm'),
             'predictions_sha256': sha256(OUTPUT / 'val_predictions.parquet'),
             'script_sha256': sha256(__file__), 'meta_training_features': list(frame),
             'test_read': False, 'submission_created': False}
    dump(OUTPUT / 'frozen_final_config.json', final)
    dump(ROOT / 'ensemble/configs/advanced_best.json', final)

    # Reload checkpoint and candidate scores; reverse each feed to check row-order invariance.
    reloaded = CatBoostRanker()
    reloaded.load_model(str(OUTPUT / 'meta.cbm'))
    reproduced = add_correction(candidates[-1][1], reloaded.predict(Pool(frame, cat_features=['category_id'])))
    np.testing.assert_array_equal(reproduced, candidates[-1][2])
    offsets = np.r_[0, np.cumsum(counts)]
    permutation = np.concatenate([np.arange(end - 1, start - 1, -1) for start, end in zip(offsets[:-1], offsets[1:])])
    for scores in [cat42, cat17, mlp]:
        np.testing.assert_array_equal(normalize(scores, counts, 'zscore')[permutation],
                                      normalize(scores[permutation], counts, 'zscore'))
    reversed_frame = make_features(cached.take(pa.array(permutation)), meta, counts,
                                   cat_expert[permutation], mlp[permutation], candidates[-1][1][permutation])
    for name in frame:
        np.testing.assert_array_equal(frame[name].to_numpy()[permutation], reversed_frame[name].to_numpy())
    reversed_scores = add_correction(candidates[-1][1][permutation],
                                    reloaded.predict(Pool(reversed_frame, cat_features=['category_id'])))
    np.testing.assert_array_equal(candidates[-1][2][permutation], reversed_scores)
    np.testing.assert_array_equal(current, align(truth, pq.read_table(OUTPUT / 'val_predictions.parquet')))
    validate_inputs(inputs, cache)
    verification = {'rows': len(truth), 'feeds': len(meta), 'checkpoint_reload_exact': True,
                    'within_feed_permutation_exact': True, 'parent_inputs_unchanged': True,
                    'baseline_addition_verified': True, 'final_score_reload_exact': True,
                    'python': platform.python_version(), 'catboost': catboost_version}
    dump(OUTPUT / 'verification.json', verification)
    with zipfile.ZipFile(OUTPUT / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['advanced.py', 'test_advanced.py', 'blend.py', 'test_blend.py']:
            archive.write(ROOT / 'ensemble' / name, 'ensemble/' + name)
        archive.write(ROOT / 'catboost/baseline.py', 'catboost/baseline.py')
    report = {'status': 'complete', 'experiments': reports, 'final_config': final,
              'verification': verification, 'seconds': time.perf_counter() - started,
              'source_snapshot_sha256': sha256(OUTPUT / 'source.zip'),
              'limitations': ['Historical val has been used by parent selection.',
                              'Meta-fit and meta-tune are on historical val; full val includes meta-fit.',
                              'Bootstrap excludes training and selection uncertainty.'],
              'EXP-ENS-006': 'not run: no second MLP val export available',
              'test_read': False, 'submission_created': False}
    dump(OUTPUT / 'result.json', report)
    dump(ROOT / 'ensemble/results/EXP-ENS-003-005.json', report)
    with (ROOT / 'ensemble_exp.md').open('a') as stream:
        stream.write('\n## Измеренные результаты EXP-ENS-003..005-r01\n\n')
        for item in reports:
            d = item['decision']
            stream.write(f"- **{d['experiment']}**: Δ по критерию принятия {d['delta']:+.10f}; "
                         f"{'оставлен' if d['accepted'] else 'отклонён'}. Конфиг: `ensemble/results/{d['experiment']}.json`.\n")
            for name, m in item['metrics'].items():
                stream.write(f"  - {name}: {m['baseline_ndcg']:.10f} → {m['candidate_ndcg']:.10f}, Δ {m['delta']:+.10f}; {m['feeds']} лент.\n")
        stream.write(f"\n- Meta full_diagnostic и selection содержат строки обучения; это не независимая оценка. Выбор метамодели только по 3458 meta-tune лентам. Все поздние метрики рассчитаны после фиксации решений.\n")
        stream.write(f"- Точные веса/seed/признаки/итерации, интервалы и SHA — в `ensemble/results/EXP-ENS-003-005.json`. Проверены повторный predict из checkpoint, сложение baseline ровно один раз и перестановка всех {len(truth)} кандидатов. Входы родителей неизменны.\n")
        stream.write(f"- Команда: `../AvitoTech_ML_Cup/.venv/bin/python -u ensemble/advanced.py`. Артефакты `{OUTPUT.relative_to(ROOT)}`, время {report['seconds']:.2f} с. Test/submission/push не выполнялись.\n")
        stream.write('- **EXP-ENS-006:** не запускался: в доступном коммите MLP нет второго val-экспорта. Это отсутствие входа, не измеренный провал гипотезы.\n')
    print(json.dumps({'status': 'complete', 'seconds': report['seconds'], 'reports': reports}), flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        with (ROOT / 'ensemble_exp.md').open('a') as stream:
            stream.write('\n## Сбой EXP-ENS-003..005\n\n```text\n' + traceback.format_exc() + '```\n')
        raise
