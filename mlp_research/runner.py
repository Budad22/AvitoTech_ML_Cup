"""Autonomous staged research with frozen hypotheses and a held-out confirmation set."""
import ctypes
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from mlp_research.cache import sha256
from mlp_research.catalog import CONTROLS, hypotheses, promote
from mlp_research.statistics import blend_metrics, conclusion, paired_interval
from mlp_research.training import Trainer
from mlp_trials.state import write_json
from scripts.gpu_suite import process_identity
from scripts.mlp_screen import controller_lock
from training_logs import RunLogger


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def finalists(results, catalog):
    by_name = {case['name']: case for case in catalog}
    selected = list(CONTROLS)
    for family, count in [('plr', 2), ('encoding', 1), ('mlp', 1), ('alternative', 1)]:
        ranked = sorted((r for r in results if r['status'] == 'completed' and
                         by_name[r['name']]['family'] == family and r['name'] != 'linear_ranker'),
                        key=lambda r: r['fast_ndcg'], reverse=True)
        selected.extend(result['name'] for result in ranked[:count])
    return list(dict.fromkeys(selected))


def write_leaderboard(directory, ledger):
    lines = ['# Локальные исследования MLP/PLR', '',
             'Fast NDCG используется для отбора; это не полный val и не результат сайта.', '',
             '| Этап | Вариант | Seed | Шаги | Fast NDCG | Статус | Секунды |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for stage, records in ledger.items():
        for result in records.values():
            score = f"{result['fast_ndcg']:.9f}" if result.get('fast_ndcg') is not None else '—'
            lines.append(f"| {stage} | {result['name']} | {result['seed']} | {result.get('steps', '—')} | "
                         f"{score} | {result['status']} | {result.get('seconds', 0):.1f} |")
    (directory / 'leaderboard.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def report(directory, trainer, names, catalog, seeds, ledger, incomplete, experiment='EXP-029'):
    by_name = {case['name']: case for case in catalog}
    metrics, scores, fast_scores, details = {}, {}, {}, []
    for name in names:
        values, predictions, fast = [], [], []
        for seed in seeds:
            folder = directory / 'runs' / f'{name}-s{seed}'
            if not (folder / 'final.json').exists():
                incomplete = True
                break
            final = read(folder / 'final.json')
            details.append(final)
            values.append(np.load(folder / 'selection_ndcg.npy')[trainer.confirmation])
            predictions.append(np.load(folder / 'selection_logits.npy'))
            fast.append(read(folder / 'result.json')['fast_ndcg'])
        if len(values) == len(seeds):
            metrics[name] = np.stack(values)
            scores[name] = np.stack(predictions)
            fast_scores[name] = float(np.mean(fast))
    if 'history_control' not in metrics:
        result = dict(decision='inconclusive_incomplete', explanation='Нет полного контроля на трёх seed.')
    else:
        users = trainer.validation.arrays['user_id'][trainer.confirmation]
        families = {name: by_name[name]['family'] for name in metrics}
        result = conclusion(metrics, users, families, fast_scores, incomplete=incomplete)
        mlp, alternative = result['best_mlp'], result['best_alternative']
        if alternative:
            blended = blend_metrics(scores[mlp], scores[alternative], trainer.validation.arrays['target'],
                                    trainer.validation.arrays['offsets'])[:, trainer.confirmation]
            result['fixed_blend_vs_alternative'] = paired_interval(
                blended, metrics[alternative], users, comparisons=max(1, (len(names) + 1) * len(names) // 2))
            result['blend'] = dict(left=mlp, right=alternative, weights=[.5, .5],
                                   confirmation_ndcg=float(blended.mean()))
    result.update(experiment=experiment, device=getattr(trainer, 'device', 'cpu'), seeds=list(seeds), final_runs=details,
                  confirmation_feeds=len(trainer.confirmation), fast_feeds=len(trainer.fast),
                  confirmation_selection_indices_sha256=sha256(trainer.cache / 'confirmation.npy'),
                  data=trainer.info, ledger=ledger, official_control_evaluated=False, test_predicted=False)
    write_json(directory / 'report.json', result)
    lines = ['# Результат локального исследования', '', result['explanation'], '',
             f"Решение: **{result['decision']}**. Область вывода — фиксированные 0,78% train и проверенные бюджеты.", '',
             '| Вариант | Confirmation NDCG, 3 seed | Δ к контролю | Интервал с поправкой на сравнения |',
             '| --- | --- | --- | --- |']
    for name, values in metrics.items():
        delta = result.get('differences', {}).get(name)
        difference = f"{delta['delta']:+.6f}" if delta else '—'
        interval = f"[{delta['lower']:+.6f}; {delta['upper']:+.6f}]" if delta else '—'
        lines.append(f'| {name} | {values.mean():.9f} | {difference} | {interval} |')
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def run(args):
    directory, cache = args.directory.resolve(), args.cache.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    catalog = hypotheses()
    by_name = {case['name']: case for case in catalog}
    seeds = (42, 43, 44)
    device = getattr(args, 'device', 'cpu')
    experiment = getattr(args, 'experiment', 'EXP-029')
    selected_path = getattr(args, 'finalists', None)
    selected_plan = read(selected_path) if selected_path else None
    if selected_path and not selected_plan:
        raise ValueError('Missing frozen finalist plan')
    if selected_plan:
        names = selected_plan['names']
        if len(names) != len(set(names)) or not set(CONTROLS).issubset(names) or any(n not in by_name for n in names):
            raise ValueError('Frozen finalists must contain known unique models and both controls')
        if selected_plan['seeds'] != list(seeds):
            raise ValueError('Frozen finalist seeds changed')
    config = dict(catalog=catalog, seeds=list(seeds), batch_size=16, threads=4, cache=str(cache),
                  steps=[200, 800, 3200, 6400], maximum_hours=args.maximum_hours)
    if device != 'cpu' or experiment != 'EXP-029':
        config.update(device=device, experiment=experiment)
    if selected_plan:
        config.update(finalists_source=str(selected_path.resolve()), finalists_sha256=sha256(selected_path),
                      mode='confirm_frozen_finalists')
    root = Path(__file__).parents[1]
    hashes = {str(path.relative_to(root)): sha256(path) for folder in
              ('mlp_research', 'mlp', 'mlp_trials', 'recsys_data', 'training_logs', 'scripts')
              for path in (root / folder).rglob('*') if path.suffix in ('.py', '.ps1')}
    if read(directory / 'research-config.json') not in (None, config):
        raise ValueError('Immutable research configuration changed')
    if read(directory / 'code-hashes.json') not in (None, hashes):
        raise ValueError('Research source snapshot changed')
    write_json(directory / 'research-config.json', config)
    write_json(directory / 'code-hashes.json', hashes)
    with controller_lock(directory / 'controller.lock'), RunLogger(directory, experiment, device.upper() + '/research') as logger:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        torch.set_num_threads(4)
        if device == 'cuda':
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
        previous = read(directory / 'status.json') or {}
        if previous.get('status') == 'completed':
            return
        state = dict(experiment=experiment, device=device, status='running', controller_pid=os.getpid(),
                     controller_identity=process_identity(os.getpid()), started_at=datetime.now(timezone.utc).isoformat(),
                     completed=previous.get('completed', 0), failed=previous.get('failed', 0), phase='prepare',
                     hypotheses=len(selected_plan['names']) if selected_plan else len(catalog),
                     cache=str(cache), directory=str(directory))
        elapsed = previous.get('elapsed_seconds', 0.)
        started = time.monotonic()
        ledger = read(directory / 'ledger.json') or {}
        trainer = None

        def update(**fields):
            state.update(fields, elapsed_seconds=elapsed + time.monotonic() - started,
                         updated_at=datetime.now(timezone.utc).isoformat())
            write_json(directory / 'status.json', state)

        def emit(event, **fields):
            if event == 'device_cache_ready':
                state['device_cache'] = fields
            update(progress=fields)
            logger.emit(event, phase=state['phase'], **fields)

        def case(name, seed, target, stage):
            records = ledger.setdefault(stage, {})
            key = f'{name}-s{seed}'
            if key in records:
                return records[key]
            if (directory / 'STOP_AFTER_CASE').exists():
                raise InterruptedError('Paused after the current case at owner request')
            if state.get('elapsed_seconds', 0) >= args.maximum_hours * 3600:
                raise TimeoutError('Research safety budget reached; no architecture verdict')
            update(phase=stage, current=dict(name=name, seed=seed, target=target), progress={})
            try:
                result = trainer.run(by_name[name], seed, target,
                                     maximum_seconds=min(7200, args.maximum_hours * 3600 - state['elapsed_seconds']))
            except InterruptedError:
                raise
            except Exception as error:
                output = directory / 'runs' / key
                output.mkdir(parents=True, exist_ok=True)
                (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                result = dict(name=name, seed=seed, status='failed', error=str(error), steps=target)
                state['failed'] += 1
                logger.emit('case_failed', phase=stage, level='ERROR', name=name, error=str(error))
            records[key] = result
            state['completed'] += 1
            write_json(directory / 'ledger.json', ledger)
            write_leaderboard(directory, ledger)
            update()
            gc.collect()
            return result

        try:
            update()
            active_root = getattr(args, 'active_root', None)
            if active_root:
                write_json(active_root / 'active-research.json', dict(directory=str(directory), experiment=experiment,
                           task_name=getattr(args, 'task_name', 'AvitoMLPResearchGPU')))
            while not (cache / 'metadata.json').exists():
                error_log = directory / 'prepare.err.log'
                if error_log.exists() and 'Traceback' in error_log.read_text(encoding='utf-8', errors='replace'):
                    raise RuntimeError('Cache preparation failed; inspect prepare.err.log')
                if state.get('elapsed_seconds', 0) > args.maximum_hours * 3600:
                    raise TimeoutError('Cache preparation safety budget reached')
                update()
                time.sleep(5)
            if selected_plan:
                for name in ('fast', 'confirmation'):
                    if sha256(cache / (name + '.npy')) != selected_plan[name + '_sha256']:
                        raise ValueError('Frozen validation partition changed: ' + name)
            trainer = Trainer(cache, directory, emit, device=device) if device != 'cpu' else Trainer(cache, directory, emit)
            write_json(directory / 'data-reference.json', dict(cache_sha256=trainer.fingerprint, data=trainer.info))
            if selected_plan:
                chosen = selected_plan['names']
            else:
                for hypothesis in catalog:
                    case(hypothesis['name'], 42, 200, 'screen_200')
                for name in CONTROLS:
                    for seed in seeds[1:]:
                        case(name, seed, 200, 'screen_200')
                selected = promote([r for r in ledger['screen_200'].values() if r['seed'] == 42], catalog)
                write_json(directory / 'promotion-800.json', dict(names=selected, criterion='family coverage plus slow learners'))
                for name in selected:
                    case(name, 42, 800, 'screen_800')
                for name in CONTROLS:
                    for seed in seeds[1:]:
                        case(name, seed, 800, 'screen_800')
                chosen = finalists([r for r in ledger['screen_800'].values() if r['seed'] == 42], catalog)
                if not all(ledger['screen_800'].get(f'{name}-s42', {}).get('status') == 'completed' for name in CONTROLS):
                    raise RuntimeError('Controls did not complete; architecture comparison is incomplete')
            write_json(directory / 'finalists.json', dict(names=chosen, seeds=list(seeds),
                       criterion='fast partition only; confirmation still sealed'))
            for name in chosen:
                for seed in seeds:
                    case(name, seed, 3200, 'confirm_3200')
            noise_stage = 'confirm_3200' if selected_plan else 'screen_800'
            controls = [r['fast_ndcg'] for r in ledger[noise_stage].values()
                        if r['name'] == 'history_control' and r['status'] == 'completed']
            noise = float(np.std(controls)) if controls else float('inf')
            rising = [r for r in ledger['confirm_3200'].values() if r['status'] == 'completed' and
                      len(r['history']) >= 2 and r['history'][-1]['fast_ndcg'] - r['history'][-2]['fast_ndcg'] > max(noise, 1e-10)]
            extend = bool(rising)
            write_json(directory / 'extension.json', dict(extend_all=extend, measured_control_seed_std=noise,
                       reason='Any finalist still improving above measured seed noise', names=[r['name'] for r in rising]))
            if extend:
                for name in chosen:
                    for seed in seeds:
                        case(name, seed, 6400, 'confirm_6400')
            # No confirmation labels are scored until every final training decision is frozen.
            stage = 'confirm_6400' if extend else 'confirm_3200'
            incomplete = not {'plr', 'encoding', 'mlp', 'alternative'}.issubset(
                {by_name[name]['family'] for name in chosen})
            for name in chosen:
                for seed in seeds:
                    if ledger[stage].get(f'{name}-s{seed}', {}).get('status') != 'completed':
                        incomplete = True
                        continue
                    update(phase='confirmation_evaluation', current=dict(name=name, seed=seed))
                    output = directory / 'runs' / f'{name}-s{seed}'
                    if not (output / 'final.json').exists():
                        trainer.final(by_name[name], seed)
            result = report(directory, trainer, chosen, catalog, seeds, ledger, incomplete, experiment=experiment)
            update(status='completed_with_errors' if incomplete or state['failed'] else 'completed', phase='completed', current=None,
                   decision=result['decision'], report=str(directory / 'report.json'))
        except InterruptedError as error:
            update(status='paused', phase='paused', error=str(error))
        except Exception as error:
            (directory / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            update(status='error', phase='error', error=str(error))
            raise
        finally:
            if trainer:
                trainer.close()
