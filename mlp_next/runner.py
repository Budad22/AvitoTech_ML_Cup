"""One autonomous queue: new signals, larger-data confirmation, optional full fit."""
import argparse
import ctypes
from datetime import datetime, timezone
from functools import partial, lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from mlp_research.cache import build, sha256
from mlp_research.statistics import paired_interval
from mlp_research.training import Trainer, evaluate
from mlp_trials.state import write_json
from scripts.gpu_suite import process_identity
from scripts.mlp_screen import controller_lock, sample_view, sample_indices, link_directory
from training_logs import RunLogger
from mlp_next.data import build_sequences, SequenceSplit
from mlp_next.models import configurations, HistoryRanker
from mlp_next.streaming import StreamingSplit, sequential_order
from mlp_next.diagnostics import exposure
from mlp_next.blend import check_blends


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


@lru_cache(maxsize=2)
def permutation(feeds, seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(feeds)


def cached_order(feeds, seed, step, batch_size):
    epoch, cursor = divmod(step, math.ceil(feeds / batch_size))
    return permutation(feeds, seed, epoch)[cursor * batch_size:(cursor + 1) * batch_size]


class Queue:
    def __init__(self, args):
        self.args, self.root = args, args.output
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.state = read(self.root / 'status.json', {})
        self.elapsed = self.state.get('elapsed_seconds', 0)
        self.state.update(experiment='EXP-031', architecture='MLP', device=args.device,
                          status='running', phase='prepare', current=None,
                          controller_pid=os.getpid(), controller_identity=process_identity(os.getpid()),
                          completed=self.state.get('completed', 0), failed=self.state.get('failed', 0),
                          hypotheses=9, directory=str(self.root))
        self.ledger = read(self.root / 'ledger.json', {})
        self.logger = RunLogger(self.root, 'EXP-031', 'MLP', echo=False)
        self.configs = configurations()

    def emit(self, event, **fields):
        self.state.update(progress=fields, updated_at=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=self.elapsed + time.monotonic() - self.started)
        write_json(self.root / 'status.json', self.state)
        self.logger.emit(event, phase=self.state['phase'], **fields)
        if self.state['elapsed_seconds'] > self.args.maximum_hours * 3600:
            raise TimeoutError('Total research budget exhausted; checkpoints preserved')

    def case(self, trainer, name, seed, target, phase):
        key = f'{phase}/{name}-s{seed}/{target}'
        self.state.update(phase=phase, current=dict(name=name, seed=seed, target=target))
        trainer.train.need_sequence = trainer.validation.need_sequence = self.configs[name]['sequence']
        if key not in self.ledger:
            self.emit('case_started', name=name, seed=seed, total=target)
            self.ledger[key] = trainer.run(self.configs[name], seed, target, maximum_seconds=7200)
            write_json(self.root / 'ledger.json', self.ledger)
            self.state['completed'] = len(self.ledger)
        return self.ledger[key]

    def trainer(self, stage, view, old_cache=None):
        folder = self.root / stage
        folder.mkdir(exist_ok=True)
        if old_cache:
            if not (folder / 'cache').exists():
                link_directory(folder / 'cache', old_cache)
        else:
            build(view, folder / 'cache', self.emit)
        build_sequences(view, folder / 'cache', folder / 'sequences', self.emit)
        return Trainer(folder / 'cache', folder, self.emit, device=self.args.device,
                       model_class=HistoryRanker, split_class=SequenceSplit, order_fn=cached_order)

    def final_metrics(self, trainer, names, stage):
        metrics, fast = {}, {}
        for name in names:
            metrics[name], fast[name] = [], []
            trainer.validation.need_sequence = self.configs[name]['sequence']
            for seed in (42, 43, 44):
                self.state.update(phase='confirmation_evaluation', current=dict(name=name, seed=seed))
                self.emit('confirmation_started', name=name, seed=seed)
                output = trainer.directory / 'runs' / f'{name}-s{seed}'
                trainer.final(self.configs[name], seed)
                values = np.load(output / 'selection_ndcg.npy')
                metrics[name].append(values[trainer.confirmation])
                fast[name].append(float(values[trainer.fast].mean()))
        summary = dict(fast_mean={n: float(np.mean(v)) for n, v in fast.items()},
                       confirmation_mean={n: float(np.mean(v)) for n, v in metrics.items()},
                       comparisons={n: paired_interval(metrics[n], metrics['history_control'],
                                       trainer.validation.arrays['user_id'][trainer.confirmation], len(names))
                                    for n in names if n != 'history_control'},
                       train_feeds=trainer.train.feeds,
                       note='Historical selection, not full official val or untouched holdout')
        write_json(self.root / (stage + '-report.json'), summary)
        return summary

    def run(self):
        torch.set_num_threads(4)
        self.emit('queue_started')
        write_json(self.root / 'research-config.json', dict(maximum_hours=self.args.maximum_hours,
                   per_model_hours=2, hypotheses=self.configs, seeds=[42, 43, 44],
                   control='history_control', selection_only=True))
        write_json(Path('Q:/AVITOCUP/active-research.json'), dict(directory=str(self.root),
                   experiment='EXP-031', task_name=None))
        write_json(self.root / 'saved-val-blends.json', check_blends(self.args.data, dict(
            history_stats=Path('Q:/AVITOCUP/experiments/MLP-SUITE-001/exports/val-for-catboost-r01/output/mlp_val_predictions.parquet'),
            softmax=Path('Q:/AVITOCUP/AvitoTech_ML_Cup/ensembles/mlp_softmax/mlp_val_predictions.parquet'),
            ftt=Path('Q:/AVITOCUP/experiments/FTT-NIGHT-004/export/ftt_val_predictions.parquet'))))
        count = len(read(self.args.data / 'manifest.json')['splits']['train']['files'])
        small, medium = sample_indices(count)
        small_view, medium_view = self.root / 'view-small', self.root / 'view-medium'
        sample_view(self.args.data, small_view, small)
        sample_view(self.args.data, medium_view, medium)
        try:
            write_json(self.root / 'exposure-diagnostics.json', exposure(medium_view))
        except (KeyError, FileNotFoundError) as error:
            write_json(self.root / 'exposure-diagnostics.json', dict(status='unavailable', reason=str(error)))
        # Existing EXP-030 controls are retained as references; do not train them again on the same small data.
        old_report = read(self.args.previous / 'report.json')
        if old_report is None:
            raise ValueError('Prior completed controls are required')
        trainer = self.trainer('small', small_view, self.args.previous / 'cache')
        names = [n for n in self.configs if n not in ('history_control', 'mlp_softmax', 'plr_softmax')]
        try:
            for name in names:
                self.case(trainer, name, 42, 20, 'smoke')
            for target in (200, 800):
                results = [self.case(trainer, n, 42, target, 'small') for n in names]
            leaders = [r['name'] for r in sorted(results, key=lambda r: -r['fast_ndcg'])[:2]]
            # Preserve a slower sequential family regardless of its first 800 steps.
            promoted = list(dict.fromkeys(leaders + ['stats_sequence128']))
            for name in promoted:
                for seed in (42, 43, 44):
                    self.case(trainer, name, seed, 6400, 'small')
            ranked = sorted(promoted, key=lambda n: -np.mean([
                self.ledger[f'small/{n}-s{s}/6400']['fast_ndcg'] for s in (42, 43, 44)]))
            selected = ranked[:2]
            write_json(self.root / 'promotion.json', dict(names=selected,
                       criterion='Mean fast score across 3 seeds; confirmation not used',
                       previous_control_report_sha256=sha256(self.args.previous / 'report.json')))
            # Score frozen new checkpoints on confirmation, without changing the promotion.
            for name in promoted:
                trainer.validation.need_sequence = self.configs[name]['sequence']
                for seed in (42, 43, 44):
                    trainer.final(self.configs[name], seed)
        finally:
            trainer.close()
        self.state.update(phase='prepare_medium', current=None)
        self.emit('medium_preparation_started')
        trainer = self.trainer('medium', medium_view)
        names = ['history_control', 'mlp_softmax', 'plr_softmax', *selected]
        try:
            for name in names:
                for seed in (42, 43, 44):
                    self.case(trainer, name, seed, 12800, 'medium')
            summary = self.final_metrics(trainer, names, 'medium')
            winner = max(names, key=lambda n: summary['fast_mean'][n])
            difference = summary['comparisons'].get(winner, {})
            promote_full = difference.get('evidence', False) and difference.get('delta', 0) >= .001
            write_json(self.root / 'full-decision.json', dict(winner=winner, train_full=promote_full,
                       comparison=difference, criterion='Fast-selected winner; paired confirmation lower>0 and delta>=0.001'))
            if promote_full:
                self.full(trainer, winner)
            write_json(self.root / 'report.json', dict(experiment='EXP-031', status='completed',
                       best_mlp=winner, baseline='history_control', medium=summary,
                       full_started=promote_full, completed_stages=len(self.ledger),
                       decision='continue_mlp' if promote_full else 'inconclusive_no_confirmed_gain',
                       test_predicted=False))
        finally:
            trainer.close()
        self.state.update(status='completed', phase='completed', current=None)
        self.emit('queue_completed')

    def full(self, previous, name):
        config = self.configs[name]
        folder = self.root / 'full'
        folder.mkdir(exist_ok=True)
        trainer = Trainer(previous.cache, folder, self.emit, device=self.args.device,
                          model_class=HistoryRanker, split_class=SequenceSplit, order_fn=sequential_order)
        trainer.train.close()
        categories = {int(k): v for k, v in previous.info['categories'].items()}
        trainer.train = StreamingSplit(self.args.data, categories, config, self.args.device)
        trainer.fingerprint = hashlib.sha256((trainer.fingerprint + sha256(self.args.data / 'manifest.json')).encode()).hexdigest()
        trainer.validation.need_sequence = config['sequence']
        self.state.update(phase='full', current=dict(name=name, seed=42))
        try:
            target = 3 * math.ceil(trainer.train.feeds / 16)
            try:
                result = trainer.run(config, 42, target, maximum_seconds=7200)
            except TimeoutError:
                result = dict(status='partial_budget', reason='Two-hour full-model budget; best checkpoint retained')
            if (folder / 'runs' / f'{name}-s42/best.pt').exists():
                trainer.final(config, 42)
            write_json(folder / 'result.json', result)
        finally:
            trainer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--maximum-hours', type=float, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with controller_lock(args.output / 'controller.lock'):
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        queue = Queue(args)
        try:
            queue.run()
        except Exception as error:
            (args.output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            queue.state.update(status='error', error=str(error))
            queue.emit('queue_failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
