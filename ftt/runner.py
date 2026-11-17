"""Resumable sequential trials; decisions require no language-model service."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from ftt.data import Cache, prepare, write_json
from ftt.model import Ranker, configuration
from ftt.training import Trainer, ndcg, save, evaluate
from ftt.policy import rank_result, select_candidates, validation_partition, paired_summary
from ftt.runtime import read_json, worker_lock, record_failure, add_run_arguments


class Queue:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        prior = self.root / 'status.json'
        self.state = read_json(prior, dict(elapsed_seconds=0))
        self.previous_seconds = self.state.get('elapsed_seconds', 0)
        self.state.update(status='running', pid=os.getpid(), experiment=getattr(args,'experiment','FTT-NIGHT-001'), device=args.device)
        self.ledger = read_json(self.root / 'ledger.json', {})
        torch.set_num_threads(4)

    def emit(self, event, **values):
        elapsed = self.previous_seconds + time.monotonic() - self.started
        if any(k in values and values[k] != self.state.get(k) for k in ('model', 'seed', 'phase')):
            for key in ('best_ndcg', 'ndcg', 'loss', 'step', 'target', 'error'):
                self.state.pop(key, None)
        self.state.update(values, event=event, heartbeat=time.time(), elapsed_seconds=elapsed,
                          completed=len(self.ledger), budget_seconds=self.args.hours * 3600)
        write_json(self.root / 'status.json', self.state)
        with (self.root / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=time.time(), event=event, **values), ensure_ascii=False) + '\n')
        if elapsed >= self.args.hours * 3600:
            raise TimeoutError('Global queue budget exhausted')

    def trial(self, number, seed, target, size):
        trainer = Trainer(root=self.root, data=self.args.data, device=self.args.device,
                          validation=self.validation, fast=self.fast, emit=self.emit, ledger=self.ledger)
        return trainer.run(number, seed, target, size)

    def screen_trial(self, number, target, size):
        if str(number) in self.excluded:
            return None
        try:
            return self.trial(number, 42, target, size)
        except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
            return self.exclude(number, error)
        except ValueError as error:
            if 'Nonfinite' not in str(error):
                raise
            return self.exclude(number, error)

    def exclude(self, number, error):
        if number == 1:
            raise error
        self.excluded[str(number)] = traceback.format_exc()
        write_json(self.root / 'excluded.json', self.excluded)
        self.emit('excluded', model=f'FTT-{number:03d}', error=str(error))
        if self.args.device == 'cuda':
            torch.cuda.empty_cache()
        return None

    def run(self):
        self.excluded = read_json(self.root / 'excluded.json', {})
        info = prepare(self.args.data, self.root / 'cache', self.emit)
        self.validation = Cache(self.root / 'cache', 'val')
        rng = np.random.default_rng(2026)
        selection, self.fast, confirmation = validation_partition(self.validation.timestamps, info['boundary'], rng)
        write_json(self.root / 'partitions.json', dict(boundary=info['boundary'], fast=self.fast.tolist(), confirmation=confirmation.tolist()))
        # Smoke trials are isolated from selection runs.
        variants = [int(n) for n in getattr(self.args,'variants','1,2,3,4,5,6,7,8').split(',')]
        if 1 not in variants or len(set(variants)) != len(variants) or any(n not in range(1,13) for n in variants):
            raise ValueError('Variants must contain control 1 and unique IDs 1..12')
        for number in variants:
            self.screen_trial(number, 20, 'smoke')
        for target in (200, 800):
            scores = [r for n in variants if (r := self.screen_trial(n, target, 'small'))]
        candidates = sorted([r for r in scores if r['number'] != 1], key=rank_result)
        if not candidates:
            write_json(self.root / 'report.json', dict(decision='inconclusive', reason='No viable extensions'))
            self.emit('completed', status='completed')
            return
        selected = select_candidates(scores, control=1, top_count=2)
        extended = [r for n in selected if (r := self.screen_trial(n, 3200, 'small'))]
        finalists = [r['number'] for r in sorted([r for r in extended if r['number'] != 1], key=rank_result)[:2]]
        if not finalists:
            raise RuntimeError('No viable finalists; no full training permitted')
        medium = [r for n in [1, *finalists] if (r := self.screen_trial(n, 3200, 'medium'))]
        growing = getattr(self.args,'force_medium_6400',False) or any({v['step']:v['ndcg'] for v in r['history']}[3200] >
                      {v['step']:v['ndcg'] for v in r['history']}[1600] for r in medium)
        target = 6400 if growing else 3200
        if growing:
            medium = [r for n in [1, *finalists] if (r := self.screen_trial(n, target, 'medium'))]
        if not any(r['number'] != 1 for r in medium):
            write_json(self.root / 'report.json', dict(decision='inconclusive', reason='No viable medium finalists'))
            self.emit('completed', status='completed')
            return
        winner = min([r for r in medium if r['number'] != 1], key=rank_result)['number']
        for seed in (43, 44):
            for n in (1, winner):
                self.trial(n, seed, target, 'medium')
        metrics = {}
        for n in (1, winner):
            metrics[n] = []
            for seed in (42, 43, 44):
                result = self.trial(n, seed, target, 'medium')
                state = torch.load(Path(result['directory']) / 'best.pt', map_location='cpu', weights_only=False)
                self.emit('confirmation', phase='confirmation', model=f'FTT-{n:03d}', seed=seed)
                train = Cache(self.root / 'cache', 'train', 'medium')
                model = Ranker(configuration(n), *train.statistics()).to(self.args.device)
                model.load_state_dict(state['model'])
                values = evaluate(model, self.validation, confirmation, self.args.device)
                metrics[n].append(values)
                np.save(Path(result['directory']) / 'confirmation_ndcg.npy', values)
        summary = paired_summary(metrics[winner], metrics[1], self.validation.users[confirmation], rng)
        qualifies = summary['qualifies']
        report = dict(winner=winner, mean_delta=summary['delta'], interval=summary['interval'],
                      per_seed=summary['per_seed'], qualifies=qualifies, full_training='not_started',
                      confirmation_feeds=len(confirmation), official_control_evaluated=False)
        write_json(self.root / 'report.json', report)
        remaining = self.args.hours*3600 - self.previous_seconds - (time.monotonic()-self.started)
        if qualifies and remaining >= 7200:
            full = self.trial(winner, 42, 10**9, 'full')
            state = torch.load(Path(full['directory']) / 'best.pt', map_location='cpu', weights_only=False)
            model = Ranker(configuration(winner), *Cache(self.root / 'cache', 'train', 'medium').statistics()).to(self.args.device)
            model.load_state_dict(state['model'])
            all_values = evaluate(model, self.validation, np.arange(self.validation.feeds), self.args.device)
            np.save(self.root / 'full_val_ndcg.npy', all_values)
            report.update(full_training='completed', full=full, full_val_ndcg=float(all_values.mean()),
                          selection_ndcg=float(all_values[selection].mean()),
                          control_ndcg=float(all_values[self.validation.timestamps >= info['boundary']].mean()),
                          official_control_evaluated=True)
            write_json(self.root / 'report.json', report)
        self.emit('screen_completed', status='completed', report=report)


def main():
    parser = argparse.ArgumentParser()
    add_run_arguments(parser, hours=8, experiment='FTT-NIGHT-001')
    parser.add_argument('--variants', default='1,2,3,4,5,6,7,8')
    parser.add_argument('--force-medium-6400', action='store_true')
    args = parser.parse_args()
    queue = Queue(args)
    lock = queue.root / 'controller.lock'
    with worker_lock(lock):
        try:
            queue.run()
        except Exception as error:
            record_failure(queue, error)
            raise


if __name__ == '__main__':
    main()
