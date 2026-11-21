"""A durable 12-hour research window with useful reserve work and isolated failures."""
import argparse
import ctypes
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from mlp_next.runner import read, cached_order
from mlp_next.models import configurations
from mlp_next.extended_models import ExtendedRanker, extended_configurations, extended_loss
from mlp_next.extended_data import ExtendedSplit, PrefetchedStream, prepare_positions
from mlp_next.streaming import sequential_order
from mlp_next.data import build_sequences
from mlp_research.cache import build, sha256
from mlp_research.training import Trainer, evaluate
from mlp_trials.state import write_json
from scripts.gpu_suite import process_identity
from scripts.mlp_screen import controller_lock, sample_view, sample_indices
from training_logs import RunLogger


def family_leaders(records, configs, count=1):
    selected = []
    for family in sorted({configs[r['name']]['family'] for r in records if r['status'] == 'completed'}):
        values = sorted((r for r in records if r['status'] == 'completed' and configs[r['name']]['family'] == family),
                        key=lambda r: (-r['fast_ndcg'], r['name']))
        selected.extend(r['name'] for r in values[:count])
    return selected


def predecessor_alive(directory):
    state = read(Path(directory) / 'status.json', {})
    pid, identity = state.get('controller_pid'), state.get('controller_identity')
    if not pid or not identity:
        return False
    # The native query returns None for an exited process; no shell exit-code race.
    # Compare creation time too, because Windows can reuse a PID.
    return process_identity(int(pid)) == int(identity)


class Overnight:
    def __init__(self, args):
        self.args, self.root = args, args.output
        self.root.mkdir(parents=True, exist_ok=True)
        self.configs = {**configurations(), **extended_configurations()}
        self.new_names = list(extended_configurations())
        self.state = read(self.root / 'status.json', {})
        self.resume_case = dict(phase=self.state.get('phase'),
                                current=dict(self.state.get('current') or {}))
        self.elapsed = self.state.get('elapsed_seconds', 0)
        self.started = time.monotonic()
        self.state.update(experiment='EXP-032', architecture='MLP', device='cuda', status='waiting',
                          phase='waiting_for_EXP-031', controller_pid=os.getpid(),
                          controller_identity=process_identity(os.getpid()), directory=str(self.root),
                          completed=self.state.get('completed', 0), failed=self.state.get('failed', 0),
                          hypotheses=len(self.new_names))
        self.state.pop('error', None)
        self.state.pop('pause_reason', None)
        self.state.pop('paused_at', None)
        self.ledger = read(self.root / 'ledger.json', {})
        self.logger = RunLogger(self.root, 'EXP-032', 'MLP', echo=False)
        self.medium_cache = self.medium_view = None
        write_json(self.root / 'research-config.json', dict(maximum_hours=args.maximum_hours,
                   minimum_hours=args.minimum_hours, hypotheses=len(self.new_names),
                   per_full_model_hours=2, status='queued'))

    def event(self, event, **fields):
        self.state.update(progress=fields, updated_at=datetime.now(timezone.utc).isoformat())
        if self.state['status'] != 'waiting':
            self.state['elapsed_seconds'] = self.elapsed + time.monotonic() - self.started
        write_json(self.root / 'status.json', self.state)
        self.logger.emit(event, phase=self.state['phase'], **fields)
        if self.state.get('elapsed_seconds', 0) >= self.args.maximum_hours * 3600:
            raise TimeoutError('Global active-time limit; checkpoint retained')

    def wait(self):
        self.event('successor_queued', predecessor=str(self.args.previous))
        write_json(Path('Q:/AVITOCUP/active-research-queue.json'), dict(
                   experiment='EXP-032', directory=str(self.root), hypotheses=len(self.new_names),
                   minimum_hours=self.args.minimum_hours, maximum_hours=self.args.maximum_hours,
                   predecessor='EXP-031'))
        while predecessor_alive(self.args.previous):
            prior = read(self.args.previous / 'status.json', {})
            self.event('waiting', predecessor_current=prior.get('current'),
                       predecessor_status=prior.get('status'))
            time.sleep(20)
        # Wait for the actual CUDA process/context, not just a terminal status file.
        while True:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
            if all(float(r.split(',')[0]) <= 10 and float(r.split(',')[1]) < 1000 for r in rows):
                break
            self.event('waiting_for_free_gpu')
            time.sleep(20)
        self.started = time.monotonic()
        self.state.update(status='running', phase='preflight')
        self.event('gpu_acquired')
        with (self.root / 'tests.log').open('a') as out, (self.root / 'tests.err.log').open('a') as err:
            result = subprocess.run([os.sys.executable, '-X', 'utf8', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                    stdout=out, stderr=err)
        if result.returncode:
            raise RuntimeError('GPU preflight failed; see tests.err.log')
        write_json(Path('Q:/AVITOCUP/active-research.json'), dict(directory=str(self.root), experiment='EXP-032', task_name=None))

    def prepare(self):
        self.state['phase'] = 'prepare'
        existing = self.args.previous / 'medium'
        if (existing / 'cache/metadata.json').exists() and (existing / 'sequences/metadata.json').exists():
            self.medium_cache, self.medium_view = existing / 'cache', self.args.previous / 'view-medium'
        else:
            self.medium_view = self.root / 'view-medium'
            count = len(read(self.args.data / 'manifest.json')['splits']['train']['files'])
            sample_view(self.args.data, self.medium_view, sample_indices(count)[1])
            self.medium_cache = self.root / 'prepared-medium/cache'
            build(self.medium_view, self.medium_cache, self.event)
            build_sequences(self.medium_view, self.medium_cache, self.medium_cache.parent / 'sequences', self.event)
        prepare_positions(self.medium_view, self.medium_cache, self.root / 'positions')
        write_json(self.root / 'research-config.json', dict(maximum_hours=self.args.maximum_hours,
                   minimum_hours=self.args.minimum_hours, per_full_model_hours=2,
                   new_hypotheses=self.new_names, configs=self.configs,
                   selection='fast2000 for all promotions; confirmation after decisions; no late control tuning',
                   data_cache=str(self.medium_cache)))

    def trainer(self, directory, *, full=False, name=None):
        trainer = Trainer(self.medium_cache, directory, self.event, batch_size=64 if full else 16, device='cuda',
                          model_class=ExtendedRanker,
                          split_class=partial(ExtendedSplit, positions=self.root / 'positions'),
                          order_fn=sequential_order if full else cached_order, loss_fn=extended_loss)
        if full:
            trainer.train.close()
            trainer.train = PrefetchedStream(self.args.data, {int(k): v for k, v in trainer.info['categories'].items()},
                                             self.configs[name], 'cuda', batch_size=64)
            trainer.fingerprint = hashlib.sha256((trainer.fingerprint + sha256(self.args.data / 'manifest.json')).encode()).hexdigest()
        return trainer

    def case(self, name, seed, target, stage, *, full=False):
        if (self.root/'STOP_AFTER_CASE').exists():
            raise InterruptedError('Owner requested a pause after the previous case')
        key = f'{stage}/{name}-s{seed}/{target}'
        if key in self.ledger:
            return self.ledger[key]
        self.state.update(phase=stage, current=dict(name=name, seed=seed, target=target))
        self.event('case_started', name=name, seed=seed, total=target, full_train=full)
        directory = self.root / stage
        trainer = self.trainer(directory, full=full, name=name)
        trainer.validation.need_sequence = self.configs[name]['sequence']
        trainer.train.need_sequence = self.configs[name]['sequence']
        config = self.configs[name]
        if full:
            epoch_steps = math.ceil(trainer.train.feeds / trainer.batch_size)
            config = dict(config, early_stop_patience=6, minimum_steps=epoch_steps)
            if config.get('schedule_steps'):
                config['schedule_steps'] = 3 * epoch_steps
        try:
            try:
                result = trainer.run(config, seed, target,
                                     maximum_seconds=min(7200 if full else 3600,
                                         self.args.maximum_hours * 3600 - self.state.get('elapsed_seconds', 0)))
            except TimeoutError as error:
                result = dict(name=name, seed=seed, status='partial_budget', reason=str(error))
            output = directory / 'runs' / f'{name}-s{seed}'
            if (output / 'best.pt').exists():
                state = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
                result.update(fast_ndcg=state['best'], best_step=state['step'])
            if (output / 'last.pt').exists():
                state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
                result.update(actual_steps=state['step'], train_seconds=state['seconds'],
                              epochs=state['step'] * trainer.batch_size / trainer.train.feeds)
            result.update(full_train=full, train_feeds=trainer.train.feeds, batch_size=trainer.batch_size)
        except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
            result = dict(name=name, seed=seed, status='failed', error=str(error), full_train=full)
            failure = directory / 'runs' / f'{name}-s{seed}' / 'failure.txt'
            failure.parent.mkdir(parents=True,exist_ok=True)
            failure.write_text(traceback.format_exc(), encoding='utf-8')
            self.state['failed'] += 1
        except ValueError as error:
            if 'Nonfinite' not in str(error):
                raise
            result = dict(name=name, seed=seed, status='failed', error=str(error), full_train=full)
            failure = directory / 'runs' / f'{name}-s{seed}' / 'failure.txt'
            failure.parent.mkdir(parents=True,exist_ok=True)
            failure.write_text(traceback.format_exc(), encoding='utf-8')
            self.state['failed'] += 1
        finally:
            trainer.close()
            gc.collect()
            torch.cuda.empty_cache()
        self.ledger[key] = result
        self.state['completed'] = len(self.ledger)
        write_json(self.root / 'ledger.json', self.ledger)
        self.event('case_completed', **result)
        return result

    def full(self, name, seed=42, stage='full_priority'):
        for result in self.ledger.values():
            if result.get('full_train') and result['name']==name and result['seed']==seed:
                return result
        # Twelve epochs is an upper step cap. The two-hour active budget usually binds first.
        feeds = read(self.args.data / 'manifest.json')['splits']['train']['feeds']
        return self.case(name, seed, 12 * math.ceil(feeds / 64), stage, full=True)

    def resume_pending_case(self):
        """Finish a saved case before checking the queue's minimum-time gate."""
        phase, current = self.resume_case['phase'], self.resume_case['current']
        if not current or phase not in ('screen', 'full_priority', 'full_new',
                'full_reserve', 'full_seed_repeat', 'full_extra_seed', 'full_EXP031_winner'):
            return
        name, seed, target = current['name'], current['seed'], current['target']
        key = f'{phase}/{name}-s{seed}/{target}'
        checkpoint = self.root / phase / 'runs' / f'{name}-s{seed}' / 'last.pt'
        if key not in self.ledger and checkpoint.exists():
            self.case(name, seed, target, phase, full=phase != 'screen')

    def run(self):
        torch.set_num_threads(4)
        self.wait()
        self.prepare()
        self.resume_pending_case()
        # Guaranteed high-priority larger-data trial: EXP-030's winner, new batch64/prefetch protocol.
        self.full('plr_softmax')
        records = [self.case(n, 42, 800, 'screen') for n in self.new_names]
        selected = family_leaders(records, self.configs)
        # Protect a more expensive history model from a purely early-speed selection.
        if 'memory_256' not in selected and any(r['name']=='memory_256' and r['status']=='completed' for r in records):
            selected.append('memory_256')
        write_json(self.root / 'promotion-3200.json', dict(names=selected, criterion='one best per new family plus long-history slow learner'))
        records = [self.case(n, 42, 3200, 'screen') for n in selected]
        finalists = [r['name'] for r in sorted((r for r in records if r['status']=='completed'),
                                               key=lambda r: -r['fast_ndcg'])[:6]]
        write_json(self.root / 'finalists.json', dict(names=finalists, seeds=[42,43], target=12800))
        confirmed = []
        for n in finalists:
            rows = [self.case(n, seed, 12800, 'screen') for seed in (42,43)]
            if all(r['status']=='completed' for r in rows):
                confirmed.append(dict(name=n, mean_fast=float(np.mean([r['fast_ndcg'] for r in rows]))))
        confirmed.sort(key=lambda r: -r['mean_fast'])
        write_json(self.root / 'ranking.json', dict(ranking=confirmed,
                   criterion='Mean fast NDCG, frozen before any new confirmation scoring'))
        prior = read(self.args.previous / 'report.json', {})
        prior_winner = prior.get('best_mlp')
        if prior_winner in self.configs and prior_winner != 'plr_softmax':
            self.full(prior_winner, stage='full_EXP031_winner')
        # Reserve queue is concrete, distinct work; no empty sleep to fill a clock budget.
        ranked = [r['name'] for r in confirmed]
        reserve = list(dict.fromkeys([*ranked, 'memory_multi4', 'experts_4', 'rankone_8',
                                      'objective_pairwise_mix', 'position_nuisance', 'all_scalars_softmax']))
        tasks = [(n,42,'full_new') for n in ranked[:2]]
        tasks += [('plr_softmax',43,'full_seed_repeat')]
        tasks += [(n,seed,'full_reserve') for seed in (42,43,44) for n in reserve]
        # Extra distinct seeds prevent a fast/early-stopped reserve from emptying the work window.
        tasks += [(n,seed,'full_extra_seed') for seed in range(45,65)
                  for n in list(dict.fromkeys(['plr_softmax',*ranked[:2]]))]
        for name, seed, stage in tasks:
            if self.state.get('elapsed_seconds', 0) >= self.args.minimum_hours*3600:
                break
            self.full(name, seed, stage)
        # Evaluate confirmation only after the adaptive training/ranking decisions are frozen.
        self.state['phase'] = 'final_evaluation'
        for record in confirmed:
            name = record['name']
            trainer = self.trainer(self.root / 'screen')
            trainer.validation.need_sequence = self.configs[name]['sequence']
            try:
                for seed in (42,43):
                    self.event('final_evaluation_started',name=name,seed=seed)
                    trainer.final(self.configs[name],seed)
            finally:
                trainer.close()
        for key, result in self.ledger.items():
            if not result.get('full_train') or 'fast_ndcg' not in result:
                continue
            stage = key.split('/')[0]
            trainer = self.trainer(self.root / stage, full=True, name=result['name'])
            trainer.validation.need_sequence = self.configs[result['name']]['sequence']
            try:
                self.event('final_evaluation_started',name=result['name'],seed=result['seed'])
                trainer.final(self.configs[result['name']],result['seed'])
            finally:
                trainer.close()
        write_json(self.root / 'report.json', dict(experiment='EXP-032', status='completed',
                   new_hypotheses=len(self.new_names), ranking=confirmed, completed_stages=len(self.ledger),
                   elapsed_seconds=self.state.get('elapsed_seconds'), minimum_hours=self.args.minimum_hours,
                   note='All scores are historical selection unless explicitly named otherwise; no test/submission.',
                   ledger=self.ledger))
        self.state.update(status='completed', phase='completed', current=None)
        self.event('queue_completed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--minimum-hours', type=float, default=18)
    parser.add_argument('--maximum-hours', type=float, default=20)
    args = parser.parse_args()
    if not 0 < args.minimum_hours < args.maximum_hours:
        parser.error('Require 0 < minimum-hours < maximum-hours')
    args.output.mkdir(parents=True, exist_ok=True)
    with controller_lock(args.output / 'controller.lock'):
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        queue = Overnight(args)
        try:
            queue.run()
        except Exception as error:
            (args.output / 'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            queue.state.update(status='error',error=str(error))
            # Persist the last state even when the budget itself triggered the exception.
            write_json(args.output / 'status.json',queue.state)
            raise


if __name__=='__main__':
    main()
