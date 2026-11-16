"""Resumable sequential trials; decisions require no language-model service."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from ftt.data import Cache, Full, digest, prepare, write_json
from ftt.model import Ranker, configuration, objective


def ndcg(y, scores):
    order = np.lexsort((y, -scores))
    discount = 1 / np.log2(np.arange(len(y)) + 2)
    ideal = (np.sort(y)[::-1] * discount).sum()
    return float((y[order] * discount).sum() / ideal) if ideal else 1.


def save(path, state):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate(model, data, feeds, device):
    model.eval()
    metrics = []
    for index in feeds:
        batch = data.batch([index], device)
        scores = model(batch).cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite validation logits')
        metrics.append(ndcg(batch['target'].cpu().numpy(), scores))
    return np.asarray(metrics)


class Queue:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        prior = self.root / 'status.json'
        self.state = json.loads(prior.read_text(encoding='utf-8')) if prior.exists() else dict(elapsed_seconds=0)
        self.previous_seconds = self.state.get('elapsed_seconds', 0)
        self.state.update(status='running', pid=os.getpid(), experiment='FTT-NIGHT-001', device=args.device)
        self.ledger = json.loads((self.root / 'ledger.json').read_text()) if (self.root / 'ledger.json').exists() else {}
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
        key = f'{size}/FTT-{number:03d}-s{seed}/{target}'
        if key in self.ledger:
            return self.ledger[key]
        config = configuration(number)
        output = self.root / size / f"{config['name']}-s{seed}"
        output.mkdir(parents=True, exist_ok=True)
        train = (Full(self.args.data, Cache(self.root / 'cache', 'train', 'medium').statistics())
                 if size == 'full' else Cache(self.root / 'cache', 'train', 'small' if size == 'smoke' else size))
        if size == 'full':
            target = min(target, 5 * math.ceil(train.feeds / 16))
        validation = self.validation
        torch.manual_seed(seed)
        model = Ranker(config, *train.statistics()).to(self.args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        identity = dict(config=config, seed=seed, cache=digest(self.root / 'cache/metadata.json'), size=size)
        step, best, history, seconds, micro = 0, -1., [], 0., 16
        if (output / 'last.pt').exists():
            state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Checkpoint fingerprint changed')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['rng'])
            if self.args.device == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            step, best, history, seconds, micro = [state[k] for k in ('step', 'best', 'history', 'seconds', 'micro')]
        started = last_emit = time.monotonic()
        def checkpoint():
            return dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.args.device == 'cuda' else None,
                        step=step, best=best, history=history, seconds=seconds+time.monotonic()-started, micro=micro)
        def validate():
            nonlocal best
            # Emit before expensive validation so status never suggests training is hung.
            self.emit('evaluating', phase=size, model=config['name'], seed=seed, step=step, target=target)
            values = evaluate(model, validation, self.fast, self.args.device)
            score = float(values.mean())
            if not np.isfinite(score):
                raise ValueError('Nonfinite or empty validation metric')
            history.append(dict(step=step, ndcg=score))
            if score > best:
                best = score
                save(output / 'best.pt', checkpoint())
            save(output / 'last.pt', checkpoint())
            self.emit('validation', phase=size, model=config['name'], seed=seed, step=step,
                      target=target, best_ndcg=best, ndcg=score)
        def validation_due(current):
            return ((current % 2000 == 0 or current % batches == 0 or current == target) if size == 'full' else
                    (current in (200, 400, 800, 1600, 3200, 6400) or current % 400 == 0 or current == target))
        save(output / 'last.pt', checkpoint())
        batches = math.ceil(train.feeds / 16)
        if target > 20 and step and validation_due(step) and not any(v['step'] == step for v in history):
            validate()
        while step < target:
            epoch, cursor = divmod(step, batches)
            chosen = (np.arange(cursor*16, min((cursor+1)*16, train.feeds)) if size == 'full' else
                      np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(train.feeds)[cursor*16:(cursor+1)*16])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            batch = value = None
            rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if self.args.device == 'cuda' else None
            try:
                for start in range(0, len(chosen), micro):
                    subset = chosen[start:start+micro]
                    batch = train.batch(subset, self.args.device)
                    value = objective(model(batch), batch['target'], batch['counts'], config['loss'])
                    if not torch.isfinite(value):
                        raise ValueError('Nonfinite training objective')
                    (value * len(subset) / len(chosen)).backward()
                    loss_value += float(value.detach()) * len(subset) / len(chosen)
            except torch.cuda.OutOfMemoryError:
                if micro == 1:
                    raise
                micro = max(1, micro // 2)
                batch = value = None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.set_rng_state(rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            for group in optimizer.param_groups:
                group['lr'] = 1e-4 * min(1., (step+1)/100)
            optimizer.step()
            step += 1
            # Persist before heartbeat/budget exceptions so resume retains real progress.
            if step % 100 == 0 or step == target:
                save(output / 'last.pt', checkpoint())
            if step == 1 or step == 20 or time.monotonic()-last_emit > 10:
                self.emit('training', phase=size, model=config['name'], seed=seed, step=step,
                          target=target, loss=loss_value, microbatch=micro)
                last_emit = time.monotonic()
            if target > 20 and validation_due(step):
                validate()
                if size == 'full' and step >= 6000 and len(history) >= 4:
                    if max(v['ndcg'] for v in history[-3:]) <= max(v['ndcg'] for v in history[:-3]):
                        break
            if seconds+time.monotonic()-started > 7200:
                save(output / 'last.pt', checkpoint())
                if size == 'full':
                    if not history or history[-1]['step'] != step:
                        validate()
                    break
                raise TimeoutError('Trial budget exhausted')
        result = dict(number=number, seed=seed, target=target, steps=step, best=best, history=history,
                      seconds=seconds+time.monotonic()-started, directory=str(output))
        self.ledger[key] = result
        write_json(self.root / 'ledger.json', self.ledger)
        return result

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
        self.excluded = json.loads((self.root / 'excluded.json').read_text()) if (self.root / 'excluded.json').exists() else {}
        info = prepare(self.args.data, self.root / 'cache', self.emit)
        self.validation = Cache(self.root / 'cache', 'val')
        selection = np.flatnonzero(self.validation.timestamps < info['boundary'])
        rng = np.random.default_rng(2026)
        self.fast = np.sort(rng.permutation(selection)[:min(2000, len(selection)//2)])
        confirmation = np.setdiff1d(selection, self.fast)
        write_json(self.root / 'partitions.json', dict(boundary=info['boundary'], fast=self.fast.tolist(), confirmation=confirmation.tolist()))
        # Smoke trials are isolated from selection runs.
        for number in range(1, 9):
            self.screen_trial(number, 20, 'smoke')
        for target in (200, 800):
            scores = [r for n in range(1, 9) if (r := self.screen_trial(n, target, 'small'))]
        rank = lambda r: (-r['best'], r['seconds'], r['number'])
        candidates = sorted([r for r in scores if r['number'] != 1], key=rank)
        if not candidates:
            write_json(self.root / 'report.json', dict(decision='inconclusive', reason='No viable extensions'))
            self.emit('completed', status='completed')
            return
        selected = [1, *[r['number'] for r in candidates[:2]]]
        remaining = [r for r in candidates if r['number'] not in selected]
        def growth(r):
            h = {v['step']:v['ndcg'] for v in r['history']}
            return (-(h[800]-h[200]), r['number'])
        if remaining:
            selected.append(min(remaining, key=growth)['number'])
        extended = [r for n in selected if (r := self.screen_trial(n, 3200, 'small'))]
        finalists = [r['number'] for r in sorted([r for r in extended if r['number'] != 1], key=rank)[:2]]
        if not finalists:
            raise RuntimeError('No viable finalists; no full training permitted')
        medium = [r for n in [1, *finalists] if (r := self.screen_trial(n, 3200, 'medium'))]
        growing = any({v['step']:v['ndcg'] for v in r['history']}[3200] >
                      {v['step']:v['ndcg'] for v in r['history']}[1600] for r in medium)
        target = 6400 if growing else 3200
        if growing:
            medium = [r for n in [1, *finalists] if (r := self.screen_trial(n, target, 'medium'))]
        if not any(r['number'] != 1 for r in medium):
            write_json(self.root / 'report.json', dict(decision='inconclusive', reason='No viable medium finalists'))
            self.emit('completed', status='completed')
            return
        winner = min([r for r in medium if r['number'] != 1], key=rank)['number']
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
        delta = np.array(metrics[winner])-np.array(metrics[1])
        _, groups, counts = np.unique(self.validation.users[confirmation], return_inverse=True, return_counts=True)
        sums = np.stack([np.bincount(groups, weights=row) for row in delta])
        draws = []
        for _ in range(20000):
            seeds = rng.integers(0, 3, 3)
            sampled = rng.integers(0, len(counts), len(counts))
            draws.append(sums[seeds].mean(0)[sampled].sum()/counts[sampled].sum())
        lower, upper = np.quantile(draws, [.025, .975])
        qualifies = bool(delta.mean() > 0 and (delta.mean(1)>0).sum() >= 2 and lower > 0)
        report = dict(winner=winner, mean_delta=float(delta.mean()), interval=[float(lower),float(upper)],
                      per_seed=delta.mean(1).tolist(), qualifies=qualifies, full_training='not_started',
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
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    queue = Queue(args)
    lock = queue.root / 'controller.lock'
    # OS-held lock is released on crashes; never use stale PID files as ownership.
    with lock.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        try:
            queue.run()
        except Exception as error:
            (queue.root / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            queue.state.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'failed',
                               error=str(error), heartbeat=time.time())
            write_json(queue.root / 'status.json', queue.state)
            raise


if __name__ == '__main__':
    main()
