"""Matched HSTU-lite pilot with checkpoints independent of the monitoring API."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from mlp_next.supervisor import guard_lock
from mlp_research.training import save
from mlp_trials.state import write_json
from .common import Split, WORK, feed_rows, ndcg_full, val_holdout_mask
from .nn_train import (EMA, FeedData, Ranker, collate, cosine_lr, evaluate, feeds_with_pos,
                       lambda_loss, make_opt, softmax_loss, to_dev)


STAGES = (20, 200, 800)
VARIANTS = (('position', True), ('control', False))
SMOKE_SETTINGS = {
    'position': (1000, .999, 'NDCG · selection'),
    'warmup': (32, .999, 'NDCG · короткий warmup'),
    'ema': (1000, .99, 'NDCG · EMA 0,99'),
    'warmup_ema': (32, .99, 'NDCG · warmup и EMA'),
}


class Pilot:
    def __init__(self, args):
        self.args = args
        self.root = Path(WORK)
        self.report = self.root / 'report.json'
        self.device = torch.device(args.device)
        self.train = Split('train')
        validation = Split('val')
        # Select whole feeds using only membership in the early timestamp partition.
        early = np.flatnonzero(~val_holdout_mask(validation))
        rng = np.random.default_rng(2026)
        self.selection = (early if args.sol_smoke or args.sol_full else
                          np.sort(rng.choice(early, min(len(early), args.val_feeds), replace=False)))
        train_feeds = np.arange(self.train.F)
        if args.sol_smoke:
            train_feeds = np.sort(np.argsort(self.train.ts, kind='stable')[-20_000:])
        self.feeds = feeds_with_pos(self.train, train_feeds)
        if len(self.feeds) < args.batch_size:
            raise ValueError('Too few positive feeds for the pilot')
        self.data = FeedData('train', np.arange(self.train.F), cmax=128, train=True,
                             emb_in_ram=args.sol_full)
        self.steps_per_epoch = (len(self.feeds) // args.batch_size if args.sol_smoke or args.sol_full else
                                math.ceil(len(self.feeds) / args.batch_size))
        if args.sol_smoke:
            args.steps = self.steps_per_epoch
        if args.sol_full:
            args.steps = 2 * self.steps_per_epoch
        self.smoke_names = (('warmup', 'ema', 'warmup_ema') if args.calibrate_smoke
                            else (args.smoke_preset,))
        plan = dict(seed=args.seed, batch_size=args.batch_size, device=args.device,
                    target=args.steps, sol_smoke=args.sol_smoke, selection_feeds=len(self.selection),
                    sol_full=args.sol_full, seed_order=[0, 1, 2] if args.sol_full else [args.seed],
                    smoke_settings={n: list(SMOKE_SETTINGS[n][:2]) for n in self.smoke_names},
                    train_meta=self.train.meta, val_meta=validation.meta,
                    selection_hash=hashlib.sha256(self.selection.tobytes()).hexdigest(),
                    train_feed_hash=hashlib.sha256(self.feeds.tobytes()).hexdigest())
        self.identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        plan_path = self.root / 'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Pilot configuration or data changed. Use a fresh output root.')
        write_json(plan_path, plan)

    def progress(self, stage, **fields):
        status = dict(step=fields.pop('step', 0), steps=self.args.steps,
                      elapsed_seconds=fields.pop('elapsed_seconds', None),
                      updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        # Standard files consumed by the shared read-only dashboard.
        write_json(self.root / 'status.json', status)
        temp = self.root / 'stage.tmp'
        temp.write_text(stage, encoding='utf-8')
        os.replace(temp, self.root / 'stage.txt')

    def case(self, name, use_pos, stop_at):
        args = self.args
        torch.manual_seed(args.seed)
        model = Ranker().to(self.device)
        if args.sol_smoke:
            warmup, decay = SMOKE_SETTINGS[name][:2]
        elif args.sol_full:
            warmup, decay = SMOKE_SETTINGS['position'][:2]
        else:
            warmup, decay = min(50, args.steps // 4), .999
        ema = EMA(model, decay)
        opt = make_opt(model, 1e-3, .02)
        shadow = Ranker().to(self.device)
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        checkpoint = directory / 'last.pt'
        step, elapsed, history = 0, 0., []
        if checkpoint.exists():
            record = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if record['identity'] != (self.identity, name):
                raise ValueError('Checkpoint identity changed')
            model.load_state_dict(record['model'])
            opt.load_state_dict(record['optimizer'])
            ema.shadow = {k: v.to(self.device) for k, v in record['ema'].items()}
            torch.set_rng_state(record['rng'])
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state_all(record['cuda_rng'])
            step, elapsed, history = record['step'], record['elapsed'], record['history']
        started = time.monotonic()

        def snapshot():
            return dict(identity=(self.identity, name), model=model.state_dict(),
                        optimizer=opt.state_dict(), ema=ema.shadow, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
                        step=step, elapsed=elapsed + time.monotonic() - started, history=history)

        def validation():
            shadow.load_state_dict(ema.shadow)
            self.progress('evaluate', step=step, variant=name)
            metric, _ = evaluate(shadow, self.device, 'val', self.selection, workers=0)
            history.append(dict(step=step, ndcg=metric))
            if metric >= max(row['ndcg'] for row in history):
                save(directory / 'best.pt', dict(cfg=model.cfg, sd=ema.shadow, val=metric, step=step))
            save(checkpoint, snapshot())
            print(f'{name} · step {step} · selection NDCG {metric:.6f}', flush=True)

        model.train()
        order_epoch = -1
        cycle = 0
        last_loss = None
        best_step = max((r['step'] for r in history), default=0)
        if step == stop_at and best_step != step:
            validation()
        while step < stop_at:
            if (self.root / 'STOP_AFTER_CHECKPOINT').exists() or elapsed + time.monotonic() - started >= args.max_seconds:
                save(checkpoint, snapshot())
                self.progress('paused', step=step, variant=name)
                return None
            epoch, batch = divmod(step, self.steps_per_epoch)
            if epoch != order_epoch:
                order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch])).permutation(self.feeds)
                self.data.crop_seed = args.seed + epoch
                order_epoch = epoch
            indexes = order[batch * args.batch_size:(batch + 1) * args.batch_size]
            values = to_dev(collate([self.data[int(i)] for i in indexes]), self.device)
            for group in opt.param_groups:
                group['lr'] = cosine_lr(step, args.steps, warmup, .001)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=self.device.type == 'cuda' and torch.cuda.is_bf16_supported()):
                scores = model(values, use_pos=use_pos)
            loss = softmax_loss(scores, values['y'], values['cmask']) + .3 * lambda_loss(scores, values['y'], values['cmask'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            ema.update(model)
            step += 1
            last_loss = float(loss.detach())
            cycle += 1
            if cycle == 1 or cycle % 10 == 0:
                self.progress('train', step=step, variant=name, loss=last_loss,
                              elapsed_seconds=elapsed + time.monotonic() - started,
                              steps_per_second=cycle / max(time.monotonic() - started, .001))
            if step == 1 or step % 25 == 0:
                save(checkpoint, snapshot())
            if step == stop_at:
                validation()
        return dict(name=name, history=history, loss=last_loss, steps=step)

    def run(self):
        if self.args.sol_full:
            self.run_sol_full()
            return
        if self.args.sol_smoke:
            self.run_sol_smoke()
            return
        results = {}
        for target in sorted(set([s for s in STAGES if s < self.args.steps] + [self.args.steps])):
            for name, use_pos in VARIANTS:
                result = self.case(name, use_pos, target)
                if result is None:
                    return
                results[name] = result
        labels = dict(position='NDCG · поправка на позицию', control='NDCG · контроль')
        scores = {labels[name]: row['history'][-1]['ndcg'] for name, row in results.items()}
        write_json(self.report, dict(experiment=self.root.parent.name, status='completed', ndcg=scores,
                   validation_feeds=len(self.selection), train_feeds=self.train.F,
                   positive_train_feeds=len(self.feeds), seed=self.args.seed, results=results))
        self.progress('completed')

    def run_sol_smoke(self):
        """Run section 6.1 on the newest 20k train feeds, then check the model."""
        targets = [*range(100, self.args.steps, 100), self.args.steps]
        from .nn_train import permutation_error
        results, scores = {}, {}
        for name in self.smoke_names:
            for target in targets:
                result = self.case(name, True, target)
                if result is None:
                    return
            checkpoint = self.root / name / 'best.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = Ranker(**state['cfg']).to(self.device)
            model.load_state_dict(state['sd'])
            self.progress('evaluate', step=self.args.steps, variant=name)
            error = permutation_error(model, self.device)
            score = state['val']
            warmup, decay, label = SMOKE_SETTINGS[name]
            scores[label] = score
            results[name] = dict(ndcg=score, best_step=state['step'], warmup=warmup, ema=decay,
                                 permutation_error=error, quality_gate=score > .342 and error < .001,
                                 history=result['history'], last_loss=result['loss'])
            print(json.dumps(dict(variant=name, **results[name]), ensure_ascii=False), flush=True)
        report = dict(experiment=self.root.parent.name, status='completed',
                      ndcg=scores, results=results,
                      validation_feeds=len(self.selection), train_feeds=len(self.feeds),
                      steps=self.args.steps, seed=self.args.seed, batch_size=self.args.batch_size,
                      quality_gate=any(row['quality_gate'] for row in results.values()))
        write_json(self.report, report)
        self.progress('completed', step=self.args.steps)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    def run_sol_full(self):
        """Run the three NN-1 seeds from section 6.2 with independent checkpoints."""
        from .nn_train import predict
        targets = [*range(3000, self.args.steps, 3000), self.args.steps]
        results = {}
        for seed in (0, 1, 2):
            self.args.seed = seed
            name = f'nn1_s{seed}'
            done = self.root / name / 'final.json'
            if done.exists():
                results[name] = json.loads(done.read_text(encoding='utf-8'))
                continue
            for target in targets:
                result = self.case(name, True, target)
                if result is None:
                    return
            state = torch.load(self.root / name / 'best.pt', map_location='cpu', weights_only=False)
            model = Ranker(**state['cfg']).to(self.device)
            model.load_state_dict(state['sd'])
            save(self.root / f'{name}.pt', state)
            self.progress('evaluate', step=self.args.steps, variant=name)
            for split in ('val', 'test'):
                data = Split(split)
                score = predict(model, split, np.arange(data.F), self.device, workers=0)
                if len(score) != data.N or not np.isfinite(score).all():
                    raise ValueError(f'Incomplete {split} prediction for {name}')
                temp = self.root / f'{name}_{split}.tmp.npy'
                np.save(temp, score, allow_pickle=False)
                os.replace(temp, self.root / f'{name}_{split}.npy')
                if split == 'val':
                    control = np.flatnonzero(val_holdout_mask(data))
                    rows, sizes = feed_rows(data.c_off, control)
                    full_val = ndcg_full(data.c_off, score, data.target)[0]
                    holdout = ndcg_full(np.r_[0, sizes.cumsum()], score[rows], data.target[rows])[0]
            results[name] = dict(seed=seed, best_step=state['step'], steps=self.args.steps,
                                 selection_ndcg=state['val'], control_ndcg=holdout, full_val_ndcg=full_val)
            write_json(done, results[name])
        scores = {f'NDCG · полная val · seed {row["seed"]}': row['full_val_ndcg']
                  for row in results.values()}
        write_json(self.report, dict(experiment=self.root.parent.name, status='completed', ndcg=scores,
                                    results=results, train_feeds=len(self.feeds)))
        self.progress('completed', step=self.args.steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--val-feeds', type=int, default=2000)
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--seed', type=int, default=99)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--sol-smoke', action='store_true')
    mode.add_argument('--sol-full', action='store_true')
    parser.add_argument('--calibrate-smoke', action='store_true')
    parser.add_argument('--smoke-preset', choices=SMOKE_SETTINGS, default='position')
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.val_feeds, args.max_seconds) < 1:
        parser.error('Batch size, steps, validation size and time limit must be positive')
    if args.calibrate_smoke and not args.sol_smoke:
        parser.error('--calibrate-smoke requires --sol-smoke')
    torch.set_num_threads(4 if args.device == 'cuda' else 2)
    root = Path(WORK)
    root.mkdir(parents=True, exist_ok=True)
    with guard_lock(root / 'pilot.lock'):
        Pilot(args).run()


if __name__ == '__main__':
    main()
