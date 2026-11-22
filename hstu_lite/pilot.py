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
from .common import Split, WORK, val_holdout_mask
from .nn_train import (EMA, FeedData, Ranker, collate, cosine_lr, evaluate, feeds_with_pos,
                       lambda_loss, make_opt, softmax_loss, to_dev)


STAGES = (20, 200, 800)
VARIANTS = (('position', True), ('control', False))


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
        self.selection = np.sort(rng.choice(early, min(len(early), args.val_feeds), replace=False))
        self.feeds = feeds_with_pos(self.train, np.arange(self.train.F))
        if len(self.feeds) < args.batch_size:
            raise ValueError('Too few positive feeds for the pilot')
        self.data = FeedData('train', np.arange(self.train.F), cmax=128, train=True)
        self.steps_per_epoch = math.ceil(len(self.feeds) / args.batch_size)
        plan = dict(seed=args.seed, batch_size=args.batch_size, device=args.device,
                    target=args.steps, fast_feeds=args.val_feeds,
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
        ema = EMA(model, .999)
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
                group['lr'] = cosine_lr(step, args.steps, min(50, args.steps // 4), .001)
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
        results = {}
        for target in sorted(set([s for s in STAGES if s < self.args.steps] + [self.args.steps])):
            for name, use_pos in VARIANTS:
                result = self.case(name, use_pos, target)
                if result is None:
                    return
                results[name] = result
        labels = dict(position='С поправкой на позицию', control='Контроль')
        scores = {labels[name]: row['history'][-1]['ndcg'] for name, row in results.items()}
        write_json(self.report, dict(experiment='EXP-HSTU-001', status='completed', ndcg=scores,
                   validation_feeds=len(self.selection), train_feeds=self.train.F,
                   positive_train_feeds=len(self.feeds), seed=self.args.seed, results=results))
        self.progress('completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--val-feeds', type=int, default=2000)
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--seed', type=int, default=99)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.val_feeds, args.max_seconds) < 1:
        parser.error('Batch size, steps, validation size and time limit must be positive')
    torch.set_num_threads(4 if args.device == 'cuda' else 2)
    root = Path(WORK)
    root.mkdir(parents=True, exist_ok=True)
    with guard_lock(root / 'pilot.lock'):
        Pilot(args).run()


if __name__ == '__main__':
    main()
