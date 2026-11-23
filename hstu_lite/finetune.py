"""Resumable NN-2 temporal calibration and grouped cross-validation (§6.3)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from mlp_next.supervisor import guard_lock
from mlp_research.training import save
from mlp_trials.state import replace_atomic, write_json, write_text
from .common import Split, WORK, feed_rank_pct, feed_rows, ndcg_full, val_folds, val_holdout_mask
from .monitor import Progress
from .nn_train import EMA, FeedData, Ranker, collate, cosine_lr, evaluate, feeds_with_pos
from .nn_train import lambda_loss, make_opt, predict, softmax_loss, to_dev


class Paused(Exception):
    pass


class Batches:
    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.count = math.ceil(len(dataset) / batch_size)
        if not self.count:
            raise ValueError('Empty training partition')
        self.epoch = -1

    def at(self, step):
        epoch, batch = divmod(step, self.count)
        if epoch != self.epoch:
            self.order = np.random.default_rng([self.seed, epoch]).permutation(len(self.dataset))
            self.dataset.crop_seed = self.seed + epoch
            self.epoch = epoch
        start = batch * self.batch_size
        return collate([self.dataset[int(i)] for i in self.order[start:start + self.batch_size]])


def save_array(path, array):
    temporary = path.with_suffix('.tmp.npy')
    np.save(temporary, array, allow_pickle=False)
    replace_atomic(temporary, path)


class Finetuning:
    def __init__(self, args, root):
        self.args, self.root = args, root
        self.monitor = Progress(root)
        self.device = torch.device(args.device)
        self.amp = self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        self.val = Split('val')
        self.train = Split('train')
        recent = np.sort(np.argsort(self.train.ts, kind='stable')[-300_000:])
        self.mix_feeds = feeds_with_pos(self.train, recent) if args.mix_train else None
        self.mix_data = (FeedData('train', self.mix_feeds, cmax=128, train=True,
                                  emb_in_ram=self.device.type == 'cuda')
                         if self.mix_feeds is not None else None)

    def fit(self, seed, tag, fitting, *, check=None, initial_seed=None):
        args = self.args
        source_seed = seed if initial_seed is None else initial_seed
        path = args.nn1 / f'nn1_s{source_seed}.pt'
        initial = torch.load(path, map_location='cpu', weights_only=False)
        torch.manual_seed(seed)
        model = Ranker(**initial['cfg']).to(self.device)
        model.load_state_dict(initial['sd'])
        shadow = Ranker(**initial['cfg']).to(self.device)
        ema = EMA(model, .99)
        optimizer = make_opt(model, args.lr, .02)
        fitting = feeds_with_pos(self.val, fitting)
        batches = Batches(FeedData('val', fitting, cmax=256, train=True), 32, seed)
        mix = Batches(self.mix_data, 32, seed + 1000) if self.mix_data is not None else None
        total = args.epochs * batches.count
        checkpoint = self.root / tag / 'last.pt'
        checkpoint.parent.mkdir(exist_ok=True)
        identity = dict(seed=seed, tag=tag, epochs=args.epochs, lr=args.lr, mix=args.mix_train,
                        device=args.device, initial=hashlib.sha256(path.read_bytes()).hexdigest(),
                        feeds=hashlib.sha256(fitting.tobytes()).hexdigest(),
                        check=hashlib.sha256(check.tobytes()).hexdigest() if check is not None else None,
                        val_meta=self.val.meta, train_meta=self.train.meta)
        step, previous, history = 0, 0., []
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['identity'] != identity:
                raise ValueError(f'Checkpoint configuration changed for {tag}')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            ema.shadow = {k: value.to(self.device) for k, value in state['ema'].items()}
            torch.set_rng_state(state['rng'])
            if self.amp or self.device.type == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            step, previous, history = state['step'], state['seconds'], state['history']
        started = time.monotonic()

        def checkpoint_now():
            save(checkpoint, dict(identity=identity, step=step, model=model.state_dict(),
                 optimizer=optimizer.state_dict(), ema=ema.shadow, history=history,
                 rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
                 seconds=previous + time.monotonic() - started))

        def validate_epoch():
            epoch = step // batches.count
            if check is None or not step or step % batches.count or len(history) == epoch:
                return
            shadow.load_state_dict(ema.shadow)
            self.monitor.publish('evaluate', step=step, steps=total, variant=tag)
            value, _ = evaluate(shadow, self.device, 'val', check, workers=0)
            history.append(dict(epoch=epoch, ndcg=value))
            checkpoint_now()
            print(f'{tag} epoch {epoch}: tuning NDCG {value:.8f}', flush=True)

        validate_epoch()
        model.train()
        while step < total:
            if (self.root / 'STOP_AFTER_CHECKPOINT').exists():
                checkpoint_now()
                self.monitor.publish('paused', step=step, steps=total, variant=tag)
                raise Paused
            values = to_dev(batches.at(step), self.device)
            for group in optimizer.param_groups:
                group['lr'] = cosine_lr(step, total, 50, args.lr)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
                scores = model(values)
            loss = softmax_loss(scores, values['y'], values['cmask']) + lambda_loss(scores, values['y'], values['cmask'])
            if mix is not None:
                extra = to_dev(mix.at(step), self.device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
                    scores = model(extra, use_pos=True)
                loss = loss + args.mix_train * softmax_loss(scores, extra['y'], extra['cmask'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{tag} nonfinite loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            step += 1
            if step == 1 or step % 10 == 0:
                self.monitor.publish('train', step=step, steps=total, variant=tag,
                                     elapsed_seconds=previous + time.monotonic() - started,
                                     loss=float(loss.detach()))
            if step == 1 or step % 100 == 0 or step % batches.count == 0:
                checkpoint_now()
            validate_epoch()
        shadow.load_state_dict(ema.shadow)
        return shadow, history

    def holdout(self):
        late = val_holdout_mask(self.val)
        tuning = np.flatnonzero(late)
        self.monitor.publish('evaluate', variant='nn1_s0')
        state = torch.load(self.args.nn1 / 'nn1_s0.pt', map_location='cpu', weights_only=False)
        base = Ranker(**state['cfg']).to(self.device)
        base.load_state_dict(state['sd'])
        baseline, _ = evaluate(base, self.device, 'val', tuning, workers=0)
        del base
        print(f'NN-1 on late tuning feeds: NDCG {baseline:.8f}', flush=True)
        _, history = self.fit(0, 'holdout', np.flatnonzero(~late), check=tuning)
        best = max(history, key=lambda row: row['ndcg'])
        return dict(ndcg={'NDCG · NN-1': baseline, 'NDCG · NN-2 подбор': best['ndcg']},
                    baseline=baseline, best_epochs=best['epoch'], best_ndcg=best['ndcg'],
                    tuning_feeds=len(tuning), history=history, lr=self.args.lr, mix_train=self.args.mix_train)

    def cv(self, seeds):
        folds = val_folds(self.val)
        test = Split('test')
        results = {}
        for seed, initial_seed in seeds:
            oof = np.full(self.val.N, np.nan, dtype=np.float32)
            test_mean = np.zeros(test.N, dtype=np.float64)
            for fold in range(5):
                fitted = np.flatnonzero(folds != fold)
                hold = np.flatnonzero(folds == fold)
                if np.intersect1d(self.val.user_id[fitted], self.val.user_id[hold]).size:
                    raise ValueError('User appears in both training and held-out folds')
                tag = f'seed{seed}-fold{fold}'
                folder = self.root / tag
                if (folder / 'fold.json').exists():
                    held_rank = np.load(folder / 'val.npy', allow_pickle=False)
                    test_rank = np.load(folder / 'test.npy', allow_pickle=False)
                else:
                    model, _ = self.fit(seed, tag, fitted, initial_seed=initial_seed)
                    self.monitor.publish('evaluate', variant=tag)
                    scores = predict(model, 'val', hold, self.device, workers=0)
                    rows, sizes = feed_rows(self.val.c_off, hold)
                    held_rank = feed_rank_pct(np.r_[0, sizes.cumsum()], scores[rows])
                    scores = predict(model, 'test', np.arange(test.F), self.device, workers=0)
                    test_rank = feed_rank_pct(test.c_off, scores)
                    save_array(folder / 'val.npy', held_rank)
                    save_array(folder / 'test.npy', test_rank)
                    write_json(folder / 'fold.json', dict(seed=seed, fold=fold, heldout_feeds=len(hold)))
                rows, _ = feed_rows(self.val.c_off, hold)
                oof[rows] = held_rank
                test_mean += test_rank / 5
            score = ndcg_full(self.val.c_off, oof, self.val.target)[0]
            save_array(self.root / f'nn2_s{seed}_oof.npy', oof)
            save_array(self.root / f'nn2_s{seed}_test.npy', test_mean.astype('float32'))
            results[f'NDCG · OOF · seed {seed}'] = score
        return dict(ndcg=results, epochs=self.args.epochs, lr=self.args.lr,
                    mix_train=self.args.mix_train, seeds=seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('holdout', 'cv'), required=True)
    parser.add_argument('--nn1', type=Path, required=True, help='NN-1 work directory')
    parser.add_argument('--cv-seed', nargs=2, metavar=('SEED', 'NN1_SEED'), type=int, action='append',
                        help='Train a CV seed from the specified NN-1 checkpoint. Repeat for more seeds.')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mix-train', type=float, default=.25)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.mix_train < 0:
        parser.error('Invalid training configuration')
    seeds = args.cv_seed or [[seed, seed] for seed in range(3)]
    if args.cv_seed and args.mode != 'cv':
        parser.error('--cv-seed requires --mode cv')
    if any(not 0 <= value < 2 ** 32 for pair in seeds for value in pair):
        parser.error('Seeds must be unsigned 32-bit integers')
    if len({seed for seed, _ in seeds}) != len(seeds):
        parser.error('CV training seeds must be unique')
    torch.set_num_threads(4 if args.device == 'cuda' else 2)
    root = Path(WORK)
    root.mkdir(parents=True, exist_ok=True)
    with guard_lock(root / 'nn2.lock'):
        if args.mode == 'cv':
            configuration = root / 'cv-config.json'
            identity = dict(seeds=seeds, epochs=args.epochs, lr=args.lr, mix_train=args.mix_train,
                            nn1=str(args.nn1.resolve()), device=args.device)
            if configuration.exists():
                if json.loads(configuration.read_bytes()) != identity:
                    raise ValueError('CV configuration changed. Use a new experiment directory.')
            else:
                if args.cv_seed and any(root.glob('seed*-fold*')):
                    raise ValueError('Custom CV seeds require a fresh experiment directory')
                write_json(configuration, identity)
        report = root / 'report.json'
        if report.exists():
            write_text(root / 'stage.txt', 'completed')
            return
        (root / 'failure.log').unlink(missing_ok=True)
        Progress(root).publish('prepare')
        try:
            stage = Finetuning(args, root)
            result = stage.holdout() if args.mode == 'holdout' else stage.cv(seeds)
            write_json(report, dict(experiment=root.parent.name, status='completed', mode=args.mode, **result))
            stage.monitor.publish('completed')
        except Paused:
            return
        except Exception:
            write_text(root / 'failure.log', traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
