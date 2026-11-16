"""Isolated, resumable full-data ablations with selection-only checkpoint choice."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from mlp.metrics import ndcg
from mlp.pipeline import prepared_batches
from mlp.training import inspect, partition, save_checkpoint
from training_logs import RunLogger
from .data import HISTORY_KINDS, batches, move, pack
from .losses import objective
from .models import KINDS, create_model, score
from .state import write_json


def fingerprint(root):
    return hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()


def schema(root, output, batch_size=64):
    categories, boundary, sizes = inspect(root, batch_size)
    value = dict(categories=categories, boundary=boundary, sizes=sizes, fingerprint=fingerprint(root))
    write_json(output, value)
    return value


@torch.no_grad()
def evaluate(model, root, reference, device, kind, window, batch_size=64, part='selection', output=None, split='val'):
    model.eval()
    sums = np.zeros(4)
    feeds = rows = 0
    writer = None
    temporary = output.with_suffix('.tmp.parquet') if output else None
    try:
        for batch in batches(root, split, batch_size, history=kind in HISTORY_KINDS, cache='ram'):
            batch = partition(batch, reference['boundary'], part)
            if batch is None:
                continue
            packed = pack(batch, reference['categories'], kind=kind, history_window=window)
            inputs = move(packed, device)
            logits = score(model, inputs, kind)
            scores = logits.cpu().numpy().astype(np.float32)
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite predictions')
            if output:
                prediction = pa.table({'feed_id': packed['table']['feed_id'], 'item_id': packed['table']['item_id'],
                                       'score': pa.array(scores, type=pa.float32())})
                if writer is None:
                    writer = pq.ParquetWriter(temporary, prediction.schema)
                writer.write_table(prediction)
            if 'target' in inputs:
                labels = inputs['target'].cpu().numpy()
                losses = F.binary_cross_entropy_with_logits(logits, inputs['target'], reduction='none').cpu().numpy()
                for start, end in zip(batch.candidate_offsets[:-1], batch.candidate_offsets[1:]):
                    sums += [ndcg(labels[start:end], scores[start:end], k) for k in (None, 10, 30)] + [float(losses[start:end].mean())]
            feeds += packed['feeds']
            rows += packed['candidates']
    finally:
        if writer:
            writer.close()
    if not feeds:
        raise ValueError('Empty evaluation partition')
    if output:
        os.replace(temporary, output)
    return dict(zip(('ndcg', 'ndcg_at_10', 'ndcg_at_30', 'logloss'), (sums / feeds).tolist()), feeds=feeds, rows=rows)


def train(args, logger):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Use --resume for an existing run')
    logger.emit('run_initializing', phase='initialize', kind=args.kind)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    reference = json.loads(args.schema.read_text(encoding='utf-8'))
    reference['categories'] = {int(k): v for k, v in reference['categories'].items()}
    if reference['fingerprint'] != fingerprint(args.data):
        raise ValueError('Prepared manifest changed')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    state = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if state:
        for key in ('kind', 'batch_size', 'epochs', 'hours', 'learning_rate', 'seed', 'history_window'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        if state['reference'] != reference:
            raise ValueError('Resume data or split changed')
    model = create_model(len(reference['categories']), args.kind, args.history_window).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    epoch, cursor, elapsed, best, stale, history = 0, 0, 0., -1., 0, []
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch, cursor, elapsed, best, stale, history = [state[k] for k in ('epoch', 'cursor', 'elapsed', 'best', 'stale', 'history')]
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['random_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    config['environment'] = dict(torch=torch.__version__, numpy=np.__version__,
                                 gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None)
    write_json(output / 'config.json', dict(config, boundary=reference['boundary'], partitions=reference['sizes']))
    started = last_save = last_report = time.monotonic()
    packer = partial(pack, kind=args.kind, history_window=args.history_window)
    total = sum((f['feeds'] + args.batch_size - 1) // args.batch_size
                for f in json.loads((args.data / 'manifest.json').read_text())['splits']['train']['files'])

    def checkpoint():
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), reference=reference, config=config,
                    epoch=epoch, cursor=cursor, elapsed=elapsed + time.monotonic() - started,
                    best=best, stale=stale, history=history, torch_rng=torch.get_rng_state(),
                    numpy_rng=np.random.get_state(), random_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None)

    # Same warm-up RNG consumption for the baseline and the two objective ablations.
    if not state:
        probe = next(batches(args.data, 'train', args.batch_size, history=args.kind in HISTORY_KINDS, cache='ram'))
        inputs = move(packer(probe, reference['categories']), args.device)
        objective(score(model, inputs, args.kind), inputs, args.kind).backward()
        optimizer.zero_grad(set_to_none=True)
        del probe, inputs
    budget = args.hours * 3600
    logger.emit('training_started', budget_seconds=budget, elapsed_seconds=elapsed, batch_size=args.batch_size,
                progress=dict(epoch=epoch + 1, epochs=args.epochs, current=cursor, total=total, unit='batch'))
    reason = 'epochs'
    exhausted = False
    while epoch < args.epochs and stale < 2:
        model.train()
        source = batches(args.data, 'train', args.batch_size, seed=args.seed + epoch, start_batch=cursor,
                         history=args.kind in HISTORY_KINDS, cache='ram')
        window_started = time.monotonic()
        window_steps = window_candidates = 0
        loader_wait = 0.
        first = True
        with prepared_batches(source, reference['categories'], device=args.device, prefetch=2, packer=packer) as ready:
            iterator = iter(ready)
            while True:
                waiting = time.monotonic()
                try:
                    packed = next(iterator)
                except StopIteration:
                    break
                loader_wait += time.monotonic() - waiting
                if elapsed + time.monotonic() - started >= budget:
                    exhausted = True
                    reason = 'budget'
                    break
                inputs = move(packed, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = objective(score(model, inputs, args.kind), inputs, args.kind)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                cursor += 1
                window_steps += 1
                window_candidates += packed['candidates']
                if first or time.monotonic() - last_report >= 10:
                    duration = max(time.monotonic() - window_started, 1e-9)
                    logger.emit('batch_completed', epoch=epoch + 1, batch=cursor, global_step=epoch * total + cursor,
                                elapsed_seconds=elapsed + time.monotonic() - started,
                                metrics=dict(train_loss=loss.item(), learning_rate=args.learning_rate,
                                             batches_per_second=window_steps / duration,
                                             candidates_per_second=window_candidates / duration,
                                             loader_wait_pct=100 * loader_wait / duration),
                                progress=dict(epoch=epoch + 1, epochs=args.epochs, current=cursor, total=total, unit='batch'))
                    first = False
                    last_report = window_started = time.monotonic()
                    window_steps = window_candidates = 0
                    loader_wait = 0.
                if time.monotonic() - last_save >= 300 or (output / 'STOP_AFTER_CHECKPOINT').exists():
                    save_checkpoint(output / 'last.pt', checkpoint())
                    last_save = time.monotonic()
                    logger.emit('checkpoint_saved')
                    if (output / 'STOP_AFTER_CHECKPOINT').exists():
                        logger.emit('paused_by_request', phase='stopped')
                        return
        # A partial epoch is eligible for selection; later control stays sealed.
        if cursor:
            logger.emit('validation_started', phase='validate')
            metrics = evaluate(model, args.data, reference, args.device, args.kind, args.history_window, args.batch_size)
            metrics.update(elapsed_seconds=elapsed + time.monotonic() - started,
                           peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0)
            improved = metrics['ndcg'] > best
            stale = 0 if improved else stale + 1
            best = max(best, metrics['ndcg'])
            history.append(dict(epoch=epoch + 1, partial_epoch=exhausted, **metrics))
            if not exhausted:
                epoch += 1
                cursor = 0
            if improved:
                save_checkpoint(output / 'best.pt', checkpoint())
            save_checkpoint(output / 'last.pt', checkpoint())
            write_json(output / 'history.json', history)
            logger.emit('epoch_completed', phase='validate', epoch=epoch if not exhausted else epoch + 1,
                        metrics=dict(val_ndcg=metrics['ndcg'], val_logloss=metrics['logloss'], best_ndcg=best))
        if exhausted:
            break
    if stale >= 2:
        reason = 'early_stopping'
    if not (output / 'best.pt').exists():
        raise RuntimeError('No trained checkpoint; increase the budget')
    result = dict(status='completed', kind=args.kind, selection_ndcg=best, epochs=epoch,
                  stop_reason=reason, history=history, elapsed_seconds=elapsed + time.monotonic() - started)
    write_json(output / 'result.json', result)
    logger.emit('training_completed', phase='completed', stop_reason=reason,
                elapsed_seconds=result['elapsed_seconds'], metrics=dict(best_ndcg=best))


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['schema', 'train', 'evaluate', 'predict'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--schema', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--kind', choices=KINDS, default='baseline')
    parser.add_argument('--part', choices=['selection', 'control', 'all'], default='selection')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hours', type=float, default=2)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--history-window', type=int, default=64)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.hours, args.learning_rate, args.history_window) <= 0:
        parser.error('Training limits must be positive')
    torch.set_num_threads(4)
    pa.set_cpu_count(2)
    if args.command == 'schema':
        schema(args.data, args.output, args.batch_size)
    elif args.command == 'train':
        if not args.schema:
            parser.error('--schema is required')
        with RunLogger(args.output, args.output.name, 'MLP/' + args.kind) as logger:
            train(args, logger)
    else:
        if not args.model:
            parser.error('--model is required')
        state = torch.load(args.model, map_location='cpu', weights_only=False)
        if state['reference']['fingerprint'] != fingerprint(args.data):
            raise ValueError('Prepared manifest changed')
        kind, window = state['config']['kind'], state['config']['history_window']
        model = create_model(len(state['reference']['categories']), kind, window).to(args.device)
        model.load_state_dict(state['model'])
        metrics = evaluate(model, args.data, state['reference'], args.device, kind, window, args.batch_size,
                           args.part, args.output if args.command == 'predict' else None,
                           'test' if args.command == 'predict' else 'val')
        if args.command == 'evaluate':
            write_json(args.output, metrics)
        print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
