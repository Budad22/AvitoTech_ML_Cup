"""Streaming training with atomic, resumable checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
from torch.nn import functional as F

from recsys_data import PreparedSplit
from recsys_data.core import FeedBatch, CANDIDATES
from .metrics import ndcg
from .model import Ranker, features
from training_logs import RunLogger


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def select(batch, indices):
    index = pa.array(indices, type=pa.int64())
    return FeedBatch(batch.features.take(index), batch.labels.take(index) if batch.labels is not None else None,
                     None, batch.embeddings)


def batches(root, split, batch_size, seed=None):
    dataset = PreparedSplit(root, split)
    if seed is not None:
        random.Random(seed).shuffle(dataset.info['files'])
    columns = ['feed_id', 'timestamp', 'user_embedding', 'candidate_count', 'candidate_embedding_present', *CANDIDATES]
    for index, batch in enumerate(dataset.iter_batches(batch_size, include_labels=split != 'test', feature_columns=columns)):
        if seed is not None:
            order = np.random.default_rng(np.random.SeedSequence([seed, index])).permutation(len(batch.features))
            batch = select(batch, order)
        yield batch


def tensors(batch, categories, device):
    dense, category, target, table = features(batch, categories)
    return (torch.from_numpy(dense).to(device), torch.from_numpy(category).long().to(device),
            torch.from_numpy(target).to(device) if target is not None else None, table)


def partition(batch, boundary, part):
    if part == 'all':
        return batch
    timestamps = batch.features['timestamp'].to_numpy()
    keep = timestamps < boundary if part == 'selection' else timestamps >= boundary
    return select(batch, np.flatnonzero(keep)) if keep.any() else None


@torch.no_grad()
def evaluate(model, root, categories, device, boundary, part='all', batch_size=64, output=None, split='val'):
    model.eval()
    sums = np.zeros(4)
    feeds = rows = 0
    writer = None
    try:
        for batch in batches(root, split, batch_size):
            batch = partition(batch, boundary, part)
            if batch is None:
                continue
            dense, category, target, table = tensors(batch, categories, device)
            score = model(dense, category)
            scores = score.cpu().numpy().astype(np.float32)
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite scores')
            if output:
                prediction = pa.table({'feed_id': table['feed_id'], 'item_id': table['item_id'],
                                       'score': pa.array(scores, type=pa.float32())})
                if writer is None:
                    writer = pq.ParquetWriter(output, prediction.schema)
                writer.write_table(prediction)
            if target is not None:
                labels = target.cpu().numpy()
                losses = F.binary_cross_entropy_with_logits(score, target, reduction='none').cpu().numpy()
                for start, end in zip(batch.candidate_offsets[:-1], batch.candidate_offsets[1:]):
                    sums += [ndcg(labels[start:end], scores[start:end], k) for k in (None, 10, 30)] + [float(losses[start:end].mean())]
            feeds += len(batch.features)
            rows += len(scores)
    finally:
        if writer:
            writer.close()
    if not feeds:
        raise ValueError('Empty evaluation partition')
    return dict(zip(('ndcg', 'ndcg_at_10', 'ndcg_at_30', 'logloss'), (sums / feeds).tolist()), feeds=feeds, rows=rows)


def inspect(root, batch_size):
    categories = set()
    training = PreparedSplit(root, 'train')
    print('Inspecting train category column and validation timestamps...', flush=True)
    for shard in training.info['files']:
        column = pq.read_table(Path(root) / 'train' / 'features' / shard['name'], columns=['category_id'])['category_id']
        categories.update(pc.unique(pc.list_flatten(column)).to_pylist())
    validation = PreparedSplit(root, 'val')
    timestamps = np.concatenate([pq.read_table(Path(root) / 'val' / 'features' / shard['name'],
                                columns=['timestamp'])['timestamp'].to_numpy() for shard in validation.info['files']])
    boundary = int(np.sort(timestamps)[min(len(timestamps) - 1, int(len(timestamps) * .8))])
    if not (timestamps < boundary).any():
        raise ValueError('Validation requires at least two distinct timestamps')
    return {int(c): i + 1 for i, c in enumerate(sorted(categories - {0}))}, boundary, {
        'selection': int((timestamps < boundary).sum()), 'control': int((timestamps >= boundary).sum())}


def train(args, logger):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Existing run: use --resume or a fresh output')
    logger.emit('run_initializing', phase='initialize', message='Loading run configuration')
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    state = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    fingerprint = hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest()
    if state:
        if state['fingerprint'] != fingerprint:
            raise ValueError('Prepared data manifest changed')
        for key in ('batch_size', 'seed', 'epochs', 'hours'):
            setattr(args, key, state['config'][key])
        categories, boundary, sizes = state['categories'], state['boundary'], state['sizes']
    else:
        categories, boundary, sizes = inspect(args.data, args.batch_size)
    model = Ranker(len(categories)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    epoch, cursor, elapsed, best, stale, history = 0, 0, 0., -1., 0, []
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch, cursor, elapsed, best, stale, history = [state[k] for k in ('epoch', 'cursor', 'elapsed', 'best', 'stale', 'history')]
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['random_rng'])
        if args.device == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    config = {**vars(args), 'data': str(args.data), 'output': str(output), 'resume': str(args.resume) if args.resume else None}
    config['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
                             'numpy': np.__version__, 'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None}
    config['code_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in Path(__file__).parent.glob('*.py')}
    write_json(output / 'config.json', dict(config, boundary=boundary, partitions=sizes))
    started = last_save = time.monotonic()

    def checkpoint():
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), categories=categories,
                    boundary=boundary, sizes=sizes, fingerprint=fingerprint, config=config,
                    epoch=epoch, cursor=cursor, elapsed=elapsed + time.monotonic() - started,
                    best=best, stale=stale, history=history, torch_rng=torch.get_rng_state(),
                    numpy_rng=np.random.get_state(), random_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None)

    # A real backward pass checks memory before committing to a long run.
    if not state:
        while True:
            try:
                probe = next(batches(args.data, 'train', args.batch_size))
                dense, category, target, _ = tensors(probe, categories, args.device)
                F.binary_cross_entropy_with_logits(model(dense, category), target).backward()
                optimizer.zero_grad(set_to_none=True)
                break
            except torch.cuda.OutOfMemoryError:
                if args.batch_size == 1:
                    raise
                args.batch_size = max(1, args.batch_size // 2)
                config['batch_size'] = args.batch_size
                torch.cuda.empty_cache()
        write_json(output / 'config.json', dict(config, boundary=boundary, partitions=sizes))
    prepared_meta = json.loads((args.data / 'manifest.json').read_text())
    total_batches = sum((file['feeds'] + args.batch_size - 1) // args.batch_size
                        for file in prepared_meta['splits']['train']['files'])
    budget = args.hours * 3600
    last_report = time.monotonic()
    logger.emit('training_started', batch_size=args.batch_size, partitions=sizes,
                budget_seconds=budget, elapsed_seconds=elapsed,
                progress=dict(epoch=epoch + 1, epochs=args.epochs, current=cursor, total=total_batches, unit='батч'))
    while epoch < args.epochs and stale < 2:
        model.train()
        for index, batch in enumerate(batches(args.data, 'train', args.batch_size, args.seed + epoch)):
            if index < cursor:
                continue
            if elapsed + time.monotonic() - started >= args.hours * 3600:
                save_checkpoint(output / 'last.pt', checkpoint())
                if not (output / 'best.pt').exists() and cursor:
                    metrics = evaluate(model, args.data, categories, args.device, boundary, 'selection', args.batch_size)
                    best = metrics['ndcg']
                    history.append(dict(epoch=epoch, partial_epoch=True, **metrics))
                    save_checkpoint(output / 'best.pt', checkpoint())
                    write_json(output / 'history.json', history)
                print('Time budget reached; progress saved.', flush=True)
                logger.emit('budget_exhausted', phase='completed', elapsed_seconds=elapsed + time.monotonic() - started)
                return
            dense, category, target, _ = tensors(batch, categories, args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(dense, category)
            losses = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            loss = torch.stack([part.mean() for part in torch.split(losses, batch.candidate_counts.tolist())]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            cursor = index + 1
            if cursor == 1 or time.monotonic() - last_report >= args.log_interval:
                measured = elapsed + time.monotonic() - started
                logger.emit('batch_completed', epoch=epoch + 1, batch=cursor, loss=loss.item(), seconds=measured,
                            elapsed_seconds=measured, global_step=epoch * total_batches + cursor,
                            metrics=dict(train_loss=loss.item(), learning_rate=optimizer.param_groups[0]['lr']),
                            progress=dict(epoch=epoch + 1, epochs=args.epochs, current=cursor, total=total_batches, unit='батч'))
                last_report = time.monotonic()
            if time.monotonic() - last_save >= 600:
                save_checkpoint(output / 'last.pt', checkpoint())
                logger.emit('checkpoint_saved', checkpoint_at=datetime.now(timezone.utc).isoformat())
                last_save = time.monotonic()
        logger.emit('validation_started', phase='validate')
        metrics = evaluate(model, args.data, categories, args.device, boundary, 'selection', args.batch_size)
        metrics['elapsed_seconds'] = elapsed + time.monotonic() - started
        metrics['peak_cuda_bytes'] = torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0
        epoch += 1
        cursor = 0
        improved = metrics['ndcg'] > best
        stale = 0 if improved else stale + 1
        best = max(best, metrics['ndcg'])
        history.append(dict(epoch=epoch, **metrics))
        if improved:
            save_checkpoint(output / 'best.pt', checkpoint())
        save_checkpoint(output / 'last.pt', checkpoint())
        write_json(output / 'history.json', history)
        logger.emit('epoch_completed', phase='validate', epoch=epoch,
                    elapsed_seconds=elapsed + time.monotonic() - started,
                    metrics=dict(val_ndcg=metrics['ndcg'], val_logloss=metrics['logloss'], best_ndcg=best))
    logger.emit('training_completed', phase='completed', elapsed_seconds=elapsed + time.monotonic() - started)
    print('Training complete.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--data', type=Path, required=True, help='Prepared dataset root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=float, default=10, help='Seconds between progress events; the first batch is always logged')
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.hours, args.log_interval) <= 0:
        parser.error('batch-size, epochs and hours must be positive')
    if args.command == 'train':
        with RunLogger(args.output, args.output.name, 'MLP') as logger:
            train(args, logger)
        return
    if not args.model:
        parser.error('--model is required')
    state = torch.load(args.model, map_location='cpu', weights_only=False)
    model = Ranker(len(state['categories'])).to(args.device)
    model.load_state_dict(state['model'])
    if args.command == 'evaluate':
        result = {part: evaluate(model, args.data, state['categories'], args.device, state['boundary'], part,
                                 args.batch_size) for part in ('selection', 'control', 'all')}
        write_json(args.output, result)
        print(json.dumps(result), flush=True)
    else:
        evaluate(model, args.data, state['categories'], args.device, state['boundary'], 'all', args.batch_size,
                 args.output, 'test')
        prediction = pq.read_table(args.output)
        manifest = json.loads((args.data / 'manifest.json').read_text())
        source = (args.data / manifest['source_relative_path']).resolve()
        template = pq.read_table(source / 'sample_submission.parquet', columns=['feed_id', 'item_id'])
        def keys(table):
            return set(zip(table['feed_id'].to_pylist(), table['item_id'].to_pylist()))
        predicted_keys = keys(prediction)
        if len(predicted_keys) != len(prediction) or len(template) != len(prediction) or predicted_keys != keys(template):
            raise ValueError('Prediction keys do not match template')
        print('Prediction schema, finiteness and exact key coverage verified.', flush=True)
