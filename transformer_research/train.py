"""Train the pinned official architecture on the shared train4/selection cache."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .core import CachedFeeds, evaluate, fit, official_parts, sha256, write_json


@contextmanager
def controller_lock(path):
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args):
    if sys.version_info[:2] != (3, 11) or not torch.__version__.startswith('2.10.0') or np.__version__ != '1.26.4':
        raise RuntimeError('This run requires the official Python 3.11 / Torch 2.10.0 / NumPy 1.26.4 environment')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if min(args.epochs, args.batch_size, args.maximum_seconds) <= 0:
        raise ValueError('Positive epochs, batch size and budget required')
    compact = args.dataset_format == 'compact'
    if compact:
        from .compact import CompactFeeds, shuffled_loader
    metadata = json.loads((args.cache / 'metadata.json').read_text(encoding='utf-8'))
    if ((metadata['splits']['train']['feeds'] != 3151208 if compact else metadata['splits']['train'] != dict(feeds=24592, rows=1925512))
            or metadata['splits']['val'] != dict(feeds=11534, rows=1101295)
            or metadata['boundary'] != 1790249984):
        raise ValueError('The registered common cohort changed')
    fast, confirmation = (np.load(args.cache / (name + '.npy')) for name in ('fast', 'confirmation'))
    if (len(fast), len(confirmation)) != (2000, 9534) or not np.array_equal(
            np.sort(np.r_[fast, confirmation]), np.arange(11534)):
        raise ValueError('Fast/confirmation partition changed')
    ranker, collate, native = official_parts(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    with controller_lock(args.output / 'controller.lock'):
        status_file = args.output / 'status.json'
        previous = json.loads(status_file.read_text(encoding='utf-8-sig')) if status_file.exists() else {}
        if previous.get('status') == 'completed':
            return
        used = [args.cache / name for name in ('metadata.json', 'fast.npy', 'confirmation.npy')]
        used += [args.cache / split / (name + '.npy') for split in ('train', 'val')
                 for name in (('floats', 'category', 'target', 'offsets', 'feed_id', 'item_id', 'item_index', 'users', 'timestamp', 'user_id') if compact else ('dense', 'category', 'target', 'offsets', 'feed_id', 'item_id'))]
        environment = {name: importlib.metadata.version(name) for name in
                       ('torch', 'numpy', 'polars', 'click', 'loguru', 'catboost')}
        identity = dict(experiment=args.experiment, native_config=native,
                        epochs=args.epochs, batch_size=args.batch_size, maximum_seconds=args.maximum_seconds,
                        device=args.device, seed=42, source_sha256=sha256(args.source),
                        cache_sha256={str(p.relative_to(args.cache)).replace('\\', '/'): sha256(p) for p in used},
                        code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                        environment=environment, checkpoint_criterion='minimum fast2000 row logloss',
                        metric='full NDCG with equal feed weights and pessimistic ties',
                        score_transform='float32 sigmoid', full_train=compact, test_read=False,
                        differences=[('native on-demand normalization; compact memmaps' if compact else 'common cached normalized inputs; train4 instead of full train'),
                                     'fast2000 instead of all validation', 'CUDA wheel of the pinned torch version'])
        config_file = args.output / 'config.json'
        if config_file.exists() and json.loads(config_file.read_text(encoding='utf-8-sig')) != identity:
            raise ValueError('Immutable run configuration changed')
        write_json(config_file, identity)
        elapsed = previous.get('elapsed_seconds', 0.)
        started = time.monotonic()
        deadline = started + max(0, args.maximum_seconds - elapsed)
        def emit(event, **fields):
            if args.device == 'cuda':
                fields.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                              gpu_peak_bytes=torch.cuda.max_memory_allocated())
            value = dict(status='running', event=event, pid=os.getpid(), device=args.device,
                         elapsed_seconds=elapsed + time.monotonic() - started,
                         updated_at=datetime.now(timezone.utc).isoformat(), **fields)
            write_json(status_file, value)
            with (args.output / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value) + '\n')
            print(json.dumps(value), flush=True)
        training = validation = None
        try:
            emit('initializing', gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None)
            dataset_type = CompactFeeds if compact else CachedFeeds
            training, validation = dataset_type(args.cache, 'train'), dataset_type(args.cache, 'val')
            torch.set_num_threads(8)
            torch.manual_seed(42)
            model = ranker(len(metadata['categories']), width=64, layers=2, dropout=.1).to(args.device)
            loader = (shuffled_loader(training, args.batch_size, collate) if compact else
                      DataLoader(training, batch_size=args.batch_size, shuffle=True, collate_fn=collate))
            fast_loader = DataLoader(Subset(validation, fast.tolist()), batch_size=args.batch_size, collate_fn=collate)
            result = fit(model, loader, fast_loader, args.output, identity, args.device, args.epochs, emit, deadline)
            write_json(args.output / 'fit-result.json', result)
            best = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
            # Commit checkpoint selection before opening the remaining selection feeds.
            write_json(args.output / 'frozen.json', dict(config=identity, selected_epoch=best['epoch'],
                       checkpoint_sha256=sha256(args.output / 'best.pt'), history=result['history']))
            model.load_state_dict(best['model'])
            full_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate)
            measured = evaluate(model, full_loader, args.device, predictions=True)
            if measured['rows'] != 1101295 or measured['feeds'] != 11534:
                raise ValueError('Final validation coverage differs')
            np.save(args.output / 'selection_scores.npy', measured['scores'], allow_pickle=False)
            np.save(args.output / 'selection_ndcg.npy', measured['per_feed'], allow_pickle=False)
            report = dict(status='completed', experiment=args.experiment, epochs=args.epochs,
                          steps=result['steps'], selected_epoch=best['epoch'],
                          parameters=sum(p.numel() for p in model.parameters()),
                          selection_ndcg=measured['ndcg'], selection_logloss=measured['logloss'],
                          fast_ndcg=float(measured['per_feed'][fast].mean()),
                          confirmation_ndcg=float(measured['per_feed'][confirmation].mean()),
                          checkpoint_sha256=sha256(args.output / 'best.pt'), history=result['history'],
                          independent_confirmation=False, full_train=compact,
                          note='Pinned official architecture; adapted common cache and validation, one seed',
                          test_read=False, submission_created=False)
            write_json(args.output / 'report.json', report)
            emit('completed', report=str(args.output / 'report.json'))
            current = json.loads(status_file.read_text(encoding='utf-8'))
            current['status'] = 'completed'
            write_json(status_file, current)
        except Exception as error:
            emit('stopped', error=str(error), traceback=traceback.format_exc())
            current = json.loads(status_file.read_text(encoding='utf-8'))
            current['status'] = ('paused' if isinstance(error, InterruptedError) else
                                 'incomplete_budget' if isinstance(error, TimeoutError) else 'error')
            write_json(status_file, current)
            raise
        finally:
            for dataset in (training, validation):
                if dataset is not None:
                    dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-format', choices=('dense', 'compact'), default='dense')
    parser.add_argument('--experiment', default='EXP-TRF-001-r01')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--maximum-seconds', type=int, default=1800)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
