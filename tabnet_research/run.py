"""Train a matched MLP/TabNet pair; no test reader or submission path exists."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def configure_source(path, dependencies=None):
    """Import only the exact pinned source, without editing the external project."""
    if dependencies:
        sys.path.insert(0, str(Path(dependencies).resolve()))
    path = Path(path).resolve()
    info = json.loads((path / 'source.json').read_text(encoding='utf-8'))
    if info['commit'] != 'd4f9edb' and not info['commit'].startswith('d4f9edb'):
        raise ValueError('Expected the reviewed MLP source d4f9edb')
    for name, expected in info['files'].items():
        if sha256(path / name) != expected:
            raise ValueError('External MLP source changed: ' + name)
    sys.path.insert(0, str(path))
    return info


@contextmanager
def controller_lock(path):
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def checkpoint(path, value):
    import torch
    temporary = path.with_suffix('.tmp.pt')
    torch.save(value, temporary)
    os.replace(temporary, path)


def import_parent_checkpoints(parent, folder, kind, seed, parent_identity, identity):
    """Fork optimizer/RNG state into a new run; never write to the parent."""
    import torch
    for name in ('last.pt', 'best.pt'):
        state = torch.load(parent / f'{kind}-s{seed}' / name,
                           map_location='cpu', weights_only=False)
        if state['identity'] != dict(parent_identity, kind=kind, seed=seed):
            raise ValueError('Parent checkpoint identity differs: ' + name)
        if name == 'last.pt' and state['step'] != parent_identity['steps']:
            raise ValueError('Parent did not finish its registered steps')
        state['identity'] = dict(identity, kind=kind, seed=seed)
        checkpoint(folder / name, state)


class BatchDevice:
    """Keep arrays on SSD and move only the current batch to the accelerator."""
    def __init__(self, source, device):
        self.source, self.device = source, device
        self.arrays, self.feeds = source.arrays, source.feeds

    def batch(self, feeds, **options):
        return {name: value if name == 'counts' else value.to(self.device)
                for name, value in self.source.batch(feeds, **options).items()}

    def close(self):
        self.source.close()


def run_case(args, kind, seed, train, validation, statistics, fast, identity, emit, deadline):
    import numpy as np
    import torch
    from mlp_research.training import feed_order, evaluate
    from tabnet_research.model import make_model, control_config, training_loss

    folder = args.output / f'{kind}-s{seed}'
    folder.mkdir(exist_ok=True)
    case_identity = dict(identity, kind=kind, seed=seed)
    torch.manual_seed(seed)
    model = make_model(kind, args.categories, statistics).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    step, best, seconds, history = 0, -1., 0., []
    if not (folder / 'last.pt').exists() and getattr(args, 'resume_from', None):
        import_parent_checkpoints(args.resume_from, folder, kind, seed,
                                  args.parent_identity, identity)
    if (folder / 'last.pt').exists():
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if state['identity'] != case_identity:
            raise ValueError('Checkpoint identity differs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        step, best, seconds, history = (state[key] for key in ('step', 'best', 'seconds', 'history'))
    initial_step = step
    started = last_report = time.monotonic()
    waiting = computing = 0.

    def state():
        return dict(identity=case_identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None,
                    step=step, best=best, seconds=seconds + time.monotonic() - started, history=history)

    for index in range(step, args.steps):
        if time.monotonic() >= deadline:
            checkpoint(folder / 'last.pt', state())
            raise TimeoutError('Campaign budget exhausted; incomplete, not a negative result')
        before = time.monotonic()
        feeds = feed_order(train.feeds, seed, index, args.batch_size)
        inputs = train.batch(feeds, history=True, context=False)
        waiting += time.monotonic() - before
        before = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        value = training_loss(model, inputs, kind)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        loss_value = float(value.detach())
        step = index + 1
        computing += time.monotonic() - before
        if step in (1, 20) or time.monotonic() - last_report >= 10:
            emit('batch_completed', kind=kind, seed=seed, step=step, total=args.steps,
                 loss=loss_value, steps_per_second=(step - initial_step) / max(time.monotonic() - started, 1e-9),
                 loader_wait_pct=100 * waiting / max(waiting + computing, 1e-9))
            last_report = time.monotonic()
        interval = getattr(args, 'evaluation_interval', 0)
        evaluate_now = step % interval == 0 if interval else step in (200, 800, 1600, 3200, 4800, 6400)
        if evaluate_now or step == args.steps:
            metric = evaluate(model, validation, fast, control_config(), batch_size=args.batch_size)
            improved = metric['ndcg'] > best
            best = max(best, metric['ndcg'])
            history.append(dict(step=step, fast_ndcg=metric['ndcg'], loss=loss_value))
            if improved:
                checkpoint(folder / 'best.pt', state())
            checkpoint(folder / 'last.pt', state())
            emit('validation_completed', kind=kind, seed=seed, step=step,
                 fast_ndcg=metric['ndcg'], best_fast_ndcg=best)
        elif step % 200 == 0:
            checkpoint(folder / 'last.pt', state())
        if (args.output / 'STOP_AFTER_CHECKPOINT').exists():
            checkpoint(folder / 'last.pt', state())
            raise InterruptedError('Paused at owner request')
    result = dict(kind=kind, seed=seed, status='completed', steps=step, fast_ndcg=best,
                  seconds=seconds + time.monotonic() - started, history=history,
                  steps_this_attempt=step - initial_step,
                  seconds_this_attempt=time.monotonic() - started,
                  parameters=sum(p.numel() for p in model.parameters()))
    write_json(folder / 'result.json', result)
    return result


def run(args):
    import numpy as np
    import torch
    from mlp_research.cache import CachedSplit
    from mlp_research.training import evaluate
    from mlp_research.statistics import paired_interval
    from tabnet_research.model import make_model, control_config

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (min(args.steps, args.batch_size, args.maximum_seconds) <= 0
            or args.evaluation_interval < 0 or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError('Positive budgets and batch size required')
    info = json.loads((args.cache / 'metadata.json').read_text(encoding='utf-8'))
    if info['splits']['train'] != dict(feeds=24592, rows=1925512):
        raise ValueError('EXP-TABNET-001 requires the common train4 cohort')
    if info['boundary'] != 1790249984 or info['splits']['val'] != dict(feeds=11534, rows=1101295):
        raise ValueError('Selection contract changed; late/test must be absent')
    args.categories = len(info['categories'])
    # Keep cache on SSD; transfer only whole-feed batches to avoid monopolizing VRAM.
    train = BatchDevice(CachedSplit(args.cache, 'train'), args.device)
    validation = BatchDevice(CachedSplit(args.cache, 'val'), args.device)
    fast = np.load(args.cache / 'fast.npy')
    confirmation = np.load(args.cache / 'confirmation.npy')
    if (len(fast), len(confirmation)) != (2000, 9534) or not np.array_equal(
            np.sort(np.r_[fast, confirmation]), np.arange(validation.feeds)):
        raise ValueError('Fast/confirmation coverage changed')
    with np.load(args.cache / 'statistics.npz') as saved:
        statistics = {name: saved[name].copy() for name in saved.files}
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'status.json'
    previous = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    if previous.get('status') == 'completed':
        return
    own = Path(__file__).parent
    identity = dict(experiment=args.experiment, seeds=args.seeds, steps=args.steps,
                    batch_size=args.batch_size, device=args.device, maximum_seconds=args.maximum_seconds,
                    cache_metadata_sha256=sha256(args.cache / 'metadata.json'),
                    cache_files={str(p.relative_to(args.cache)): sha256(p) for p in sorted(args.cache.rglob('*'))
                                 if p.is_file() and p.suffix in ('.npy', '.npz')},
                    source_sha256=sha256(args.mlp_source / 'source.json'),
                    tabnet_version=importlib.metadata.version('pytorch-tabnet'),
                    code_sha256={p.name: sha256(p) for p in sorted(own.glob('*.py'))},
                    control=control_config(), lambda_sparse=.001, n_d=32, n_a=32, n_steps=3,
                    optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001),
                    test_read=False, submission_created=False)
    if args.evaluation_interval:
        identity['evaluation_interval'] = args.evaluation_interval
    if args.resume_from:
        args.resume_from = args.resume_from.resolve()
        if args.resume_from == args.output.resolve():
            raise ValueError('Continuation must use a new output directory')
        args.parent_identity = json.loads((args.resume_from / 'config.json').read_text(encoding='utf-8'))
        parent_report = json.loads((args.resume_from / 'report.json').read_text(encoding='utf-8'))
        if (parent_report['status'] != 'completed'
                or args.parent_identity['experiment'] == args.experiment
                or args.steps <= args.parent_identity['steps']):
            raise ValueError('Continuation requires a completed smaller parent and a new experiment ID')
        for name in ('seeds', 'batch_size', 'device', 'cache_metadata_sha256', 'cache_files',
                     'source_sha256', 'tabnet_version', 'control', 'lambda_sparse', 'n_d', 'n_a',
                     'n_steps', 'optimizer'):
            if args.parent_identity[name] != identity[name]:
                raise ValueError('Continuation changes more than the training budget: ' + name)
        if args.parent_identity['code_sha256']['model.py'] != sha256(own / 'model.py'):
            raise ValueError('Parent model code differs')
        for saved in parent_report['runs']:
            path = args.resume_from / f"{saved['kind']}-s{saved['seed']}" / 'best.pt'
            if sha256(path) != saved['checkpoint_sha256']:
                raise ValueError('Parent best checkpoint no longer matches its report')
        identity['parent'] = dict(experiment=args.parent_identity['experiment'],
                                  config_sha256=sha256(args.resume_from / 'config.json'),
                                  checkpoints={f'{kind}-s{seed}/{name}':
                                      sha256(args.resume_from / f'{kind}-s{seed}' / name)
                                      for kind in ('mlp', 'tabnet') for seed in args.seeds
                                      for name in ('last.pt', 'best.pt')})
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Immutable experiment config changed')
    write_json(config_path, identity)
    torch.set_num_threads(4)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    elapsed = previous.get('elapsed_seconds', 0.)
    deadline = started + max(0, args.maximum_seconds - elapsed)

    def emit(event, **fields):
        if args.device == 'cuda':
            fields.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                          gpu_peak_bytes=torch.cuda.max_memory_allocated())
        value = dict(status='running', event=event, pid=os.getpid(), device=args.device,
                     elapsed_seconds=elapsed + time.monotonic() - started,
                     updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        write_json(status_path, value)
        with (args.output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    with controller_lock(args.output / 'controller.lock'):
        try:
            emit('initializing', gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None)
            if os.name == 'nt':
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            ledger = []
            for seed in args.seeds:
                for kind in ('mlp', 'tabnet'):
                    ledger.append(run_case(args, kind, seed, train, validation, statistics, fast,
                                           identity, emit, deadline))
            # Freeze training/checkpoint decisions before opening confirmation metrics.
            write_json(args.output / 'frozen.json', dict(config=identity, runs=ledger))
            metrics, details = {}, []
            for kind in ('mlp', 'tabnet'):
                values = []
                for seed in args.seeds:
                    folder = args.output / f'{kind}-s{seed}'
                    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                    model = make_model(kind, args.categories, statistics).to(args.device)
                    model.load_state_dict(state['model'])
                    measured = evaluate(model, validation, np.arange(validation.feeds), control_config(),
                                        batch_size=args.batch_size, predictions=True)
                    np.save(folder / 'selection_logits.npy', measured['scores'])
                    np.save(folder / 'selection_ndcg.npy', measured['per_feed'])
                    values.append(measured['per_feed'][confirmation])
                    details.append(dict(kind=kind, seed=seed, best_step=state['step'],
                                        selection_ndcg=measured['ndcg'],
                                        confirmation_ndcg=float(values[-1].mean()),
                                        checkpoint_sha256=sha256(folder / 'best.pt')))
                metrics[kind] = np.stack(values)
            comparison = paired_interval(metrics['tabnet'], metrics['mlp'],
                                         validation.arrays['user_id'][confirmation], comparisons=1)
            fast_delta = float(np.mean([r['fast_ndcg'] for r in ledger if r['kind'] == 'tabnet']) -
                               np.mean([r['fast_ndcg'] for r in ledger if r['kind'] == 'mlp']))
            report = dict(status='completed', runs=details, fast_delta=fast_delta,
                          practical_fast_threshold=.001, confirmation_comparison=comparison,
                          promote_candidate=fast_delta >= .001, independent_confirmation=False,
                          note='Historical selection reused; one seed is preliminary; larger train not evaluated',
                          test_read=False, submission_created=False)
            write_json(args.output / 'report.json', report)
            emit('completed', report=str(args.output / 'report.json'))
            current = json.loads(status_path.read_text())
            current['status'] = 'completed'
            write_json(status_path, current)
        except Exception as error:
            emit('stopped', error=str(error), traceback=traceback.format_exc())
            current = json.loads(status_path.read_text())
            current['status'] = 'paused' if isinstance(error, InterruptedError) else 'incomplete_budget' if isinstance(error, TimeoutError) else 'error'
            write_json(status_path, current)
            raise
        finally:
            train.close()
            validation.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlp-source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int, default=6400)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--maximum-seconds', type=int, default=7200)
    parser.add_argument('--experiment', default='EXP-TABNET-001-r01')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--evaluation-interval', type=int, default=0)
    args = parser.parse_args()
    configure_source(args.mlp_source, args.dependencies)
    run(args)


if __name__ == '__main__':
    main()
