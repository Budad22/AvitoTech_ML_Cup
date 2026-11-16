"""Matched whole-feed steps, stateful continuation, and checkpoint selection on fast val."""
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from mlp.metrics import ndcg
from mlp_research.cache import sha256
from mlp_research.device_data import DeviceSplit
from mlp_research.models import ResearchRanker, loss
from mlp_trials.state import write_json


def save(path, state):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05)


def feed_order(feeds, seed, step, batch_size):
    batches = math.ceil(feeds / batch_size)
    epoch, cursor = divmod(step, batches)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(feeds)
    return order[cursor * batch_size:(cursor + 1) * batch_size]


@torch.inference_mode()
def evaluate(model, dataset, feeds, config, batch_size=32, predictions=False):
    model.eval()
    metrics, all_scores = [], []
    for start in range(0, len(feeds), batch_size):
        chosen = feeds[start:start + batch_size]
        inputs = dataset.batch(chosen, history=config['history'], context=config['context'])
        scores = model(inputs).cpu().numpy().astype('float32')
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite research predictions')
        labels = inputs['target'].cpu().numpy()
        offsets = np.r_[0, inputs['counts'].numpy().cumsum()]
        metrics.extend(ndcg(labels[a:b], scores[a:b]) for a, b in zip(offsets[:-1], offsets[1:]))
        if predictions:
            all_scores.append(scores)
    values = np.asarray(metrics, dtype='float64')
    if not len(values):
        raise ValueError('Empty research validation')
    return dict(ndcg=float(values.mean()), per_feed=values,
                scores=np.concatenate(all_scores) if predictions else None)


class Trainer:
    def __init__(self, cache, directory, emit, batch_size=16, device='cpu'):
        self.cache, self.directory, self.emit = Path(cache), Path(directory), emit
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        self.info = json.loads((self.cache / 'metadata.json').read_text(encoding='utf-8'))
        self.fingerprint = sha256(self.cache / 'metadata.json')
        self.train = DeviceSplit(cache, 'train', device)
        self.validation = DeviceSplit(cache, 'val', device)
        with np.load(self.cache / 'statistics.npz') as statistics:
            self.statistics = {key: statistics[key].copy() for key in statistics.files}
        self.fast = np.load(self.cache / 'fast.npy')
        self.confirmation = np.load(self.cache / 'confirmation.npy')
        self.batch_size = batch_size
        if device == 'cuda':
            emit('device_cache_ready', device=device, gpu=torch.cuda.get_device_name(),
                 train_resident_bytes=self.train.resident_bytes, val_resident_bytes=self.validation.resident_bytes)

    def close(self):
        self.train.close()
        self.validation.close()

    def run(self, config, seed, target, *, maximum_seconds=7200):
        output = self.directory / 'runs' / f"{config['name']}-s{seed}"
        output.mkdir(parents=True, exist_ok=True)
        identity = dict(config=config, seed=seed, cache=self.fingerprint, batch_size=self.batch_size)
        if self.device != 'cpu':
            identity['device'] = self.device
        torch.manual_seed(seed)
        model = ResearchRanker(len(self.info['categories']), config, self.statistics).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        step, best, history, seconds = 0, -1., [], 0.
        if (output / 'last.pt').exists():
            state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Research checkpoint identity changed')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['torch_rng'])
            if self.device == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            step, best, history, seconds = (state[key] for key in ('step', 'best', 'history', 'seconds'))
        started = last_report = time.monotonic()
        initial_step = step
        wait = compute = 0.
        last_loss = None

        def state():
            return dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(), step=step, best=best, history=history,
                        cuda_rng=torch.cuda.get_rng_state_all() if self.device == 'cuda' else None,
                        seconds=seconds + time.monotonic() - started)

        for index in range(step, target):
            if seconds + time.monotonic() - started > maximum_seconds:
                save(output / 'last.pt', state())
                raise TimeoutError('Per-model safety budget reached; insufficient training is not a negative result')
            before = time.monotonic()
            chosen = feed_order(self.train.feeds, seed, index, self.batch_size)
            inputs = self.train.batch(chosen, history=config['history'], context=config['context'])
            wait += time.monotonic() - before
            before = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            objective = loss(model(inputs), inputs, config['loss'])
            if not torch.isfinite(objective):
                raise ValueError('Nonfinite research training loss')
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            last_loss = float(objective.detach())
            step = index + 1
            compute += time.monotonic() - before
            if step == 1 or step == 20 or time.monotonic() - last_report >= 10:
                self.emit('batch_completed', name=config['name'], seed=seed, step=step, total=target,
                          device=self.device,
                          loss=last_loss, steps_per_second=(step - initial_step) /
                          max(time.monotonic() - started, 1e-9), loader_wait_pct=100 * wait / max(wait + compute, 1e-9))
                last_report = time.monotonic()
            if step in (200, 800, 1600, 3200, 4800, 6400) or step == target:
                metric = evaluate(model, self.validation, self.fast, config)
                improved = metric['ndcg'] > best
                best = max(best, metric['ndcg'])
                history.append(dict(step=step, fast_ndcg=metric['ndcg'], loss=last_loss))
                if improved:
                    save(output / 'best.pt', state())
                save(output / 'last.pt', state())
                self.emit('validation_completed', name=config['name'], seed=seed, step=step,
                          total=target, fast_ndcg=metric['ndcg'], best_ndcg=best)
            elif step % 200 == 0:
                save(output / 'last.pt', state())
            if (self.directory / 'STOP_AFTER_CHECKPOINT').exists():
                save(output / 'last.pt', state())
                raise InterruptedError('Paused through checkpoint at owner request')
        result = dict(name=config['name'], seed=seed, status='completed', steps=step, fast_ndcg=best,
                      device=self.device,
                      history=history, seconds=seconds + time.monotonic() - started,
                      parameters=sum(parameter.numel() for parameter in model.parameters()), output=str(output))
        write_json(output / 'result.json', result)
        return result

    def final(self, config, seed):
        output = self.directory / 'runs' / f"{config['name']}-s{seed}"
        state = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
        model = ResearchRanker(len(self.info['categories']), config, self.statistics).to(self.device)
        model.load_state_dict(state['model'])
        metrics = evaluate(model, self.validation, np.arange(self.validation.feeds), config, predictions=True)
        np.save(output / 'selection_logits.npy', metrics['scores'])
        np.save(output / 'selection_ndcg.npy', metrics['per_feed'])
        result = dict(name=config['name'], seed=seed, best_step=state['step'],
                      selection_ndcg=metrics['ndcg'], confirmation_ndcg=float(metrics['per_feed'][self.confirmation].mean()),
                      checkpoint_sha256=sha256(output / 'best.pt'))
        write_json(output / 'final.json', result)
        return result
