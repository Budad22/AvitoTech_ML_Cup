"""Whole-feed optimization, validation, and checkpoint recovery."""
import math
import os
import time

import numpy as np
import torch

from ftt.data import Cache, Full, digest
from ftt.atomic import write_json, replace
from ftt.model import Ranker, configuration, objective


def ndcg(y, scores):
    order = np.lexsort((y, -scores))
    discount = 1 / np.log2(np.arange(len(y)) + 2)
    ideal = (np.sort(y)[::-1] * discount).sum()
    return float((y[order] * discount).sum() / ideal) if ideal else 1.


def save(path, state):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    replace(temporary, path)


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


class Trainer:
    def __init__(self, *, root, data, device, validation, fast, emit, ledger):
        self.root = root
        self.data = data
        self.device = device
        self.validation = validation
        self.fast = fast
        self.emit = emit
        self.ledger = ledger

    def run(self, number, seed, target, size):
        key = f'{size}/FTT-{number:03d}-s{seed}/{target}'
        if key in self.ledger:
            return self.ledger[key]
        config = configuration(number)
        output = self.root / size / f"{config['name']}-s{seed}"
        output.mkdir(parents=True, exist_ok=True)
        train = (Full(self.data, Cache(self.root / 'cache', 'train', 'medium').statistics())
                 if size == 'full' else Cache(self.root / 'cache', 'train', 'small' if size == 'smoke' else size))
        if size == 'full':
            target = min(target, 5 * math.ceil(train.feeds / 16))
        validation = self.validation
        torch.manual_seed(seed)
        model = Ranker(config, *train.statistics()).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate',1e-4), weight_decay=1e-5)
        identity = dict(config=config, seed=seed, cache=digest(self.root / 'cache/metadata.json'), size=size)
        step, best, history, seconds, micro = 0, -1., [], 0., 16
        if (output / 'last.pt').exists():
            state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Checkpoint fingerprint changed')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['rng'])
            if self.device == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            step, best, history, seconds, micro = [state[k] for k in ('step', 'best', 'history', 'seconds', 'micro')]
        started = last_emit = time.monotonic()
        def checkpoint():
            return dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device == 'cuda' else None,
                        step=step, best=best, history=history, seconds=seconds+time.monotonic()-started, micro=micro)
        def validate():
            nonlocal best
            # Emit before expensive validation so status never suggests training is hung.
            self.emit('evaluating', phase=size, model=config['name'], seed=seed, step=step, target=target)
            values = evaluate(model, validation, self.fast, self.device)
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
            cuda_rng = torch.cuda.get_rng_state_all() if self.device == 'cuda' else None
            try:
                for start in range(0, len(chosen), micro):
                    subset = chosen[start:start+micro]
                    batch = train.batch(subset, self.device)
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
                group['lr'] = config.get('learning_rate',1e-4) * min(1., (step+1)/100)
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
