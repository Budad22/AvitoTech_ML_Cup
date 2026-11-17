"""Pinned official architecture, cached whole feeds, and faithful resumable fitting."""
import ast
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

OFFICIAL_SHA = '87b731df2a5507957c075e93a7f219c1fa4f59b9b7bb9752c59c43087bb2cc06'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def replace_file(temporary, destination):
    for attempt in range(10):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    replace_file(temporary, path)


def save(path, value):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(value, temporary)
    replace_file(temporary, path)


def official_parts(source):
    if sha256(source) != OFFICIAL_SHA:
        raise ValueError('Official source hash changed')
    tree = ast.parse(Path(source).read_text(encoding='utf-8'), filename=str(source))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in ('Ranker', 'collate_feeds')]
    namespace = dict(torch=torch, nn=nn, F=F, pad_sequence=pad_sequence, __name__='official_baseline_pinned')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    config = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'TRAIN_CONFIG' for target in node.targets))
    return namespace['Ranker'], namespace['collate_feeds'], config


class CachedFeeds(Dataset):
    """Use only official inputs; keys/targets never enter model features."""
    def __init__(self, root, split):
        if split not in ('train', 'val'):
            raise ValueError('Only train and validation are supported')
        folder = Path(root) / split
        self.arrays = {name: np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                       for name in ('dense', 'category', 'target', 'offsets', 'feed_id', 'item_id')}
        self.rows = len(self.arrays['target'])
        if self.arrays['dense'].shape != (self.rows, 519):
            raise ValueError('Unexpected dense feature schema')
        if len(self.arrays['offsets']) != len(self.arrays['feed_id']) + 1:
            raise ValueError('Feed offsets disagree')

    def __len__(self):
        return len(self.arrays['feed_id'])

    def __getitem__(self, feed):
        left, right = self.arrays['offsets'][feed:feed + 2]
        dense = self.arrays['dense'][left:right]
        features = np.empty((right - left, 134), dtype=np.float32)
        features[:, :5] = dense[:, :5]
        features[:, 5:133] = dense[:, 133:261]
        features[:, 133] = dense[:, 517]
        return (torch.from_numpy(features),
                torch.from_numpy(np.array(self.arrays['category'][left:right], copy=True)).long(),
                torch.from_numpy(np.array(dense[0, 5:133], copy=True)),
                torch.from_numpy(np.array(self.arrays['target'][left:right], dtype=np.float32, copy=True)))

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()


def ndcg(target, score):
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite prediction')
    discount = 1 / np.log2(np.arange(len(target)) + 2)
    denominator = discount[:int(target.sum())].sum()
    return float(target[np.lexsort((target, -score))] @ discount / denominator) if denominator else 1.


@torch.inference_mode()
def evaluate(model, loader, device, predictions=False):
    model.eval()
    total_loss, rows = 0., 0
    per_feed, all_scores = [], []
    for batch in loader:
        features, category, users, padding, target = (value.to(device) for value in batch)
        logits = model(features, category, users, padding)[~padding]
        labels = target[~padding]
        total_loss += float(F.binary_cross_entropy_with_logits(logits, labels, reduction='sum'))
        scores = logits.sigmoid().float().cpu().numpy()
        labels = labels.cpu().numpy()
        counts = (~padding).sum(dim=1).cpu().numpy()
        offsets = np.r_[0, counts.cumsum()]
        per_feed.extend(ndcg(labels[left:right], scores[left:right])
                        for left, right in zip(offsets[:-1], offsets[1:]))
        rows += len(scores)
        if predictions:
            all_scores.append(scores)
    if rows == 0:
        raise ValueError('Empty validation')
    values = np.asarray(per_feed, dtype=np.float64)
    return dict(logloss=total_loss / rows, ndcg=float(values.mean()), rows=rows, feeds=len(values),
                per_feed=values, scores=np.concatenate(all_scores) if predictions else None)


def fit(model, train_loader, fast_loader, output, identity, device, epochs, emit, deadline):
    """Retain native DataLoader shuffling and restore its epoch iterator exactly."""
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    epoch, cursor, step, best_loss = 0, 0, 0, float('inf')
    epoch_rng, history, loss_sum, row_sum = None, [], 0., 0
    restored = None
    if (output / 'last.pt').exists():
        restored = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        if restored['identity'] != identity:
            raise ValueError('Checkpoint identity changed')
        model.load_state_dict(restored['model'])
        optimizer.load_state_dict(restored['optimizer'])
        epoch, cursor, step, best_loss, epoch_rng, history, loss_sum, row_sum = (
            restored[name] for name in ('epoch', 'cursor', 'step', 'best_loss', 'epoch_rng', 'history', 'loss_sum', 'row_sum'))
        torch.set_rng_state(restored['rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(restored['cuda_rng'])
    started = last_report = time.monotonic()
    initial_step = step

    def state():
        return dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None,
                    epoch=epoch, cursor=cursor, step=step, best_loss=best_loss, epoch_rng=epoch_rng,
                    history=history, loss_sum=loss_sum, row_sum=row_sum)

    while epoch < epochs:
        if cursor:
            # Rebuild the same native shuffle, then restore RNG after skipped CPU batches.
            current_rng = torch.get_rng_state()
            torch.set_rng_state(epoch_rng)
            iterator = iter(train_loader)
            if hasattr(train_loader.batch_sampler, 'skip'):
                train_loader.batch_sampler.skip(cursor)
            else:
                for _ in range(cursor):
                    next(iterator)
            torch.set_rng_state(current_rng)
        else:
            epoch_rng = torch.get_rng_state()
            iterator = iter(train_loader)
        model.train()
        for batch in iterator:
            if time.monotonic() >= deadline:
                save(output / 'last.pt', state())
                raise TimeoutError('Training budget exhausted; checkpoint saved, experiment incomplete')
            features, category, users, padding, target = (value.to(device) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, category, users, padding)[~padding]
            loss = F.binary_cross_entropy_with_logits(logits, target[~padding])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite objective')
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
            loss_sum += loss_value * logits.numel()
            row_sum += logits.numel()
            cursor += 1
            step += 1
            if step in (1, 20) or time.monotonic() - last_report >= 10:
                emit('batch_completed', epoch=epoch + 1, batch=cursor, step=step,
                     total_steps=epochs * len(train_loader), loss=loss_value,
                     steps_per_second=(step - initial_step) / max(time.monotonic() - started, 1e-9))
                last_report = time.monotonic()
            if step % 100 == 0:
                save(output / 'last.pt', state())
            if (output / 'STOP_AFTER_CHECKPOINT').exists():
                save(output / 'last.pt', state())
                raise InterruptedError('Paused at owner request')
        measured = evaluate(model, fast_loader, device)
        history.append(dict(epoch=epoch + 1, step=step, train_logloss=loss_sum / row_sum,
                            fast_logloss=measured['logloss'], fast_ndcg=measured['ndcg']))
        improved = measured['logloss'] < best_loss
        best_loss = min(best_loss, measured['logloss'])
        epoch += 1
        cursor, epoch_rng, loss_sum, row_sum = 0, None, 0., 0
        if improved:
            save(output / 'best.pt', state())
        save(output / 'last.pt', state())
        emit('epoch_completed', **history[-1], best_fast_logloss=best_loss)
    return dict(steps=step, epochs=epoch, history=history, seconds_this_attempt=time.monotonic() - started)
