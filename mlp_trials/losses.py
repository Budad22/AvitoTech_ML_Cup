"""Equal-feed BCE, listwise softmax, and full-NDCG weighted pairwise loss."""
import torch
from torch.nn import functional as F

from mlp.pipeline import feed_loss
from .data import pad


def objective(scores, inputs, kind):
    if kind not in ('softmax', 'lambda'):
        return feed_loss(scores, inputs['target'], inputs['weights'], len(inputs['counts']))
    logits = pad(scores, inputs)
    labels = pad(inputs['target'], inputs)
    positions = torch.arange(logits.shape[1], device=logits.device)[None, :]
    mask = positions < inputs['counts'][:, None]
    positives = labels.sum(dim=1)
    if kind == 'softmax':
        logp = F.log_softmax(logits.masked_fill(~mask, -1e4), dim=1)
        per_feed = -(labels * logp).sum(dim=1) / positives.clamp_min(1)
        valid = positives > 0
    else:
        # Stable two-pass sorting preserves the official negative-first tie rule.
        negative_first = labels.argsort(dim=1, stable=True)
        sorted_scores = logits.masked_fill(~mask, -1e4).gather(1, negative_first)
        order = negative_first.gather(1, sorted_scores.argsort(dim=1, descending=True, stable=True))
        ranks = torch.empty_like(order).scatter(1, order, positions.expand_as(order))
        discount = 1 / torch.log2(ranks.float() + 2)
        ideal_discount = 1 / torch.log2(positions.float() + 2)
        idcg = (ideal_discount * (positions < positives[:, None])).sum(dim=1).clamp_min(1e-8)
        pairs = (labels[:, :, None] > labels[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        weights = ((discount[:, :, None] - discount[:, None, :]).abs() / idcg[:, None, None]) * pairs
        differences = logits[:, :, None] - logits[:, None, :]
        denominator = weights.sum(dim=(1, 2))
        per_feed = (F.softplus(-differences) * weights.detach()).sum(dim=(1, 2)) / denominator.clamp_min(1e-8)
        valid = denominator > 0
    return (per_feed * valid).sum() / valid.sum().clamp_min(1)
