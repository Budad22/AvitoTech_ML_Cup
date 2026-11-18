import torch
from torch import nn
from torch.nn import functional as F


class PositionBCE(nn.Module):
    def __init__(self, max_position):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(max_position + 1))

    def forward(self, scores, labels, valid, pos):
        adjusted = scores + self.bias[pos.masked_fill(~valid, 0)]
        loss = F.binary_cross_entropy_with_logits(adjusted, labels, reduction="none")
        return loss.masked_fill(~valid, 0).sum(1) / valid.sum(1)


def swap_weights(scores, labels, item_id):
    # Stable sorts apply keys in reverse priority: score, target, item ID.
    order = item_id.argsort(stable=True)
    order = order[labels[order].argsort(stable=True)]
    order = order[scores.detach()[order].argsort(descending=True, stable=True)]
    discount = torch.arange(2, len(scores) + 2, dtype=scores.dtype, device=scores.device).log2().reciprocal()
    ranks = torch.empty_like(discount)
    ranks[order] = discount
    positive = labels.bool()
    ideal = (discount * (torch.arange(len(scores), device=scores.device) < positive.sum())).sum()
    return (ranks[positive, None] - ranks[None, ~positive]).abs() / ideal.clamp_min(1)


def lambda_rank(scores, labels, valid, item_ids):
    losses = []
    for s, y, m, ids in zip(scores, labels, valid, item_ids):
        s, y, ids = s[m], y[m], ids[m]
        positive = y.bool()
        margins = s[None, ~positive] - s[positive, None]
        weights = swap_weights(s, y, ids)
        losses.append((weights * F.softplus(margins)).sum() / max(margins.numel(), 1))
    return torch.stack(losses)
