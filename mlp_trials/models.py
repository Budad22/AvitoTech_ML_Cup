"""Candidate-equivariant MLP ablations; grouping indices never enter learned layers."""
import torch
from torch import nn

from mlp.model import Ranker
from .data import pad
from .periodic import PeriodicEmbeddings

KINDS = ('baseline', 'history_stats', 'history_pool', 'history_attention',
         'softmax', 'lambda', 'context_stats', 'context_pool', 'cross', 'raw_norm', 'low_lr',
         'plr', 'history_stats_plr')
BASE_KINDS = {'baseline', 'softmax', 'lambda', 'low_lr'}
PLR_KINDS = {'plr', 'history_stats_plr'}


def head(size):
    return nn.Sequential(nn.Linear(size, 256), nn.ReLU(), nn.Dropout(.1),
                         nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 1))


def history_stats(inputs):
    """Similarity windows and observed code frequencies, including missing-vector events."""
    vectors, real = inputs['h_vectors'], inputs['h_mask']
    present = real & inputs['h_present']
    item = pad(inputs['dense'][:, 133:261], inputs)
    similarities = torch.bmm(item, vectors.transpose(1, 2))
    category = pad(inputs['category'], inputs)
    same = category[:, :, None] == inputs['h_category'][:, None, :]
    # Unknown category codes do not provide evidence of a shared category.
    same = same & (category[:, :, None] != 0)
    columns = []
    for window in (8, 32, 128):
        valid = present[:, :window]
        values = similarities[:, :, :window]
        count = valid.sum(dim=1).clamp_min(1)[:, None]
        masked = values.masked_fill(~valid[:, None, :], -2)
        maximum = masked.max(dim=2).values.clamp_min(-1)
        maximum = maximum * valid.any(dim=1)[:, None]
        top = masked.topk(min(3, masked.shape[2]), dim=2).values
        top_mask = top > -2
        top_mean = (top * top_mask).sum(dim=2) / top_mask.sum(dim=2).clamp_min(1)
        mean = (values * valid[:, None, :]).sum(dim=2) / count
        category_share = (same[:, :, :window] * real[:, None, :window]).sum(dim=2) / real[:, :window].sum(dim=1).clamp_min(1)[:, None]
        pool = (vectors[:, :window] * valid[:, :, None]).sum(dim=1)
        pool = torch.nn.functional.normalize(pool, dim=-1)
        pool_cosine = (item * pool[:, None, :]).sum(dim=2)
        columns.extend((maximum, top_mean, mean, category_share, pool_cosine))
    scalars = torch.stack(columns, dim=-1)[inputs['feed'], inputs['row']]
    per_feed = [torch.log1p(inputs['h_lengths'].float())[:, None],
                ((real & ~inputs['h_present']).sum(dim=1) / real.sum(dim=1).clamp_min(1))[:, None]]
    for key in ('h_type', 'h_source'):
        histogram = vectors.new_zeros((len(real), 257))
        histogram.scatter_add_(1, inputs[key], real.float())
        per_feed.append(histogram[:, 1:] / real.sum(dim=1).clamp_min(1)[:, None])
    return torch.cat((scalars, torch.cat(per_feed, dim=1)[inputs['feed']]), dim=1)


class TrialRanker(nn.Module):
    def __init__(self, categories, kind, history_window=64):
        super().__init__()
        self.kind = kind
        self.category = nn.Embedding(categories + 1, 16, padding_idx=0)
        size = 535
        if kind in PLR_KINDS:
            self.numerical = PeriodicEmbeddings()
            size += 5 * (16 - 1)  # Replace five scalars with five 16D embeddings.
        if kind == 'raw_norm':
            size += 2
        elif kind in ('history_stats', 'history_stats_plr'):
            size += 529
        elif kind in ('context_stats', 'context_pool'):
            size += 17
            if kind == 'context_pool':
                self.context = nn.Sequential(nn.Linear(size, 64), nn.ReLU())
                size += 128
        elif kind in ('history_pool', 'history_attention'):
            self.query = nn.Linear(128, 32)
            self.history_vector = nn.Linear(128, 32)
            self.event_type = nn.Embedding(257, 8, padding_idx=0)
            self.event_source = nn.Embedding(257, 8, padding_idx=0)
            self.position = nn.Embedding(history_window, 4)
            self.metadata = nn.Linear(37, 32)
            if kind == 'history_attention':
                self.attention = nn.Linear(32, 1)
            size += 97
        elif kind == 'cross':
            self.user_projection = nn.Linear(128, 32)
            self.item_projection = nn.Linear(128, 32)
            size += 64
            self.cross_layers = nn.ModuleList([nn.Sequential(nn.Linear(size, 64, bias=False), nn.Linear(64, size)) for _ in range(2)])
            self.deep = nn.Sequential(nn.Linear(size, 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 128), nn.ReLU())
            self.cross_output = nn.Linear(size, 128)
            self.output = nn.Linear(256, 1)
        self.network = head(size) if kind != 'cross' else nn.Identity()

    def forward(self, inputs):
        dense = inputs['dense']
        encoded = dense
        if self.kind in PLR_KINDS:
            encoded = torch.cat((self.numerical(dense[:, :5]).flatten(1), dense[:, 5:]), dim=1)
        base = torch.cat((encoded, self.category(inputs['category'])), dim=1)
        if self.kind == 'raw_norm':
            base = torch.cat((base, inputs['norms']), dim=1)
        elif self.kind in ('history_stats', 'history_stats_plr'):
            base = torch.cat((base, history_stats(inputs)), dim=1)
        elif self.kind in ('context_stats', 'context_pool'):
            base = torch.cat((base, inputs['context']), dim=1)
            if self.kind == 'context_pool':
                pooled = pad(self.context(base), inputs)
                mask = torch.arange(pooled.shape[1], device=pooled.device)[None, :] < inputs['counts'][:, None]
                mean = pooled.sum(dim=1) / inputs['counts'][:, None]
                maximum = pooled.masked_fill(~mask[:, :, None], -torch.inf).max(dim=1).values
                base = torch.cat((base, mean[inputs['feed']], maximum[inputs['feed']]), dim=1)
        elif self.kind in ('history_pool', 'history_attention'):
            real = inputs['h_mask']
            metadata = torch.cat((self.category(inputs['h_category']), self.event_type(inputs['h_type']),
                                  self.event_source(inputs['h_source']), self.position(inputs['h_position']),
                                  inputs['h_present'].float()[:, :, None]), dim=-1)
            encoded = torch.relu(self.history_vector(inputs['h_vectors']) + self.metadata(metadata))
            query = self.query(dense[:, 133:261])
            if self.kind == 'history_pool':
                history = (encoded * real[:, :, None]).sum(dim=1) / real.sum(dim=1).clamp_min(1)[:, None]
                history = history[inputs['feed']]
            else:
                q = pad(query, inputs)[:, :, None, :]
                h = encoded[:, None, :, :]
                energy = self.attention(torch.relu(q + h + q * h)).squeeze(-1)
                weights = torch.softmax(energy.masked_fill(~real[:, None, :], -1e4), dim=-1) * real[:, None, :]
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                history = torch.bmm(weights, encoded)[inputs['feed'], inputs['row']]
            has = real.any(dim=1)[inputs['feed'], None].float()
            extra = torch.cat((history, history * query, (history - query).abs()), dim=1) * has
            base = torch.cat((base, extra, has), dim=1)
        elif self.kind == 'cross':
            user = self.user_projection(dense[:, 5:133])
            item = self.item_projection(dense[:, 133:261])
            base = torch.cat((base, user * item, (user - item).abs()), dim=1)
            crossed = base
            for layer in self.cross_layers:
                crossed = base * layer(crossed) + crossed
            return self.output(torch.cat((self.deep(base), self.cross_output(crossed)), dim=1)).squeeze(-1)
        return self.network(base).squeeze(-1)


def create_model(categories, kind, history_window=64):
    if kind not in KINDS:
        raise ValueError(f'Unknown trial kind: {kind}')
    return Ranker(categories) if kind in BASE_KINDS else TrialRanker(categories, kind, history_window)


def score(model, inputs, kind):
    return model(inputs['dense'], inputs['category']) if kind in BASE_KINDS else model(inputs)
