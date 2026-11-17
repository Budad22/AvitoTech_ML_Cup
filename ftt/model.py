"""Feature-token Transformer; candidate ordering is never an input."""
import math

import torch
from torch import nn
from torch.nn import functional as F


from ftt.catalog import configuration


class Ranker(nn.Module):
    def __init__(self, config, mean, scale, edges):
        super().__init__()
        self.config = config
        d = config['width']
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clamp_min(1e-6))
        self.register_buffer('edges', torch.as_tensor(edges, dtype=torch.float32))
        self.frequency = nn.Parameter(torch.randn(5, 16) * .1)
        n = {'linear': 1, 'plr': 32, 'ple': 16}[config['encoding']]
        self.numeric_weight = nn.Parameter(torch.randn(5, n, d) / math.sqrt(n))
        self.numeric_bias = nn.Parameter(torch.zeros(5, d))
        self.category = nn.Embedding(256, d)
        self.groups = nn.ModuleList([nn.Linear(128, d) for _ in range(3)])
        self.flags = nn.Linear(4, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * .02)
        layer = nn.TransformerEncoderLayer(d, 4, d * 4, config.get('dropout',.1), activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config['depth'], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 1)
        if config['history']:
            self.event = nn.Embedding(256, d)
            self.source = nn.Embedding(256, d)
            self.history_projection = nn.Linear(128, d)
            self.recency = nn.Embedding(50, d)
            self.empty = nn.Parameter(torch.zeros(1, 1, d))
            self.history_attention = nn.MultiheadAttention(d, 4, dropout=.1, batch_first=True)
        if config['context']:
            self.context_layer = nn.TransformerEncoderLayer(d, 4, d * 4, .1,
                                                            activation='gelu', batch_first=True, norm_first=True)
        if config.get('interactions'):
            self.interaction_projection = nn.Linear(8, d)
        if config.get('similarity_tokens'):
            self.similarity_weight = nn.Parameter(torch.randn(8,d)*.02)
            self.similarity_bias = nn.Parameter(torch.zeros(8,d))
        if config.get('score_skip'):
            self.score_skip = nn.Linear(8,1)
            nn.init.zeros_(self.score_skip.weight)
            nn.init.zeros_(self.score_skip.bias)

    def interaction_features(self, batch):
        rows = []
        start = 0
        for i, count in enumerate(batch['counts']):
            item = batch['item'][start:start+count]
            user = batch['user'][start:start+count]
            history = batch['history_vectors'][i]
            valid = batch['history_mask'][i] & (history.square().sum(-1) > 0)
            similarity = item @ history.T
            weights = valid.to(similarity.dtype)
            n = weights.sum().clamp_min(1)
            mean = (similarity*weights).sum(1)/n
            maximum = similarity.masked_fill(~valid, -torch.inf).max(1).values
            maximum = torch.where(valid.any(), maximum, torch.zeros_like(maximum))
            std = (((similarity-mean[:,None]).square()*weights).sum(1)/n).sqrt()
            matches = (batch['category'][start:start+count,None] == batch['history_category'][i,None])
            category_rate = (matches*weights).sum(1)/n
            rows.append(torch.stack([(item*user).sum(1),
                       (item*batch['history_mean'][start:start+count]).sum(1), maximum, mean, std,
                       category_rate, weights.sum().log1p().expand(count), (~valid.any()).float().expand(count)],1))
            start += count
        return torch.cat(rows)

    def forward(self, batch):
        x = (batch['numeric'] - self.mean) / self.scale
        kind = self.config['encoding']
        if kind == 'plr':
            angle = x[..., None] * self.frequency * (2 * math.pi)
            x = torch.cat([angle.sin(), angle.cos()], -1)
        elif kind == 'ple':
            edge = self.edges
            width = edge[:, 1:] - edge[:, :-1]
            x = ((batch['numeric'][..., None] - edge[:, :-1]) / width.clamp_min(1e-6)).clamp(0, 1)
            x = x * (width > 1e-6)
        else:
            x = x[..., None]
        tokens = [(x.transpose(0, 1) @ self.numeric_weight).transpose(0, 1) + self.numeric_bias,
                  self.category(batch['category'])[:, None]]
        tokens += [projection(batch[key])[:, None] for projection, key in
                   zip(self.groups, ('user', 'item', 'history_mean'))]
        tokens.append(self.flags(batch['flags'])[:, None])
        if self.config.get('remove_vector_tokens'):
            tokens = [tokens[0],tokens[1],tokens[-1]]
        if self.config.get('interactions'):
            similarities = self.interaction_features(batch)
            if self.config.get('similarity_tokens'):
                tokens.append(similarities[...,None]*self.similarity_weight+self.similarity_bias)
            else:
                tokens.append(self.interaction_projection(similarities)[:,None])
        if self.config['history']:
            h = self.history_projection(batch['history_vectors'])
            h = h + self.category(batch['history_category']) + self.event(batch['history_event'])
            h = h + self.source(batch['history_source']) + self.recency.weight[None]
            h = torch.cat([self.empty.expand(len(h), -1, -1), h], 1)
            mask = F.pad(~batch['history_mask'], (1, 0), value=False)
            # Process each feed once; candidate queries attend only to their own history.
            attended = []
            start = 0
            for i, count in enumerate(batch['counts']):
                query = (tokens[3][start:start + count] + tokens[0][start:start + count].mean(1, keepdim=True)
                         + tokens[1][start:start + count]).transpose(0, 1)
                value = self.history_attention(query, h[i:i+1], h[i:i+1],
                                               key_padding_mask=mask[i:i+1], need_weights=False)[0]
                attended.append(value.transpose(0, 1))
                start += count
            tokens.append(torch.cat(attended))
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), *tokens], 1)
        features = self.encoder(tokens)[:, 0]
        if self.config['context']:
            parts = features.split(batch['counts'])
            padded = nn.utils.rnn.pad_sequence(parts, batch_first=True)
            lengths = torch.tensor(batch['counts'], device=x.device)
            mask = torch.arange(padded.shape[1], device=x.device)[None] >= lengths[:, None]
            values = self.context_layer(padded, src_key_padding_mask=mask)
            features = torch.cat([values[i, :n] for i, n in enumerate(batch['counts'])])
        score = self.output(self.norm(features)).squeeze(-1)
        if self.config.get('score_skip'):
            score = score + self.score_skip(similarities).squeeze(-1)
        return score


def objective(scores, target, counts, kind):
    losses = []
    start = 0
    for count in counts:
        s, y = scores[start:start+count], target[start:start+count]
        if kind == 'bce':
            losses.append(F.binary_cross_entropy_with_logits(s, y))
        else:
            losses.append(-(s.log_softmax(0) * y).sum() / y.sum().clamp_min(1))
        start += count
    return torch.stack(losses).mean()
