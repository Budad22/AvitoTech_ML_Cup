"""Small feature encoders and architecture controls over exactly the same inputs."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def activation(name):
    return {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[name]()


class NumericalEmbedding(nn.Module):
    def __init__(self, config, statistics):
        super().__init__()
        self.config = config
        n = 22 if config['scalar_history'] else 5
        self.register_buffer('mean', torch.tensor(statistics['mean'][:n]))
        self.register_buffer('scale', torch.tensor(statistics['scale'][:n]))
        quantiles = torch.tensor(statistics['quantiles'])
        self.register_buffer('quantiles', quantiles)
        if config['encoding'] == 'plr':
            self.frequencies = nn.Parameter(torch.randn(n, config['frequencies']) * config['sigma'],
                                            requires_grad=not config['frozen'])
            size = 2 * config['frequencies']
        elif config['encoding'] == 'linear':
            size = 1
        else:
            size = config['bins']
            indices = torch.linspace(0, 256, size + 1).round().long()
            self.register_buffer('edges', quantiles[:, indices])
        groups = 1 if config['shared'] else n
        self.weight = nn.Parameter(torch.empty(groups, size, config['embedding']))
        self.bias = nn.Parameter(torch.empty(groups, config['embedding']))
        nn.init.uniform_(self.weight, -size ** -.5, size ** -.5)
        nn.init.uniform_(self.bias, -size ** -.5, size ** -.5)

    def forward(self, values):
        config = self.config
        if config['normalize'] == 'standard':
            values = (values - self.mean) / self.scale
        elif config['normalize'] == 'quantile':
            positions = torch.searchsorted(self.quantiles.contiguous(), values.T.contiguous(), right=True).clamp(1, 256)
            lower = self.quantiles.gather(1, positions - 1)
            upper = self.quantiles.gather(1, positions)
            fraction = ((values.T - lower) / (upper - lower).clamp_min(1e-8)).clamp(0, 1)
            values = ((positions - 1 + fraction) / 256).T
        if config['encoding'] == 'plr':
            angles = 2 * math.pi * values[:, :, None] * self.frequencies
            encoded = torch.cat((angles.cos(), angles.sin()), dim=-1)
        elif config['encoding'] == 'linear':
            encoded = values[:, :, None]
        elif config['encoding'] == 'ple':
            width = self.edges[:, 1:] - self.edges[:, :-1]
            encoded = ((values[:, :, None] - self.edges[:, :-1]) / width.clamp_min(1e-8)).clamp(0, 1)
            encoded = encoded * (width > 1e-8)
        else:
            centers = (self.edges[:, :-1] + self.edges[:, 1:]) / 2
            width = (self.edges[:, -1] - self.edges[:, 0]).clamp_min(1e-3) / config['bins']
            encoded = torch.exp(-.5 * ((values[:, :, None] - centers) / width[:, None]) ** 2)
        return torch.relu((encoded.transpose(0, 1) @ self.weight).transpose(0, 1) + self.bias).flatten(1)


class Residual(nn.Module):
    def __init__(self, size, width, dropout):
        super().__init__()
        self.stem = nn.Linear(size, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
                                                  nn.Dropout(dropout), nn.Linear(width, width)) for _ in range(2)])
        self.output = nn.Linear(width, 1)

    def forward(self, values):
        values = torch.relu(self.stem(values))
        for block in self.blocks:
            values = values + block(values)
        return self.output(values)


class CrossNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(size, 32, bias=False), nn.Linear(32, size)) for _ in range(2)])
        self.deep = nn.Sequential(nn.Linear(size, 128), nn.ReLU())
        self.cross = nn.Linear(size, 128)
        self.output = nn.Linear(256, 1)

    def forward(self, values):
        crossed = values
        for layer in self.layers:
            crossed = values * layer(crossed) + crossed
        return self.output(torch.cat((self.deep(values), self.cross(crossed)), dim=1))


class ResearchRanker(nn.Module):
    def __init__(self, categories, config, statistics):
        super().__init__()
        self.config = config
        self.category = nn.Embedding(categories + 1, 16, padding_idx=0)
        self.numerical = NumericalEmbedding(config, statistics) if config['encoding'] != 'raw' else None
        size = 535 + 529 * config['history'] + 17 * config['context']
        if self.numerical is not None:
            scalars = 22 if config['scalar_history'] else 5
            size += scalars * (config['embedding'] - (not config['raw_skip']))
        backbone = config['backbone']
        width = config['width']
        if backbone == 'linear':
            self.network = nn.Linear(size, 1)
        elif backbone == 'residual':
            self.network = Residual(size, width, config['dropout'])
        elif backbone == 'cross':
            self.network = CrossNetwork(size)
        elif backbone == 'gated':
            self.network = nn.Sequential(nn.Linear(size, 2 * width), nn.GLU(), nn.Dropout(config['dropout']),
                                         nn.Linear(width, width), nn.GLU(), nn.Dropout(config['dropout']),
                                         nn.Linear(width // 2, 1))
        elif backbone == 'transformer':
            dimensions = [1] * 5 + [128] * 4 + [2, 16] + ([529] if config['history'] else [])
            dimensions += [17] if config['context'] else []
            self.projections = nn.ModuleList([nn.Linear(dimension, 32) for dimension in dimensions])
            self.token_identity = nn.Parameter(torch.randn(len(dimensions) + 1, 32) * .02)
            layer = nn.TransformerEncoderLayer(32, 4, 64, config['dropout'], activation='gelu',
                                                batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, config['depth'], enable_nested_tensor=False)
            self.network = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))
        else:
            layers = []
            for index in range(config['depth']):
                output = width if index < config['depth'] - 1 else width // 2
                layers.append(nn.Linear(size, output))
                if config['layer_norm']:
                    layers.append(nn.LayerNorm(output))
                layers.extend((activation(config['activation']), nn.Dropout(config['dropout'])))
                size = output
            self.network = nn.Sequential(*layers, nn.Linear(size, 1))

    def forward(self, inputs):
        config, dense = self.config, inputs['dense']
        category = self.category(inputs['category'])
        if config['backbone'] == 'transformer':
            groups = [dense[:, i:i + 1] for i in range(5)]
            groups += [dense[:, start:start + 128] for start in (5, 133, 261, 389)]
            groups += [dense[:, 517:519], category]
            groups += [inputs['history']] if config['history'] else []
            groups += [inputs['context']] if config['context'] else []
            tokens = torch.stack([layer(group) for layer, group in zip(self.projections, groups)], dim=1)
            cls = tokens.new_zeros((len(tokens), 1, 32))
            tokens = torch.cat((cls, tokens), dim=1) + self.token_identity
            return self.network(self.transformer(tokens)[:, 0]).squeeze(-1)
        history = inputs.get('history')
        if self.numerical is not None:
            scalars = dense[:, :5]
            if config['scalar_history']:
                scalars = torch.cat((scalars, history[:, :17]), dim=1)
            numeric = self.numerical(scalars)
            if not config['raw_skip']:
                dense = dense[:, 5:]
                if config['scalar_history']:
                    history = history[:, 17:]
        parts = [dense, category]
        if config['history']:
            parts.append(history)
        if config['context']:
            parts.append(inputs['context'])
        if self.numerical is not None:
            parts.append(numeric)
        return self.network(torch.cat(parts, dim=1)).squeeze(-1)


def loss(logits, inputs, kind):
    bce = (F.binary_cross_entropy_with_logits(logits, inputs['target'], reduction='none') * inputs['weights']).sum()
    bce = bce / len(inputs['counts'])
    if kind == 'bce':
        return bce
    terms = []
    for scores, targets in zip(logits.split(inputs['counts'].tolist()), inputs['target'].split(inputs['counts'].tolist())):
        positives = targets.sum()
        terms.append((torch.logsumexp(scores, dim=0) - (scores * targets).sum() / positives.clamp_min(1)) * (positives > 0))
    listwise = torch.stack(terms).mean()
    return listwise if kind == 'softmax' else bce + .1 * listwise
