"""Periodic -> per-feature Linear -> ReLU embeddings (PLR).

Reference: Gorishniy et al., NeurIPS 2022, https://arxiv.org/abs/2203.05556.
Only continuous scalar features belong here, not category codes or vector indices.
"""
import math

import torch
from torch import nn


class PeriodicEmbeddings(nn.Module):
    """Map [candidates, features] to independent trainable feature embeddings."""

    def __init__(self, features=5, frequencies=16, embedding_size=16, frequency_scale=1.):
        super().__init__()
        if min(features, frequencies, embedding_size) <= 0 or frequency_scale <= 0:
            raise ValueError('PLR dimensions and frequency scale must be positive')
        self.frequencies = nn.Parameter(torch.empty(features, frequencies))
        self.weight = nn.Parameter(torch.empty(features, 2 * frequencies, embedding_size))
        self.bias = nn.Parameter(torch.empty(features, embedding_size))
        # Competition f1-f5 are already in [0, 1]; no validation-fitted transform.
        nn.init.normal_(self.frequencies, std=frequency_scale)
        bound = (2 * frequencies) ** -.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values):
        if values.ndim != 2 or values.shape[1] != self.frequencies.shape[0]:
            raise ValueError('PLR expects [candidates, numerical_features]')
        angles = (2 * math.pi) * values[:, :, None] * self.frequencies
        periodic = torch.cat((angles.cos(), angles.sin()), dim=-1)
        projected = (periodic.transpose(0, 1) @ self.weight).transpose(0, 1)
        return torch.relu(projected + self.bias)
