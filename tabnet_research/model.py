"""Change only the backbone; retain the winning MLP's PLR and feature assembly."""
import torch
from torch import nn
from pytorch_tabnet.tab_network import TabNetNoEmbeddings

from mlp_research.catalog import hypotheses
from mlp_research.models import ResearchRanker


def control_config():
    return next(case for case in hypotheses() if case['name'] == 'plr_softmax')


class TabNetHead(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.network = TabNetNoEmbeddings(size, 1, n_d=32, n_a=32, n_steps=3,
                                         gamma=1.3, virtual_batch_size=128,
                                         mask_type='sparsemax')
        # The upstream low-level module keeps this tensor as an ordinary attribute.
        # Register it so .to(device) and checkpoint round trips include it.
        encoder = self.network.encoder
        matrix = encoder.group_attention_matrix
        del encoder.group_attention_matrix
        encoder.register_buffer('group_attention_matrix', matrix)
        self.mask_entropy = None

    def forward(self, values):
        logits, self.mask_entropy = self.network(values)
        return logits


def make_model(kind, categories, statistics):
    model = ResearchRanker(categories, control_config(), statistics)
    if kind == 'tabnet':
        model.network = TabNetHead(model.network[0].in_features)
    elif kind != 'mlp':
        raise ValueError('Unknown model kind: ' + kind)
    return model


def training_loss(model, inputs, kind, lambda_sparse=.001):
    from mlp_research.models import loss
    value = loss(model(inputs), inputs, 'softmax')
    if kind == 'tabnet':
        # Upstream returns sum(p*log(p)); subtracting it penalizes dense masks.
        value = value - lambda_sparse * model.network.mask_entropy
    if not torch.isfinite(value):
        raise ValueError('Nonfinite training objective')
    return value
