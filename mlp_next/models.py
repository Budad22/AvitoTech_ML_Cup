"""Add candidate-conditioned history to the existing strong statistics path."""
import torch
from torch import nn

from mlp_research.catalog import hypotheses
from mlp_research.models import ResearchRanker


def configurations():
    old = {c['name']: c for c in hypotheses()}
    cases = {name: dict(old[name], sequence=False, window=128)
             for name in ('history_control', 'mlp_softmax', 'plr_softmax')}
    for name, base, window, encoder in (
            ('stats_din64', 'mlp_softmax', 64, False),
            ('stats_din128', 'mlp_softmax', 128, False),
            ('stats_din256', 'mlp_softmax', 256, False),
            ('stats_sequence128', 'mlp_softmax', 128, True),
            ('plr_stats_din128', 'plr_softmax', 128, False)):
        cases[name] = dict(cases[base], name=name, sequence=True, window=window,
                           sequence_encoder=encoder, description=name)
    cases['plr_context'] = dict(cases['plr_softmax'], name='plr_context', context=True)
    return cases


class HistoryRanker(ResearchRanker):
    def __init__(self, categories, config, statistics):
        super().__init__(categories, config, statistics)
        if not config.get('sequence'):
            return
        self.query = nn.Linear(128, 32)
        self.history_projection = nn.Linear(128, 32)
        self.history_category = nn.Embedding(categories + 1, 32, padding_idx=0)
        self.event = nn.Embedding(257, 32, padding_idx=0)
        self.source = nn.Embedding(257, 32, padding_idx=0)
        self.position = nn.Embedding(256, 32)
        self.present = nn.Linear(1, 32)
        self.attention = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))
        self.correction = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))
        # Start from the strong statistics model; learn only useful extra information.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        if config.get('sequence_encoder'):
            layer = nn.TransformerEncoderLayer(32, 4, 64, .1, batch_first=True,
                                                norm_first=True, activation='gelu')
            self.history_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def forward(self, inputs):
        score = super().forward(inputs)
        if not self.config.get('sequence'):
            return score
        window = self.config['window']
        valid = inputs['h_mask'][:, :window]
        h = self.history_projection(inputs['h_vectors'][:, :window].float())
        h = h + self.history_category(inputs['h_category'][:, :window])
        h = h + self.event(inputs['h_type'][:, :window]) + self.source(inputs['h_source'][:, :window])
        h = h + self.position.weight[:window] + self.present(inputs['h_present'][:, :window, None].float())
        h = torch.relu(h)
        if hasattr(self, 'history_encoder'):
            # A real sentinel keeps entirely empty histories finite.
            h = torch.cat((h.new_zeros(len(h), 1, 32), h), 1)
            padding = torch.cat((valid.new_zeros(len(valid), 1), ~valid), 1)
            h = self.history_encoder(h, src_key_padding_mask=padding)[:, 1:]
        query = self.query(inputs['dense'][:, 133:261])
        extra, offset = [], 0
        for i, count in enumerate(inputs['counts'].tolist()):
            q = query[offset:offset + count]
            values = h[i].expand(count, -1, -1)
            queries = q[:, None].expand_as(values)
            energy = self.attention(torch.cat((queries, values, queries * values), -1)).squeeze(-1)
            weights = energy.masked_fill(~valid[i], -1e4).softmax(-1) * valid[i]
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            interest = (weights[..., None] * values).sum(1)
            correction = self.correction(torch.cat((interest, q * interest, (q - interest).abs()), -1)).squeeze(-1)
            extra.append(correction * valid[i].any())
            offset += count
        return score + torch.cat(extra)
