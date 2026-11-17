"""Predeclared new families; the old EXP-031 definitions remain untouched."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from mlp_next.models import configurations, HistoryRanker
from mlp_research.models import ResearchRanker, loss as original_loss


def extended_configurations():
    base = configurations()['plr_softmax']
    cases = {}

    def add(name, family, **changes):
        cases[name] = dict(base, name=name, description=name, family=family, **changes)

    for name, changes in (
            ('memory_multi4', dict(heads=4)), ('memory_multi8', dict(heads=8)),
            ('memory_cosine', dict(heads=4, cosine_bias=True)),
            ('memory_top8', dict(heads=4, top_k=8)), ('memory_top32', dict(heads=4, top_k=32)),
            ('memory_type4', dict(heads=4, typed_gate=True)),
            ('memory_no_order', dict(heads=4, no_order=True)),
            ('memory_256', dict(heads=4, window=256)),
            ('memory_recent32', dict(heads=4, window=32)),
            ('memory_dual', dict(heads=4, dual_window=True))):
        add(name, 'memory', sequence=True, memory=True, **changes)
    for heads in (4, 8):
        add(f'experts_{heads}', 'experts', experts=heads)
        add(f'rankone_{heads}', 'ensemble', branches=heads)
    for dimension in (32, 64):
        add(f'bilinear_{dimension}', 'interactions', bilinear=dimension)
    add('group_dropout', 'regularization', group_dropout=.15)
    add('memory_dropout', 'memory', memory=True, sequence=True, heads=4, history_dropout=.15)
    for name, objective in (('smooth', 'smooth'), ('temperature', 'temperature'),
                            ('pairwise', 'pairwise'), ('pairwise_mix', 'pairwise_mix'),
                            ('focal', 'focal')):
        add('objective_' + name, 'objective', loss=objective)
    add('position_nuisance', 'bias', position_nuisance=True)
    add('warmup_cosine', 'schedule', schedule_steps=25600, warmup_steps=400)
    add('all_scalars_softmax', 'encoding', scalar_history=True, normalize='standard')
    return cases


class ExtendedRanker(HistoryRanker):
    def __init__(self, categories, config, statistics):
        # New memory replaces only the learned residual; statistics/PLR are retained.
        parent = dict(config, sequence=False) if config.get('memory') else config
        super().__init__(categories, parent, statistics)
        self.config = config
        if config.get('memory'):
            heads = config['heads']
            self.memory_query = nn.Linear(128, heads * 32)
            self.memory_key = nn.Linear(128, 32)
            self.memory_category = nn.Embedding(categories + 1, 32, padding_idx=0)
            self.memory_type = nn.Embedding(257, 32, padding_idx=0)
            self.memory_source = nn.Embedding(257, 32, padding_idx=0)
            self.memory_position = nn.Embedding(256, 32)
            self.memory_present = nn.Linear(1, 32)
            self.memory_output = nn.Linear(heads * 32, 1)
            nn.init.zeros_(self.memory_output.weight)
            nn.init.zeros_(self.memory_output.bias)
            if config.get('typed_gate'):
                self.type_gate = nn.Embedding(257, heads)
        if config.get('experts'):
            count = config['experts']
            self.expert_gate = nn.Linear(529, count)
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(1064, 128), nn.ReLU(), nn.Linear(128, 1))
                                         for _ in range(count)])
            for expert in self.experts:
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)
        if config.get('branches'):
            count = config['branches']
            self.shared_branch = nn.Linear(1064, 128)
            self.branch_input = nn.Parameter(torch.randint(0, 2, (count, 1064)).float() * 2 - 1)
            self.branch_hidden = nn.Parameter(torch.ones(count, 128))
            self.branch_bias = nn.Parameter(torch.zeros(count, 128))
            self.branch_output = nn.Linear(128, 1)
            nn.init.zeros_(self.branch_output.weight)
            nn.init.zeros_(self.branch_output.bias)
        if config.get('bilinear'):
            dimension = config['bilinear']
            self.bilinear_user = nn.Linear(128, dimension, bias=False)
            self.bilinear_item = nn.Linear(128, dimension, bias=False)
            self.bilinear_output = nn.Linear(dimension, 1)
            nn.init.zeros_(self.bilinear_output.weight)
            nn.init.zeros_(self.bilinear_output.bias)
        if config.get('position_nuisance'):
            self.exposure_bias = nn.Embedding(6, 1)
            nn.init.zeros_(self.exposure_bias.weight)

    def memory_score(self, inputs):
        config = self.config
        window, heads = config['window'], config['heads']
        valid = inputs['h_mask'][:, :window]
        if self.training and config.get('history_dropout'):
            valid = valid & (torch.rand_like(valid.float()) >= config['history_dropout'])
        raw = inputs['h_vectors'][:, :window].float()
        h = self.memory_key(raw) + self.memory_category(inputs['h_category'][:, :window])
        h = h + self.memory_type(inputs['h_type'][:, :window]) + self.memory_source(inputs['h_source'][:, :window])
        h = h + self.memory_present(inputs['h_present'][:, :window, None].float())
        if not config.get('no_order'):
            h = h + self.memory_position.weight[:window]
        h = torch.tanh(h)
        q = self.memory_query(inputs['dense'][:, 133:261]).reshape(-1, heads, 32)
        pieces, offset = [], 0
        for i, count in enumerate(inputs['counts'].tolist()):
            current = q[offset:offset + count]
            energy = torch.einsum('nhd,td->nht', current, h[i]) / math.sqrt(32)
            cosine = inputs['dense'][offset:offset + count, 133:261] @ raw[i].T
            if config.get('cosine_bias'):
                energy = energy + cosine[:, None] * 5
            if config.get('typed_gate'):
                energy = energy + self.type_gate(inputs['h_type'][i, :window]).T
            mask = valid[i].expand(count, heads, -1)
            if config.get('top_k'):
                nearest = cosine.masked_fill(~valid[i], -torch.inf).topk(min(window, config['top_k']), -1).indices
                selected = torch.zeros_like(cosine, dtype=torch.bool).scatter_(1, nearest, True)
                mask = mask & selected[:, None]
            if config.get('dual_window'):
                recent = torch.arange(window, device=h.device) < 32
                group = torch.arange(heads, device=h.device) < heads // 2
                mask = mask & (recent[None, None] == group[None, :, None])
            weights = energy.masked_fill(~mask, -1e4).softmax(-1) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            interest = torch.einsum('nht,td->nhd', weights, h[i]).flatten(1)
            pieces.append(self.memory_output(interest).squeeze(-1) * valid[i].any())
            offset += count
        return torch.cat(pieces)

    def forward(self, inputs):
        config = self.config
        if self.training and config.get('group_dropout'):
            inputs = dict(inputs)
            dense = inputs['dense'].clone()
            for a, b in ((5, 133), (133, 261), (261, 517)):
                keep = (torch.rand(len(dense), 1, device=dense.device) >= config['group_dropout']).to(dense.dtype)
                dense[:, a:b] = dense[:, a:b] * keep / (1 - config['group_dropout'])
            inputs['dense'] = dense
        score = ResearchRanker.forward(self, inputs) if config.get('memory') else super().forward(inputs)
        if config.get('memory'):
            score = score + self.memory_score(inputs)
        if config.get('experts') or config.get('branches'):
            raw = torch.cat((inputs['dense'], self.category(inputs['category']), inputs['history']), -1)
            if config.get('experts'):
                values = torch.cat([expert(raw) for expert in self.experts], -1)
                score = score + (self.expert_gate(inputs['history']).softmax(-1) * values).sum(-1)
            else:
                branches = F.relu(self.shared_branch(raw[:, None] * self.branch_input) * self.branch_hidden + self.branch_bias)
                score = score + self.branch_output(branches).mean(1).squeeze(-1)
        if config.get('bilinear'):
            product = self.bilinear_user(inputs['dense'][:, 5:133]) * self.bilinear_item(inputs['dense'][:, 133:261])
            score = score + self.bilinear_output(product).squeeze(-1)
        if self.training and config.get('position_nuisance'):
            if 'position_bucket' not in inputs:
                raise ValueError('Position nuisance training requires train-only diagnostic positions')
            score = score + self.exposure_bias(inputs['position_bucket']).squeeze(-1)
        return score


def extended_loss(scores, inputs, kind):
    if kind in ('bce', 'softmax', 'mixed'):
        return original_loss(scores, inputs, kind)
    if kind == 'temperature':
        return original_loss(scores / .5, inputs, 'softmax')
    if kind == 'focal':
        bce = F.binary_cross_entropy_with_logits(scores, inputs['target'], reduction='none')
        pt = torch.exp(-bce)
        return (bce * (1 - pt).square() * inputs['weights']).sum() / len(inputs['counts'])
    if kind not in ('smooth', 'pairwise', 'pairwise_mix'):
        raise ValueError('Unknown research objective: ' + kind)
    terms, offset = [], 0
    for count in inputs['counts'].tolist():
        values, target = scores[offset:offset + count], inputs['target'][offset:offset + count]
        if kind == 'smooth':
            distribution = .95 * target / target.sum().clamp_min(1) + .05 / count
            terms.append(-(distribution * values.log_softmax(0)).sum() * (target.sum() > 0))
        else:
            positive, negative = values[target > 0], values[target == 0]
            if len(positive) and len(negative):
                # Stable pair loss; average per feed, without unstable rank-dependent normalization.
                terms.append(F.softplus(negative[None] - positive[:, None]).mean())
            else:
                terms.append(values.sum() * 0)
        offset += count
    value = torch.stack(terms).mean()
    return original_loss(scores, inputs, 'softmax') + .1 * value if kind == 'pairwise_mix' else value
