import math

import torch
from torch import nn
from torch.nn import functional as F


def encoder(layers):
    layer = nn.TransformerEncoderLayer(
        d_model=128, nhead=4, dim_feedforward=256, dropout=0.1,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, layers, norm=nn.LayerNorm(128), enable_nested_tensor=False,
    )


def known_equal(left, right):
    return (left[:, :, None] == right[:, None, :]) & (left[:, :, None] > 0)


class HistoryRanker(nn.Module):
    def __init__(self, query_chunk=None):
        super().__init__()
        self.query_chunk = query_chunk
        self.category = nn.Embedding(54, 16)
        self.event_type = nn.Embedding(11, 8)
        self.event_source = nn.Embedding(31, 8)
        self.recency = nn.Embedding(256, 16)
        self.history_projection = nn.Linear(177, 128)
        self.history_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.history_encoder = encoder(2)
        self.candidate_projection = nn.Sequential(
            nn.Linear(413, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128),
        )
        self.query_norm = nn.LayerNorm(128)
        self.query = nn.Linear(128, 128)
        self.key_value = nn.Linear(128, 256)
        self.match_bias = nn.Parameter(torch.zeros(2, 4))
        self.cross_output = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.1)
        self.candidate_encoder = encoder(1)
        self.score = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def cross_attention(self, q, history, data, history_valid):
        batch, candidates, _ = q.shape
        projected = self.query(self.query_norm(q)).reshape(batch, candidates, 4, 32)
        key, value = self.key_value(history).reshape(batch, -1, 2, 4, 32).unbind(2)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        outputs = []
        stride = self.query_chunk or candidates
        for start in range(0, candidates, stride):
            part = slice(start, start + stride)
            exact = known_equal(data["item_id"][:, part], data["history_item_id"])
            category = known_equal(data["category"][:, part], data["history_category"])
            bias = (
                exact[:, None] * self.match_bias[0, :, None, None]
                + category[:, None] * self.match_bias[1, :, None, None]
            )
            bias = F.pad(bias, (1, 0)).masked_fill(
                ~history_valid[:, None, None, :], -math.inf,
            )
            # SDPA also accepts a trainable additive mask on the math backend.
            attended = F.scaled_dot_product_attention(
                projected[:, part].transpose(1, 2), key, value, attn_mask=bias,
                dropout_p=0.1 if self.training else 0.0,
            )
            outputs.append(attended.transpose(1, 2).reshape(batch, -1, 128))
        return q + self.dropout(self.cross_output(torch.cat(outputs, dim=1)))

    def forward(self, data):
        user_raw = data["user"].float()
        user = F.normalize(user_raw, dim=-1).unsqueeze(1)
        item = F.normalize(data["item"].float(), dim=-1)
        history_item = F.normalize(data["history_item"].float(), dim=-1)
        batch, candidates, _ = item.shape
        events = history_item.shape[1]
        recency = self.recency(torch.arange(events, device=item.device)).expand(batch, -1, -1)
        history = self.history_projection(torch.cat((
            history_item, data["history_missing"].unsqueeze(-1).float(),
            self.category(data["history_category"]), self.event_type(data["history_type"]),
            self.event_source(data["history_source"]), recency,
        ), dim=-1))
        history_valid = F.pad(data["history_valid"], (1, 0), value=True)
        history = torch.cat((self.history_token.expand(batch, -1, -1), history), dim=1)
        history = self.history_encoder(history, src_key_padding_mask=~history_valid)

        product = user * item
        q = self.candidate_projection(torch.cat((
            user.expand(-1, candidates, -1), item, product, product.sum(-1, keepdim=True),
            self.category(data["category"]), data["floats"], data["percentile"],
            data["missing"].unsqueeze(-1).float(),
            (~user_raw.any(dim=-1)).float().reshape(batch, 1, 1).expand(-1, candidates, -1),
        ), dim=-1))
        q = self.cross_attention(q, history, data, history_valid)
        contextual = self.candidate_encoder(q, src_key_padding_mask=~data["valid"])
        return self.score(contextual).squeeze(-1).float()
