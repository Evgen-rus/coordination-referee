"""Small jointly trained full-sequence classifier and onset localizer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

TOKEN_DIM = 64
CATEGORY_DIM = 8
REF_DIM = 16
META_DIM = 12
MODEL_DIM = 128
HIDDEN_DIM = 64
N_CLASSES = 7
N_FAULT_CLASSES = 6
N_AGENT_SLOTS = 32
N_TASK_SLOTS = 256
N_ARTIFACT_SLOTS = 128


class FaultFirstReferee(nn.Module):
    def __init__(self):
        super().__init__()
        self.words = nn.EmbeddingBag(1 << 15 | 1, TOKEN_DIM, mode="mean")
        self.refs = nn.EmbeddingBag(N_ARTIFACT_SLOTS + 1, REF_DIM,
                                    mode="mean", padding_idx=0)
        self.event_type = nn.Embedding(5, CATEGORY_DIM, padding_idx=0)
        self.agent = nn.Embedding(N_AGENT_SLOTS + 1, CATEGORY_DIM, padding_idx=0)
        self.task = nn.Embedding(N_TASK_SLOTS + 1, CATEGORY_DIM, padding_idx=0)
        self.artifact = nn.Embedding(N_ARTIFACT_SLOTS + 1, CATEGORY_DIM,
                                     padding_idx=0)
        self.topology = nn.Embedding(7, CATEGORY_DIM, padding_idx=0)
        input_dim = TOKEN_DIM + REF_DIM + 8 * CATEGORY_DIM + META_DIM
        self.project = nn.Sequential(nn.Linear(input_dim, MODEL_DIM),
                                     nn.LayerNorm(MODEL_DIM), nn.GELU())
        self.encoder = nn.GRU(MODEL_DIM, HIDDEN_DIM, num_layers=2,
                              batch_first=True, bidirectional=True,
                              dropout=0.15)
        self.attention = nn.Linear(2 * HIDDEN_DIM, 1)
        self.label_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(2 * HIDDEN_DIM, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, N_CLASSES))
        self.turn_projection = nn.Sequential(
            nn.Linear(4 * HIDDEN_DIM, 32), nn.Tanh())
        self.class_queries = nn.Parameter(torch.empty(N_FAULT_CLASSES, 32))
        self.turn_bias = nn.Parameter(torch.zeros(N_FAULT_CLASSES))
        nn.init.xavier_uniform_(self.class_queries)

    def forward(self, batch):
        device = self.class_queries.device
        token_ids = torch.as_tensor(batch["token_ids"], dtype=torch.long,
                                    device=device)
        token_offsets = torch.as_tensor(batch["token_offsets"],
                                        dtype=torch.long, device=device)
        ref_ids = torch.as_tensor(batch["ref_ids"], dtype=torch.long,
                                  device=device)
        ref_offsets = torch.as_tensor(batch["ref_offsets"], dtype=torch.long,
                                      device=device)
        cats = torch.as_tensor(batch["categories"], dtype=torch.long,
                               device=device)
        meta = torch.as_tensor(batch["metadata"], dtype=torch.float32,
                               device=device)
        lengths = np_lengths = batch["lengths"].tolist()

        text = self.words(token_ids, token_offsets)
        refs = self.refs(ref_ids, ref_offsets)
        cat_embeds = torch.cat((
            self.event_type(cats[:, 0:1]).squeeze(1),
            self.agent(cats[:, 1:2]).squeeze(1),
            self.agent(cats[:, 2:3]).squeeze(1),
            self.agent(cats[:, 3:4]).squeeze(1),
            self.task(cats[:, 4:5]).squeeze(1),
            self.artifact(cats[:, 5:6]).squeeze(1),
            self.agent(cats[:, 6:7]).squeeze(1),
            self.topology(cats[:, 7:8]).squeeze(1),
        ), dim=1)
        event_vectors = self.project(torch.cat((text, refs, cat_embeds, meta), dim=1))
        split_vectors = torch.split(event_vectors, lengths)
        padded = pad_sequence(split_vectors, batch_first=True)
        packed = pack_padded_sequence(padded, np_lengths, batch_first=True,
                                      enforce_sorted=False)
        encoded_packed, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded_packed, batch_first=True,
                                         total_length=padded.shape[1])
        mask = torch.arange(encoded.shape[1], device=device)[None, :] < \
            torch.as_tensor(lengths, device=device)[:, None]
        attn = self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(attn, dim=1)
        context = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        label_logits = self.label_head(context)
        context_for_events = context.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        turn_vectors = self.turn_projection(torch.cat((encoded, context_for_events),
                                                       dim=-1))
        turn_scores = torch.einsum("btd,cd->btc", turn_vectors,
                                   self.class_queries) + self.turn_bias
        return label_logits, turn_scores, mask
