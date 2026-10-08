"""StepFinder model components adapted for Exp19's event and turn contract.

Architecture source: https://github.com/taiyu-zhu/StepFinder
Copyright (c) 2026 taiyu-zhu. Used and adapted under the MIT License; see
THIRD_PARTY_NOTICES.md. Exp19 adds role-string embeddings, class conditioning,
metadata projection, and a message-only output mask.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


CONTENT_DIM = 384
CONTENT_PROJECTION = 128
AGENT_DIM = 32
HIDDEN_PER_DIRECTION = 64
HIDDEN_DIM = 2 * HIDDEN_PER_DIRECTION
N_FAULT_CLASSES = 6
META_DIM = 12
SCALES = (1, 2)
ALPHA = 0.1
BETA = 0.9
DROPOUT = 0.5
LAMBDA_TEMPORAL = 0.9


class StepFinderLocalizer(nn.Module):
    def __init__(self, n_roles: int):
        super().__init__()
        if n_roles < 1:
            raise ValueError("role vocabulary must include the unknown role")
        self.role_embedding = nn.Embedding(n_roles, AGENT_DIM)
        self.class_embedding = nn.Embedding(N_FAULT_CLASSES, AGENT_DIM)
        self.class_projection = nn.Linear(AGENT_DIM, CONTENT_PROJECTION,
                                          bias=False)
        self.metadata_projection = nn.Linear(META_DIM, CONTENT_PROJECTION,
                                             bias=False)
        self.input_projection = nn.Linear(CONTENT_DIM, CONTENT_PROJECTION)
        self.input_ln = nn.LayerNorm(CONTENT_PROJECTION)
        self.bilstm = nn.LSTM(
            input_size=CONTENT_PROJECTION,
            hidden_size=HIDDEN_PER_DIRECTION,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.output_ln = nn.LayerNorm(HIDDEN_DIM)
        self.query = nn.Linear(HIDDEN_DIM, 64, bias=False)
        self.key = nn.Linear(HIDDEN_DIM, 64, bias=False)
        self.value = nn.Linear(HIDDEN_DIM, 64, bias=False)
        self.attention_out = nn.Linear(64, HIDDEN_DIM)
        self.agent_gate = nn.Linear(AGENT_DIM, HIDDEN_DIM)
        self.score_mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    @staticmethod
    def _multi_scale_difference(hidden: torch.Tensor,
                                valid_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = hidden.shape
        mask = valid_mask.to(hidden.dtype)
        diffs = []
        for scale in SCALES:
            if steps <= scale:
                norm = hidden.new_zeros((batch, steps))
            elif scale == 1:
                delta = hidden[:, 1:] - hidden[:, :-1]
                norm = torch.cat([hidden.new_zeros((batch, 1)),
                                  delta.norm(dim=-1)], dim=1)
            else:
                delta = (hidden[:, scale:] - scale *
                         hidden[:, 1:steps - scale + 1] +
                         (scale - 1) * hidden[:, :steps - scale])
                norm = torch.cat([hidden.new_zeros((batch, scale)),
                                  delta.norm(dim=-1)], dim=1)
            mean = (norm * mask).sum(1, keepdim=True) / mask.sum(
                1, keepdim=True).clamp_min(1.0)
            diffs.append(norm / (mean + 1e-6))
        return torch.stack(diffs).mean(0)

    def forward(self, content: torch.Tensor, metadata: torch.Tensor,
                role_ids: torch.Tensor, class_ids: torch.Tensor,
                valid_mask: torch.Tensor, message_mask: torch.Tensor):
        if content.ndim != 3 or content.shape[-1] != CONTENT_DIM:
            raise ValueError("content must have shape [batch, events, 384]")
        if metadata.shape[:2] != content.shape[:2] or metadata.shape[-1] != META_DIM:
            raise ValueError("metadata must align to events and have 12 columns")
        if role_ids.shape != content.shape[:2]:
            raise ValueError("role ids must align to events")
        if class_ids.shape != (content.shape[0],):
            raise ValueError("one fault class is required per sequence")
        if valid_mask.shape != content.shape[:2] or message_mask.shape != content.shape[:2]:
            raise ValueError("event and message masks must align to events")
        if torch.any(message_mask.bool() & ~valid_mask.bool()):
            raise ValueError("message output mask cannot include padding")
        if torch.any(message_mask.sum(1) == 0):
            raise ValueError("scored sequences must contain a message event")

        batch, steps, _ = content.shape
        valid = valid_mask.bool()
        mask_f = valid.unsqueeze(-1).to(content.dtype)
        agent = self.role_embedding(role_ids)
        class_context = self.class_projection(self.class_embedding(class_ids))
        x = (self.input_projection(content) +
             self.metadata_projection(metadata) + class_context.unsqueeze(1))
        x = self.input_ln(x) * mask_f
        lengths = valid.sum(1).cpu().long()
        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                      enforce_sorted=False)
        out_packed, _ = self.bilstm(packed)
        hidden, _ = pad_packed_sequence(out_packed, batch_first=True,
                                        total_length=steps)
        hidden = self.output_ln(hidden * mask_f) * mask_f

        # Agent-aware scaled dot-product attention and masked global role gate.
        q = self.query(hidden).view(batch, steps, 2, 32)
        k = self.key(hidden).view(batch, steps, 2, 32)
        v = self.value(hidden).view(batch, steps, 2, 32)
        scores = torch.einsum("bthd,bshd->bhts", q, k) / (32 ** 0.5)
        role_unit = F.normalize(agent, p=2, dim=-1)
        role_bias = torch.bmm(role_unit, role_unit.transpose(1, 2))
        scores = scores + ALPHA * role_bias.unsqueeze(1)
        scores = scores.masked_fill(~valid[:, None, None, :], -1e9)
        attention = F.softmax(scores, dim=-1)
        attended = torch.einsum("bhts,bshd->bthd", attention, v).reshape(
            batch, steps, 64)
        hidden = (hidden + self.attention_out(attended)) * mask_f
        mean_role = (agent * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
        gate = torch.sigmoid(self.agent_gate(mean_role))
        hidden = hidden * (1.0 + ALPHA * gate.unsqueeze(1)) * mask_f

        logits = (self.score_mlp(hidden).squeeze(-1) +
                  BETA * self._multi_scale_difference(hidden, valid))
        # Gamma is fixed at zero: no position-dependent late-step bias.
        logits = logits.masked_fill(~message_mask.bool(), -1e9)

        if steps > 1:
            next_hidden = self.temporal_head(hidden[:, :-1])
            next_target = hidden[:, 1:].detach()
            next_mask = valid[:, 1:].unsqueeze(-1).to(hidden.dtype)
            temporal_loss = ((next_hidden - next_target).square() *
                             next_mask).sum() / next_mask.sum().clamp_min(1.0)
        else:
            temporal_loss = hidden.new_zeros(())
        return logits, temporal_loss


def compute_loss(logits: torch.Tensor, target_event: torch.Tensor,
                 temporal_loss: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or target_event.shape != (logits.shape[0],):
        raise ValueError("one target message event is required per sequence")
    return F.cross_entropy(logits, target_event) + LAMBDA_TEMPORAL * temporal_loss
