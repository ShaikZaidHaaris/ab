"""Spatio-temporal kNN mixer."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ...config import CoreConfig, EncoderConfig
from ...utils.geometry import motion_compensated_positions
from ...utils.nn import FeedForward, MaskedSelfAttention, ResidualBlock


class SpatioTemporalEncoder(nn.Module):
    """Motion compensated spatio-temporal encoder."""

    def __init__(self, core: CoreConfig, encoder: EncoderConfig):
        super().__init__()
        self.core = core
        self.encoder = encoder
        self.attn_block = ResidualBlock(
            MaskedSelfAttention(core.model_dim, core.num_heads, core.dropout),
            model_dim=core.model_dim,
            dropout=core.dropout,
        )
        self.ffn_block = ResidualBlock(
            lambda x, **_: FeedForward(core.model_dim, core.model_dim * 2, core.dropout)(x),
            model_dim=core.model_dim,
            dropout=core.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Apply motion-compensated attention across time."""

        bsz, time, num_nodes, _ = x.shape
        offsets = [i for i in range(-self.encoder.temporal_window, self.encoder.temporal_window + 1) if i != 0]
        outputs = x
        for _ in range(self.encoder.iterations_st):
            updated: List[torch.Tensor] = []
            for t in range(time):
                current_feat = outputs[:, t]
                current_pos = positions[:, t]
                mask = torch.zeros(bsz, num_nodes, num_nodes, device=x.device, dtype=torch.bool)
                for offset in offsets:
                    target_idx = t + offset
                    if target_idx < 0 or target_idx >= time:
                        continue
                    predicted = motion_compensated_positions(current_pos, velocities[:, t], offset)
                    target_pos = positions[:, target_idx]
                    diff = predicted.unsqueeze(2) - target_pos.unsqueeze(1)
                    dist = diff.norm(dim=-1)
                    topk = dist.topk(min(self.encoder.k_temporal, num_nodes), largest=False).indices
                    for b in range(bsz):
                        mask[b].scatter_(1, topk[b], True)
                mask = mask | torch.eye(num_nodes, device=x.device, dtype=torch.bool)
                attn = self.attn_block(current_feat, mask=mask)
                attn = self.ffn_block(attn)
                updated.append(attn.unsqueeze(1))
            outputs = torch.cat(updated, dim=1)
        return outputs


__all__ = ["SpatioTemporalEncoder"]
