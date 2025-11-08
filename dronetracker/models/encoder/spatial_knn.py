"""Spatial kNN attention encoder stage."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...config import CoreConfig, EncoderConfig
from ...utils.geometry import pairwise_distances, knn_indices
from ...utils.nn import FeedForward, MaskedSelfAttention, ResidualBlock


class SpatialKNNEncoder(nn.Module):
    """Per-frame kNN attention encoder."""

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

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, N, D).
            positions: Tensor of shape (B, T, N, 3).
        """

        bsz, time, num_nodes, _ = x.shape
        outputs = []
        for t in range(time):
            frame_feats = x[:, t]
            frame_pos = positions[:, t]
            distances = pairwise_distances(frame_pos)
            indices = knn_indices(distances, self.encoder.k_spatial)
            mask = torch.zeros(bsz, num_nodes, num_nodes, device=x.device, dtype=torch.bool)
            for b in range(bsz):
                mask[b].scatter_(1, indices[b], True)
                mask[b] = mask[b] | torch.eye(num_nodes, device=x.device, dtype=torch.bool)
            frame_out = self.attn_block(frame_feats, mask=mask)
            frame_out = self.ffn_block(frame_out)
            outputs.append(frame_out.unsqueeze(1))
        return torch.cat(outputs, dim=1)


__all__ = ["SpatialKNNEncoder"]
