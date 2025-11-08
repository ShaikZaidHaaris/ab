"""Scene-aware decoder composed of self-attention and cross attention blocks."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import CoreConfig
from ..utils.nn import FeedForward, MaskedSelfAttention, ResidualBlock


class DecoderLayer(nn.Module):
    """Single decoder layer with gated multi-source cross attention."""

    def __init__(self, core: CoreConfig):
        super().__init__()
        self.self_attn = ResidualBlock(
            MaskedSelfAttention(core.model_dim, core.num_heads, core.dropout),
            model_dim=core.model_dim,
            dropout=core.dropout,
        )
        self.source_proj = nn.Linear(core.model_dim, core.model_dim)
        self.cross_spatial = nn.MultiheadAttention(core.model_dim, core.num_heads, batch_first=True, dropout=core.dropout)
        self.cross_st = nn.MultiheadAttention(core.model_dim, core.num_heads, batch_first=True, dropout=core.dropout)
        self.cross_social = nn.MultiheadAttention(core.model_dim, core.num_heads, batch_first=True, dropout=core.dropout)
        self.ffn = ResidualBlock(
            lambda x, **_: FeedForward(core.model_dim, core.model_dim * 2, core.dropout)(x),
            model_dim=core.model_dim,
            dropout=core.dropout,
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(core.model_dim),
            nn.Linear(core.model_dim, core.model_dim),
            nn.GELU(),
            nn.Linear(core.model_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, queries: torch.Tensor, sources: Dict[str, torch.Tensor]) -> torch.Tensor:
        q = self.self_attn(queries)
        gates = self.gate_mlp(q)
        spatial, _ = self.cross_spatial(q, sources["spatial"], sources["spatial"])
        st, _ = self.cross_st(q, sources["st"], sources["st"])
        social, _ = self.cross_social(q, sources["social"], sources["social"])
        q = q + gates[..., 0:1] * spatial + gates[..., 1:2] * st + gates[..., 2:3] * social
        q = self.ffn(q)
        return q


class SceneDecoder(nn.Module):
    """Stack multiple decoder layers."""

    def __init__(self, core: CoreConfig, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(core) for _ in range(num_layers)])
        self.identity_proj = nn.Linear(core.model_dim, core.model_dim)
        self.norm = nn.LayerNorm(core.model_dim)

    def forward(self, queries: torch.Tensor, sources: Dict[str, torch.Tensor], identity_embeddings: torch.Tensor, identity_gate: torch.Tensor) -> torch.Tensor:
        x = queries
        for layer in self.layers:
            x = layer(x, sources)
        identity = self.identity_proj(identity_embeddings)
        x = self.norm((1 - identity_gate) * x + identity_gate * identity)
        return x


__all__ = ["SceneDecoder"]
