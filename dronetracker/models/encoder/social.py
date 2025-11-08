"""Social interaction graph encoder."""
from __future__ import annotations

import torch
import torch.nn as nn

from ...config import CoreConfig, EncoderConfig
from ...utils.geometry import pairwise_distances, knn_indices
from ...utils.nn import FeedForward


class GraphAttentionLayer(nn.Module):
    """Simple graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        self.proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.Parameter(torch.randn(num_heads, self.head_dim * 2))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, num_nodes, _ = x.shape
        h = self.proj(x).view(bsz, num_nodes, self.num_heads, self.head_dim)
        h_i = h.unsqueeze(2)
        h_j = h.unsqueeze(1)
        attn_input = torch.cat([h_i, h_j], dim=-1)
        logits = (attn_input * self.attn).sum(dim=-1)
        logits = self.leaky_relu(logits)
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(logits, dim=2)
        weights = self.dropout(weights)
        context = torch.einsum("bijd, bjhd -> bihd", weights, h)
        return context.reshape(bsz, num_nodes, self.num_heads * self.head_dim)


class SocialInteractionEncoder(nn.Module):
    """Stacked GAT encoder for social reasoning."""

    def __init__(self, core: CoreConfig, encoder: EncoderConfig):
        super().__init__()
        self.encoder = encoder
        self.layer1 = GraphAttentionLayer(core.model_dim, core.model_dim, num_heads=2, dropout=core.dropout)
        self.layer2 = GraphAttentionLayer(core.model_dim, core.model_dim, num_heads=2, dropout=core.dropout)
        self.pool = FeedForward(core.model_dim, core.model_dim * 2, core.dropout)
        self.norm = nn.LayerNorm(core.model_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, time, num_nodes, dim = x.shape
        outputs = []
        for t in range(time):
            feats = x[:, t]
            pos = positions[:, t]
            distances = pairwise_distances(pos)
            indices = knn_indices(distances, self.encoder.k_social)
            mask = torch.zeros(bsz, num_nodes, num_nodes, dtype=torch.bool, device=x.device)
            for b in range(bsz):
                mask[b].scatter_(1, indices[b], True)
            mask = mask | torch.eye(num_nodes, device=x.device, dtype=torch.bool)
            h1 = self.layer1(feats, mask)
            h2 = self.layer2(h1, mask)
            pooled = self.pool(h2)
            outputs.append(self.norm(pooled).unsqueeze(1))
        return torch.cat(outputs, dim=1)


__all__ = ["SocialInteractionEncoder"]
