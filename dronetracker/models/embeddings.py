"""Feature embedding layers."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CoreConfig


class DetectionEmbedding(nn.Module):
    """Embed raw detection features into model space."""

    def __init__(self, config: CoreConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, config.model_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.mlp(features)
        return self.norm(x)


__all__ = ["DetectionEmbedding"]
