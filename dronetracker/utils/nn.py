"""Neural network utilities used across the tracker."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Simple transformer feed-forward block."""

    def __init__(self, model_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or model_dim * 2
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MaskedSelfAttention(nn.Module):
    """Scaled dot-product attention with binary masks."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply masked self-attention.

        Args:
            x: Input tensor with shape (B, N, D).
            mask: Attention mask with shape (B, N, N).
            key_padding_mask: Optional boolean mask with shape (B, N).
        """

        bsz, num_tokens, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        if key_padding_mask is not None:
            padding = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~padding, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(bsz, num_tokens, self.model_dim)
        return self.out_proj(context)


class ResidualBlock(nn.Module):
    """LayerNorm + residual helper."""

    def __init__(self, module: nn.Module, model_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.module(x, *args, **kwargs)
        x = self.dropout(x)
        return residual + x


def topk_mask(scores: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Return boolean mask keeping top-k entries along a dimension."""

    if k >= scores.shape[dim]:
        return torch.ones_like(scores, dtype=torch.bool)
    topk = scores.topk(k, dim=dim).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(dim, topk, True)
    return mask


__all__ = ["FeedForward", "MaskedSelfAttention", "ResidualBlock", "topk_mask"]
