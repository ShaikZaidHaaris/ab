"""Gumbel-Sinkhorn operator implementation."""
from __future__ import annotations

from typing import Optional

import torch


def gumbel_sinkhorn(
    scores: torch.Tensor,
    iterations: int,
    tau: float,
    gumbel_noise: bool = False,
    training: bool = False,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Apply the Gumbel-Sinkhorn operator to obtain a soft assignment matrix."""

    logits = scores / tau
    if gumbel_noise and training:
        noise = -torch.empty_like(logits).exponential_().log()
        logits = logits + noise

    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)

    return torch.exp(logits).clamp_min(epsilon)


__all__ = ["gumbel_sinkhorn"]
