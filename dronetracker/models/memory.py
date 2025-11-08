"""Memory bank structures for track state management."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from ..config import CoreConfig, IdentityConfig, TrackManagementConfig


@dataclass
class TrackState:
    """State container for a single track slot."""

    position_velocity: torch.Tensor
    covariance: torch.Tensor
    process_noise: torch.Tensor
    exemplar_bank: torch.Tensor
    exemplar_quality: torch.Tensor
    mu_identity: torch.Tensor
    kappa: torch.Tensor
    confidence: torch.Tensor
    age: torch.Tensor
    miss_count: torch.Tensor
    track_id: torch.Tensor
    last_features: torch.Tensor
    soft_assign_history: torch.Tensor


@dataclass
class MemoryBank:
    """Memory bank for all active track slots."""

    state: TrackState
    active_mask: torch.Tensor

    def to(self, device: torch.device) -> "MemoryBank":
        for field_name, value in self.state.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self.state, field_name, value.to(device))
        self.active_mask = self.active_mask.to(device)
        return self

    @classmethod
    def initialize(
        cls,
        batch_size: int,
        config: CoreConfig,
        identity: IdentityConfig,
        track_config: TrackManagementConfig,
        device: Optional[torch.device] = None,
    ) -> "MemoryBank":
        num_slots = config.max_detections
        model_dim = config.model_dim
        device = device or torch.device("cpu")

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(batch_size, num_slots, *shape, device=device)

        state = TrackState(
            position_velocity=zeros(6),
            covariance=torch.ones(batch_size, num_slots, 6, device=device),
            process_noise=torch.ones(batch_size, num_slots, 6, device=device),
            exemplar_bank=zeros(identity.exemplar_bank_size, model_dim),
            exemplar_quality=zeros(identity.exemplar_bank_size),
            mu_identity=torch.zeros(batch_size, num_slots, model_dim, device=device),
            kappa=torch.ones(batch_size, num_slots, device=device),
            confidence=torch.full((batch_size, num_slots), track_config.birth_confidence, device=device),
            age=torch.zeros(batch_size, num_slots, device=device),
            miss_count=torch.zeros(batch_size, num_slots, device=device),
            track_id=torch.arange(num_slots, device=device).expand(batch_size, -1),
            last_features=zeros(2, model_dim),
            soft_assign_history=zeros(2, num_slots),
        )

        active_mask = torch.zeros(batch_size, num_slots, dtype=torch.bool, device=device)
        return cls(state=state, active_mask=active_mask)


__all__ = ["MemoryBank", "TrackState"]
