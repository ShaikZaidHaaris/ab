"""Configuration objects for the drone tracking architecture."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CoreConfig:
    """Core hyper-parameters shared across the architecture."""

    time_window: int = 100
    max_detections: int = 80
    feature_dim: int = 16
    model_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    precision: str = "bf16"
    use_flash_attention: bool = True


@dataclass
class HierarchicalQueryConfig:
    num_coarse_queries: int = 16
    grid_shape: Tuple[int, int] = (4, 4)
    max_fine_per_region: int = 10
    total_queries: int = 80
    inactivity_threshold: float = 0.1
    allocation_buffer: float = 1.2


@dataclass
class EncoderConfig:
    k_spatial: int = 15
    k_temporal: int = 6
    temporal_window: int = 3
    k_social: int = 15
    iterations_st: int = 2


@dataclass
class AssociationConfig:
    candidates_per_query: int = 24
    candidates_per_detection: int = 16
    association_layers: int = 2
    sinkhorn_iterations: int = 5
    sinkhorn_temperature_start: float = 1.0
    sinkhorn_temperature_end: float = 0.2
    dustbin_bias_init: float = -2.0


@dataclass
class IdentityConfig:
    exemplar_bank_size: int = 8
    kappa_max: float = 100.0
    strong_match_threshold: float = 0.7
    cosine_threshold: float = 0.35
    blend_rate: float = 0.2


@dataclass
class PredictionConfig:
    num_modes: int = 3
    horizon: int = 10
    corridor_threshold: float = 0.3


@dataclass
class TrackManagementConfig:
    birth_confidence: float = 0.5
    death_miss_threshold: int = 5
    death_confidence_threshold: float = 0.2
    search_mode_trigger: int = 2


@dataclass
class TrainingConfig:
    warmup_epochs: int = 10
    total_epochs: int = 50
    base_lr: float = 2e-4
    final_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    sinkhorn_temperature_schedule: Tuple[float, float] = (1.0, 0.2)
    curriculum: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "phase1": (1, 15),
            "phase2": (16, 35),
            "phase3": (36, 50),
        }
    )


@dataclass
class LossConfig:
    w_assoc: float = 1.0
    w_traj: float = 2.0
    w_id: float = 0.5
    w_state: float = 0.3
    w_calib: float = 0.1
    alpha_focal: float = 0.3
    beta_margin: float = 0.1
    lambda_dustbin: float = 0.01
    lambda_smooth: float = 1e-4
    tau_id: float = 0.07
    lambda_purity: float = 0.05
    margin_purity: float = 0.5
    huber_delta: float = 1.0
    lambda_nll: float = 0.01


@dataclass
class DroneTrackerConfig:
    core: CoreConfig = field(default_factory=CoreConfig)
    queries: HierarchicalQueryConfig = field(default_factory=HierarchicalQueryConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    track_management: TrackManagementConfig = field(default_factory=TrackManagementConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    losses: LossConfig = field(default_factory=LossConfig)


DEFAULT_CONFIG = DroneTrackerConfig()

__all__ = [
    "CoreConfig",
    "HierarchicalQueryConfig",
    "EncoderConfig",
    "AssociationConfig",
    "IdentityConfig",
    "PredictionConfig",
    "TrackManagementConfig",
    "TrainingConfig",
    "LossConfig",
    "DroneTrackerConfig",
    "DEFAULT_CONFIG",
]
