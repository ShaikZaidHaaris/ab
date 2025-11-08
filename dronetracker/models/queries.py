"""Hierarchical query initialization logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CoreConfig, HierarchicalQueryConfig
from ..models.memory import MemoryBank
from ..utils.geometry import grid_centers


@dataclass
class RegionAllocation:
    activity: torch.Tensor
    allocations: torch.Tensor
    region_centers: torch.Tensor


class CoarseQueryStage(nn.Module):
    """Compute coarse query embeddings and activity."""

    def __init__(self, core: CoreConfig, queries: HierarchicalQueryConfig):
        super().__init__()
        self.queries = queries
        self.embeddings = nn.Parameter(torch.randn(queries.num_coarse_queries, core.model_dim))
        self.activity_head = nn.Sequential(
            nn.LayerNorm(core.model_dim),
            nn.Linear(core.model_dim, core.model_dim // 2),
            nn.GELU(),
            nn.Linear(core.model_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, frame_features: torch.Tensor, region_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = frame_features.size(0)
        coarse = self.embeddings.unsqueeze(0).expand(bsz, -1, -1)
        attention_logits = torch.matmul(coarse, frame_features.transpose(1, 2))
        attention = F.softmax(attention_logits.masked_fill(region_mask == 0, float("-inf")), dim=-1)
        pooled = attention @ frame_features
        activity = self.activity_head(pooled).squeeze(-1)
        return pooled, activity


class FineQueryAllocation(nn.Module):
    """Allocate fine queries per region based on activity and detection counts."""

    def __init__(self, core: CoreConfig, queries: HierarchicalQueryConfig):
        super().__init__()
        self.queries = queries
        self.fine_pool = nn.Parameter(torch.randn(queries.num_coarse_queries * queries.max_fine_per_region, core.model_dim))
        self.state_projector = nn.Sequential(
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, core.model_dim),
        )
        self.identity_projector = nn.Linear(core.model_dim, core.model_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(4, core.model_dim // 2),
            nn.GELU(),
            nn.Linear(core.model_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        coarse_outputs: torch.Tensor,
        activity: torch.Tensor,
        detections_per_region: torch.Tensor,
        memory: MemoryBank,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_regions, dim = coarse_outputs.shape
        allocations = torch.zeros_like(activity, dtype=torch.int64)
        for b in range(bsz):
            for r in range(num_regions):
                if activity[b, r] < self.queries.inactivity_threshold:
                    continue
                dets = detections_per_region[b, r]
                desired = torch.ceil(dets * self.queries.allocation_buffer)
                allocations[b, r] = int(torch.clamp(desired, max=self.queries.max_fine_per_region).item())
        total_queries = allocations.sum(dim=1).clamp(max=self.queries.total_queries)
        max_queries = int(total_queries.max().item()) or 1
        fine = self.fine_pool[:max_queries].unsqueeze(0).expand(bsz, -1, -1)
        state_proj = self.state_projector(memory.state.position_velocity[:, :max_queries])
        identity_proj = self.identity_projector(memory.state.mu_identity[:, :max_queries])
        track_stats = torch.stack(
            [
                memory.state.confidence[:, :max_queries],
                memory.state.miss_count[:, :max_queries],
                memory.state.age[:, :max_queries],
                memory.state.kappa[:, :max_queries],
            ],
            dim=-1,
        )
        gate = self.gate_head(track_stats)
        warm = F.layer_norm(
            fine
            + coarse_outputs[:, :max_queries]
            + gate * identity_proj
            + (1 - gate) * state_proj,
            normalized_shape=(dim,),
        )
        return warm, allocations


def build_region_mask(
    positions: torch.Tensor,
    region_centers: torch.Tensor,
    extent: Tuple[float, float] = (1000.0, 1000.0),
) -> torch.Tensor:
    """Mask detections per region based on grid centers."""

    bsz, num_dets, _ = positions.shape
    num_regions = region_centers.shape[0]
    mask = torch.zeros(bsz, num_regions, num_dets, device=positions.device)
    width, height = extent
    rows, cols = int(torch.sqrt(torch.tensor(num_regions)).item()), int(torch.sqrt(torch.tensor(num_regions)).item())
    cell_w = width / cols
    cell_h = height / rows
    for idx, center in enumerate(region_centers):
        row = idx // cols
        col = idx % cols
        x_min = center[0] - cell_w / 2
        x_max = center[0] + cell_w / 2
        y_min = center[1] - cell_h / 2
        y_max = center[1] + cell_h / 2
        in_region = (
            (positions[..., 0] >= x_min)
            & (positions[..., 0] < x_max)
            & (positions[..., 1] >= y_min)
            & (positions[..., 1] < y_max)
        )
        mask[:, idx] = in_region.float()
    return mask


__all__ = [
    "RegionAllocation",
    "CoarseQueryStage",
    "FineQueryAllocation",
    "build_region_mask",
]
