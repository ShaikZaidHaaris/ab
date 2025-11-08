"""High-level inference loop implementing the architecture."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import DEFAULT_CONFIG, DroneTrackerConfig
from ..models.association.core import AssociationModule
from ..models.decoder import SceneDecoder
from ..models.embeddings import DetectionEmbedding
from ..models.encoder.spatial_knn import SpatialKNNEncoder
from ..models.encoder.spatiotemporal import SpatioTemporalEncoder
from ..models.encoder.social import SocialInteractionEncoder
from ..models.memory import MemoryBank
from ..models.prediction.core import PredictionHeads
from ..models.queries import CoarseQueryStage, FineQueryAllocation, build_region_mask
from ..utils.geometry import grid_centers


class DroneTracker(nn.Module):
    def __init__(self, config: DroneTrackerConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.embedding = DetectionEmbedding(config.core)
        self.spatial_encoder = SpatialKNNEncoder(config.core, config.encoder)
        self.st_encoder = SpatioTemporalEncoder(config.core, config.encoder)
        self.social_encoder = SocialInteractionEncoder(config.core, config.encoder)
        self.coarse_queries = CoarseQueryStage(config.core, config.queries)
        self.fine_queries = FineQueryAllocation(config.core, config.queries)
        self.decoder = SceneDecoder(config.core)
        self.association = AssociationModule(config.core, config.association, config.identity)
        self.prediction = PredictionHeads(config.core, config.prediction)
        self.memory = MemoryBank.initialize(1, config.core, config.identity, config.track_management)
        self.register_buffer(
            "region_centers",
            grid_centers(config.queries.grid_shape, (1000.0, 1000.0)).to(torch.float32),
        )

    def forward(self, detections: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, time, num_dets, feat = detections.shape
        embedded = self.embedding(detections.view(-1, feat)).view(bsz, time, num_dets, -1)
        positions = detections[..., :3]
        velocities = detections[..., 3:6]
        spatial = self.spatial_encoder(embedded, positions)
        st = self.st_encoder(spatial, positions, velocities)
        social = self.social_encoder(st, positions)
        frame_features = spatial[:, -1]
        region_mask = build_region_mask(positions[:, -1], self.region_centers)
        coarse, activity = self.coarse_queries(frame_features, region_mask)
        detections_per_region = region_mask.sum(dim=-1)
        warm, allocations = self.fine_queries(coarse, activity, detections_per_region, self.memory)
        sources = {
            "spatial": frame_features,
            "st": st.reshape(bsz, -1, frame_features.size(-1)),
            "social": social[:, -1],
        }
        identity_gate = torch.sigmoid(self.memory.state.confidence.unsqueeze(-1))
        decoded = self.decoder(warm, sources, self.memory.state.mu_identity, identity_gate)
        association_inputs = {
            "positions": positions[:, -1],
            "velocities": velocities[:, -1],
            "features": frame_features,
        }
        assoc = self.association(
            self.memory.state.position_velocity,
            self.memory.state.covariance,
            self.memory.state.position_velocity[..., 3:6],
            association_inputs,
            decoded,
            self.memory.state.mu_identity,
        )
        predictions = self.prediction(decoded, self.memory.state.position_velocity)
        return {
            "assignments": assoc.smoothed,
            "predictions": predictions,
            "decoder": decoded,
        }


__all__ = ["DroneTracker"]
