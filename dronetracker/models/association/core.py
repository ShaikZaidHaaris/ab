"""Association module implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ...config import AssociationConfig, CoreConfig, IdentityConfig
from ...utils.geometry import gaussian_probability, mahalanobis_distance
from ...utils.nn import FeedForward, MaskedSelfAttention, ResidualBlock
from ...utils.sinkhorn import gumbel_sinkhorn


@dataclass
class AssociationOutput:
    scores: torch.Tensor
    assignments: torch.Tensor
    smoothed: torch.Tensor


class CandidatePruner(nn.Module):
    """Physics-based candidate gating."""

    def __init__(self, association: AssociationConfig):
        super().__init__()
        self.association = association
        self.weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))

    def forward(
        self,
        query_states: torch.Tensor,
        query_covariance: torch.Tensor,
        query_velocity: torch.Tensor,
        detection_positions: torch.Tensor,
        detection_velocities: torch.Tensor,
        detection_features: torch.Tensor,
        query_features: torch.Tensor,
        kappa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_queries, _ = query_states.shape
        num_dets = detection_positions.shape[1]
        diff = detection_positions.unsqueeze(1) - query_states.unsqueeze(2)
        gate_distance = (diff.abs() <= 3.5 * torch.sqrt(query_covariance.unsqueeze(2))).all(dim=-1)
        maha = mahalanobis_distance(diff, query_covariance.unsqueeze(2))
        threshold = torch.distributions.chi2.Chi2(6).icdf(torch.tensor(0.95)).to(diff.device)
        gate_maha = maha <= threshold
        cos_similarity = torch.nn.functional.cosine_similarity(
            query_features.unsqueeze(2), detection_features.unsqueeze(1), dim=-1
        )
        speed_diff = detection_velocities.norm(dim=-1).unsqueeze(1) - query_velocity.norm(dim=-1).unsqueeze(2)
        alignment = torch.nn.functional.cosine_similarity(query_velocity.unsqueeze(2), diff, dim=-1)
        spatial_score = torch.exp(-diff.norm(dim=-1))
        total_score = (
            self.weights[0] * cos_similarity * (1 + 0.1 * kappa.unsqueeze(2))
            + self.weights[1] * spatial_score
            + self.weights[2] * alignment
            - self.weights[3] * speed_diff.abs()
        )
        score = total_score.masked_fill(~(gate_distance & gate_maha), float("-inf"))
        topk_queries = score.topk(min(self.association.candidates_per_query, num_dets), dim=-1)
        return score, topk_queries.indices


class AssociationTransformer(nn.Module):
    """Bidirectional transformer for track-detection associations."""

    def __init__(self, core: CoreConfig, association: AssociationConfig, identity: IdentityConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "track_self": ResidualBlock(
                            MaskedSelfAttention(core.model_dim, core.num_heads, core.dropout),
                            model_dim=core.model_dim,
                            dropout=core.dropout,
                        ),
                        "det_self": ResidualBlock(
                            MaskedSelfAttention(core.model_dim, core.num_heads, core.dropout),
                            model_dim=core.model_dim,
                            dropout=core.dropout,
                        ),
                        "co_attn": nn.MultiheadAttention(core.model_dim, core.num_heads, batch_first=True, dropout=core.dropout),
                        "ffn_track": ResidualBlock(
                            lambda x, **_: FeedForward(core.model_dim, core.model_dim * 2, core.dropout)(x),
                            model_dim=core.model_dim,
                            dropout=core.dropout,
                        ),
                        "ffn_det": ResidualBlock(
                            lambda x, **_: FeedForward(core.model_dim, core.model_dim * 2, core.dropout)(x),
                            model_dim=core.model_dim,
                            dropout=core.dropout,
                        ),
                    }
                )
                for _ in range(association.association_layers)
            ]
        )
        self.identity_proj = nn.Linear(core.model_dim, core.model_dim)
        self.identity_gate = nn.Parameter(torch.tensor(0.5))
        self.score_head_query = nn.Linear(core.model_dim, core.model_dim)
        self.score_head_det = nn.Linear(core.model_dim, core.model_dim)
        self.bias_unmatched_query = nn.Parameter(torch.full((1,), association.dustbin_bias_init))
        self.bias_unmatched_det = nn.Parameter(torch.full((1,), association.dustbin_bias_init))

    def forward(
        self,
        query_tokens: torch.Tensor,
        detection_tokens: torch.Tensor,
        identity_memory: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = query_tokens
        d = detection_tokens
        identity = self.identity_proj(identity_memory)
        for layer in self.layers:
            q = layer["track_self"](q, mask=None)
            d = layer["det_self"](d, mask=None)
            cross_q, _ = layer["co_attn"](q, d, d)
            cross_d, _ = layer["co_attn"](d, q, q)
            q = q + cross_q + self.identity_gate * identity
            d = d + cross_d
            q = layer["ffn_track"](q)
            d = layer["ffn_det"](d)
        score = torch.einsum("bid, bjd -> bij", self.score_head_query(q), self.score_head_det(d))
        return score


class AssociationModule(nn.Module):
    def __init__(self, core: CoreConfig, association: AssociationConfig, identity: IdentityConfig):
        super().__init__()
        self.pruner = CandidatePruner(association)
        self.transformer = AssociationTransformer(core, association, identity)
        self.association = association

    def forward(
        self,
        query_state: torch.Tensor,
        query_covariance: torch.Tensor,
        query_velocity: torch.Tensor,
        detections: Dict[str, torch.Tensor],
        query_tokens: torch.Tensor,
        identity_embeddings: torch.Tensor,
        training: bool = False,
    ) -> AssociationOutput:
        score, _ = self.pruner(
            query_state,
            query_covariance,
            query_velocity,
            detections["positions"],
            detections["velocities"],
            detections["features"],
            query_tokens,
            identity_embeddings,
        )
        refined_scores = self.transformer(query_tokens, detections["features"], identity_embeddings, mask=None)
        dustbin_query = self.transformer.bias_unmatched_query
        dustbin_det = self.transformer.bias_unmatched_det
        padded_scores = torch.cat([refined_scores, dustbin_query.expand_as(refined_scores[..., :1])], dim=-1)
        padded_scores = torch.cat([padded_scores, dustbin_det.expand_as(padded_scores[:, :1])], dim=1)
        tau = torch.linspace(
            self.association.sinkhorn_temperature_start,
            self.association.sinkhorn_temperature_end,
            steps=1,
            device=refined_scores.device,
        )[0]
        assignment = gumbel_sinkhorn(
            padded_scores,
            iterations=self.association.sinkhorn_iterations,
            tau=tau,
            gumbel_noise=True,
            training=training,
        )
        smoothed = 0.7 * assignment + 0.3 * assignment.detach()
        return AssociationOutput(scores=refined_scores, assignments=assignment, smoothed=smoothed)


__all__ = ["AssociationModule", "AssociationOutput"]
