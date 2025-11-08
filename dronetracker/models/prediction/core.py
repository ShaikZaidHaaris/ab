"""Prediction heads for current state, multi-modal trajectories, and confidence."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ...config import CoreConfig, PredictionConfig
from ...utils.nn import FeedForward


class CurrentStateHead(nn.Module):
    def __init__(self, core: CoreConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(core.model_dim),
            nn.Linear(core.model_dim, core.model_dim),
            nn.GELU(),
            nn.Linear(core.model_dim, core.model_dim // 2),
            nn.GELU(),
            nn.Linear(core.model_dim // 2, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TrajectoryTransformer(nn.Module):
    def __init__(self, core: CoreConfig, prediction: PredictionConfig):
        super().__init__()
        self.prediction = prediction
        self.token_proj = nn.Linear(core.model_dim + 64 + 6, core.model_dim)
        self.pe = nn.Parameter(torch.randn(prediction.horizon, 64))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=core.model_dim,
                    nhead=core.num_heads,
                    dim_feedforward=core.model_dim * 2,
                    dropout=core.dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(2)
            ]
        )
        self.mode_embeddings = nn.Parameter(torch.randn(prediction.num_modes, 64))
        self.mean_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(core.model_dim + 64, core.model_dim),
                    nn.GELU(),
                    nn.Linear(core.model_dim, 6),
                )
                for _ in range(prediction.num_modes)
            ]
        )
        self.var_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(core.model_dim + 64, core.model_dim // 2),
                    nn.GELU(),
                    nn.Linear(core.model_dim // 2, 6),
                    nn.Softplus(),
                )
                for _ in range(prediction.num_modes)
            ]
        )
        self.mode_head = nn.Sequential(
            nn.LayerNorm(core.model_dim),
            nn.Linear(core.model_dim, core.model_dim // 2),
            nn.GELU(),
            nn.Linear(core.model_dim // 2, prediction.num_modes),
            nn.Softmax(dim=-1),
        )

    def forward(self, track_tokens: torch.Tensor, current_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, num_tracks, dim = track_tokens.shape
        temporal = self.pe.unsqueeze(0).unsqueeze(0).expand(bsz, num_tracks, -1, -1)
        state = current_state.unsqueeze(2).expand(-1, -1, self.prediction.horizon, -1)
        tokens = torch.cat([track_tokens.unsqueeze(2).expand(-1, -1, self.prediction.horizon, -1), temporal, state], dim=-1)
        tokens = self.token_proj(tokens)
        tokens = tokens.view(bsz * num_tracks, self.prediction.horizon, dim)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = tokens.view(bsz, num_tracks, self.prediction.horizon, dim)
        outputs = {
            "means": [],
            "vars": [],
            "mode_probs": None,
        }
        mode_probs = self.mode_head(track_tokens)
        outputs["mode_probs"] = mode_probs
        for idx in range(self.prediction.num_modes):
            mode_embed = self.mode_embeddings[idx].view(1, 1, 1, -1)
            concat = torch.cat([tokens, mode_embed.expand_as(tokens)], dim=-1)
            mean = self.mean_head[idx](concat)
            var = self.var_head[idx](concat)
            outputs["means"].append(mean)
            outputs["vars"].append(var)
        outputs["means"] = torch.stack(outputs["means"], dim=2)
        outputs["vars"] = torch.stack(outputs["vars"], dim=2)
        return outputs


class ConfidenceHead(nn.Module):
    def __init__(self, core: CoreConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(core.model_dim),
            nn.Linear(core.model_dim, core.model_dim // 2),
            nn.GELU(),
            nn.Linear(core.model_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class PredictionHeads(nn.Module):
    def __init__(self, core: CoreConfig, prediction: PredictionConfig):
        super().__init__()
        self.state = CurrentStateHead(core)
        self.trajectory = TrajectoryTransformer(core, prediction)
        self.confidence = ConfidenceHead(core)

    def forward(self, tokens: torch.Tensor, memory_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        state = self.state(tokens)
        traj = self.trajectory(tokens, state)
        conf = self.confidence(tokens)
        return {"state": state, "trajectory": traj, "confidence": conf}


__all__ = ["PredictionHeads"]
