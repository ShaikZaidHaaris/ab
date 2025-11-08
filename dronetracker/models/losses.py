"""Loss functions for the tracker."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ..config import LossConfig


def association_loss(pred: torch.Tensor, target: torch.Tensor, config: LossConfig) -> torch.Tensor:
    ce = F.cross_entropy(pred, target)
    focal = (-(1 - torch.softmax(pred, dim=-1)) ** config.alpha_focal * F.log_softmax(pred, dim=-1)).mean()
    margin = torch.relu(torch.cosine_similarity(pred, target, dim=-1) - config.margin_purity).mean()
    return config.w_assoc * (ce + config.beta_margin * margin + config.alpha_focal * focal)


def trajectory_nll(outputs: Dict[str, torch.Tensor], target: torch.Tensor, config: LossConfig) -> torch.Tensor:
    means = outputs["means"]
    variances = outputs["vars"]
    mode_probs = outputs["mode_probs"]
    diff = target.unsqueeze(2) - means
    log_prob = -0.5 * (diff**2 / variances).sum(dim=-1) - 0.5 * torch.log(variances).sum(dim=-1)
    mixture = torch.logsumexp(torch.log(mode_probs.unsqueeze(-1)) + log_prob, dim=2)
    smooth = config.lambda_smooth * torch.diff(target, n=2, dim=-2).pow(2).sum()
    return config.w_traj * (-mixture.mean() + smooth)


def identity_loss(embeddings: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, config: LossConfig) -> torch.Tensor:
    logits = torch.matmul(embeddings, positives.t()) / config.tau_id
    labels = torch.arange(embeddings.size(0), device=embeddings.device)
    info_nce = F.cross_entropy(logits, labels)
    purity = config.lambda_purity * torch.relu(torch.cosine_similarity(embeddings, negatives) - config.margin_purity).mean()
    return config.w_id * (info_nce + purity)


def state_loss(pred: torch.Tensor, target: torch.Tensor, config: LossConfig) -> torch.Tensor:
    return config.w_state * F.huber_loss(pred, target, delta=config.huber_delta)


def calibration_loss(confidence: torch.Tensor, error: torch.Tensor, config: LossConfig, bins: int = 10) -> torch.Tensor:
    bin_boundaries = torch.linspace(0, 1, bins + 1, device=confidence.device)
    ece = torch.zeros(1, device=confidence.device)
    for i in range(bins):
        mask = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
        if mask.any():
            ece += torch.abs(confidence[mask].mean() - error[mask].mean()) * mask.float().mean()
    nll_temp = config.lambda_nll * F.binary_cross_entropy(confidence, 1 - error)
    return config.w_calib * (ece + nll_temp)


__all__ = [
    "association_loss",
    "trajectory_nll",
    "identity_loss",
    "state_loss",
    "calibration_loss",
]
