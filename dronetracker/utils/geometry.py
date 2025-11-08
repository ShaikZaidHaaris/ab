"""Geometry utilities for distance metrics and motion compensation."""
from __future__ import annotations

import math
from typing import Tuple

import torch


def pairwise_distances(points: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute pairwise Euclidean distances between points.

    Args:
        points: Tensor of shape (..., N, 3).
        eps: Numerical stability constant.

    Returns:
        Pairwise distance tensor of shape (..., N, N).
    """

    diff = points.unsqueeze(-2) - points.unsqueeze(-3)
    dist_sq = (diff**2).sum(dim=-1).clamp_min(eps)
    return dist_sq.sqrt()


def motion_compensated_positions(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    offset: int,
    delta_t: float = 1.0,
) -> torch.Tensor:
    """Predict positions under constant velocity assumption."""

    return positions + velocities * (offset * delta_t)


def mahalanobis_distance(
    diff: torch.Tensor,
    covariance: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Mahalanobis distance for diagonal covariance matrices."""

    inv_cov = (covariance + eps).reciprocal()
    return (diff**2 * inv_cov).sum(dim=-1)


def knn_indices(distances: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the k nearest neighbours (including self)."""

    k = min(k, distances.shape[-1])
    return distances.topk(k, largest=False).indices


def build_knn_mask(indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Construct an adjacency mask from kNN indices."""

    batch_shape = indices.shape[:-1]
    device = indices.device
    mask = torch.zeros(*batch_shape, num_nodes, dtype=torch.bool, device=device)
    arange_idx = torch.arange(num_nodes, device=device)
    expanded_arange = arange_idx.view((1,) * (mask.ndim - 1) + (num_nodes,))
    mask.scatter_(-1, indices, True)
    mask[..., arange_idx] = True
    mask = mask.unsqueeze(-2).expand(*batch_shape, num_nodes, num_nodes)
    return mask


def grid_centers(grid_shape: Tuple[int, int], extent: Tuple[float, float]) -> torch.Tensor:
    """Return 2D grid centers for hierarchical coarse queries."""

    rows, cols = grid_shape
    width, height = extent
    xs = torch.linspace(-width / 2.0, width / 2.0, cols)
    ys = torch.linspace(-height / 2.0, height / 2.0, rows)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
    return centers


def gaussian_probability(diff: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """Compute unnormalised Gaussian probability for gating."""

    maha = mahalanobis_distance(diff, covariance)
    return torch.exp(-0.5 * maha)


__all__ = [
    "pairwise_distances",
    "motion_compensated_positions",
    "mahalanobis_distance",
    "knn_indices",
    "build_knn_mask",
    "grid_centers",
    "gaussian_probability",
]
