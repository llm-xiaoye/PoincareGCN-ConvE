"""Numerically stable operations on a Poincare ball with curvature -c."""

from __future__ import annotations

import math

import torch


def artanh(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def project(value: torch.Tensor, curvature: float, eps: float = 1e-5) -> torch.Tensor:
    max_norm = (1.0 - eps) / math.sqrt(curvature)
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    projected = value * (max_norm / norm)
    return torch.where(norm > max_norm, projected, value)


def expmap0(tangent: torch.Tensor, curvature: float) -> torch.Tensor:
    sqrt_c = math.sqrt(curvature)
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    mapped = torch.tanh(sqrt_c * norm) * tangent / (sqrt_c * norm)
    return project(mapped, curvature)


def logmap0(point: torch.Tensor, curvature: float) -> torch.Tensor:
    sqrt_c = math.sqrt(curvature)
    point = project(point, curvature)
    norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return artanh(sqrt_c * norm) * point / (sqrt_c * norm)


def mobius_add(left: torch.Tensor, right: torch.Tensor, curvature: float) -> torch.Tensor:
    left_squared = (left * left).sum(dim=-1, keepdim=True)
    right_squared = (right * right).sum(dim=-1, keepdim=True)
    inner = (left * right).sum(dim=-1, keepdim=True)
    numerator = (
        (1.0 + 2.0 * curvature * inner + curvature * right_squared) * left
        + (1.0 - curvature * left_squared) * right
    )
    denominator = (
        1.0
        + 2.0 * curvature * inner
        + curvature * curvature * left_squared * right_squared
    ).clamp_min(1e-15)
    return project(numerator / denominator, curvature)


def mobius_matvec(
    matrix: torch.Tensor, point: torch.Tensor, curvature: float
) -> torch.Tensor:
    tangent = logmap0(point, curvature)
    transformed = torch.nn.functional.linear(tangent, matrix)
    return expmap0(transformed, curvature)


def distance(left: torch.Tensor, right: torch.Tensor, curvature: float) -> torch.Tensor:
    difference = mobius_add(-left, right, curvature)
    norm = difference.norm(dim=-1).clamp_min(1e-15)
    return 2.0 * artanh(math.sqrt(curvature) * norm) / math.sqrt(curvature)


def pairwise_distance(
    queries: torch.Tensor, candidates: torch.Tensor, curvature: float
) -> torch.Tensor:
    return distance(queries[:, None, :], candidates[None, :, :], curvature)


def hyperbolic_concat(
    left: torch.Tensor, right: torch.Tensor, curvature: float
) -> torch.Tensor:
    tangent = torch.cat([logmap0(left, curvature), logmap0(right, curvature)], dim=-1)
    return expmap0(tangent, curvature)
