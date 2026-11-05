import torch

from poincare_gcn_conve.geometry import (
    distance,
    expmap0,
    logmap0,
    mobius_add,
    pairwise_distance,
    project,
)


def test_exp_log_roundtrip_and_ball_projection():
    tangent = torch.tensor([[0.1, -0.2], [0.3, 0.2]])
    point = expmap0(tangent, 1.0)
    assert torch.all(point.norm(dim=-1) < 1.0)
    assert torch.allclose(logmap0(point, 1.0), tangent, atol=1e-5)
    projected = project(torch.tensor([[3.0, 4.0]]), 1.0)
    assert float(projected.norm()) < 1.0


def test_mobius_identity_and_distance():
    point = expmap0(torch.tensor([[0.2, 0.1]]), 1.0)
    zero = torch.zeros_like(point)
    assert torch.allclose(mobius_add(point, zero, 1.0), point, atol=1e-6)
    assert torch.allclose(distance(point, point, 1.0), torch.zeros(1), atol=1e-6)
    distances = pairwise_distance(point, torch.cat([point, zero]), 1.0)
    assert distances.shape == (1, 2)
    assert distances[0, 0] < distances[0, 1]
