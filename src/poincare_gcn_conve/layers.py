"""Hyperbolic layers implemented through tangent-space operations."""

from __future__ import annotations

import torch
from torch import nn

from .geometry import expmap0, logmap0, mobius_add, mobius_matvec


class HyperbolicLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, curvature: float) -> None:
        super().__init__()
        self.curvature = curvature
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias_tangent = nn.Parameter(torch.zeros(output_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = mobius_matvec(self.weight, inputs, self.curvature)
        bias = expmap0(self.bias_tangent, self.curvature)
        return mobius_add(transformed, bias, self.curvature)


class HyperbolicDropout(nn.Module):
    def __init__(self, probability: float, curvature: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(probability)
        self.curvature = curvature

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return expmap0(self.dropout(logmap0(inputs, self.curvature)), self.curvature)


class MobiusReLU(nn.Module):
    def __init__(self, curvature: float) -> None:
        super().__init__()
        self.curvature = curvature

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return expmap0(torch.relu(logmap0(inputs, self.curvature)), self.curvature)


class HyperbolicBatchNorm1d(nn.Module):
    def __init__(self, features: int, curvature: float) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(features)
        self.curvature = curvature

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tangent = logmap0(inputs, self.curvature)
        if self.training and tangent.shape[0] == 1:
            normalized = torch.nn.functional.batch_norm(
                tangent,
                self.normalization.running_mean,
                self.normalization.running_var,
                self.normalization.weight,
                self.normalization.bias,
                training=False,
                momentum=0.0,
                eps=self.normalization.eps,
            )
        else:
            normalized = self.normalization(tangent)
        return expmap0(normalized, self.curvature)


class HyperbolicConv2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        input_height: int,
        input_width: int,
        curvature: float,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(input_channels, output_channels, kernel_size)
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_size = (
            output_channels
            * (input_height - kernel_size + 1)
            * (input_width - kernel_size + 1)
        )
        self.curvature = curvature

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = self.input_channels * self.input_height * self.input_width
        if inputs.shape[-1] != expected:
            raise ValueError(f"Expected flattened size {expected}, got {inputs.shape[-1]}.")
        tangent = logmap0(inputs, self.curvature).reshape(
            -1, self.input_channels, self.input_height, self.input_width
        )
        convolved = self.convolution(tangent).flatten(start_dim=1)
        return expmap0(convolved, self.curvature)
