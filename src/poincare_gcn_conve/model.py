"""PoincareGCN-ConvE for hyperbolic knowledge graph link prediction."""

from __future__ import annotations

import geoopt
import torch
from torch import nn

from .geometry import expmap0, hyperbolic_concat, logmap0, pairwise_distance, project
from .layers import (
    HyperbolicBatchNorm1d,
    HyperbolicConv2d,
    HyperbolicDropout,
    HyperbolicLinear,
    MobiusReLU,
)


class PoincareCompGCNLayer(nn.Module):
    """Relation-aware graph convolution performed through the origin tangent space."""

    def __init__(self, input_size: int, output_size: int, curvature: float, dropout: float) -> None:
        super().__init__()
        self.curvature = curvature
        self.message = nn.Linear(input_size, output_size, bias=False)
        self.self_loop = nn.Linear(input_size, output_size, bias=False)
        self.relation_update = nn.Linear(input_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        entities: torch.Tensor,
        relations: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entity_tangent = logmap0(entities, self.curvature)
        relation_tangent = logmap0(relations, self.curvature)
        sources, targets = edge_index
        composed = entity_tangent[sources] * relation_tangent[edge_type]
        messages = self.message(composed)
        aggregated = torch.zeros(
            entities.shape[0], messages.shape[-1], dtype=messages.dtype, device=messages.device
        )
        aggregated.index_add_(0, targets, messages)
        degrees = torch.zeros(
            entities.shape[0], dtype=messages.dtype, device=messages.device
        )
        degrees.index_add_(0, targets, torch.ones_like(targets, dtype=messages.dtype))
        aggregated = aggregated / degrees.clamp_min(1.0).unsqueeze(-1)
        updated_entities = self.activation(self.self_loop(entity_tangent) + aggregated)
        updated_entities = self.dropout(updated_entities)
        updated_relations = self.relation_update(relation_tangent)
        return (
            expmap0(updated_entities, self.curvature),
            expmap0(updated_relations, self.curvature),
        )


class PoincareGCNConvE(nn.Module):
    """Hyperbolic CompGCN encoder with a ConvE-style hyperbolic decoder."""

    def __init__(
        self,
        number_of_entities: int,
        number_of_relations: int,
        embedding_dim: int = 100,
        gcn_dim: int = 100,
        num_filters: int = 250,
        kernel_size: int = 6,
        reshape_height: int = 10,
        reshape_width: int = 20,
        curvature: float = 1.0,
        graph_dropout: float = 0.1,
        hidden_dropout: float = 0.2,
        feature_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if reshape_height * reshape_width != 2 * gcn_dim:
            raise ValueError("ConvE grid area must equal 2 * gcn_dim.")
        self.number_of_entities = number_of_entities
        self.number_of_relations = number_of_relations
        self.curvature = curvature
        self.manifold = geoopt.PoincareBall(c=curvature)
        self.entity_embeddings = geoopt.ManifoldParameter(
            self.manifold.random_normal((number_of_entities, embedding_dim), std=1e-2),
            manifold=self.manifold,
        )
        self.relation_embeddings = geoopt.ManifoldParameter(
            self.manifold.random_normal((2 * number_of_relations, embedding_dim), std=1e-2),
            manifold=self.manifold,
        )

        self.gcn = PoincareCompGCNLayer(
            embedding_dim, gcn_dim, curvature, graph_dropout
        )
        self.graph_dropout = HyperbolicDropout(graph_dropout, curvature)
        self.graph_norm = HyperbolicBatchNorm1d(gcn_dim, curvature)
        self.convolution = HyperbolicConv2d(
            1,
            num_filters,
            kernel_size,
            reshape_height,
            reshape_width,
            curvature,
        )
        self.feature_dropout = HyperbolicDropout(feature_dropout, curvature)
        self.convolution_norm = HyperbolicBatchNorm1d(
            self.convolution.output_size, curvature
        )
        self.activation = MobiusReLU(curvature)
        self.hidden = HyperbolicLinear(self.convolution.output_size, gcn_dim, curvature)
        self.hidden_dropout = HyperbolicDropout(hidden_dropout, curvature)
        self.hidden_norm = HyperbolicBatchNorm1d(gcn_dim, curvature)
        self.entity_bias = nn.Parameter(torch.zeros(number_of_entities))

    def initial_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            project(self.entity_embeddings, self.curvature),
            project(self.relation_embeddings, self.curvature),
        )

    def encode_graph(
        self, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entities, relations = self.initial_embeddings()
        entities, relations = self.gcn(entities, relations, edge_index, edge_type)
        entities = self.graph_norm(self.graph_dropout(entities))
        return entities, relations

    def query_embeddings(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        entity_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        combined = hyperbolic_concat(
            entity_embeddings[heads], relation_embeddings[relations], self.curvature
        )
        convolved = self.convolution(combined)
        convolved = self.feature_dropout(convolved)
        convolved = self.convolution_norm(convolved)
        convolved = self.activation(convolved)
        hidden = self.hidden(convolved)
        hidden = self.hidden_dropout(hidden)
        hidden = self.hidden_norm(hidden)
        return self.activation(hidden)

    def score_all(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        entity_embeddings, relation_embeddings = self.encode_graph(edge_index, edge_type)
        queries = self.query_embeddings(
            heads, relations, entity_embeddings, relation_embeddings
        )
        distances = pairwise_distance(queries, entity_embeddings, self.curvature)
        return -distances + self.entity_bias

    def loss(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.score_all(heads, relations, edge_index, edge_type)
        return nn.functional.binary_cross_entropy_with_logits(scores, labels)
