import torch

from poincare_gcn_conve.data import EncodedTriple, build_message_graph
from poincare_gcn_conve.layers import HyperbolicLinear, MobiusReLU
from poincare_gcn_conve.model import PoincareGCNConvE


def make_model() -> PoincareGCNConvE:
    return PoincareGCNConvE(
        number_of_entities=5,
        number_of_relations=2,
        embedding_dim=8,
        gcn_dim=8,
        num_filters=4,
        kernel_size=2,
        reshape_height=4,
        reshape_width=4,
        graph_dropout=0.0,
        hidden_dropout=0.0,
        feature_dropout=0.0,
    )


def graph():
    return build_message_graph(
        [EncodedTriple(0, 0, 1), EncodedTriple(1, 1, 2), EncodedTriple(2, 0, 3)], 2
    )


def test_hyperbolic_layers_remain_in_ball():
    inputs = torch.randn(4, 6) * 0.05
    outputs = HyperbolicLinear(6, 3, 1.0)(inputs)
    activated = MobiusReLU(1.0)(outputs)
    assert torch.all(activated.norm(dim=-1) < 1.0)


def test_model_scores_and_loss_backpropagate():
    model = make_model()
    model.eval()
    edge_index, edge_type = graph()
    scores = model.score_all(torch.tensor([0, 1]), torch.tensor([0, 1]), edge_index, edge_type)
    assert scores.shape == (2, 5)
    labels = torch.zeros_like(scores)
    labels[0, 1] = 1.0
    labels[1, 2] = 1.0
    loss = model.loss(
        torch.tensor([0, 1]), torch.tensor([0, 1]), labels, edge_index, edge_type
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.entity_embeddings.grad is not None
