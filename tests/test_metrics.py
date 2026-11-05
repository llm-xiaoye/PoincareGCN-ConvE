from poincare_gcn_conve.data import EncodedTriple
from poincare_gcn_conve.metrics import build_filter_map, summarize_ranks


def test_filter_map_includes_forward_and_inverse_queries():
    filters = build_filter_map([EncodedTriple(0, 1, 2)], number_of_relations=3)
    assert filters[(0, 1)] == {2}
    assert filters[(2, 4)] == {0}


def test_ranking_metrics():
    metrics = summarize_ranks([1, 2, 4], [1, 3, 5])
    assert metrics["mrr"] == (1.0 + 0.5 + 0.25) / 3
    assert metrics["hits@1"] == 1 / 3
    assert metrics["hits@3"] == 2 / 3
    assert metrics["hits@5"] == 1.0
