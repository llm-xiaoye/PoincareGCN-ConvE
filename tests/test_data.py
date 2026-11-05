import pytest

from poincare_gcn_conve.data import (
    Triple,
    build_message_graph,
    build_queries,
    encode_triples,
    infer_schema,
    read_triples,
)


def test_read_schema_encode_and_queries(tmp_path):
    path = tmp_path / "triples.tsv"
    path.write_text("a\tr\tb\nb\tr\tc\n", encoding="utf-8")
    raw = read_triples(path)
    entities, relations = infer_schema(raw)
    encoded = encode_triples(raw, entities, relations, "test")
    edge_index, edge_type = build_message_graph(encoded, len(relations))
    assert edge_index.shape == (2, 4)
    assert edge_type.tolist() == [0, 1, 0, 1]
    queries = build_queries(encoded, len(relations))
    assert len(queries) == 4


def test_duplicate_and_unknown_schema_are_rejected(tmp_path):
    path = tmp_path / "triples.tsv"
    path.write_text("a\tr\tb\na\tr\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_triples(path)
    with pytest.raises(ValueError, match="absent from training"):
        encode_triples([Triple("a", "r", "z")], {"a": 0}, {"r": 0}, "dev")
