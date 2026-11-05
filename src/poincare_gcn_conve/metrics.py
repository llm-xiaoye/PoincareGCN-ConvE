"""Filtered link-prediction ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from .data import EncodedTriple


def build_filter_map(
    triples: Iterable[EncodedTriple], number_of_relations: int
) -> dict[tuple[int, int], set[int]]:
    filters: dict[tuple[int, int], set[int]] = defaultdict(set)
    for triple in triples:
        filters[(triple.head, triple.relation)].add(triple.tail)
        filters[(triple.tail, triple.relation + number_of_relations)].add(triple.head)
    return filters


def summarize_ranks(ranks: Sequence[int], hits: Sequence[int]) -> dict[str, float | int]:
    if not ranks:
        raise ValueError("Cannot compute ranking metrics without ranks.")
    metrics: dict[str, float | int] = {
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "count": len(ranks),
    }
    for value in hits:
        metrics[f"hits@{value}"] = sum(rank <= value for rank in ranks) / len(ranks)
    return metrics
