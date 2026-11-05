"""Knowledge graph triple parsing, schema validation, splitting, and query batching."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str


@dataclass(frozen=True)
class EncodedTriple:
    head: int
    relation: int
    tail: int


def read_triples(path: str | Path) -> list[Triple]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Triple file not found: {source}")
    triples: list[Triple] = []
    seen: set[Triple] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 3:
                raise ValueError(
                    f"{source}:{line_number}: expected HEAD<TAB>RELATION<TAB>TAIL."
                )
            if any(not value for value in columns):
                raise ValueError(f"{source}:{line_number}: triple fields must be non-empty.")
            triple = Triple(*columns)
            if triple in seen:
                raise ValueError(f"{source}:{line_number}: duplicate triple {triple}.")
            seen.add(triple)
            triples.append(triple)
    if not triples:
        raise ValueError(f"Triple file contains no triples: {source}")
    return triples


def infer_schema(triples: Sequence[Triple]) -> tuple[dict[str, int], dict[str, int]]:
    entities = sorted({entity for triple in triples for entity in (triple.head, triple.tail)})
    relations = sorted({triple.relation for triple in triples})
    return (
        {entity: index for index, entity in enumerate(entities)},
        {relation: index for index, relation in enumerate(relations)},
    )


def encode_triples(
    triples: Sequence[Triple],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    source: str,
) -> list[EncodedTriple]:
    encoded: list[EncodedTriple] = []
    for index, triple in enumerate(triples):
        unknown_entities = [
            entity for entity in (triple.head, triple.tail) if entity not in entity_to_id
        ]
        if unknown_entities:
            raise ValueError(
                f"{source}: triple {index} contains entities absent from training: "
                f"{unknown_entities}."
            )
        if triple.relation not in relation_to_id:
            raise ValueError(
                f"{source}: triple {index} contains relation absent from training: "
                f"{triple.relation!r}."
            )
        encoded.append(
            EncodedTriple(
                entity_to_id[triple.head],
                relation_to_id[triple.relation],
                entity_to_id[triple.tail],
            )
        )
    return encoded


def build_message_graph(
    triples: Sequence[EncodedTriple], number_of_relations: int
) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[int] = []
    targets: list[int] = []
    relations: list[int] = []
    for triple in triples:
        sources.extend([triple.head, triple.tail])
        targets.extend([triple.tail, triple.head])
        relations.extend([triple.relation, triple.relation + number_of_relations])
    return (
        torch.tensor([sources, targets], dtype=torch.long),
        torch.tensor(relations, dtype=torch.long),
    )


def build_queries(
    triples: Sequence[EncodedTriple], number_of_relations: int
) -> list[tuple[int, int, tuple[int, ...]]]:
    targets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for triple in triples:
        targets[(triple.head, triple.relation)].add(triple.tail)
        targets[(triple.tail, triple.relation + number_of_relations)].add(triple.head)
    return [
        (head, relation, tuple(sorted(tails)))
        for (head, relation), tails in sorted(targets.items())
    ]


class QueryDataset(Dataset[tuple[int, int, tuple[int, ...]]]):
    def __init__(self, queries: Sequence[tuple[int, int, tuple[int, ...]]]) -> None:
        self.queries = list(queries)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> tuple[int, int, tuple[int, ...]]:
        return self.queries[index]


def make_query_collate(number_of_entities: int, label_smoothing: float):
    def collate(batch: Sequence[tuple[int, int, tuple[int, ...]]]) -> dict[str, torch.Tensor]:
        heads = torch.tensor([row[0] for row in batch], dtype=torch.long)
        relations = torch.tensor([row[1] for row in batch], dtype=torch.long)
        labels = torch.zeros((len(batch), number_of_entities), dtype=torch.float)
        for row_index, (_, _, targets) in enumerate(batch):
            labels[row_index, list(targets)] = 1.0
        if label_smoothing:
            labels = labels * (1.0 - label_smoothing) + label_smoothing / number_of_entities
        return {"heads": heads, "relations": relations, "labels": labels}

    return collate


def create_query_loader(
    queries: Sequence[tuple[int, int, tuple[int, ...]]],
    number_of_entities: int,
    batch_size: int,
    label_smoothing: float,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        QueryDataset(queries),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_query_collate(number_of_entities, label_smoothing),
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def write_triples(path: str | Path, triples: Sequence[Triple]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for triple in triples:
            stream.write(f"{triple.head}\t{triple.relation}\t{triple.tail}\n")


def split_triples(
    triples: Sequence[Triple],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Triple], list[Triple], list[Triple]]:
    if any(ratio <= 0 for ratio in (train_ratio, dev_ratio, test_ratio)):
        raise ValueError("All split ratios must be positive.")
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")
    if len(triples) < 3:
        raise ValueError("At least three triples are required for splitting.")
    shuffled = list(triples)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * train_ratio))
    dev_count = max(1, int(len(shuffled) * dev_ratio))
    if train_count + dev_count >= len(shuffled):
        train_count = len(shuffled) - 2
        dev_count = 1
    train = shuffled[:train_count]
    dev = shuffled[train_count : train_count + dev_count]
    test = shuffled[train_count + dev_count :]

    train_entities = {entity for triple in train for entity in (triple.head, triple.tail)}
    train_relations = {triple.relation for triple in train}
    kept_dev: list[Triple] = []
    kept_test: list[Triple] = []
    for source, destination in ((dev, kept_dev), (test, kept_test)):
        for triple in source:
            if (
                triple.head not in train_entities
                or triple.tail not in train_entities
                or triple.relation not in train_relations
            ):
                train.append(triple)
                train_entities.update((triple.head, triple.tail))
                train_relations.add(triple.relation)
            else:
                destination.append(triple)
    if not kept_dev or not kept_test:
        raise ValueError(
            "Coverage-preserving split produced an empty dev/test set. Add more triples or "
            "provide a manually curated split."
        )
    return train, kept_dev, kept_test
