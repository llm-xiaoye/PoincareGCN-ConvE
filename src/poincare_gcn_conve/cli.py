"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import infer_schema, read_triples, split_triples, write_triples
from .runtime import evaluate_checkpoint, predict_links, train


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate TSV knowledge graph triples.")
    parser.add_argument("data")
    args = parser.parse_args()
    triples = read_triples(args.data)
    entities, relations = infer_schema(triples)
    _print_json(
        {"triples": len(triples), "entities": len(entities), "relations": len(relations)}
    )


def split_main() -> None:
    parser = argparse.ArgumentParser(description="Split triples with train-schema coverage.")
    parser.add_argument("data")
    parser.add_argument("output_dir")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_data, dev_data, test_data = split_triples(
        read_triples(args.data),
        args.train_ratio,
        args.dev_ratio,
        args.test_ratio,
        args.seed,
    )
    output = Path(args.output_dir)
    write_triples(output / "train.tsv", train_data)
    write_triples(output / "dev.tsv", dev_data)
    write_triples(output / "test.tsv", test_data)
    _print_json({"train": len(train_data), "dev": len(dev_data), "test": len(test_data)})


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train PoincareGCN-ConvE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    _print_json(
        train(
            load_config(args.config),
            args.train,
            args.dev,
            args.test,
            args.output,
            args.device,
        )
    )


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Run filtered link-prediction evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--filter-data", action="append", default=[])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    _print_json(
        evaluate_checkpoint(
            args.checkpoint, args.data, args.filter_data, args.device
        )
    )


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate tail entities.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    _print_json(
        predict_links(
            args.checkpoint, args.head, args.relation, args.top_k, args.device
        )
    )
