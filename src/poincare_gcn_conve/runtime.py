"""Training, filtered evaluation, checkpointing, and link prediction workflows."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .data import (
    EncodedTriple,
    build_message_graph,
    build_queries,
    create_query_loader,
    encode_triples,
    infer_schema,
    read_triples,
)
from .metrics import build_filter_map, summarize_ranks
from .model import PoincareGCNConvE


def save_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available.")
    return device


def _build_model(
    config: dict[str, Any], entity_to_id: dict[str, int], relation_to_id: dict[str, int]
) -> PoincareGCNConvE:
    return PoincareGCNConvE(
        number_of_entities=len(entity_to_id),
        number_of_relations=len(relation_to_id),
        **config["model"],
    )


def _move_graph(
    edge_index: torch.Tensor, edge_type: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return edge_index.to(device), edge_type.to(device)


def evaluate(
    model: PoincareGCNConvE,
    triples: list[EncodedTriple],
    all_true_triples: list[EncodedTriple],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    hits: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    filters = build_filter_map(all_true_triples, model.number_of_relations)
    queries: list[tuple[int, int, int]] = []
    for triple in triples:
        queries.append((triple.head, triple.relation, triple.tail))
        queries.append(
            (triple.tail, triple.relation + model.number_of_relations, triple.head)
        )
    ranks: list[int] = []
    with torch.inference_mode():
        entity_embeddings, relation_embeddings = model.encode_graph(edge_index, edge_type)
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            heads = torch.tensor([row[0] for row in batch], device=device)
            relations = torch.tensor([row[1] for row in batch], device=device)
            query_embeddings = model.query_embeddings(
                heads, relations, entity_embeddings, relation_embeddings
            )
            from .geometry import pairwise_distance

            scores = -pairwise_distance(
                query_embeddings, entity_embeddings, model.curvature
            ) + model.entity_bias
            for row_index, (head, relation, target) in enumerate(batch):
                target_score = float(scores[row_index, target])
                for filtered in filters[(head, relation)]:
                    if filtered != target:
                        scores[row_index, filtered] = -torch.inf
                rank = int((scores[row_index] > target_score).sum().item()) + 1
                ranks.append(rank)
    return summarize_ranks(ranks, hits)


def _save_checkpoint(
    artifact_dir: Path,
    model: PoincareGCNConvE,
    config: dict[str, Any],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    train_triples: list[EncodedTriple],
    best_epoch: int,
    best_dev_metrics: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    package = {
        "model_state": model.state_dict(),
        "config": config,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "train_triples": [
            [triple.head, triple.relation, triple.tail] for triple in train_triples
        ],
        "best_epoch": best_epoch,
        "best_dev_metrics": best_dev_metrics,
        "format_version": 1,
    }
    torch.save(package, artifact_dir / "model.pt")
    save_json(artifact_dir / "config.json", config)
    save_json(artifact_dir / "entity_to_id.json", entity_to_id)
    save_json(artifact_dir / "relation_to_id.json", relation_to_id)
    save_json(artifact_dir / "best_dev_metrics.json", best_dev_metrics)


def load_checkpoint(
    checkpoint: str | Path, device: torch.device
) -> tuple[PoincareGCNConvE, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    package = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {
        "model_state",
        "config",
        "entity_to_id",
        "relation_to_id",
        "train_triples",
    }
    missing = required - set(package)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    model = _build_model(
        package["config"], package["entity_to_id"], package["relation_to_id"]
    )
    model.load_state_dict(package["model_state"])
    model.to(device)
    model.eval()
    return model, package


def train(
    config: dict[str, Any],
    train_file: str | Path,
    dev_file: str | Path,
    test_file: str | Path | None,
    output_dir: str | Path,
    requested_device: str = "auto",
) -> dict[str, Any]:
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(requested_device)
    train_raw = read_triples(train_file)
    dev_raw = read_triples(dev_file)
    test_raw = read_triples(test_file) if test_file else []
    entity_to_id, relation_to_id = infer_schema(train_raw)
    train_triples = encode_triples(train_raw, entity_to_id, relation_to_id, "train")
    dev_triples = encode_triples(dev_raw, entity_to_id, relation_to_id, "dev")
    test_triples = encode_triples(test_raw, entity_to_id, relation_to_id, "test")
    edge_index, edge_type = build_message_graph(train_triples, len(relation_to_id))
    edge_index, edge_type = _move_graph(edge_index, edge_type, device)
    queries = build_queries(train_triples, len(relation_to_id))
    training = config["training"]
    loader = create_query_loader(
        queries,
        len(entity_to_id),
        int(training["batch_size"]),
        float(training["label_smoothing"]),
        True,
        int(training["num_workers"]),
        seed,
    )
    model = _build_model(config, entity_to_id, relation_to_id).to(device)
    optimizer_name = training.get("optimizer", "riemannian_adam")
    if optimizer_name == "riemannian_adam":
        import geoopt

        optimizer_class = geoopt.optim.RiemannianAdam
    elif optimizer_name == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError("training.optimizer must be 'riemannian_adam' or 'adamw'.")
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    hits = [int(value) for value in config["evaluation"]["hits"]]
    evaluation_batch_size = int(config["evaluation"]["batch_size"])
    all_true = [*train_triples, *dev_triples, *test_triples]
    best_mrr = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, Any]] = []
    artifact_dir = Path(output_dir)

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for batch in progress:
            heads = batch["heads"].to(device)
            relations = batch["relations"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(heads, relations, labels, edge_index, edge_type)
            loss.backward()
            if float(training["gradient_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        dev_metrics = evaluate(
            model,
            dev_triples,
            all_true,
            edge_index,
            edge_type,
            hits,
            evaluation_batch_size,
            device,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(loader)),
            "dev_metrics": dev_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} loss={epoch_record['train_loss']:.4f} "
            f"dev_mrr={float(dev_metrics['mrr']):.4f}"
        )
        if float(dev_metrics["mrr"]) > best_mrr + float(training["min_delta"]):
            best_mrr = float(dev_metrics["mrr"])
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(
                artifact_dir,
                model,
                config,
                entity_to_id,
                relation_to_id,
                train_triples,
                best_epoch,
                dev_metrics,
            )
        else:
            patience_counter += 1
            if patience_counter >= int(training["patience"]):
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    save_json(artifact_dir / "training_history.json", history)
    best_model, package = load_checkpoint(artifact_dir, device)
    result: dict[str, Any] = {
        "device": str(device),
        "best_epoch": package["best_epoch"],
        "dev_metrics": package["best_dev_metrics"],
    }
    if test_triples:
        result["test_metrics"] = evaluate(
            best_model,
            test_triples,
            all_true,
            edge_index,
            edge_type,
            hits,
            evaluation_batch_size,
            device,
        )
    save_json(artifact_dir / "metrics.json", result)
    return result


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_file: str | Path,
    filter_files: list[str] | None = None,
    requested_device: str = "auto",
) -> dict[str, Any]:
    device = resolve_device(requested_device)
    model, package = load_checkpoint(checkpoint, device)
    entity_to_id = package["entity_to_id"]
    relation_to_id = package["relation_to_id"]
    test_triples = encode_triples(
        read_triples(data_file), entity_to_id, relation_to_id, "evaluation"
    )
    train_triples = [EncodedTriple(*values) for values in package["train_triples"]]
    all_true = [*train_triples, *test_triples]
    for path in filter_files or []:
        all_true.extend(
            encode_triples(read_triples(path), entity_to_id, relation_to_id, str(path))
        )
    edge_index, edge_type = build_message_graph(train_triples, len(relation_to_id))
    edge_index, edge_type = _move_graph(edge_index, edge_type, device)
    return evaluate(
        model,
        test_triples,
        all_true,
        edge_index,
        edge_type,
        [int(value) for value in package["config"]["evaluation"]["hits"]],
        int(package["config"]["evaluation"]["batch_size"]),
        device,
    )


def predict_links(
    checkpoint: str | Path,
    head: str,
    relation: str,
    top_k: int,
    requested_device: str = "auto",
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    device = resolve_device(requested_device)
    model, package = load_checkpoint(checkpoint, device)
    entity_to_id = package["entity_to_id"]
    relation_to_id = package["relation_to_id"]
    if head not in entity_to_id:
        raise ValueError(f"Unknown head entity: {head!r}.")
    if relation not in relation_to_id:
        raise ValueError(f"Unknown relation: {relation!r}.")
    train_triples = [EncodedTriple(*values) for values in package["train_triples"]]
    edge_index, edge_type = build_message_graph(train_triples, len(relation_to_id))
    edge_index, edge_type = _move_graph(edge_index, edge_type, device)
    with torch.inference_mode():
        scores = model.score_all(
            torch.tensor([entity_to_id[head]], device=device),
            torch.tensor([relation_to_id[relation]], device=device),
            edge_index,
            edge_type,
        )[0]
    id_to_entity = {identifier: entity for entity, identifier in entity_to_id.items()}
    count = min(top_k, len(entity_to_id))
    values, indices = scores.topk(count)
    return {
        "head": head,
        "relation": relation,
        "predictions": [
            {"tail": id_to_entity[int(index)], "score": float(value)}
            for value, index in zip(values.cpu(), indices.cpu(), strict=True)
        ],
    }
