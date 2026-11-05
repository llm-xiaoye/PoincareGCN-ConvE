"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"seed", "data", "model", "training", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    model = config["model"]
    for field in ("embedding_dim", "gcn_dim", "num_filters", "kernel_size"):
        if int(model.get(field, 0)) <= 0:
            raise ValueError(f"model.{field} must be positive.")
    if float(model.get("curvature", 0.0)) <= 0:
        raise ValueError("model.curvature must be positive.")
    if int(model["reshape_height"]) * int(model["reshape_width"]) != (
        2 * int(model["gcn_dim"])
    ):
        raise ValueError(
            "model.reshape_height * model.reshape_width must equal 2 * model.gcn_dim."
        )
    if int(model["kernel_size"]) > min(
        int(model["reshape_height"]), int(model["reshape_width"])
    ):
        raise ValueError("model.kernel_size does not fit the configured ConvE grid.")
    training = config["training"]
    for field in ("epochs", "batch_size"):
        if int(training.get(field, 0)) <= 0:
            raise ValueError(f"training.{field} must be positive.")
    smoothing = float(training.get("label_smoothing", 0.0))
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("training.label_smoothing must be in [0, 1).")
    hits = config["evaluation"].get("hits", [])
    if not hits or any(int(k) <= 0 for k in hits):
        raise ValueError("evaluation.hits must contain positive integers.")
