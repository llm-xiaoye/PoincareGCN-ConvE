from copy import deepcopy

from poincare_gcn_conve.config import load_config
from poincare_gcn_conve.runtime import evaluate_checkpoint, predict_links, train


def test_offline_training_checkpoint_roundtrip(tmp_path):
    config = deepcopy(load_config("configs/offline_smoke.yaml"))
    config["training"]["epochs"] = 1
    artifact = tmp_path / "artifact"
    result = train(
        config,
        "examples/data/train.tsv",
        "examples/data/dev.tsv",
        "examples/data/test.tsv",
        artifact,
        "cpu",
    )
    assert (artifact / "model.pt").is_file()
    assert "test_metrics" in result
    metrics = evaluate_checkpoint(
        artifact,
        "examples/data/test.tsv",
        ["examples/data/dev.tsv"],
        "cpu",
    )
    assert "mrr" in metrics
    prediction = predict_links(artifact, "overheat", "caused_by", 3, "cpu")
    assert len(prediction["predictions"]) == 3
