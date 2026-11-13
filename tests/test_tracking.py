"""Tests for MLflow helpers that do not need a workspace."""

from pathlib import Path

from geo_embeddings.tracking import file_sha256, flatten_metrics, metric_key


def test_metric_key() -> None:
    assert metric_key("Corn+Sorghum") == "corn_sorghum"
    assert metric_key("Fallow/Idle") == "fallow_idle"
    assert metric_key("Small grains") == "small_grains"


def test_flatten_metrics() -> None:
    results = {
        "class_names": {"0": "Shrubland", "1": "Corn+Sorghum"},
        "pooled": {"overall_accuracy": 0.9, "kappa": 0.8, "macro_f1": 0.85},
        "block_bootstrap": {
            "overall_accuracy": {"value": 0.9, "ci": [0.88, 0.92]},
            "kappa": {"value": 0.8, "ci": [0.77, 0.83]},
            "f1_per_class": {"0": {"value": 0.95, "ci": [0.9, 0.97], "n_blocks": 100},
                             "1": {"value": 0.7, "ci": [0.6, 0.8], "n_blocks": 12}},
        },
    }
    m = flatten_metrics(results)
    assert m["overall_accuracy"] == 0.9 and m["kappa_ci_hi"] == 0.83
    assert m["f1_corn_sorghum"] == 0.7 and m["f1_corn_sorghum_ci_lo"] == 0.6
    assert m["blocks_shrubland"] == 100.0


def test_file_sha256(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"abc")
    assert file_sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
