"""Correctness checks for the frozen ResNet-18 head (LeoVesinML)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import Sample
from src.model import (
    assert_split_unlocked,
    build_frozen_resnet18,
    feature_provenance,
    load_checked_feature_cache,
    load_logit_cache,
    load_model_config,
    save_array_artifact,
)
from src.predict import (
    condition_a_report,
    expected_calibration_error,
    infer_split_logits,
    negative_log_likelihood,
    temperature_preserves_argmax,
)
from src.train import fit_linear_head, logits_from_head, select_best_epoch


def test_config_uses_protocol_defaults():
    cfg = load_model_config()
    assert cfg["model"]["architecture"] == "resnet18"
    assert cfg["model"]["weights"] == "IMAGENET1K_V1"
    assert cfg["model"]["num_classes"] == 10
    assert cfg["training"]["train_split"] == "training"
    assert cfg["training"]["val_split"] == "model_validation"
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["optimizer"] == "AdamW"
    assert cfg["training"]["lr"] == 0.001
    assert cfg["training"]["max_epochs"] == 30
    assert "final_test" not in cfg["extraction"]["splits"]
    assert "final_test" not in cfg["logit_export_splits"]


def test_final_test_stays_locked():
    with pytest.raises(RuntimeError, match="final_test"):
        assert_split_unlocked("final_test")


def test_select_lowest_nll_then_earliest_epoch():
    history = [
        {"epoch": 1, "val_nll": 0.5},
        {"epoch": 2, "val_nll": 0.4},
        {"epoch": 3, "val_nll": 0.4},
        {"epoch": 4, "val_nll": 0.45},
    ]
    assert select_best_epoch(history)["epoch"] == 2


def test_fit_rejects_held_out_split():
    features = np.zeros((2, 4), dtype=np.float32)
    labels = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="training"):
        fit_linear_head(
            features,
            labels,
            features,
            labels,
            split_name="calibration",
            source_ids=["train:1", "train:2"],
            forbidden_ids=set(),
            max_epochs=1,
        )


def test_fit_rejects_overlapping_ids():
    features = np.zeros((1, 4), dtype=np.float32)
    labels = np.zeros(1, dtype=np.int64)
    with pytest.raises(ValueError, match="held-out"):
        fit_linear_head(
            features,
            labels,
            features,
            labels,
            split_name="training",
            source_ids=["train:1"],
            forbidden_ids={"train:1"},
            max_epochs=1,
        )


def test_fit_trains_a_ten_class_head_and_reloads():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 8)).astype(np.float32)
    labels = rng.integers(0, 10, size=40)
    val_features = rng.normal(size=(12, 8)).astype(np.float32)
    val_labels = rng.integers(0, 10, size=12)
    result = fit_linear_head(
        features,
        labels,
        val_features,
        val_labels,
        split_name="training",
        source_ids=[f"train:{i}" for i in range(40)],
        forbidden_ids={f"train:{i}" for i in range(100, 120)},
        seed=42,
        max_epochs=3,
        batch_size=8,
        num_classes=10,
    )
    assert result["weight"].shape == (10, 8)
    assert result["bias"].shape == (10,)
    assert result["selected_epoch"] in {1, 2, 3}
    assert result["train_split"] == "training"
    logits = logits_from_head(val_features, result["weight"], result["bias"])
    assert logits.shape == (12, 10)
    assert np.isfinite(logits).all()


def test_nll_of_uniform_logits():
    logits = np.zeros((1, 10), dtype=np.float64)
    nll = negative_log_likelihood(logits, np.array([3]))
    assert nll == pytest.approx(-np.log(0.1), abs=1e-8)


def test_ece_bins_and_confidence_one():
    uncertain = expected_calibration_error(np.array([0.5]), np.array([False]))
    assert uncertain["ece"] == pytest.approx(0.5)
    assert uncertain["bins"][7]["count"] == 1

    certain = expected_calibration_error(np.array([1.0]), np.array([True]))
    assert certain["ece"] == pytest.approx(0.0)
    assert certain["bins"][-1]["count"] == 1
    assert certain["bins"][0]["count"] == 0

    edge = expected_calibration_error(np.array([1.0 / 15.0]), np.array([True]))
    assert edge["bins"][0]["count"] == 0
    assert edge["bins"][1]["count"] == 1


def test_condition_a_accepts_every_image():
    logits = np.array(
        [
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1])
    report = condition_a_report(logits, labels, [f"c{i}" for i in range(10)])
    assert report["coverage"] == 1.0
    assert report["rejection_rate"] == 0.0
    assert report["accepted_count"] == 2
    assert report["full_set_accuracy"] == pytest.approx(0.5)
    assert report["accepted_accuracy"] == report["full_set_accuracy"]
    assert report["correct_accepted_count"] == 1


def test_positive_temperature_preserves_argmax():
    logits = np.array([[0.2, 3.0, -1.0, 0.0, 0.4, 0.1, -2.0, 0.3, 1.0, 0.5]])
    assert temperature_preserves_argmax(logits, 0.7)
    assert temperature_preserves_argmax(logits, 2.5)


def test_logit_artifact_roundtrip(tmp_path):
    path = tmp_path / "split_clean.npz"
    logits = np.arange(20, dtype=np.float32).reshape(2, 10)
    labels = np.array([1, 4], dtype=np.int64)
    save_array_artifact(
        path,
        arrays={
            "logits": logits,
            "labels": labels,
            "source_ids": np.asarray(["train:1", "train:2"]),
        },
        metadata={"split": "calibration"},
    )
    loaded = load_logit_cache(path)
    assert loaded["source_ids"] == ["train:1", "train:2"]
    assert np.array_equal(loaded["labels"], labels)
    assert np.allclose(loaded["logits"], logits)


def _provenance(**overrides):
    base = feature_provenance(
        split="training",
        corruption="clean",
        severity="none",
        feature_dim=4,
        batch_size=8,
        device="cpu",
        weights_enum="ResNet18_Weights.IMAGENET1K_V1",
        weights_url="https://example.invalid/resnet18.pth",
        transform={"resize_size": 256, "crop_size": 224},
        model_config_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
    )
    base.update(overrides)
    return base


def test_feature_cache_requires_hash_and_provenance(tmp_path):
    path = tmp_path / "training_clean.npz"
    provenance = _provenance()
    features = np.arange(8, dtype=np.float32).reshape(2, 4)
    labels = np.array([1, 3], dtype=np.int64)
    ids = ["train:1", "train:2"]
    save_array_artifact(
        path,
        arrays={"features": features, "labels": labels, "source_ids": np.asarray(ids)},
        metadata=dict(provenance),
    )
    cache, sidecar = load_checked_feature_cache(
        path,
        expected_ids=ids,
        expected_labels=[1, 3],
        provenance=provenance,
    )
    assert sidecar["npz_sha256"]
    assert np.array_equal(cache["features"], features)

    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["device"] = "mps"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance"):
        load_checked_feature_cache(
            path,
            expected_ids=ids,
            expected_labels=[1, 3],
            provenance=provenance,
        )

    sidecar["device"] = "cpu"
    sidecar["npz_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        load_checked_feature_cache(
            path,
            expected_ids=ids,
            expected_labels=[1, 3],
            provenance=provenance,
        )


class _TimerStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 10)

    def assert_frozen_batchnorm(self) -> None:
        return None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(images.new_zeros(images.shape[0], 1))


def test_inference_timer_includes_image_fetch(monkeypatch):
    events: list[str] = []
    real_clock = time.perf_counter

    def clock() -> float:
        events.append("clock")
        return real_clock()

    monkeypatch.setattr("src.predict.time.perf_counter", clock)
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    class Dataset:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> Sample:
            events.append("fetch")
            return Sample(
                source_id=f"train:{index}",
                split="model_validation",
                label=index,
                corruption="clean",
                severity="none",
                image=image,
            )

    result = infer_split_logits(
        _TimerStub(),
        Dataset(),
        lambda _image: torch.zeros(3, 4, 4),
        torch.device("cpu"),
        batch_size=2,
        time_it=True,
        warmup_batches=0,
    )
    assert result["timed_images"] == 2
    assert events[:3] == ["clock", "fetch", "fetch"]
    assert "in_memory_image_fetch" in result["includes"]


def test_backbone_batchnorm_does_not_drift_and_head_is_ten_way():
    model, _, provenance = build_frozen_resnet18(load_model_config(), pretrained=False)
    assert provenance["original_classifier_removed"] is True
    assert model.head.out_features == 10
    assert model.feature_dim == 512
    before = model.batchnorm_buffer_sha256()
    model.train()
    model.assert_frozen_batchnorm()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 10)
    assert model.batchnorm_buffer_sha256() == before
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
