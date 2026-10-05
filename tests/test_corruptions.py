"""Determinism and config-driven corruption tests (Telman3000)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.corruptions import (
    BLUR_RADII,
    JPEG_QUALITIES,
    NOISE_STDS,
    apply_corruption,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg,
    corruption_specs,
    image_sha256,
    jpeg_encoder_options,
    noise_seed_u64,
)
from src.data import Cifar10SplitDataset, load_config, load_manifest, prepare_all


@pytest.fixture(scope="module")
def ensure_manifest():
    prepare_all(write_panel=False)
    return load_manifest(verify=True)


def _clean_sample(ensure_manifest):
    ds = Cifar10SplitDataset("model_validation", corruption="clean")
    return ds[0]


def test_ten_specs_default():
    specs = corruption_specs()
    assert len(specs) == 10
    assert specs[0]["corruption"] == "clean"
    noise = [s for s in specs if s["corruption"] == "gaussian_noise"]
    assert [s["param_value"] for s in noise] == list(NOISE_STDS)


def test_corruption_specs_follow_yaml_overrides():
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["corruptions"]["families"]["gaussian_noise"]["std_in_01"] = [0.01, 0.02, 0.03]
    cfg["corruptions"]["families"]["jpeg"]["qualities"] = [90, 50, 10]
    cfg["corruptions"]["families"]["jpeg"]["subsampling"] = 0
    cfg["corruptions"]["families"]["jpeg"]["optimize"] = True
    cfg["corruptions"]["families"]["jpeg"]["progressive"] = True

    specs = corruption_specs(cfg)
    noise = [s for s in specs if s["corruption"] == "gaussian_noise"]
    jpeg = [s for s in specs if s["corruption"] == "jpeg"]
    assert [s["param_value"] for s in noise] == [0.01, 0.02, 0.03]
    assert [s["param_value"] for s in jpeg] == [90, 50, 10]
    assert all(s["jpeg_optimize"] is True for s in jpeg)
    assert all(s["jpeg_progressive"] is True for s in jpeg)

    # Defaults unchanged when no config is passed.
    default_noise = [
        s["param_value"] for s in corruption_specs() if s["corruption"] == "gaussian_noise"
    ]
    assert default_noise == list(NOISE_STDS)


def test_jpeg_encoder_options_from_config():
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["corruptions"]["families"]["jpeg"]["optimize"] = True
    opts = jpeg_encoder_options(cfg)
    assert opts["optimize"] is True
    assert opts["subsampling"] == 0


def test_noise_seed_stable():
    a = noise_seed_u64(42, "final_test", "test:7", "gaussian_noise", "mild")
    b = noise_seed_u64(42, "final_test", "test:7", "gaussian_noise", "mild")
    c = noise_seed_u64(42, "final_test", "test:8", "gaussian_noise", "mild")
    assert a == b
    assert a != c


def test_corruptions_deterministic(ensure_manifest):
    sample = _clean_sample(ensure_manifest)
    kwargs = dict(
        protocol_seed=42,
        split=sample.split,
        source_id=sample.source_id,
    )
    for fn, args in (
        (apply_gaussian_blur, (sample.image, 1.0)),
        (apply_gaussian_noise, (sample.image, 0.06)),
        (apply_jpeg, (sample.image, 40)),
    ):
        if fn is apply_gaussian_noise:
            a = fn(*args, severity="medium", **kwargs)
            b = fn(*args, severity="medium", **kwargs)
        else:
            a = fn(*args)
            b = fn(*args)
        assert np.array_equal(a, b)
        assert a.dtype == np.uint8
        assert a.shape == (32, 32, 3)


def test_dataset_retains_id_and_label(ensure_manifest):
    clean = Cifar10SplitDataset("model_validation", corruption="clean")[0]
    noisy = Cifar10SplitDataset(
        "model_validation",
        corruption="gaussian_noise",
        severity="medium",
        param_value=0.06,
    ).get_by_source_id(clean.source_id)
    assert noisy.source_id == clean.source_id
    assert noisy.label == clean.label
    assert noisy.split == clean.split
    assert not np.array_equal(noisy.image, clean.image)


def test_corrupted_test_only_from_test_split(ensure_manifest):
    ds = Cifar10SplitDataset(
        "final_test",
        corruption="jpeg",
        severity="severe",
        param_value=15,
    )
    sample = ds[0]
    assert sample.source_id.startswith("test:")
    assert sample.split == "final_test"


def test_image_hash_changes_with_corruption(ensure_manifest):
    sample = _clean_sample(ensure_manifest)
    clean_h = image_sha256(sample.image)
    blur_h = image_sha256(apply_gaussian_blur(sample.image, 1.5))
    assert clean_h != blur_h


def test_default_protocol_levels_exposed():
    assert BLUR_RADII == (0.5, 1.0, 1.5)
    assert NOISE_STDS == (0.03, 0.06, 0.12)
    assert JPEG_QUALITIES == (70, 40, 15)
