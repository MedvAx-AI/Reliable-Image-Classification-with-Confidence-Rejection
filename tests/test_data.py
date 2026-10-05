"""Integrity tests for stratified CIFAR-10 splits (Telman3000)."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    SPLIT_NAMES,
    assert_split_integrity,
    build_stratified_splits,
    load_config,
    load_manifest,
    prepare_all,
    sha256_file,
    sha256_json,
    verify_manifest_directory,
)


@pytest.fixture(scope="module")
def prepared():
    result = prepare_all(write_panel=False)
    manifest = load_manifest(verify=True)
    return result, manifest


def test_counts_and_disjointness(prepared):
    _, manifest = prepared
    assert_split_integrity(manifest)
    assert manifest["seed"] == 42
    assert sum(manifest["counts"][n]["total"] for n in SPLIT_NAMES) == 17000


def test_deterministic_manifest_ids(prepared):
    _, first = prepared
    second = build_stratified_splits(load_config())
    assert first["split_ids"] == second["split_ids"]
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_final_test_from_official_test(prepared):
    _, manifest = prepared
    for row in manifest["splits"]["final_test"]:
        assert row["partition"] == "test"
        assert row["source_id"].startswith("test:")


def test_train_splits_from_official_train(prepared):
    _, manifest = prepared
    for name in ("training", "model_validation", "calibration", "policy_selection"):
        for row in manifest["splits"][name]:
            assert row["partition"] == "train"
            assert row["source_id"].startswith("train:")


def test_class_balance(prepared):
    _, manifest = prepared
    for name in SPLIT_NAMES:
        counts = list(manifest["counts"][name]["per_class"].values())
        assert len(set(counts)) == 1


def test_verify_detects_cross_split_record_leak(prepared, tmp_path):
    """Copying a calibration record into training must fail on-disk verification."""
    src = Path(prepared[0]["manifests_dir"])
    dst = tmp_path / "leaky"
    shutil.copytree(src, dst)

    train_path = dst / "training.json"
    calib_path = dst / "calibration.json"
    with train_path.open("r", encoding="utf-8") as f:
        train = json.load(f)
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)

    leaked = copy.deepcopy(calib["records"][0])
    leaked["split"] = "training"
    # Keep class balance: replace one training record of same class.
    same_class_idx = next(
        i for i, r in enumerate(train["records"]) if r["label"] == leaked["label"]
    )
    train["records"][same_class_idx] = leaked
    train["ids"] = [r["source_id"] for r in train["records"]]
    train["sha256"] = sha256_json(train["ids"])
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)

    # Update master hash for training file so hash check is not the only failure mode;
    # leakage must still be caught from records.
    master_path = dst / "manifest.json"
    with master_path.open("r", encoding="utf-8") as f:
        master = json.load(f)
    master["split_file_hashes"]["training"] = sha256_file(train_path)
    # Keep stale master split_ids (attacker only edits the split file).
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    with pytest.raises(AssertionError):
        verify_manifest_directory(dst)


def test_verify_detects_tampered_file_hash(prepared, tmp_path):
    src = Path(prepared[0]["manifests_dir"])
    dst = tmp_path / "hashed"
    shutil.copytree(src, dst)

    master_path = dst / "manifest.json"
    with master_path.open("r", encoding="utf-8") as f:
        master = json.load(f)
    master["split_file_hashes"]["calibration"] = "0" * 64
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    with pytest.raises(AssertionError, match="hash mismatch"):
        verify_manifest_directory(dst)


def test_custom_manifest_directory(tmp_path):
    cfg = load_config()
    out = tmp_path / "custom_splits"
    panel = tmp_path / "custom_panel"
    result = prepare_all(
        config=cfg,
        out_dir=out,
        panel_dir=panel,
        write_panel=True,
    )
    assert Path(result["manifests_dir"]) == out
    assert (out / "manifest.json").exists()
    assert (out / "training.json").exists()
    assert panel.exists()

    loaded = load_manifest(manifests_dir=out, verify=True)
    assert loaded["seed"] == 42
    assert Path(loaded["_manifests_dir"]) == out
