"""Checked logit/policy handoff shared by calibration and later evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data import REPO_ROOT, sha256_file, sha256_json, sha256_text_file, verify_manifest_directory
from src.corruptions import SEVERITY_NAMES
from src.metrics import validate_logits


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_evaluation_config(path: Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((path or REPO_ROOT / "configs/evaluation.yaml").read_text(encoding="utf-8"))
    # Reject silent protocol drift in settings consumed by the public interfaces.
    if cfg["metrics"]["ece_bins"] != 15:
        raise ValueError("the frozen protocol requires 15 ECE bins")
    if cfg["rejection"]["threshold_grid"] != {"start": 0.0, "stop": 1.0, "step": 0.01}:
        raise ValueError("the frozen protocol requires the 0.00..1.00 threshold grid")
    if cfg["calibration"]["optimizer"] != "golden_section_log_temperature" or cfg["calibration"]["initial_log_temperature"] != 0:
        raise ValueError("unsupported optimizer or initialization")
    if cfg["rejection"]["selection_rule"] != "maximum coverage; smallest threshold breaks ties":
        raise ValueError("unsupported threshold selection rule")
    return cfg


def current_provenance(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verify_manifest_directory(root / "artifacts/splits")
    checkpoint = read_json(root / "artifacts/model/checkpoint_manifest.json")
    model_cfg = yaml.safe_load((root / "configs/model.yaml").read_text(encoding="utf-8"))
    data_cfg = yaml.safe_load((root / "configs/data.yaml").read_text(encoding="utf-8"))
    provenance = {"checkpoint_sha256": sha256_file(root / "artifacts/model/linear_head.npz"),
                  "model_config_sha256": sha256_json(model_cfg),
                  "data_config_sha256": sha256_json(data_cfg),
                  "split_manifest_sha256": manifest["manifest_sha256"],
                  "protocol_sha256": sha256_text_file(root / "docs/EXPERIMENT_PROTOCOL.md"),
                  "class_order": manifest["class_order"]}
    for field, key in (("checkpoint_sha256", "head_sha256"), ("model_config_sha256", "model_config_sha256"),
                       ("data_config_sha256", "data_config_sha256"), ("split_manifest_sha256", "split_manifest_sha256")):
        if provenance[field] != checkpoint[key]:
            raise ValueError(f"checkpoint provenance mismatch: {field}")
    return provenance, manifest


def load_checked_logits(
    path: Path, *, expected_split: str, manifest: dict[str, Any],
    provenance: dict[str, Any], require_clean: bool = False,
) -> dict[str, Any]:
    """Verify bytes, provenance and exact ordered manifest IDs/labels before use."""
    sidecar = read_json(path.with_suffix(".json"))
    if sidecar.get("split") != expected_split:
        raise ValueError("logit split does not match its declared purpose")
    if sidecar.get("final_test") != (expected_split == "final_test"):
        raise ValueError("logit final_test flag does not match its split")
    if require_clean and (sidecar.get("corruption"), sidecar.get("severity")) != ("clean", "none"):
        raise ValueError("fitting requires clean logits")
    condition = sidecar.get("corruption"), sidecar.get("severity")
    allowed = {("clean", "none")} | {(family, severity) for family in ("gaussian_blur", "gaussian_noise", "jpeg")
                                    for severity in SEVERITY_NAMES}
    if condition not in allowed:
        raise ValueError("unknown corruption/severity condition")
    for field in ("checkpoint_sha256", "model_config_sha256", "split_manifest_sha256", "class_order"):
        if sidecar.get(field) != provenance[field]:
            raise ValueError(f"logit provenance mismatch: {field}")
    actual_hash = sha256_file(path)
    if sidecar.get("npz_sha256") != actual_hash:
        raise ValueError("logit npz SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as payload:
        logits, labels = payload["logits"], payload["labels"]
        source_ids = payload["source_ids"].tolist()
    validate_logits(logits, labels)
    records = manifest["splits"][expected_split]
    if source_ids != [r["source_id"] for r in records] or labels.tolist() != [r["label"] for r in records]:
        raise ValueError("logit rows do not match ordered split IDs/labels")
    if logits.shape != (len(records), len(provenance["class_order"])) or sidecar.get("n") != len(records):
        raise ValueError("logit shape/count mismatch")
    return {"logits": logits, "labels": labels, "source_ids": source_ids, "metadata": sidecar,
            "provenance": {"npz_sha256": actual_hash, "sidecar_sha256": sha256_text_file(path.with_suffix(".json")),
                           "split": expected_split,
                           "split_file_sha256": manifest["split_file_hashes"][expected_split],
                           "split_ids_sha256": sha256_json(source_ids)}}


def load_frozen_policy(root: Path = REPO_ROOT, artifact_dir: Path | None = None) -> dict[str, Any]:
    """Fail on drift; loading a policy never refits T or thresholds."""
    folder = artifact_dir or root / "artifacts"
    policy = read_json(folder / "policy.json")
    temperature = read_json(folder / "temperature.json")
    provenance, manifest = current_provenance(root)
    for field, expected in provenance.items():
        if policy.get(field) != expected or temperature.get(field) != expected:
            raise ValueError(f"frozen policy provenance mismatch: {field}")
    config_hash = sha256_json(load_evaluation_config(root / "configs/evaluation.yaml"))
    if policy.get("evaluation_config_sha256") != config_hash or temperature.get("evaluation_config_sha256") != config_hash:
        raise ValueError("evaluation settings changed after policy freeze")
    if policy.get("temperature_artifact_sha256") != sha256_text_file(folder / "temperature.json"):
        raise ValueError("temperature artifact changed after policy freeze")
    if policy.get("temperature") != temperature.get("temperature"):
        raise ValueError("policy and temperature disagree")
    if not np.isfinite(policy["temperature"]) or policy["temperature"] <= 0:
        raise ValueError("frozen temperature must be finite and positive")
    for name in ("raw", "calibrated"):
        tau = policy[name]["threshold"]
        if not np.isfinite(tau) or not 0 <= tau <= 1 or policy[name]["selection_split"] != "policy_selection":
            raise ValueError("invalid frozen operating threshold")
    if policy.get("locked_before_final_test") is not True or policy.get("final_test_evaluated") is not False:
        raise ValueError("policy was not locked before final test")
    for split, record in (("calibration", temperature["input"]), ("policy_selection", policy["input"])):
        checked = load_checked_logits(root / f"artifacts/logits/{split}_clean.npz", expected_split=split,
                                      manifest=manifest, provenance=provenance, require_clean=True)
        if checked["provenance"] != record:
            raise ValueError(f"fitting input changed after policy freeze: {split}")
    return policy
