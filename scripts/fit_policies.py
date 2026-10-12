"""Freeze Issue #3 policies from committed clean calibration/policy logits only."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.calibration import fit_temperature
from src.data import sha256_json, sha256_text_file
from src.metrics import confidence_from_logits
from src.policy_artifacts import current_provenance, load_checked_logits, load_evaluation_config, write_json
from src.rejection import select_threshold


def fit_policies(*, root: Path = ROOT, output_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    cfg = load_evaluation_config(root / "configs/evaluation.yaml")
    provenance, manifest = current_provenance(root)
    inputs = {}
    for split in ("calibration", "policy_selection"):
        inputs[split] = load_checked_logits(root / f"artifacts/logits/{split}_clean.npz",
                                            expected_split=split, manifest=manifest,
                                            provenance=provenance, require_clean=True)
    # Entire split verification runs before any fit or write; final-test logits are never opened.
    calibration, selection = inputs["calibration"], inputs["policy_selection"]
    fit_cfg = cfg["calibration"]
    fit_started = time.perf_counter()
    fit = fit_temperature(calibration["logits"], calibration["labels"], split="calibration",
                          **{key: fit_cfg[key] for key in ("log_temperature_bounds", "tolerance",
                                                         "objective_tolerance", "max_iterations")})
    fit_seconds = time.perf_counter() - fit_started
    timestamp = datetime.now(timezone.utc).isoformat()
    shared = {"schema_version": 1, "saved_utc": timestamp, **provenance,
              "evaluation_config_sha256": sha256_json(cfg), "final_test_evaluated": False}
    temperature = {**shared, **fit, "input": calibration["provenance"], "fit_seconds": fit_seconds,
                   "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                               "os": platform.platform(), "device": "cpu", "inputs": "cached logits"}}
    selection_started = time.perf_counter()
    correct = selection["logits"].argmax(axis=1) == selection["labels"]
    thresholds = {}
    for name, temp in (("raw", 1.0), ("calibrated", fit["temperature"])):
        conf = confidence_from_logits(selection["logits"], temp)
        thresholds[name] = select_threshold(conf, correct, split="policy_selection",
                                            **{key: cfg["rejection"][key] for key in (
                                                "target_accepted_accuracy", "minimum_coverage", "fallback_threshold")})
    selection_seconds = time.perf_counter() - selection_started
    folder = output_dir or root / "artifacts"
    # Both computations have succeeded before publishing either artifact.
    write_json(folder / "temperature.json", temperature)
    policy = {**shared, "temperature": fit["temperature"],
              "temperature_status": fit["status"], "raw": thresholds["raw"], "calibrated": thresholds["calibrated"],
              "input": selection["provenance"], "temperature_artifact_sha256": sha256_text_file(folder / "temperature.json"),
              "selection_seconds": selection_seconds, "total_cached_fit_seconds": time.perf_counter() - started,
              "locked_before_final_test": True, "clean_fitted_values_apply_to_all_corruptions": True}
    write_json(folder / "policy.json", policy)
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="optional separate reproduction folder")
    args = parser.parse_args()
    policy = fit_policies(output_dir=args.output_dir)
    print(f"T={policy['temperature']:.8f} ({policy['temperature_status']}); "
          f"raw tau={policy['raw']['threshold']:.2f}, calibrated tau={policy['calibrated']['threshold']:.2f}; "
          "final-test logits were not read")
