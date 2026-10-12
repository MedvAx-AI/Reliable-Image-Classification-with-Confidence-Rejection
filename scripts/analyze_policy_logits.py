"""Export A-D metrics, decisions, fixed-grid sensitivity and a risk-coverage plot."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import sha256_json, sha256_text_file
from src.policy_artifacts import (current_provenance, load_checked_logits, load_evaluation_config,
                                  load_frozen_policy, read_json, write_json)
from src.rejection import evaluate_policies, plot_risk_coverage, sensitivity_analysis


def analyze_logits(path: Path, output_dir: Path, *, allow_final_test: bool = False) -> dict:
    policy = load_frozen_policy()
    provenance, manifest = current_provenance()
    sidecar = read_json(path.with_suffix(".json"))
    split = sidecar["split"]
    if split == "final_test" and not allow_final_test:
        raise ValueError("final_test reporting requires --allow-final-test and a frozen policy")
    if split not in {"model_validation", "final_test"}:
        raise ValueError("report only model_validation or explicitly unlocked final_test logits")
    data = load_checked_logits(path, expected_split=split, manifest=manifest, provenance=provenance)
    cfg = load_evaluation_config()
    started = time.perf_counter()
    comparison = evaluate_policies(data["logits"], data["labels"], temperature=policy["temperature"],
                                  raw_threshold=policy["raw"]["threshold"],
                                  calibrated_threshold=policy["calibrated"]["threshold"],
                                  n_bins=cfg["metrics"]["ece_bins"], class_order=provenance["class_order"])
    sensitivity = sensitivity_analysis(data["logits"], data["labels"], data["source_ids"],
                                       temperature=policy["temperature"],
                                       matched_coverages=tuple(cfg["diagnostics"]["matched_coverages"]))
    cached_seconds = time.perf_counter() - started
    policy_hash = sha256_text_file(ROOT / "artifacts/policy.json")
    run_id = sha256_json({"input": data["provenance"], "policy": policy_hash})[:16]
    identity = {"run_id": run_id, "split": split, "corruption": sidecar["corruption"], "severity": sidecar["severity"],
                **provenance, "input": data["provenance"],
                "policy_artifact_sha256": policy_hash,
                "evaluation_config_sha256": policy["evaluation_config_sha256"],
                "final_test": split == "final_test", "cached_policy_evaluation_seconds": cached_seconds,
                "runtime_excludes": ["image_fetch", "transforms", "backbone", "head", "plotting", "file_writes"]}
    report = {**identity, "metrics": comparison["metrics"]}
    write_json(output_dir / "metrics.json", report)
    write_json(output_dir / "sensitivity.json", {**identity, **sensitivity})
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["run_id", "checkpoint_sha256", "evaluation_config_sha256", "split", "corruption", "severity",
                   "condition", "temperature", "threshold", "n", "correct", "full_set_accuracy",
                   "accepted_count", "correct_accepted_count", "coverage", "rejection_rate", "accepted_accuracy",
                   "selective_risk", "nll", "ece", "ece_bins"]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in comparison["metrics"]:
            writer.writerow({**identity, **row})
    with (output_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["run_id", "split", "source_id", "true_class", "predicted_class", "corruption", "severity", "condition",
                   "raw_confidence", "calibrated_confidence", "active_threshold", "accepted", "correct", "output"]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for condition, decision in comparison["decisions"].items():
            for i, sample_id in enumerate(data["source_ids"]):
                prediction, label = int(decision["latent_predictions"][i]), int(data["labels"][i])
                writer.writerow({"run_id": run_id, "split": split, "source_id": sample_id, "true_class": label, "predicted_class": prediction,
                                 "corruption": sidecar["corruption"], "severity": sidecar["severity"],
                                 "condition": condition, "raw_confidence": float(decision["raw_confidence"][i]),
                                 "calibrated_confidence": float(decision["calibrated_confidence"][i]),
                                 "active_threshold": decision["threshold"], "accepted": bool(decision["accepted"][i]),
                                 "correct": prediction == label, "output": decision["output"][i]})
    with (output_dir / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["confidence", "threshold", "n", "correct", "full_set_accuracy", "accepted_count",
                   "correct_accepted_count", "coverage", "rejection_rate", "accepted_accuracy", "selective_risk"]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name in ("raw", "calibrated"):
            for row in sensitivity[name]["threshold_sweep"]:
                writer.writerow({"confidence": name, **row})
    plot_risk_coverage(sensitivity, str(output_dir / "risk_coverage.png"),
                       title=f"{split}: {sidecar['corruption']}/{sidecar['severity']}, N={len(data['labels'])}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, default=ROOT / "artifacts/logits/model_validation_clean.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/policy_development")
    parser.add_argument("--allow-final-test", action="store_true", help="report final cached logits using frozen policies")
    args = parser.parse_args()
    report = analyze_logits(args.logits, args.output_dir, allow_final_test=args.allow_final_test)
    for row in report["metrics"]:
        print(f"{row['condition']}: full accuracy={row['full_set_accuracy']:.4f}, "
              f"accepted accuracy={row['accepted_accuracy']}, coverage={row['coverage']:.4f}, N={row['n']}")
