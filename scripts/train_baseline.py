#!/usr/bin/env python3
"""Extract features, train the ten-class head, and save the development baseline.

Owner: LeoVesinML / Issue #2. Does not evaluate final_test.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.data import (
    Cifar10SplitDataset,
    assert_split_integrity,
    download_cifar10,
    load_config,
    load_manifest,
    resolve_path,
    sha256_file,
    sha256_json,
)
from src.model import (
    assert_split_unlocked,
    build_frozen_resnet18,
    choose_device,
    config_sha256,
    load_feature_cache,
    load_head_npz,
    load_logit_cache,
    load_model_config,
    pil_from_uint8,
    save_array_artifact,
    save_head_npz,
    synchronize,
)
from src.predict import (
    condition_a_report,
    infer_split_logits,
    temperature_preserves_argmax,
    time_cached_logit_eval,
    write_json,
    write_predictions_csv,
)
from src.train import fit_linear_head, logits_from_head


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ram_bytes() -> int | None:
    try:
        if sys.platform == "darwin":
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        return int(os_sysconf())
    except Exception:
        return None


def os_sysconf() -> int:
    import os

    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def package_versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import pandas
    import PIL
    import pytest
    import torchvision
    import yaml

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "pyyaml": yaml.__version__,
        "pytest": pytest.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
    }


def environment_record(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        gpu = "Apple MPS"
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": sys.version,
        "packages": package_versions(),
        "device": str(device),
        "accelerator": gpu,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "ram_bytes": _ram_bytes(),
    }


def feature_path(features_dir: Path, split: str) -> Path:
    return features_dir / f"{split}_clean.npz"


def logit_path(logits_dir: Path, split: str) -> Path:
    return logits_dir / f"{split}_clean.npz"


def _expected_rows(manifest: dict[str, Any], split: str) -> tuple[list[str], list[int]]:
    rows = manifest["splits"][split]
    return [row["source_id"] for row in rows], [int(row["label"]) for row in rows]


def _batches(dataset, batch_size: int):
    from src.predict import iter_image_batches

    return iter_image_batches(dataset, batch_size)


def extract_or_load(
    model,
    transform,
    device: torch.device,
    *,
    split: str,
    data_config: dict[str, Any],
    manifest: dict[str, Any],
    features_dir: Path,
    batch_size: int,
    reextract: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = feature_path(features_dir, split)
    expected_ids, expected_labels = _expected_rows(manifest, split)
    if path.exists() and not reextract:
        cache = load_feature_cache(path)
        if cache["source_ids"] != expected_ids or cache["labels"].tolist() != expected_labels:
            raise RuntimeError(f"{split}: cached features do not match the manifest")
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        print(f"Loaded cached features for {split} ({len(cache['source_ids'])} rows)")
        return cache, {
            "split": split,
            "seconds": sidecar.get("extraction_seconds"),
            "cached": True,
            "n": len(cache["source_ids"]),
            "npz_sha256": sidecar.get("npz_sha256"),
        }

    print(f"Extracting clean features for {split} ...")
    assert_split_unlocked(split)
    dataset = Cifar10SplitDataset(
        split,
        corruption="clean",
        severity="none",
        config=data_config,
        manifest=manifest,
    )
    source_ids: list[str] = []
    labels: list[int] = []
    chunks: list[np.ndarray] = []
    synchronize(device)
    started = time.perf_counter()
    for batch in _batches(dataset, batch_size):
        tensors = torch.stack([transform(pil_from_uint8(sample.image)) for sample in batch]).to(device)
        with torch.no_grad():
            feats = model.backbone(tensors)
        synchronize(device)
        chunks.append(feats.detach().cpu().numpy().astype(np.float32))
        source_ids.extend(sample.source_id for sample in batch)
        labels.extend(int(sample.label) for sample in batch)
    seconds = time.perf_counter() - started
    features = np.concatenate(chunks, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)
    if source_ids != expected_ids or label_array.tolist() != expected_labels:
        raise RuntimeError(f"{split}: extracted rows do not match the manifest")
    meta = {
        "split": split,
        "corruption": "clean",
        "severity": "none",
        "n": len(source_ids),
        "feature_dim": int(features.shape[1]),
        "extraction_seconds": seconds,
        "batch_size": batch_size,
        "device": str(device),
        "includes_backbone": True,
    }
    saved = save_array_artifact(
        path,
        arrays={
            "features": features,
            "labels": label_array,
            "source_ids": np.asarray(source_ids),
        },
        metadata=meta,
    )
    print(f"  {split}: {len(source_ids)} images in {seconds:.1f}s ({saved['npz_sha256'][:12]})")
    cache = {"features": features, "labels": label_array, "source_ids": source_ids}
    return cache, {"split": split, "seconds": seconds, "cached": False, "n": len(source_ids), "npz_sha256": saved["npz_sha256"]}


def cpu_feature_gap(model, transform, dataset, device: torch.device, n: int = 32) -> dict[str, Any] | None:
    """Compare backbone features on the run device against a CPU copy of the same weights."""
    if device.type == "cpu":
        return None
    n = min(n, len(dataset))
    samples = [dataset[i] for i in range(n)]
    tensors = torch.stack([transform(pil_from_uint8(sample.image)) for sample in samples])
    cpu_backbone = copy.deepcopy(model.backbone).cpu().eval()
    with torch.no_grad():
        on_device = model.backbone(tensors.to(device)).detach().cpu().numpy()
        on_cpu = cpu_backbone(tensors).numpy()
    diff = np.abs(on_device.astype(np.float64) - on_cpu.astype(np.float64))
    return {
        "n": n,
        "run_device": str(device),
        "reference_device": "cpu",
        "max_abs_feature_diff": float(diff.max()),
        "mean_abs_feature_diff": float(diff.mean()),
    }


def held_out_ids(manifest: dict[str, Any], train_split: str) -> set[str]:
    forbidden: set[str] = set()
    for name, rows in manifest["splits"].items():
        if name == train_split:
            continue
        forbidden.update(row["source_id"] for row in rows)
    return forbidden


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    model_cfg = load_model_config(args.config)
    data_cfg = load_config(args.data_config)
    device = choose_device(args.device or model_cfg["extraction"]["device"])
    features_dir = resolve_path(model_cfg["paths"]["features_dir"])
    logits_dir = resolve_path(model_cfg["paths"]["logits_dir"])
    checkpoint_path = resolve_path(model_cfg["paths"]["checkpoint_npz"])
    manifest_path = resolve_path(model_cfg["paths"]["checkpoint_manifest"])

    print("Checking CIFAR-10 archive and split manifests ...")
    download_cifar10(resolve_path(data_cfg["dataset"]["root"]), download=True)
    manifest = load_manifest(config=data_cfg, verify=True)
    assert_split_integrity(manifest)

    print(f"Loading ResNet18_Weights.{model_cfg['model']['weights']} on {device} ...")
    model, transform, provenance = build_frozen_resnet18(model_cfg, pretrained=True)
    model.to(device)
    model.assert_frozen_batchnorm()
    bn_before = model.batchnorm_buffer_sha256()

    extraction_log = []
    caches = {}
    for split in model_cfg["extraction"]["splits"]:
        cache, info = extract_or_load(
            model,
            transform,
            device,
            split=split,
            data_config=data_cfg,
            manifest=manifest,
            features_dir=features_dir,
            batch_size=int(model_cfg["extraction"]["batch_size"]),
            reextract=args.reextract,
        )
        caches[split] = cache
        extraction_log.append(info)

    bn_after = model.batchnorm_buffer_sha256()
    if bn_before != bn_after:
        raise RuntimeError("BatchNorm buffers changed during feature extraction")

    gap_dataset = Cifar10SplitDataset(
        "model_validation",
        corruption="clean",
        severity="none",
        config=data_cfg,
        manifest=manifest,
    )
    device_gap = cpu_feature_gap(model, transform, gap_dataset, device, n=32)
    if device_gap is not None:
        print(
            "CPU vs "
            f"{device} feature gap on {device_gap['n']} images: "
            f"max {device_gap['max_abs_feature_diff']:.3e}"
        )

    train_split = model_cfg["training"]["train_split"]
    val_split = model_cfg["training"]["val_split"]
    train_cache = caches[train_split]
    val_cache = caches[val_split]
    forbidden = held_out_ids(manifest, train_split)
    print("Training the ten-class head on cached training features ...")
    train_started = time.perf_counter()
    fit = fit_linear_head(
        train_cache["features"],
        train_cache["labels"],
        val_cache["features"],
        val_cache["labels"],
        split_name=train_split,
        source_ids=train_cache["source_ids"],
        forbidden_ids=forbidden,
        seed=int(model_cfg["training"]["seed"]),
        lr=float(model_cfg["training"]["lr"]),
        weight_decay=float(model_cfg["training"]["weight_decay"]),
        batch_size=int(model_cfg["training"]["feature_batch_size"]),
        max_epochs=int(model_cfg["training"]["max_epochs"]),
        num_classes=int(model_cfg["model"]["num_classes"]),
    )
    train_seconds = time.perf_counter() - train_started
    head_sha = save_head_npz(checkpoint_path, fit["weight"], fit["bias"])
    print(
        f"Selected epoch {fit['selected_epoch']} "
        f"val NLL {fit['selected_val_nll']:.6f} head {head_sha[:12]}"
    )

    class_order = list(data_cfg["dataset"]["class_order"])
    model_hash = config_sha256(model_cfg)
    data_manifest_sha = manifest["manifest_sha256"]
    logit_files = {}
    exported_logits = {}
    for split in model_cfg["logit_export_splits"]:
        cache = caches[split]
        logits = logits_from_head(cache["features"], fit["weight"], fit["bias"])
        meta = {
            "split": split,
            "corruption": "clean",
            "severity": "none",
            "n": int(logits.shape[0]),
            "class_order": class_order,
            "checkpoint_sha256": head_sha,
            "model_config_sha256": model_hash,
            "split_manifest_sha256": data_manifest_sha,
            "source": "cached_backbone_features_times_selected_head",
            "final_test": False,
        }
        saved = save_array_artifact(
            logit_path(logits_dir, split),
            arrays={
                "logits": logits,
                "labels": cache["labels"],
                "source_ids": np.asarray(cache["source_ids"]),
            },
            metadata=meta,
        )
        logit_files[split] = saved
        exported_logits[split] = {
            "logits": logits,
            "labels": cache["labels"],
            "source_ids": cache["source_ids"],
        }
        print(f"Exported {split} logits {logits.shape} {saved['npz_sha256'][:12]}")

    # Isolation check: a positive temperature does not change labels or the head file.
    probe = exported_logits["calibration"]["logits"]
    if not temperature_preserves_argmax(probe, 1.5):
        raise RuntimeError("temperature scaling changed argmax on calibration logits")
    if sha256_file(checkpoint_path) != head_sha:
        raise RuntimeError("checkpoint bytes changed during the temperature isolation check")

    val = exported_logits["model_validation"]
    report = condition_a_report(val["logits"], val["labels"], class_order)
    saved_utc = _utc_now()
    baseline = {
        "saved_utc": saved_utc,
        "role": "development raw-confidence no-rejection baseline",
        "predates_comparisons": True,
        "final_test_evaluated": False,
        "split": "model_validation",
        "corruption": "clean",
        "severity": "none",
        "class_order": class_order,
        "checkpoint_sha256": head_sha,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "model_config_sha256": model_hash,
        "split_manifest_sha256": data_manifest_sha,
        "selected_epoch": fit["selected_epoch"],
        "selected_val_nll": fit["selected_val_nll"],
        "selection_rule": fit["selection_rule"],
        "logit_npz_sha256": logit_files["model_validation"]["npz_sha256"],
        "metrics": {k: v for k, v in report.items() if k != "per_class"},
        "per_class": report["per_class"],
    }
    write_json(resolve_path(model_cfg["paths"]["baseline_json"]), baseline)
    write_predictions_csv(
        resolve_path(model_cfg["paths"]["predictions_csv"]),
        source_ids=val["source_ids"],
        labels=val["labels"],
        logits=val["logits"],
        class_order=class_order,
    )
    print(
        f"Baseline accuracy {report['full_set_accuracy']:.4f} "
        f"NLL {report['nll']:.4f} ECE {report['ece']:.4f}"
    )

    fixture_n = 8
    fixture = {
        "description": (
            "First 8 model-validation clean logits, in manifest order. "
            "Canonical file is artifacts/logits/model_validation_clean.npz."
        ),
        "class_order": class_order,
        "checkpoint_sha256": head_sha,
        "samples": [
            {
                "source_id": val["source_ids"][i],
                "label": int(val["labels"][i]),
                "logits": [float(x) for x in val["logits"][i].tolist()],
            }
            for i in range(fixture_n)
        ],
    }
    write_json(resolve_path(model_cfg["paths"]["dev_fixture"]), fixture)

    checkpoint_manifest = {
        "saved_utc": saved_utc,
        "provenance": provenance,
        "head_sha256": head_sha,
        "head_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "model_config_sha256": model_hash,
        "data_config_sha256": sha256_json(data_cfg),
        "split_manifest_sha256": data_manifest_sha,
        "seed": int(model_cfg["training"]["seed"]),
        "selected_epoch": fit["selected_epoch"],
        "selected_val_nll": fit["selected_val_nll"],
        "selection_rule": fit["selection_rule"],
        "train_split": train_split,
        "val_split": val_split,
        "n_train": fit["n_train"],
        "n_val": fit["n_val"],
        "held_out_splits_excluded_from_weight_updates": [
            "model_validation",
            "calibration",
            "policy_selection",
            "final_test",
        ],
        "final_test_evaluated": False,
        "batchnorm_sha256_before_extraction": bn_before,
        "batchnorm_sha256_after_extraction": bn_after,
        "batchnorm_buffers_unchanged": True,
        "optimizer": fit["optimizer"],
        "lr": fit["lr"],
        "weight_decay": fit["weight_decay"],
        "feature_batch_size": fit["batch_size"],
        "max_epochs": fit["max_epochs"],
        "history": fit["history"],
        "extraction": extraction_log,
        "logit_files": {split: info["npz_sha256"] for split, info in logit_files.items()},
        "temperature_isolation": {
            "probe_split": "calibration",
            "temperature": 1.5,
            "argmax_unchanged": True,
            "checkpoint_sha256_unchanged": True,
            "weights_updated": False,
        },
        "training_seconds": train_seconds,
        "device": str(device),
    }
    write_json(manifest_path, checkpoint_manifest)

    reload_info = None
    if not args.skip_reload:
        reload_info = verify_reload(args, device_name=str(device))

    cached_eval = time_cached_logit_eval(val["logits"], val["labels"])
    runtime = {
        "saved_utc": _utc_now(),
        "environment": environment_record(device),
        "seed": int(model_cfg["training"]["seed"]),
        "extraction_batch_size": int(model_cfg["extraction"]["batch_size"]),
        "extraction": extraction_log,
        "extraction_seconds_total": sum(float(row["seconds"] or 0.0) for row in extraction_log),
        "training_seconds": train_seconds,
        "full_image_inference": None if reload_info is None else reload_info["full_image_inference"],
        "cached_logit_evaluation": cached_eval,
        "cross_device_feature_gap": device_gap,
        "checkpoint_sha256": head_sha,
        "baseline_path": model_cfg["paths"]["baseline_json"],
        "final_test_evaluated": False,
        "notes": [
            "Full image inference time includes in-memory fetch, ImageNet transforms, the ResNet-18 backbone, and the linear head.",
            "Cached logit evaluation is softmax/NLL on saved logits and does not run the backbone.",
            "Feature caches under artifacts/features/ are local and gitignored. The linear head npz is the retrieval artifact.",
        ],
    }
    write_json(resolve_path(model_cfg["paths"]["runtime_manifest"]), runtime)
    summary = {
        "checkpoint_sha256": head_sha,
        "selected_epoch": fit["selected_epoch"],
        "selected_val_nll": fit["selected_val_nll"],
        "baseline_accuracy": report["full_set_accuracy"],
        "baseline_nll": report["nll"],
        "baseline_ece": report["ece"],
        "training_seconds": train_seconds,
        "extraction_seconds_total": runtime["extraction_seconds_total"],
        "batchnorm_buffers_unchanged": True,
        "final_test_evaluated": False,
        "reload": None
        if reload_info is None
        else {
            "max_abs_logit_diff": reload_info["max_abs_logit_diff"],
            "argmax_agreement": reload_info["argmax_agreement"],
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def verify_reload(args: argparse.Namespace, device_name: str | None = None) -> dict[str, Any]:
    model_cfg = load_model_config(args.config)
    data_cfg = load_config(args.data_config)
    device = choose_device(device_name or args.device or model_cfg["extraction"]["device"])
    manifest = load_manifest(config=data_cfg, verify=True)
    download_cifar10(resolve_path(data_cfg["dataset"]["root"]), download=True)

    print(f"Fresh load check on {device} ...")
    model, transform, _ = build_frozen_resnet18(model_cfg, pretrained=True)
    weight, bias = load_head_npz(resolve_path(model_cfg["paths"]["checkpoint_npz"]))
    model.load_head_arrays(weight, bias)
    model.to(device)
    bn_before = model.batchnorm_buffer_sha256()

    dataset = Cifar10SplitDataset(
        "model_validation",
        corruption="clean",
        severity="none",
        config=data_cfg,
        manifest=manifest,
    )
    inferred = infer_split_logits(
        model,
        dataset,
        transform,
        device,
        batch_size=int(model_cfg["extraction"]["batch_size"]),
        time_it=True,
        warmup_batches=1,
    )
    bn_after = model.batchnorm_buffer_sha256()
    if bn_before != bn_after:
        raise RuntimeError("BatchNorm buffers changed during fresh-load inference")

    saved = load_logit_cache(logit_path(resolve_path(model_cfg["paths"]["logits_dir"]), "model_validation"))
    if inferred["source_ids"] != saved["source_ids"]:
        raise RuntimeError("fresh-load source IDs do not match the saved logit artifact")
    if not np.array_equal(inferred["labels"], saved["labels"]):
        raise RuntimeError("fresh-load labels do not match the saved logit artifact")
    diff = np.max(np.abs(inferred["logits"].astype(np.float64) - saved["logits"].astype(np.float64)))
    raw_pred = inferred["logits"].argmax(axis=1)
    saved_pred = saved["logits"].argmax(axis=1)
    agreement = float(np.mean(raw_pred == saved_pred))
    atol = float(model_cfg["reproducibility"]["same_device_logit_atol"])
    timed = float(inferred["timed_seconds"] or 0.0)
    timed_images = int(inferred["timed_images"] or 0)
    payload = {
        "saved_utc": _utc_now(),
        "device": str(device),
        "split": "model_validation",
        "max_abs_logit_diff": float(diff),
        "same_device_logit_atol": atol,
        "within_atol": bool(diff <= atol),
        "argmax_agreement": agreement,
        "batchnorm_buffers_unchanged": True,
        "full_image_inference": {
            "seconds": timed,
            "images": timed_images,
            "images_per_second": (timed_images / timed) if timed else None,
            "warmup_batches_excluded": 1,
            "batch_size": int(model_cfg["extraction"]["batch_size"]),
            "includes_backbone": True,
            "includes": inferred["includes"],
        },
    }
    write_json(resolve_path(model_cfg["paths"]["reload_check"]), payload)
    print(
        f"Reload max |Δlogit| {diff:.3e} argmax agreement {agreement:.4f} "
        f"inference {timed:.1f}s for {timed_images} images"
    )
    if diff > atol:
        raise RuntimeError(
            f"fresh load logit diff {diff:.3e} exceeds same-device atol {atol:.3e}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    parser.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument("--device", default=None, help="auto, cpu, mps, or cuda. Default: config extraction.device")
    parser.add_argument("--reextract", action="store_true", help="Ignore cached features and run the backbone again")
    parser.add_argument(
        "--verify-reload-only",
        action="store_true",
        help="Reload the saved head and compare model-validation logits. Does not retrain.",
    )
    parser.add_argument("--skip-reload", action="store_true", help="Skip the fresh-load logit check")
    args = parser.parse_args()
    if args.verify_reload_only:
        verify_reload(args)
        return 0
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
