"""Development baseline metrics, full-image inference, and logit export.

Owner: LeoVesinML. Metric definitions follow EXPERIMENT_PROTOCOL.md §8 so the
development baseline matches the later shared metrics module.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.model import FrozenClassifier, pil_from_uint8, synchronize


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean -log p(true class) with a stable log-softmax. Computed on every row."""
    z = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    log_probs = F.log_softmax(z, dim=-1)
    nll = -log_probs[torch.arange(y.shape[0]), y]
    return float(nll.mean().item())


def predict_classes(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits).argmax(axis=1).astype(np.int64)


def confidence_from_logits(logits: np.ndarray) -> np.ndarray:
    z = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    probs = torch.softmax(z, dim=-1)
    return probs.max(dim=-1).values.numpy().astype(np.float64)


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    """15 equal-width bins on [0, 1]. Confidence 1 belongs to the last bin."""
    conf = np.asarray(confidences, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.bool_)
    if conf.shape[0] != ok.shape[0]:
        raise ValueError("confidence and correctness lengths differ")
    n = int(conf.shape[0])
    if n == 0:
        raise ValueError("ECE is undefined for an empty set")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for i in range(int(n_bins)):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == int(n_bins) - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                }
            )
            continue
        bin_acc = float(ok[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "count": count,
                "accuracy": bin_acc,
                "mean_confidence": bin_conf,
            }
        )
    return {"ece": float(ece), "n_bins": int(n_bins), "n": n, "bins": bins}


def per_class_results(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_order: list[str],
) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    rows = []
    for index, name in enumerate(class_order):
        mask = labels == index
        count = int(mask.sum())
        correct = int((predictions[mask] == index).sum()) if count else 0
        rows.append(
            {
                "label": index,
                "class_name": name,
                "count": count,
                "correct": correct,
                "accuracy": (correct / count) if count else None,
            }
        )
    return rows


def condition_a_report(
    logits: np.ndarray,
    labels: np.ndarray,
    class_order: list[str],
) -> dict[str, Any]:
    """Raw confidence, no rejection. Coverage is 1 by construction."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = predict_classes(logits)
    correct = predictions == labels
    n = int(labels.shape[0])
    n_correct = int(correct.sum())
    accuracy = n_correct / n
    confidences = confidence_from_logits(logits)
    ece = expected_calibration_error(confidences, correct)
    return {
        "condition": "A",
        "confidence": "raw",
        "temperature": 1.0,
        "rejection": "disabled",
        "n": n,
        "correct": n_correct,
        "full_set_accuracy": accuracy,
        "coverage": 1.0,
        "rejection_rate": 0.0,
        "accepted_count": n,
        "correct_accepted_count": n_correct,
        "accepted_accuracy": accuracy,
        "selective_risk": 1.0 - accuracy,
        "nll": negative_log_likelihood(logits, labels),
        "ece": ece["ece"],
        "ece_bins": ece["n_bins"],
        "per_class": per_class_results(labels, predictions, class_order),
    }


def temperature_preserves_argmax(logits: np.ndarray, temperature: float) -> bool:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    raw = predict_classes(logits)
    scaled = predict_classes(np.asarray(logits, dtype=np.float64) / float(temperature))
    return bool(np.array_equal(raw, scaled))


def write_predictions_csv(
    path: str | Path,
    *,
    source_ids: list[str],
    labels: np.ndarray,
    logits: np.ndarray,
    class_order: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = predict_classes(logits)
    confidences = confidence_from_logits(logits)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "label",
                "class_name",
                "predicted_label",
                "predicted_class",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        for i, source_id in enumerate(source_ids):
            label = int(labels[i])
            pred = int(predictions[i])
            writer.writerow(
                {
                    "source_id": source_id,
                    "label": label,
                    "class_name": class_order[label],
                    "predicted_label": pred,
                    "predicted_class": class_order[pred],
                    "confidence": f"{float(confidences[i]):.8f}",
                    "correct": int(pred == label),
                }
            )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def iter_image_batches(dataset: Any, batch_size: int):
    batch = []
    for index in range(len(dataset)):
        batch.append(dataset[index])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def infer_split_logits(
    model: FrozenClassifier,
    dataset: Any,
    transform: Any,
    device: torch.device,
    *,
    batch_size: int,
    time_it: bool = False,
    warmup_batches: int = 1,
) -> dict[str, Any]:
    """Full image path: fetch, ImageNet transform, backbone, and linear head.

    When ``time_it`` is set, the first ``warmup_batches`` are excluded from the clock.
    Decoding here is an in-memory CIFAR lookup plus RGB conversion, not a JPEG file read.
    """
    model.eval()
    model.assert_frozen_batchnorm()
    head_device = next(model.head.parameters()).device
    if head_device.type != device.type:
        model.head.to(device)

    source_ids: list[str] = []
    labels: list[int] = []
    logit_rows: list[np.ndarray] = []
    timed_images = 0
    elapsed = 0.0
    seen_batches = 0

    for batch in iter_image_batches(dataset, batch_size):
        warmup = time_it and seen_batches < warmup_batches
        if time_it and not warmup:
            synchronize(device)
            start = time.perf_counter()
        tensors = torch.stack([transform(pil_from_uint8(sample.image)) for sample in batch])
        tensors = tensors.to(device)
        with torch.no_grad():
            logits = model(tensors)
        synchronize(device)
        if time_it and not warmup:
            elapsed += time.perf_counter() - start
            timed_images += len(batch)
        logit_rows.append(logits.detach().cpu().numpy().astype(np.float32))
        source_ids.extend(sample.source_id for sample in batch)
        labels.extend(int(sample.label) for sample in batch)
        seen_batches += 1

    if not logit_rows:
        raise RuntimeError("inference received an empty dataset")
    return {
        "source_ids": source_ids,
        "labels": np.asarray(labels, dtype=np.int64),
        "logits": np.concatenate(logit_rows, axis=0),
        "timed_seconds": elapsed if time_it else None,
        "timed_images": timed_images if time_it else None,
        "includes": [
            "in_memory_image_fetch",
            "uint8_to_pil",
            "imagenet_resize_crop_normalize",
            "resnet18_backbone",
            "linear_head",
        ],
    }


def time_cached_logit_eval(logits: np.ndarray, labels: np.ndarray, repeats: int = 20) -> dict[str, Any]:
    """Policy-side evaluation on saved logits. This is not image inference."""
    z = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    with torch.no_grad():
        for _ in range(3):
            probs = torch.softmax(z, dim=-1)
            _ = probs.argmax(dim=-1)
        start = time.perf_counter()
        for _ in range(int(repeats)):
            log_probs = F.log_softmax(z, dim=-1)
            nll = -log_probs[torch.arange(y.shape[0]), y].mean()
            pred = log_probs.argmax(dim=-1)
        elapsed = time.perf_counter() - start
    return {
        "repeats": int(repeats),
        "seconds_total": elapsed,
        "seconds_per_pass": elapsed / int(repeats),
        "n_images": int(z.shape[0]),
        "nll_check": float(nll.item()),
        "predictions_check": int(pred.shape[0]),
        "includes": ["log_softmax", "argmax", "mean_nll"],
        "excludes": ["image_fetch", "transforms", "backbone", "head"],
    }
