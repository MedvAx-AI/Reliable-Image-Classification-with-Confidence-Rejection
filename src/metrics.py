"""Shared full-set calibration and selective metrics (Issue #3)."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_logits(logits: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError("logits must be nonempty finite [N, K], K >= 2")
    if labels is not None:
        y = np.asarray(labels)
        if y.shape != (len(z),) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("labels must be integer [N] aligned with logits")
        if np.any(y < 0) or np.any(y >= z.shape[1]):
            raise ValueError("label outside the class range")
    return z


def log_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = validate_logits(logits)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Center before division so a shared large offset cannot overflow softmax.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scaled = (z - z.max(axis=1, keepdims=True)) / temperature
        return scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))


def confidence_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return np.exp(log_probabilities(logits, temperature).max(axis=1))


def negative_log_likelihood(
    logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0,
) -> float:
    z = validate_logits(logits, labels)
    log_probs = log_probabilities(z, temperature)
    return float(-log_probs[np.arange(len(z)), labels].mean())


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, *, n_bins: int = 15,
) -> dict[str, Any]:
    conf = np.asarray(confidences, dtype=np.float64)
    ok = np.asarray(correct)
    if conf.ndim != 1 or not len(conf) or ok.shape != conf.shape or ok.dtype != np.bool_:
        raise ValueError("confidence and boolean correctness must be nonempty aligned vectors")
    if not np.isfinite(conf).all() or np.any(conf < 0) or np.any(conf > 1):
        raise ValueError("confidence must be finite in [0, 1]")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    edges = np.linspace(0, 1, n_bins + 1)
    # Exact interior boundaries enter the bin on their right; 1 enters the last.
    assignments = np.minimum(np.searchsorted(edges, conf, side="right") - 1, n_bins - 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        mask = assignments == i
        count = int(mask.sum())
        accuracy = float(ok[mask].mean()) if count else None
        mean_confidence = float(conf[mask].mean()) if count else None
        if count:
            ece += count / len(conf) * abs(accuracy - mean_confidence)
        bins.append({"bin": i, "lo": float(edges[i]), "hi": float(edges[i + 1]),
                     "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    return {"ece": float(ece), "n_bins": n_bins, "n": len(conf), "bins": bins}


def selective_counts(correct: np.ndarray, accepted: np.ndarray) -> dict[str, Any]:
    ok, mask = np.asarray(correct), np.asarray(accepted)
    if ok.ndim != 1 or not len(ok) or mask.shape != ok.shape:
        raise ValueError("correct and accepted must be nonempty aligned vectors")
    if ok.dtype != np.bool_ or mask.dtype != np.bool_:
        raise ValueError("correct and accepted must be boolean")
    count = int(mask.sum())
    correct_count = int((ok & mask).sum())
    accuracy = correct_count / count if count else None
    return {"n": len(ok), "correct": int(ok.sum()), "full_set_accuracy": float(ok.mean()),
            "accepted_count": count, "correct_accepted_count": correct_count,
            "coverage": count / len(ok), "rejection_rate": 1 - count / len(ok),
            "accepted_accuracy": accuracy,
            "selective_risk": 1 - accuracy if count else None}


def classification_metrics(
    logits: np.ndarray, labels: np.ndarray, *, temperature: float = 1.0,
    accepted: np.ndarray | None = None, n_bins: int = 15,
    class_order: list[str] | None = None,
) -> dict[str, Any]:
    z = validate_logits(logits, labels)
    y = np.asarray(labels)
    correct = z.argmax(axis=1) == y
    mask = np.ones(len(z), dtype=bool) if accepted is None else np.asarray(accepted)
    conf = confidence_from_logits(z, temperature)
    ece = expected_calibration_error(conf, correct, n_bins=n_bins)
    result = selective_counts(correct, mask)
    result.update(nll=negative_log_likelihood(z, y, temperature), ece=ece["ece"],
                  ece_bins=n_bins, reliability_bins=ece["bins"])
    names = class_order if class_order is not None else [str(i) for i in range(z.shape[1])]
    if len(names) != z.shape[1]:
        raise ValueError("class order length must equal logit class count")
    rows = []
    for i, name in enumerate(names):
        subset = y == i
        if subset.any():
            row = selective_counts(correct[subset], mask[subset])
        else:
            row = {"n": 0, "correct": 0, "accepted_count": 0, "correct_accepted_count": 0,
                   "full_set_accuracy": None, "coverage": None, "rejection_rate": None,
                   "accepted_accuracy": None, "selective_risk": None}
        rows.append({"label": i, "class_name": name, **row})
    result["per_class"] = rows
    return result
