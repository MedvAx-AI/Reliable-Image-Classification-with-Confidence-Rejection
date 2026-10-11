"""Train the ten-class linear head on cached training features only.

Owner: LeoVesinML. Protocol: EXPERIMENT_PROTOCOL.md §3.
Checkpoint selection uses model-validation NLL. Held-out splits never update weights.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import set_seed

TRAIN_SPLIT = "training"


def select_best_epoch(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Lowest validation NLL. Equal NLL keeps the earliest epoch."""
    if not history:
        raise ValueError("empty training history")
    best = history[0]
    for row in history[1:]:
        if float(row["val_nll"]) < float(best["val_nll"]):
            best = row
    return best


def mean_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean negative log-likelihood via log-softmax. Matches cross-entropy."""
    return float(F.cross_entropy(logits, labels, reduction="mean").item())


def assert_fit_inputs(
    *,
    split_name: str,
    source_ids: list[str],
    forbidden_ids: set[str],
) -> None:
    if split_name != TRAIN_SPLIT:
        raise ValueError(
            f"the linear head may be fit only on '{TRAIN_SPLIT}', got '{split_name}'"
        )
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("training source IDs contain duplicates")
    overlap = set(source_ids) & forbidden_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} training IDs also appear in a held-out split"
        )


def fit_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    split_name: str,
    source_ids: list[str],
    forbidden_ids: set[str],
    seed: int = 42,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 256,
    max_epochs: int = 30,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Fit one linear layer. Returns the selected state and the full epoch history."""
    assert_fit_inputs(split_name=split_name, source_ids=source_ids, forbidden_ids=forbidden_ids)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    val_features = np.asarray(val_features, dtype=np.float32)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != len(labels):
        raise ValueError("training features and labels are misaligned")
    if val_features.ndim != 2 or val_features.shape[0] != len(val_labels):
        raise ValueError("validation features and labels are misaligned")
    if features.shape[1] != val_features.shape[1]:
        raise ValueError("train/val feature dimensions differ")
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("training labels are outside 0..num_classes-1")

    set_seed(seed)
    head = nn.Linear(features.shape[1], num_classes)
    initial = copy.deepcopy(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    x_val = torch.from_numpy(val_features)
    y_val = torch.from_numpy(val_labels)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    history: list[dict[str, Any]] = []
    states: list[dict[str, torch.Tensor]] = []
    for epoch in range(1, int(max_epochs) + 1):
        head.train()
        permutation = torch.randperm(x.shape[0], generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, x.shape[0], int(batch_size)):
            index = permutation[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x[index]), y[index])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(index.shape[0])
            seen += int(index.shape[0])
        head.eval()
        with torch.no_grad():
            val_logits = head(x_val)
            val_nll = mean_nll(val_logits, y_val)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "val_nll": val_nll,
        }
        history.append(row)
        states.append(copy.deepcopy(head.state_dict()))

    best = select_best_epoch(history)
    head.load_state_dict(states[int(best["epoch"]) - 1])
    head.eval()
    trained = head.state_dict()
    changed = any(not torch.equal(trained[key], initial[key]) for key in trained)
    if not changed:
        raise RuntimeError("linear head weights did not change during training")
    if head.out_features != num_classes:
        raise RuntimeError(f"expected {num_classes} outputs, got {head.out_features}")

    weight = head.weight.detach().cpu().numpy().astype(np.float32)
    bias = head.bias.detach().cpu().numpy().astype(np.float32)
    return {
        "weight": weight,
        "bias": bias,
        "history": history,
        "selected_epoch": int(best["epoch"]),
        "selected_val_nll": float(best["val_nll"]),
        "selection_rule": "lowest model-validation NLL; earliest epoch breaks ties",
        "train_split": split_name,
        "n_train": int(features.shape[0]),
        "n_val": int(val_features.shape[0]),
        "optimizer": "AdamW",
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "seed": int(seed),
        "num_classes": int(num_classes),
    }


def logits_from_head(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply the saved linear head on CPU. Same math as nn.Linear."""
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    w = torch.from_numpy(np.asarray(weight, dtype=np.float32))
    b = torch.from_numpy(np.asarray(bias, dtype=np.float32))
    with torch.no_grad():
        logits = F.linear(x, w, b)
    return logits.numpy().astype(np.float32)
