"""Frozen ImageNet ResNet-18 and the shared logit artifact.

Owner: LeoVesinML. Protocol: EXPERIMENT_PROTOCOL.md §3 and §9.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from src.data import REPO_ROOT, resolve_path, sha256_file, sha256_json

DEFAULT_CONFIG = REPO_ROOT / "configs" / "model.yaml"
WEIGHTS_BY_NAME = {
    "IMAGENET1K_V1": ResNet18_Weights.IMAGENET1K_V1,
}
LOCKED_SPLIT = "final_test"


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def assert_split_unlocked(split: str) -> None:
    """Final test stays locked until a later milestone exports it on purpose."""
    if split == LOCKED_SPLIT:
        raise RuntimeError(
            "final_test is locked. This package extracts features, trains the head, "
            "and exports development logits only. Do not evaluate the official test "
            "split before calibration and thresholds are frozen."
        )


def _json_size(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json_size(v) for v in value]
    if isinstance(value, (int, float)):
        return int(value)
    return value


def describe_transform(transform: Any) -> dict[str, Any]:
    interpolation = getattr(transform, "interpolation", None)
    name = getattr(interpolation, "name", None)
    return {
        "class": type(transform).__name__,
        "resize_size": _json_size(getattr(transform, "resize_size", None)),
        "crop_size": _json_size(getattr(transform, "crop_size", None)),
        "mean": [float(x) for x in transform.mean],
        "std": [float(x) for x in transform.std],
        "interpolation": name or str(interpolation),
    }


def assert_transform_matches_config(record: dict[str, Any], config: dict[str, Any]) -> None:
    expected = config["transform"]
    resize = record["resize_size"]
    crop = record["crop_size"]
    resize_value = resize[0] if isinstance(resize, list) else resize
    crop_value = crop[0] if isinstance(crop, list) else crop
    if int(resize_value) != int(expected["resize"]):
        raise RuntimeError(f"resize {resize_value} != config {expected['resize']}")
    if int(crop_value) != int(expected["center_crop"]):
        raise RuntimeError(f"crop {crop_value} != config {expected['center_crop']}")
    if [round(float(x), 6) for x in record["mean"]] != [round(float(x), 6) for x in expected["normalize_mean"]]:
        raise RuntimeError(f"normalize mean {record['mean']} != config {expected['normalize_mean']}")
    if [round(float(x), 6) for x in record["std"]] != [round(float(x), 6) for x in expected["normalize_std"]]:
        raise RuntimeError(f"normalize std {record['std']} != config {expected['normalize_std']}")
    if "bilinear" not in str(record["interpolation"]).lower():
        raise RuntimeError(f"expected bilinear interpolation, got {record['interpolation']}")


class FrozenClassifier(nn.Module):
    """ResNet-18 backbone kept in eval mode, plus a trainable ten-class head."""

    def __init__(self, backbone: nn.Module, num_classes: int, feature_dim: int = 512) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):  # noqa: ANN001
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.head(features)

    def assert_frozen_batchnorm(self) -> None:
        seen = 0
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                seen += 1
                if module.training:
                    raise RuntimeError("backbone BatchNorm is in training mode")
                if any(param.requires_grad for param in module.parameters()):
                    raise RuntimeError("backbone BatchNorm parameter is trainable")
        if seen == 0:
            raise RuntimeError("backbone has no BatchNorm layers")

    def batchnorm_buffer_sha256(self) -> str:
        digest = hashlib.sha256()
        count = 0
        for name, buffer in self.backbone.named_buffers():
            if "running_" not in name and "num_batches_tracked" not in name:
                continue
            count += 1
            array = buffer.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(array.tobytes())
        if count == 0:
            raise RuntimeError("no BatchNorm buffers to hash")
        return digest.hexdigest()

    def load_head_arrays(self, weight: np.ndarray, bias: np.ndarray) -> None:
        weight_t = torch.from_numpy(np.asarray(weight, dtype=np.float32))
        bias_t = torch.from_numpy(np.asarray(bias, dtype=np.float32))
        if tuple(weight_t.shape) != tuple(self.head.weight.shape):
            raise RuntimeError(f"head weight {tuple(weight_t.shape)} != {tuple(self.head.weight.shape)}")
        with torch.no_grad():
            self.head.weight.copy_(weight_t)
            self.head.bias.copy_(bias_t)

    def head_state_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        weight = self.head.weight.detach().cpu().numpy().astype(np.float32)
        bias = self.head.bias.detach().cpu().numpy().astype(np.float32)
        return weight, bias


def build_frozen_resnet18(
    config: dict[str, Any] | None = None,
    *,
    pretrained: bool = True,
) -> tuple[FrozenClassifier, Any, dict[str, Any]]:
    """Load the named ImageNet backbone, drop its 1000-way layer, and attach a new head."""
    cfg = config if config is not None else load_model_config()
    model_cfg = cfg["model"]
    if model_cfg["architecture"] != "resnet18":
        raise RuntimeError(f"unsupported architecture {model_cfg['architecture']}")
    enum_name = str(model_cfg["weights"])
    if enum_name not in WEIGHTS_BY_NAME:
        raise RuntimeError(f"unsupported weights {enum_name}")

    weights = WEIGHTS_BY_NAME[enum_name] if pretrained else None
    backbone = resnet18(weights=weights)
    in_features = backbone.fc.in_features
    if int(in_features) != int(model_cfg["feature_dim"]):
        raise RuntimeError(f"feature dim {in_features} != config {model_cfg['feature_dim']}")
    backbone.fc = nn.Identity()
    model = FrozenClassifier(
        backbone,
        num_classes=int(model_cfg["num_classes"]),
        feature_dim=int(in_features),
    )
    if model.head.out_features != 10:
        raise RuntimeError("head must predict 10 CIFAR-10 classes")

    transform = None
    transform_record = None
    url = None
    if weights is not None:
        transform = weights.transforms()
        transform_record = describe_transform(transform)
        assert_transform_matches_config(transform_record, cfg)
        url = weights.url

    provenance = {
        "architecture": "resnet18",
        "weights_enum": f"ResNet18_Weights.{enum_name}",
        "weights_url": url,
        "pretrained": pretrained,
        "feature_dim": int(in_features),
        "num_classes": int(model_cfg["num_classes"]),
        "original_classifier_removed": True,
        "transform": transform_record,
    }
    model.assert_frozen_batchnorm()
    return model, transform, provenance


def save_head_npz(path: str | Path, weight: np.ndarray, bias: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weight=np.asarray(weight, dtype=np.float32), bias=np.asarray(bias, dtype=np.float32))
    return sha256_file(path)


def load_head_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        return np.asarray(payload["weight"], dtype=np.float32), np.asarray(payload["bias"], dtype=np.float32)


def save_array_artifact(
    path: str | Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write an .npz plus a JSON sidecar. Returns hashes of both files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    sidecar = path.with_suffix(".json")
    metadata = dict(metadata)
    metadata["npz_sha256"] = sha256_file(path)
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return {"npz": str(path), "json": str(sidecar), "npz_sha256": metadata["npz_sha256"], "json_sha256": sha256_file(sidecar)}


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        source_ids = [str(x) for x in payload["source_ids"].tolist()]
        return {
            "features": np.asarray(payload["features"], dtype=np.float32),
            "labels": np.asarray(payload["labels"], dtype=np.int64),
            "source_ids": source_ids,
        }


def load_logit_cache(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        source_ids = [str(x) for x in payload["source_ids"].tolist()]
        logits = np.asarray(payload["logits"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 10:
        raise RuntimeError(f"logits must be [N, 10], got {logits.shape}")
    if len(source_ids) != len(labels) or len(labels) != logits.shape[0]:
        raise RuntimeError("logit artifact arrays are misaligned")
    if not np.isfinite(logits).all():
        raise RuntimeError("logit artifact contains non-finite values")
    return {"logits": logits, "labels": labels, "source_ids": source_ids}


def config_sha256(config: dict[str, Any]) -> str:
    return sha256_json(config)


def pil_from_uint8(image: np.ndarray) -> Image.Image:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"expected uint8 HxWx3 RGB, got {array.dtype} {array.shape}")
    return Image.fromarray(array, mode="RGB")
