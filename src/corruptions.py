"""Deterministic CIFAR-10 corruptions: Gaussian blur, Gaussian noise, JPEG.

Protocol: EXPERIMENT_PROTOCOL.md §7. Apply at native 32×32 before model transforms.
Noise seeds derive from SHA-256 of `seed:split:source_id:family:severity`.
"""

from __future__ import annotations

import hashlib
import io
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter

BLUR_RADII = (0.5, 1.0, 1.5)
NOISE_STDS = (0.03, 0.06, 0.12)
JPEG_QUALITIES = (70, 40, 15)

SEVERITY_NAMES = ("mild", "medium", "severe")


def noise_seed_u64(
    protocol_seed: int,
    split: str,
    source_id: str,
    family: str,
    severity: str | int | float,
) -> int:
    """First 8 bytes of SHA-256 as unsigned 64-bit integer (never Python hash())."""
    key = f"{protocol_seed}:{split}:{source_id}:{family}:{severity}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _as_uint8_hwc(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            raise TypeError(f"expected uint8 image, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got shape {arr.shape}")
    return arr


def apply_gaussian_blur(image: np.ndarray | Image.Image, radius: float) -> np.ndarray:
    """Pillow GaussianBlur at native resolution; returns uint8 RGB array."""
    pil = Image.fromarray(_as_uint8_hwc(image), mode="RGB")
    out = pil.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(out, dtype=np.uint8)


def apply_gaussian_noise(
    image: np.ndarray | Image.Image,
    std: float,
    *,
    protocol_seed: int,
    split: str,
    source_id: str,
    severity: str | float,
) -> np.ndarray:
    """Independent Gaussian noise per RGB channel in [0,1], clip + round to uint8."""
    arr = _as_uint8_hwc(image).astype(np.float64) / 255.0
    seed = noise_seed_u64(protocol_seed, split, source_id, "gaussian_noise", severity)
    rng = np.random.default_rng(seed)
    noisy = arr + rng.normal(loc=0.0, scale=float(std), size=arr.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return np.rint(noisy * 255.0).astype(np.uint8)


def apply_jpeg(
    image: np.ndarray | Image.Image,
    quality: int,
    *,
    subsampling: int = 0,
    optimize: bool = False,
    progressive: bool = False,
) -> np.ndarray:
    """In-memory JPEG encode/decode with fixed encoder settings."""
    pil = Image.fromarray(_as_uint8_hwc(image), mode="RGB")
    buf = io.BytesIO()
    pil.save(
        buf,
        format="JPEG",
        quality=int(quality),
        subsampling=int(subsampling),
        optimize=bool(optimize),
        progressive=bool(progressive),
    )
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    return np.asarray(decoded, dtype=np.uint8)


def jpeg_encoder_options(config: dict | None = None) -> dict:
    """JPEG encoder kwargs from config, with protocol defaults."""
    jpeg = {}
    if config is not None:
        jpeg = dict(config.get("corruptions", {}).get("families", {}).get("jpeg", {}))
    return {
        "subsampling": int(jpeg.get("subsampling", 0)),
        "optimize": bool(jpeg.get("optimize", False)),
        "progressive": bool(jpeg.get("progressive", False)),
    }


def corruption_specs(config: dict | None = None) -> list[dict]:
    """Clean control + nine independent corruption settings.

    When ``config`` is provided, severity values and JPEG encoder options come from
    ``config["corruptions"]``; otherwise the protocol defaults are used.
    """
    families = {}
    if config is not None:
        families = dict(config.get("corruptions", {}).get("families", {}))

    blur_radii = tuple(
        families.get("gaussian_blur", {}).get("radii_px", BLUR_RADII)
    )
    noise_stds = tuple(
        families.get("gaussian_noise", {}).get("std_in_01", NOISE_STDS)
    )
    jpeg_qualities = tuple(families.get("jpeg", {}).get("qualities", JPEG_QUALITIES))
    jpeg_opts = jpeg_encoder_options(config)

    if not (len(blur_radii) == len(noise_stds) == len(jpeg_qualities) == 3):
        raise ValueError(
            "each corruption family must define exactly 3 severity levels "
            f"(got blur={len(blur_radii)}, noise={len(noise_stds)}, jpeg={len(jpeg_qualities)})"
        )

    specs: list[dict] = [
        {
            "corruption": "clean",
            "severity": "none",
            "severity_index": 0,
            "param_name": None,
            "param_value": None,
        }
    ]
    for idx, (name, radius) in enumerate(zip(SEVERITY_NAMES, blur_radii), start=1):
        specs.append(
            {
                "corruption": "gaussian_blur",
                "severity": name,
                "severity_index": idx,
                "param_name": "radius_px",
                "param_value": float(radius),
            }
        )
    for idx, (name, std) in enumerate(zip(SEVERITY_NAMES, noise_stds), start=1):
        specs.append(
            {
                "corruption": "gaussian_noise",
                "severity": name,
                "severity_index": idx,
                "param_name": "std_in_01",
                "param_value": float(std),
            }
        )
    for idx, (name, quality) in enumerate(zip(SEVERITY_NAMES, jpeg_qualities), start=1):
        specs.append(
            {
                "corruption": "jpeg",
                "severity": name,
                "severity_index": idx,
                "param_name": "quality",
                "param_value": int(quality),
                "jpeg_subsampling": jpeg_opts["subsampling"],
                "jpeg_optimize": jpeg_opts["optimize"],
                "jpeg_progressive": jpeg_opts["progressive"],
            }
        )
    return specs


def apply_corruption(
    image: np.ndarray | Image.Image,
    corruption: str,
    *,
    severity: str | int | float | None = None,
    param_value: float | int | None = None,
    protocol_seed: int = 42,
    split: str = "unknown",
    source_id: str = "unknown",
    jpeg_subsampling: int = 0,
    jpeg_optimize: bool = False,
    jpeg_progressive: bool = False,
) -> np.ndarray:
    """Apply one named corruption. Labels/IDs are unchanged by the caller."""
    if corruption == "clean":
        return _as_uint8_hwc(image).copy()

    if corruption == "gaussian_blur":
        radius = float(param_value if param_value is not None else severity)
        return apply_gaussian_blur(image, radius)

    if corruption == "gaussian_noise":
        std = float(param_value if param_value is not None else severity)
        sev_key = severity if severity is not None else std
        return apply_gaussian_noise(
            image,
            std,
            protocol_seed=protocol_seed,
            split=split,
            source_id=source_id,
            severity=sev_key,
        )

    if corruption == "jpeg":
        quality = int(param_value if param_value is not None else severity)
        return apply_jpeg(
            image,
            quality,
            subsampling=jpeg_subsampling,
            optimize=jpeg_optimize,
            progressive=jpeg_progressive,
        )

    raise ValueError(f"unknown corruption: {corruption}")


def image_sha256(image: np.ndarray | Image.Image) -> str:
    arr = _as_uint8_hwc(image)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def iter_corrupted_variants(
    image: np.ndarray | Image.Image,
    *,
    protocol_seed: int,
    split: str,
    source_id: str,
    specs: Iterable[dict] | None = None,
    config: dict | None = None,
) -> list[dict]:
    """Return clean + corrupted uint8 images with metadata (same source_id/label expected)."""
    out = []
    for spec in specs if specs is not None else corruption_specs(config):
        arr = apply_corruption(
            image,
            spec["corruption"],
            severity=spec["severity"],
            param_value=spec["param_value"],
            protocol_seed=protocol_seed,
            split=split,
            source_id=source_id,
            jpeg_subsampling=int(spec.get("jpeg_subsampling", 0)),
            jpeg_optimize=bool(spec.get("jpeg_optimize", False)),
            jpeg_progressive=bool(spec.get("jpeg_progressive", False)),
        )
        out.append(
            {
                **spec,
                "source_id": source_id,
                "split": split,
                "image": arr,
                "image_sha256": image_sha256(arr),
            }
        )
    return out
