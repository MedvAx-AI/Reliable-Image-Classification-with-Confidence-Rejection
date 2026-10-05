"""CIFAR-10 retrieval, stratified seed-42 splits, and shared sample interface.

Owner: Telman3000. Protocol: EXPERIMENT_PROTOCOL.md §2 and §9.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity

from src.corruptions import apply_corruption, corruption_specs, jpeg_encoder_options

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"

# Official archive + mirrors (Toronto SSL cert is often broken on some hosts).
CIFAR10_FILENAME = "cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_URLS = (
    "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-10-python.tar.gz",
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
)

SPLIT_NAMES = (
    "training",
    "model_validation",
    "calibration",
    "policy_selection",
    "final_test",
)


@dataclass(frozen=True)
class Sample:
    source_id: str
    split: str
    label: int
    corruption: str
    severity: str
    image: np.ndarray  # uint8 HxWx3 RGB

    def metadata(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("image")
        return d


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_source_id(partition: str, index: int) -> str:
    if partition not in {"train", "test"}:
        raise ValueError(f"partition must be train|test, got {partition}")
    return f"{partition}:{int(index)}"


def parse_source_id(source_id: str) -> tuple[str, int]:
    partition, idx = source_id.split(":", 1)
    return partition, int(idx)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def archive_paths(root: str | Path) -> dict[str, Path]:
    """Locate the torchvision CIFAR-10 archive under root."""
    root = Path(root)
    archive = root / CIFAR10_FILENAME
    return {"archive": archive, "extracted": root / "cifar-10-batches-py"}


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(url: str, dst: Path, *, timeout: int = 60) -> None:
    """Download url → dst; tolerate expired upstream TLS when needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    contexts: list[ssl.SSLContext | None]
    if url.startswith("http://"):
        contexts = [None]
    else:
        contexts = [
            ssl.create_default_context(),
            ssl._create_unverified_context(),  # fallback for expired toronto.edu cert
        ]
    last_err: Exception | None = None
    for ctx in contexts:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cifar10-prep/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp, tmp.open(
                "wb"
            ) as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(dst)
            return
        except Exception as exc:  # noqa: BLE001 - try next SSL mode / caller tries next URL
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def ensure_cifar10_archive(root: str | Path) -> Path:
    """Ensure archive exists, MD5 matches official checksum, and batches are extracted."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = archive_paths(root)
    archive = paths["archive"]
    extracted = paths["extracted"]

    if archive.exists() and _md5_file(archive) != CIFAR10_MD5:
        archive.unlink()

    if not archive.exists():
        errors: list[str] = []
        for url in CIFAR10_URLS:
            try:
                print(f"Downloading CIFAR-10 from {url} ...")
                _download_url(url, archive)
                if _md5_file(archive) != CIFAR10_MD5:
                    archive.unlink(missing_ok=True)
                    raise RuntimeError(f"MD5 mismatch for {url}")
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {exc}")
                archive.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                "Could not download CIFAR-10 from any mirror:\n" + "\n".join(errors)
            )

    if not check_integrity(str(archive), CIFAR10_MD5):
        raise RuntimeError(f"CIFAR-10 archive failed integrity check: {archive}")

    # Extract if torchvision marker folder is missing
    if not extracted.exists():
        print(f"Extracting {archive} ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=root)

    return archive


def download_cifar10(root: str | Path, *, download: bool = True) -> tuple[CIFAR10, CIFAR10]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if download:
        ensure_cifar10_archive(root)
    # Archive already verified; avoid torchvision's own TLS-fragile download.
    train = CIFAR10(root=str(root), train=True, download=False)
    test = CIFAR10(root=str(root), train=False, download=False)
    return train, test


def build_stratified_splits(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create five disjoint class-balanced manifests with seed 42."""
    cfg = config if config is not None else load_config()
    seed = int(cfg["split"]["seed"])
    per_class = cfg["split"]["per_class"]
    block_order = list(cfg["split"]["train_block_order"])
    class_order = list(cfg["dataset"]["class_order"])
    root = resolve_path(cfg["dataset"]["root"])

    train_ds, test_ds = download_cifar10(root, download=True)

    if list(train_ds.classes) != class_order:
        raise RuntimeError(
            f"class order mismatch: got {train_ds.classes}, expected {class_order}"
        )

    train_targets = np.asarray(train_ds.targets, dtype=np.int64)
    test_targets = np.asarray(test_ds.targets, dtype=np.int64)

    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    unused_train: list[dict[str, Any]] = []
    unused_test: list[dict[str, Any]] = []

    rng = np.random.default_rng(seed)

    for class_idx in range(10):
        train_idxs = np.flatnonzero(train_targets == class_idx)
        train_idxs = np.sort(train_idxs)
        perm = rng.permutation(train_idxs)

        needed_train = sum(int(per_class[name]) for name in block_order)
        if len(perm) < needed_train:
            raise RuntimeError(f"class {class_idx}: not enough train images")

        cursor = 0
        for split_name in block_order:
            n = int(per_class[split_name])
            chosen = perm[cursor : cursor + n]
            cursor += n
            for idx in chosen.tolist():
                splits[split_name].append(
                    {
                        "source_id": make_source_id("train", idx),
                        "partition": "train",
                        "index": int(idx),
                        "label": class_idx,
                        "class_name": class_order[class_idx],
                        "split": split_name,
                    }
                )

        for idx in perm[cursor:].tolist():
            unused_train.append(
                {
                    "source_id": make_source_id("train", idx),
                    "partition": "train",
                    "index": int(idx),
                    "label": class_idx,
                    "class_name": class_order[class_idx],
                    "split": "unused_train",
                }
            )

        test_idxs = np.flatnonzero(test_targets == class_idx)
        test_idxs = np.sort(test_idxs)
        test_perm = rng.permutation(test_idxs)
        n_test = int(per_class["final_test"])
        if len(test_perm) < n_test:
            raise RuntimeError(f"class {class_idx}: not enough test images")

        for idx in test_perm[:n_test].tolist():
            splits["final_test"].append(
                {
                    "source_id": make_source_id("test", idx),
                    "partition": "test",
                    "index": int(idx),
                    "label": class_idx,
                    "class_name": class_order[class_idx],
                    "split": "final_test",
                }
            )
        for idx in test_perm[n_test:].tolist():
            unused_test.append(
                {
                    "source_id": make_source_id("test", idx),
                    "partition": "test",
                    "index": int(idx),
                    "label": class_idx,
                    "class_name": class_order[class_idx],
                    "split": "unused_test",
                }
            )

    # Stable order within each split: by label then source index
    for name in SPLIT_NAMES:
        splits[name].sort(key=lambda r: (r["label"], r["index"]))

    unused_train.sort(key=lambda r: (r["label"], r["index"]))
    unused_test.sort(key=lambda r: (r["label"], r["index"]))

    archive = archive_paths(root)["archive"]
    archive_sha = sha256_file(archive) if archive.exists() else None

    counts = {
        name: {
            "total": len(splits[name]),
            "per_class": {
                class_order[c]: sum(1 for r in splits[name] if r["label"] == c)
                for c in range(10)
            },
        }
        for name in SPLIT_NAMES
    }

    manifest: dict[str, Any] = {
        "dataset": cfg["dataset"]["name"],
        "seed": seed,
        "numpy_version": np.__version__,
        "class_order": class_order,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_archive_md5_expected": cfg["dataset"]["official_archive_md5"],
        "archive_sha256": archive_sha,
        "split_sizes_expected": {
            "training": 10000,
            "model_validation": 2000,
            "calibration": 1500,
            "policy_selection": 1500,
            "final_test": 2000,
        },
        "counts": counts,
        "splits": splits,
        # Full unused ID lists are large; counts + regenerate-from-seed are enough for Git.
        "unused_train_count": len(unused_train),
        "unused_test_count": len(unused_test),
    }

    # Compact ID lists for quick loading
    manifest["split_ids"] = {
        name: [r["source_id"] for r in splits[name]] for name in SPLIT_NAMES
    }
    manifest["manifest_sha256"] = sha256_json(
        {k: manifest[k] for k in ("seed", "class_order", "split_ids", "counts")}
    )
    return manifest


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve relative paths against the repo root (or an explicit base)."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or REPO_ROOT) / p


def manifests_dir_from_config(config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else load_config()
    return resolve_path(cfg["paths"]["manifests_dir"])


def sample_panel_dir_from_config(config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else load_config()
    return resolve_path(cfg["paths"]["sample_panel_dir"])


def assert_split_integrity(manifest: dict[str, Any]) -> None:
    """Fail loudly on overlap, wrong counts, record/ID mismatch, or mixed partitions.

    Validates **actual records** when present — not only master ``split_ids`` lists.
    """
    if "splits" not in manifest:
        raise AssertionError("manifest has no splits/records to validate")

    expected = manifest["split_sizes_expected"]
    all_ids: list[str] = []

    for name in SPLIT_NAMES:
        records = manifest["splits"][name]
        record_ids = [r["source_id"] for r in records]

        if len(records) != expected[name]:
            raise AssertionError(
                f"{name}: expected {expected[name]} records, got {len(records)}"
            )
        if len(record_ids) != len(set(record_ids)):
            raise AssertionError(f"{name}: duplicate source IDs in records")

        # Prefer validating IDs derived from records; master lists must match if present.
        if "split_ids" in manifest:
            listed = manifest["split_ids"][name]
            if listed != record_ids:
                raise AssertionError(
                    f"{name}: master split_ids do not match record source_ids"
                )

        # Recompute class counts from records (do not trust stored counts alone).
        class_order = manifest["class_order"]
        per_class = {c: 0 for c in class_order}
        for row in records:
            per_class[class_order[int(row["label"])]] += 1
            if row.get("split") not in (None, name):
                raise AssertionError(
                    f"{name}: record {row['source_id']} has split={row.get('split')}"
                )
            if name == "final_test":
                if row["partition"] != "test":
                    raise AssertionError("final_test must use official test partition")
            else:
                if row["partition"] != "train":
                    raise AssertionError(f"{name} must use official train partition")
            want_id = make_source_id(row["partition"], row["index"])
            if row["source_id"] != want_id:
                raise AssertionError(
                    f"{name}: source_id {row['source_id']} != {want_id}"
                )

        want_per = expected[name] // 10
        for cls, n in per_class.items():
            if n != want_per:
                raise AssertionError(f"{name}/{cls}: expected {want_per}, got {n}")

        if "counts" in manifest:
            stored = manifest["counts"][name]
            if stored["total"] != len(records):
                raise AssertionError(
                    f"{name}: stored total {stored['total']} != {len(records)} records"
                )
            if stored["per_class"] != per_class:
                raise AssertionError(f"{name}: stored per_class does not match records")

        all_ids.extend(record_ids)

    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("source IDs overlap across splits (leakage)")

    if manifest["unused_train_count"] != 50000 - (
        expected["training"]
        + expected["model_validation"]
        + expected["calibration"]
        + expected["policy_selection"]
    ):
        raise AssertionError("unexpected unused_train_count")
    if manifest["unused_test_count"] != 10000 - expected["final_test"]:
        raise AssertionError("unexpected unused_test_count")


def verify_manifest_directory(manifests_dir: str | Path) -> dict[str, Any]:
    """Load manifests from disk and validate records, IDs, and saved file hashes."""
    manifests_dir = Path(manifests_dir)
    master_path = manifests_dir / "manifest.json"
    if not master_path.exists():
        raise FileNotFoundError(f"missing master manifest: {master_path}")

    with master_path.open("r", encoding="utf-8") as f:
        master = json.load(f)

    expected_hashes = master.get("split_file_hashes")
    if not expected_hashes:
        raise AssertionError("master manifest missing split_file_hashes")

    splits: dict[str, list[dict[str, Any]]] = {}
    for name in SPLIT_NAMES:
        split_path = manifests_dir / f"{name}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split file: {split_path}")

        actual_hash = sha256_file(split_path)
        if actual_hash != expected_hashes.get(name):
            raise AssertionError(
                f"{name}.json hash mismatch: expected {expected_hashes.get(name)}, "
                f"got {actual_hash}"
            )

        with split_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        records = payload["records"]
        ids = payload.get("ids") or [r["source_id"] for r in records]
        if ids != [r["source_id"] for r in records]:
            raise AssertionError(f"{name}.json: ids field does not match records")
        if payload.get("sha256") and payload["sha256"] != sha256_json(ids):
            raise AssertionError(f"{name}.json: embedded sha256 does not match ids")
        if payload.get("count") not in (None, len(records)):
            raise AssertionError(f"{name}.json: count does not match records length")

        splits[name] = records

    combined = {
        **master,
        "splits": splits,
        # Re-derive IDs from records so master lists cannot hide leakage.
        "split_ids": {name: [r["source_id"] for r in splits[name]] for name in SPLIT_NAMES},
    }
    # Keep master counts only if present; integrity recomputes from records.
    assert_split_integrity(combined)

    # Master split_ids (if present) must equal record-derived IDs.
    if "split_ids" in master:
        for name in SPLIT_NAMES:
            if master["split_ids"][name] != combined["split_ids"][name]:
                raise AssertionError(
                    f"{name}: master split_ids disagree with on-disk records"
                )

    content_hash = sha256_json(
        {
            "seed": master["seed"],
            "class_order": master["class_order"],
            "split_ids": combined["split_ids"],
            "counts": master["counts"],
        }
    )
    if master.get("manifest_sha256") != content_hash:
        raise AssertionError("master manifest_sha256 does not match on-disk records")

    combined["_manifests_dir"] = str(manifests_dir)
    return combined


def save_manifests(
    manifest: dict[str, Any],
    out_dir: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write master index + per-split JSON files (full records live in per-split files)."""
    cfg = config if config is not None else load_config()
    if out_dir is None:
        out = manifests_dir_from_config(cfg)
    else:
        out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    assert_split_integrity(manifest)

    paths: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        p = out / f"{name}.json"
        payload = {
            "split": name,
            "seed": manifest["seed"],
            "class_order": manifest["class_order"],
            "count": len(manifest["splits"][name]),
            "records": manifest["splits"][name],
            "ids": [r["source_id"] for r in manifest["splits"][name]],
        }
        payload["sha256"] = sha256_json(payload["ids"])
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        paths[name] = p

    # Compact master: IDs + metadata only (no duplicated record bodies).
    master = {
        "dataset": manifest["dataset"],
        "seed": manifest["seed"],
        "numpy_version": manifest["numpy_version"],
        "class_order": manifest["class_order"],
        "created_utc": manifest["created_utc"],
        "official_archive_md5_expected": manifest["official_archive_md5_expected"],
        "archive_sha256": manifest.get("archive_sha256"),
        "split_sizes_expected": manifest["split_sizes_expected"],
        "counts": manifest["counts"],
        "unused_train_count": manifest["unused_train_count"],
        "unused_test_count": manifest["unused_test_count"],
        "split_ids": {
            name: [r["source_id"] for r in manifest["splits"][name]]
            for name in SPLIT_NAMES
        },
        "manifest_sha256": manifest["manifest_sha256"],
        "split_file_hashes": {name: sha256_file(paths[name]) for name in SPLIT_NAMES},
    }
    master_path = out / "manifest.json"
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
        f.write("\n")
    paths["manifest"] = master_path

    summary = {
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_sha256": manifest.get("archive_sha256"),
        "seed": manifest["seed"],
        "numpy_version": manifest["numpy_version"],
        "counts": {k: v["total"] for k, v in manifest["counts"].items()},
        "split_file_hashes": master["split_file_hashes"],
        "created_utc": manifest["created_utc"],
    }
    summary_path = out / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    paths["summary"] = summary_path

    # Final gate: re-read from disk so save cannot skip on-disk checks.
    verify_manifest_directory(out)
    return paths


def load_manifest(
    path: str | Path | None = None,
    *,
    manifests_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Load master index and attach per-split records from the same directory.

    ``path`` may be the master ``manifest.json`` or omitted. Companion split files are
    always read from that file's parent directory (or ``manifests_dir`` / config).
    """
    cfg = config if config is not None else load_config()
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = resolve_path(p)
        directory = p.parent if p.name == "manifest.json" else p
        master_path = directory / "manifest.json" if p.name != "manifest.json" else p
    elif manifests_dir is not None:
        directory = resolve_path(manifests_dir)
        master_path = directory / "manifest.json"
    else:
        directory = manifests_dir_from_config(cfg)
        master_path = directory / "manifest.json"

    if verify:
        return verify_manifest_directory(directory)

    with master_path.open("r", encoding="utf-8") as f:
        master = json.load(f)

    if "splits" not in master:
        splits: dict[str, list[dict[str, Any]]] = {}
        for name in SPLIT_NAMES:
            split_path = directory / f"{name}.json"
            with split_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            splits[name] = payload["records"]
        master["splits"] = splits
        master["split_ids"] = {
            name: [r["source_id"] for r in splits[name]] for name in SPLIT_NAMES
        }
    master["_manifests_dir"] = str(directory)
    return master


class Cifar10SplitDataset:
    """Load samples for one split, optionally with a corruption setting."""

    def __init__(
        self,
        split: str,
        *,
        corruption: str = "clean",
        severity: str = "none",
        param_value: float | int | None = None,
        config: dict[str, Any] | None = None,
        manifest: dict[str, Any] | None = None,
        manifests_dir: str | Path | None = None,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split}; expected one of {SPLIT_NAMES}")
        self.cfg = config if config is not None else load_config()
        if manifest is not None:
            self.manifest = manifest
        else:
            self.manifest = load_manifest(
                manifests_dir=manifests_dir,
                config=self.cfg,
            )
        self.split = split
        self.corruption = corruption
        self.severity = severity
        self.param_value = param_value
        self.protocol_seed = int(self.cfg["split"]["seed"])
        self.records = self.manifest["splits"][split]

        root = resolve_path(self.cfg["dataset"]["root"])
        self.train_ds, self.test_ds = download_cifar10(root, download=False)

        jpeg_opts = jpeg_encoder_options(self.cfg)
        self.jpeg_kwargs = {
            "jpeg_subsampling": jpeg_opts["subsampling"],
            "jpeg_optimize": jpeg_opts["optimize"],
            "jpeg_progressive": jpeg_opts["progressive"],
        }

    def __len__(self) -> int:
        return len(self.records)

    def _raw_image(self, record: dict[str, Any]) -> np.ndarray:
        ds = self.train_ds if record["partition"] == "train" else self.test_ds
        img, label = ds[record["index"]]
        if int(label) != int(record["label"]):
            raise RuntimeError(
                f"label mismatch for {record['source_id']}: "
                f"dataset={label}, manifest={record['label']}"
            )
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def __getitem__(self, i: int) -> Sample:
        record = self.records[i]
        clean = self._raw_image(record)
        image = apply_corruption(
            clean,
            self.corruption,
            severity=self.severity,
            param_value=self.param_value,
            protocol_seed=self.protocol_seed,
            split=self.split,
            source_id=record["source_id"],
            **self.jpeg_kwargs,
        )
        return Sample(
            source_id=record["source_id"],
            split=self.split,
            label=int(record["label"]),
            corruption=self.corruption,
            severity=str(self.severity),
            image=image,
        )

    def get_by_source_id(self, source_id: str) -> Sample:
        for i, record in enumerate(self.records):
            if record["source_id"] == source_id:
                return self[i]
        raise KeyError(source_id)


def write_corruption_sample_panel(
    *,
    split: str = "model_validation",
    n_images: int = 3,
    out_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    manifests_dir: str | Path | None = None,
) -> Path:
    """Save a visual QA grid: clean + configured corruptions for development images."""
    from PIL import ImageDraw

    cfg = config if config is not None else load_config()
    if out_dir is None:
        out = sample_panel_dir_from_config(cfg)
    else:
        out = resolve_path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    ds = Cifar10SplitDataset(
        split,
        corruption="clean",
        config=cfg,
        manifests_dir=manifests_dir,
    )
    specs = corruption_specs(cfg)
    jpeg_opts = jpeg_encoder_options(cfg)
    cell = 32
    pad = 2
    cols = len(specs)
    rows = n_images
    canvas = Image.new(
        "RGB",
        (cols * (cell + pad) + pad, rows * (cell + pad) + pad + 14),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)

    meta_rows = []
    for r in range(rows):
        sample = ds[r]
        for c, spec in enumerate(specs):
            arr = apply_corruption(
                sample.image,
                spec["corruption"],
                severity=spec["severity"],
                param_value=spec["param_value"],
                protocol_seed=int(cfg["split"]["seed"]),
                split=split,
                source_id=sample.source_id,
                jpeg_subsampling=int(spec.get("jpeg_subsampling", jpeg_opts["subsampling"])),
                jpeg_optimize=bool(spec.get("jpeg_optimize", jpeg_opts["optimize"])),
                jpeg_progressive=bool(
                    spec.get("jpeg_progressive", jpeg_opts["progressive"])
                ),
            )
            tile = Image.fromarray(arr, mode="RGB")
            x = pad + c * (cell + pad)
            y = 14 + pad + r * (cell + pad)
            canvas.paste(tile, (x, y))
            meta_rows.append(
                {
                    "row": r,
                    "source_id": sample.source_id,
                    "label": sample.label,
                    "class_name": cfg["dataset"]["class_order"][sample.label],
                    **{
                        k: spec[k]
                        for k in (
                            "corruption",
                            "severity",
                            "param_name",
                            "param_value",
                        )
                    },
                }
            )
        draw.text((pad, 14 + pad + r * (cell + pad) - 12), sample.source_id, fill=(0, 0, 0))

    panel_path = out / f"{split}_corruption_panel.png"
    canvas.save(panel_path)
    with (out / f"{split}_corruption_panel.json").open("w", encoding="utf-8") as f:
        json.dump({"split": split, "cells": meta_rows}, f, indent=2)
        f.write("\n")
    return panel_path


def prepare_all(
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    panel_dir: str | Path | None = None,
    write_panel: bool = True,
) -> dict[str, Any]:
    """Download CIFAR-10, build/save manifests, run integrity checks, optional panel."""
    cfg = config if config is not None else load_config(config_path)
    manifests_out = (
        resolve_path(out_dir) if out_dir is not None else manifests_dir_from_config(cfg)
    )
    manifest = build_stratified_splits(cfg)
    paths = save_manifests(manifest, out_dir=manifests_out, config=cfg)
    panel = None
    if write_panel:
        panel = write_corruption_sample_panel(
            config=cfg,
            out_dir=panel_dir,
            manifests_dir=manifests_out,
        )
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_sha256": manifest.get("archive_sha256"),
        "counts": {k: v["total"] for k, v in manifest["counts"].items()},
        "paths": {k: str(v) for k, v in paths.items()},
        "manifests_dir": str(manifests_out),
        "panel": str(panel) if panel is not None else None,
    }
