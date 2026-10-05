# Dataset retrieval and split procedure

Owner: **Telman3000** ([Issue #1](https://github.com/MedvAx-AI/Reliable-Image-Classification-with-Confidence-Rejection/issues/1)).

Executable preparation (not just a plan) lives in `src/data.py`, `src/corruptions.py`, and `scripts/prepare_data.py`. Settings are in `configs/data.yaml`. Frozen ID lists are written to `artifacts/splits/`.

## Source

Use [official CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). The official Python archive MD5 is `c58f30108f718f92721af3b95e74349a`. Preparation records a SHA-256 of the archive under `data/raw/`.

`scripts/prepare_data.py` downloads via mirrors (Toronto HTTP, brainchip, etc.) when the default torchvision HTTPS URL fails TLS, then verifies the official MD5 before extract.

Class order (must match torchvision):

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

## Install (data package)

From the repository root, with Python 3.11+:

```bash
python -m pip install torch torchvision numpy pillow pyyaml pytest
```

Pin/tested environment for the full project is owned by LeoVesinML; the packages above are sufficient to reproduce the data pipeline.

## Prepare manifests + corruption QA panel

```bash
# from repository root
python scripts/prepare_data.py
```

What this does:

1. Downloads CIFAR-10 into `data/raw/` (gitignored).
2. Builds five disjoint seed-**42** stratified subsets (17,000 images total).
3. Writes:
   - `artifacts/splits/manifest.json` — full records + unused IDs
   - `artifacts/splits/{training,model_validation,calibration,policy_selection,final_test}.json`
   - `artifacts/splits/summary.json` — counts and file hashes
4. Writes a development visual panel to `artifacts/corruption_samples/`.

Re-check an existing manifest without rebuilding:

```bash
python scripts/prepare_data.py --verify-only
```

Run unit tests:

```bash
python -m pytest tests/test_data.py tests/test_corruptions.py -q
```

## Split contract

| Split | Images | Per class | Official partition | Used for |
|---|---:|---:|---|---|
| `training` | 10,000 | 1,000 | train | Fit linear head |
| `model_validation` | 2,000 | 200 | train | Checkpoint selection / development |
| `calibration` | 1,500 | 150 | train | Temperature fit only |
| `policy_selection` | 1,500 | 150 | train | Rejection thresholds only |
| `final_test` | 2,000 | 200 | test | Final reporting only |

IDs are stable strings: `train:<index>` / `test:<index>`. Remaining official images are listed as unused and are not used in the experiment.

Integrity guarantees (enforced in code + tests):

- pairwise disjoint source IDs across the five splits
- class counts match the table
- `final_test` uses only official test indices
- identical seed + NumPy RNG path reproduces the same IDs

## Corruptions

Applied at **native 32×32** before ResNet resize/normalize (see protocol §7):

| Family | Mild / medium / severe |
|---|---|
| Gaussian blur (Pillow) | radius 0.5 / 1.0 / 1.5 |
| Gaussian noise | std 0.03 / 0.06 / 0.12 in [0,1], clip + round uint8 |
| JPEG | quality 70 / 40 / 15; `subsampling=0`, `optimize=False`, `progressive=False` |

Noise seeds: SHA-256 of `seed:split:source_id:family:severity` → first 8 bytes as uint64 (never Python `hash()`).

Corruption **does not** change `source_id`, `label`, or `split`. Corrupted test images are derived only from `final_test` clean images.

## Shared Python interface

```python
from src.data import Cifar10SplitDataset, load_manifest
from src.corruptions import corruption_specs

manifest = load_manifest()
ds = Cifar10SplitDataset(
    "final_test",
    corruption="gaussian_blur",
    severity="severe",
    param_value=1.5,
)
sample = ds[0]
# sample.source_id, sample.split, sample.label, sample.corruption, sample.severity, sample.image (uint8 HxWx3)

for spec in corruption_specs():
    print(spec)
```

Hand off to teammates:

- **M1:** `artifacts/splits/*.json` + this README
- **M2:** `src/corruptions.py` + sample panel for MedvAx-AI evaluation alignment

## Git policy

Do **not** commit `data/raw/`. Do commit small manifests under `artifacts/splits/` and the QA panel under `artifacts/corruption_samples/`.
