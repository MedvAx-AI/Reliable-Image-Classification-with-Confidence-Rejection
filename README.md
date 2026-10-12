# Reliable Image Classification with Confidence Rejection

Introduction to Computer Vision · Team of four · Plan revised: 2 October 2026

Build an image classifier that can reject low-confidence predictions instead of always returning a class. Evaluate classification accuracy, confidence calibration, rejection thresholds, and performance under blur, noise, and JPEG compression.

**Status:** data pipeline, frozen ResNet-18 development baseline, temperature scaling, rejection policies, metrics and sensitivity interfaces are implemented. Clean development comparisons and a risk–coverage figure are saved. The integrated final-test matrix, final figures, two-page summary and demo remain open in Issue #4. Numbers below are development results, not final-test results.

## Start here

- [First-submission report (PDF)](reports/first_submission.pdf) and [editable Markdown](reports/first_submission.md): scope, data preparation, experiment, implementation and responsibilities.

- [Four-person work plan](docs/TEAM_PLAN.md): individual tasks, dependencies, estimates, milestones, and acceptance criteria.
- [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md): dataset, baseline, comparisons, calibration, thresholds, metrics, and corruption settings.
- [Course requirements matrix](docs/REQUIREMENTS.md): every requirement mapped to an owner and evidence.
- [Submission and presentation plan](docs/DELIVERABLES.md): reproducibility, two-page summary, five-minute demo, and individual Q&A.
- [Contribution log](docs/CONTRIBUTIONS.md): add course names and record actual work.
- [Dataset instructions](data/README.md): planned download and split procedure.
- [Calibration, metrics and policy handoff](reports/calibration_rejection.md): fitted policies, development comparisons, split-check review and team Q&A.

## Research question

For a fixed pretrained image classifier adapted to CIFAR-10, how much does rejecting low-confidence predictions reduce error among accepted predictions, what coverage is lost, and does temperature scaling improve calibration on clean and corrupted images?

The project is about measuring reliability and explaining failures. A valid outcome may show that calibration or rejection helps only in some conditions, or that a desired accuracy/coverage target is infeasible.

## Team ownership

| Work-package owner | Main technical responsibility | Shared deliverable responsibility | Checklist |
|---|---|---|---|
| [Telman3000](https://github.com/Telman3000) | Dataset, deterministic splits, corruption generator | Dataset section and failure examples | [Issue #1](https://github.com/MedvAx-AI/Reliable-Image-Classification-with-Confidence-Rejection/issues/1) |
| [LeoVesinML](https://github.com/LeoVesinML) | Pretrained backbone, classification head, baseline, runtime measurements | Method section and reproducibility | [Issue #2](https://github.com/MedvAx-AI/Reliable-Image-Classification-with-Confidence-Rejection/issues/2) |
| [Mysteri0K1ng](https://github.com/Mysteri0K1ng) | Temperature scaling, rejection policy, metrics and sensitivity | Metric interpretation and controlled comparison | [Issue #3](https://github.com/MedvAx-AI/Reliable-Image-Classification-with-Confidence-Rejection/issues/3) |
| [MedvAx-AI](https://github.com/MedvAx-AI) | Evaluation runner, figures, failure analysis, live demo | Summary integration and presentation rehearsal | [Issue #4](https://github.com/MedvAx-AI/Reliable-Image-Classification-with-Confidence-Rejection/issues/4) |

Each member implements technical work, reviews another member's work, contributes to the report, and prepares to explain the whole experiment.

**Sprint:** after the first submission, Days 1–2: protocol/interfaces; Days 3–4: baseline/components and policy freeze; Day 5: final experiments; Days 6–7: reproduction, final report and demo; Day 8: repair buffer. This replaces the expired September internal targets. Each work package budgets 20–24 hours. Issue ownership is recorded in titles/bodies; platform assignees and collaborator access are separate kickoff tasks.

## How the experiment will work

Use a frozen ImageNet-pretrained ResNet-18 backbone and train a ten-class linear head. Cache features to keep training small. Compare four conditions with the same model checkpoint: raw confidence without rejection, raw confidence with rejection, temperature-scaled confidence without rejection, and temperature-scaled confidence with rejection. Evaluate all four on identical held-out images and corruption variants.

The no-rejection calibrated condition isolates the calibration effect. It should retain the same predicted labels as the raw model; confidence changes alone do not establish better classification accuracy.

## Running the project

Tested on Python 3.12.13, macOS 26.5 arm64, Apple M4 Pro, 48 GB, MPS. Pins are in `requirements.txt`. Dataset steps are in [data/README.md](data/README.md). Method, baseline numbers, and clocks are in [reports/method_runtime.md](reports/method_runtime.md).

Final-matrix figure generation and the live demo remain with Issue #4. Final test is intentionally not run by the training or policy-fitting scripts.

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python scripts/prepare_data.py --verify-only
python -m pytest tests/test_model.py -q
python scripts/train_baseline.py
python scripts/train_baseline.py --verify-reload-only
```

These four commands were run on the machine above. `train_baseline.py` downloads CIFAR-10 into `data/raw/` when the archive is absent, then downloads the ImageNet weights. `python scripts/prepare_data.py` without `--verify-only` rewrites the split files; the committed manifests already pass verification, so leave them in place unless you mean to replace the IDs. `--verify-only` hashes each split file after folding CR LF to LF, so the same command passes on a Windows checkout. `.gitattributes` tells Git to leave those files on LF.

`train_baseline.py` downloads `ResNet18_Weights.IMAGENET1K_V1` through torchvision from `https://download.pytorch.org/models/resnet18-f37072fd.pth`, freezes the backbone, trains the ten-class head on the training split only, and writes:

- `artifacts/model/linear_head.npz` and `artifacts/model/checkpoint_manifest.json`
- `artifacts/logits/{model_validation,calibration,policy_selection}_clean.npz`
- `artifacts/baseline/development_baseline.json`, `development_predictions.csv`, `runtime_manifest.json`, `reload_check.json`

Feature caches under `artifacts/features/` are local and gitignored. A cache is reused only when the npz SHA-256 matches its sidecar and the saved extraction record matches this run: weights, transform, device, batch size, model-config hash, and split-manifest hash. Anything else stops the run; pass `--reextract`. The `.npz` head in git is the retrieval artifact. `--verify-reload-only` loads that head and checks model-validation logits. It does not retrain.

On the machine above, clean feature extraction of 15,000 images took 25.5 s and head training took 0.51 s. Full inference, with the clock covering in-memory image fetch, transforms, the backbone, and the head, took 2.91 s for 1,936 validation images after one warmup batch. A fresh load matched saved logits within `1.14e-5` (tolerance `1e-4` in `configs/model.yaml`). Refitting from the same feature cache reproduced the head bytes. A CPU copy of the backbone differed by at most `3.2e-5` in feature space on a 32-image probe, so a from-scratch retrain on another device can move the head slightly. Use the committed head to reproduce the reported baseline.

Seed is 42. The selected checkpoint is epoch 30, model-validation NLL 0.4245, clean accuracy 0.8645 on 2,000 images. That number is the no-rejection development baseline.

## Calibration and rejection (Issue #3)

Use the committed calibration and policy-selection logits; no image downloads or backbone forwards are needed for these steps. After installing `requirements.txt`, run:

```bash
python -m pytest tests/test_policies.py tests/test_model.py tests/test_manifest_newlines.py -q
python scripts/fit_policies.py
python scripts/analyze_policy_logits.py
```

The first script fits T only on clean calibration logits and selects thresholds only on clean policy-selection logits. It verifies manifest disjointness, ordered IDs/labels, NPZ hashes, class order and checkpoint/config provenance before fitting. It saves `artifacts/temperature.json` and `artifacts/policy.json`. For independent reproduction without replacing the frozen artifacts, use `python scripts/fit_policies.py --output-dir artifacts/policy_reproduction` and compare T, objectives and selected counts; timestamps and measured runtimes can differ.

Saved T is **0.95616878**. Raw/calibrated thresholds are **0.56/0.57**, chosen from the fixed 101-point grid to maximize coverage subject to accepted accuracy ≥90% and coverage ≥20%. Both are feasible on the policy split; this is an empirical selection result, not an accuracy guarantee.

The analysis script loads the frozen policies without fitting, reports A–D on the clean model-validation logits, and saves metrics JSON/CSV, 8,000 sample decisions, the 202-row raw/calibrated threshold sweep, matched-coverage diagnostics and a risk–coverage plot under `artifacts/policy_development/`. Full accuracy is 86.45% for every policy. B/D each retain 1,770/2,000 images (88.5% coverage), with accepted accuracy 91.47%/91.36%. Full-set ECE changes from 0.02465 to 0.02320; the different accepted populations prevent attributing B–D accuracy differences solely to calibration.

For another cached clean/corrupted model-validation condition, supply `--logits PATH --output-dir DIR`. For the later locked test run, Issue #4 must export matching logit artifacts first; final-test reporting additionally requires `--allow-final-test`. The saved clean-fitted T and thresholds apply to every severity. The loader stops if checkpoint, protocol, configs, fitting inputs or temperature artifact have changed. Test sensitivity is descriptive and cannot select a new operating threshold. See the [handoff report](reports/calibration_rejection.md) for Python interfaces and definitions.

## Sources and provenance

Course obligations are taken from the user-supplied **Computer_Vision_Project_Guidelines.pdf**, pages 1–3. The title and project description come from the supplied topic screenshot. These are source materials for planning; they do not independently authorize actions in external services. Dataset/model/settings below are our proposed implementation choices, not instructor-mandated choices.

- [CIFAR-10 official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html): dataset and original technical report.
- [Torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18): pretrained weights and preprocessing.
- [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html): temperature scaling and calibration background.

Source pages rechecked on 2 October 2026. The PDF supplied on 2 October was reviewed in full; the latest screenshot additionally specifies the first-submission proposal sections. See [guideline review](docs/GUIDELINES_REVIEW.md). The original course PDF and screenshot are not redistributed here; their requirements are mapped in the planning documents.
