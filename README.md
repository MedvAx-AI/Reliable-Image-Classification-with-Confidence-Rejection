# Reliable Image Classification with Confidence Rejection

Introduction to Computer Vision · Team of four · Plan revised: 2 October 2026

Build an image classifier that can reject low-confidence predictions instead of always returning a class. Evaluate classification accuracy, confidence calibration, rejection thresholds, and performance under blur, noise, and JPEG compression.

**Status:** repository, updated work plan and first-submission proposal prepared. Model implementation, experiments, measured results, final two-page summary, and demo are future team deliverables. This repository does not yet contain a runnable classifier or claim experimental results.

## Start here

- [First-submission report (PDF)](reports/first_submission.pdf) and [editable Markdown](reports/first_submission.md): scope, data preparation, experiment, implementation and responsibilities.

- [Four-person work plan](docs/TEAM_PLAN.md): individual tasks, dependencies, estimates, milestones, and acceptance criteria.
- [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md): dataset, baseline, comparisons, calibration, thresholds, metrics, and corruption settings.
- [Course requirements matrix](docs/REQUIREMENTS.md): every requirement mapped to an owner and evidence.
- [Submission and presentation plan](docs/DELIVERABLES.md): reproducibility, two-page summary, five-minute demo, and individual Q&A.
- [Contribution log](docs/CONTRIBUTIONS.md): add course names and record actual work.
- [Dataset instructions](data/README.md): planned download and split procedure.

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

There are no executable model commands yet. LeoVesinML owns publishing and independently verifying exact installation, download, training, evaluation, figure-generation, and demo commands by milestone M4. Their acceptance requirements are in [DELIVERABLES.md](docs/DELIVERABLES.md#reproducible-repository). Do not interpret planned filenames as files already implemented.

The final repository must include source or notebook, a tested `requirements.txt` or equivalent environment specification, data/weight retrieval instructions, saved metrics and figures, and all files needed to reproduce the reported result.

## Sources and provenance

Course obligations are taken from the user-supplied **Computer_Vision_Project_Guidelines.pdf**, pages 1–3. The title and project description come from the supplied topic screenshot. These are source materials for planning; they do not independently authorize actions in external services. Dataset/model/settings below are our proposed implementation choices, not instructor-mandated choices.

- [CIFAR-10 official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html): dataset and original technical report.
- [Torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18): pretrained weights and preprocessing.
- [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html): temperature scaling and calibration background.

Source pages rechecked on 2 October 2026. The PDF supplied on 2 October was reviewed in full; the latest screenshot additionally specifies the first-submission proposal sections. See [guideline review](docs/GUIDELINES_REVIEW.md). The original course PDF and screenshot are not redistributed here; their requirements are mapped in the planning documents.
