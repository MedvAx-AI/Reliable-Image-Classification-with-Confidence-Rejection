# Reliable Image Classification with Confidence Rejection

Introduction to Computer Vision · Final experiment completed 12 October 2026

We compare raw/scaled confidence, with and without rejection, using one frozen ResNet-18/linear-head checkpoint on CIFAR-10. The final run contains **40 aggregate rows, 80,000 decisions and ten image conditions**. Clean accepted accuracy rises from **85.05%** to **90.90%**, retaining **87.90%** of images. Severe corruption exposes confident errors; clean-fitted calibration/rejection do not guarantee robustness.

The technical implementation and submission artifacts are complete. Actual timed rehearsal, individual understanding, instructor registration/deadline confirmation and course upload remain human tasks. See the [completion audit](docs/COMPLETION_AUDIT.md).

## Submission package

- [Exactly two-page technical summary](reports/technical_summary.pdf), [editable source](reports/technical_summary.md).
- [Three slides](reports/slides.pdf), [five-minute script and Q&A](reports/demo_script.md).
- [Measured failures](reports/failure_analysis.md), [complete metrics](results/final_metrics.csv), [runtime/provenance](results/run_manifest.json).
- [First-working-model milestone](reports/first_working_model.pdf), [original proposal](reports/first_submission.md).
- [Team plan](docs/TEAM_PLAN.md), [frozen protocol](docs/EXPERIMENT_PROTOCOL.md), [requirements](docs/REQUIREMENTS.md), [contributions](docs/CONTRIBUTIONS.md).

## Reproduce reported results

Run commands from the repository root. Tested final environment: Windows 11, Python **3.13.13**, Ryzen 7 5800H, about 16 GiB RAM, CPU, eight PyTorch threads. No GPU is required. First-use image/weight downloads require Internet; subsequent live inference works offline from local caches.

```powershell
git clone https://github.com/MedvAx-AI/Reliable-Image-Classification-with-Confidence-Rejection.git
cd Reliable-Image-Classification-with-Confidence-Rejection
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-evaluation-lock.txt
python scripts/prepare_data.py --verify-only
python -m pytest -q
python scripts/verify_results.py
python scripts/evaluate_final.py --from-cache --output-dir reproduced/cached
```

Activation is optional: use `.venv\Scripts\python.exe` directly if PowerShell activation is unavailable. The lock records all installed packages and the official PyTorch CPU index; top-level pins are in `requirements-evaluation.txt`. The original macOS/MPS baseline environment is separately pinned in `requirements.txt`; the Windows CPU lock is not a universal GPU/macOS environment.

Checked-cache replay downloads no images/backbone. It checks checkpoint, splits, fitting inputs, policy, data/model/corruption/prediction code, NPZ and sidecar hashes before regenerating outputs. It performs no fitting or image forwards. Both cached replay and a full fresh-environment rerun (20,000 forwards) reproduced six CSV/decision/figure artifacts byte-for-byte; maximum logit difference was zero. Evidence: `results/verification.json`. The independent verifier recomputes 40 metric sets, per-class counts, 80,000 decisions and 2,020 sweep counts.

For full inference from pixels using the **committed head and locked policies**:

```powershell
python scripts/evaluate_final.py --device cpu --output-dir reproduced/full
python scripts/analyze_failures.py
python demo.py --device cpu
```

The runner downloads/verifies official CIFAR-10 through torchvision and gets ImageNet `IMAGENET1K_V1` weights from [official PyTorch](https://download.pytorch.org/models/resnet18-f37072fd.pth). Each of 2,000 final images is forwarded once under each of ten conditions. Custom output directories keep recomputed logits/metrics separate from the committed run. Failure analysis regenerates the saved deterministic cases from committed decisions without tuning parameters.

Open [the local live demo](http://127.0.0.1:8765), select an example and press **Run classifier**. It runs the backbone and saved head on the displayed pixels and shows all four locked policies. It binds only to this computer; Ctrl+C stops it. Saved slides/panel are an explicitly identified fallback.

Original final timed image processing: **445.99 s for 19,360 images**, excluding the first 64 per condition, archive setup/download and weight loading. The clock includes fetch, native corruption, hashing, preprocessing, backbone and head. The full matrix has 20,000 forwards; elapsed rerun times vary with background load. Seeds and encoder settings are fixed. Cross-device logits may differ: use the existing **1e-4** reload tolerance rather than promising bitwise GPU/MPS agreement.

## Data, model and fitting

[Data instructions](data/README.md) document official source/checksum, class order and seed-42 disjoint source IDs: 10,000 training / 2,000 model validation / 1,500 calibration / 1,500 policy selection / 2,000 official final-test images. Native 32x32 images are independently corrupted before ImageNet bilinear resize 256, center crop 224 and normalization. No augmentation is used. Blur radii are 0.5/1/1.5, noise standard deviations 0.03/0.06/0.12, JPEG qualities 70/40/15.

Raw data/features and pretrained weights stay outside Git. The trained head and checked final-logit bundles are included. Head SHA-256: `9a02afa9f5595437a35dcac020ccdf9fafc1944088948bdd781cb03f92eb107c`. Run **a9d534f5ff01664d** records checkpoint/split/protocol/policy hashes, native-image hashes, conditions, environments, clocks and artifact hashes. Do not regenerate fitting manifests or replace policies when reproducing final results: committed inputs already verify.

The frozen backbone includes frozen BatchNorm. Only the 512-to-10 head is trained with cross-entropy, AdamW lr 0.001, weight decay 0.0001, batch 256 and 30 epochs. Epoch 30 minimizes model-validation NLL, not test error. Original M4 Pro/MPS clocks: 25.45 s feature extraction of 15,000 development images, 0.51 s head training, 2.91 s full inference for 1,936 timed validation images. These clocks are separate from CPU final inference; see [method/runtime](reports/method_runtime.md).

For training/fitting reproduction, use a separate clone and the original baseline environment, so reported final artifacts are preserved:

```bash
python scripts/train_baseline.py
python scripts/train_baseline.py --verify-reload-only
python scripts/fit_policies.py --output-dir artifacts/policy_reproduction
python scripts/analyze_policy_logits.py
```

Training uses training only; model validation selects the epoch. Calibration fits T=**0.9561687801607236**. Policy selection separately chooses raw/calibrated tau=**0.56/0.57** from 101 predefined grid points: maximize coverage subject to accepted accuracy >=90% and coverage >=20%; smallest tau wins ties, fixed 0.80 fallback if infeasible. Acceptance uses confidence >= tau. Neither fitting script evaluates final test. Device-dependent retraining can change the head slightly; use the committed head for published results. [Policy handoff](reports/calibration_rejection.md) details development evidence and interfaces.

## Final findings and metric definitions

A: raw always-predict; B: raw reject; C: calibrated always-predict; D: calibrated reject. Positive T preserves argmax labels. Full accuracy includes all images; accepted accuracy includes only accepted images and always accompanies coverage/counts. With zero acceptance, accuracy/risk are null. NLL and 15-bin equal-width ECE use all images, including rejected predictions. Test threshold sweeps/matched-coverage diagnostics are descriptive and cannot reselect a policy.

| Final condition | Full accuracy A-D | B accepted accuracy / coverage | D accepted accuracy / coverage |
|---|---:|---:|---:|
| Clean | 85.05% | 90.85% / 88.00% | 90.90% / 87.90% |
| Blur radius 1.5 | 22.55% | 32.42% / 36.40% | 31.81% / 38.35% |
| Noise std 0.12 | 16.20% | 17.46% / 45.25% | 17.40% / 46.85% |
| JPEG quality 15 | 44.95% | 61.30% / 46.90% | 60.58% / 47.95% |

Clean ECE changes 0.02011→0.01870; NLL is nearly unchanged. D accepts 1,598 correct among 1,758 accepted images, rejects 103 correct predictions and still accepts 160 errors. Under corruption, fitted T sharpens wrong confidence, worsening ECE/NLL. Raw/calibrated risk-coverage curves nearly overlap. One seed, a small subset and custom corruptions limit generalization; this is not CIFAR-10-C or an OOD/safety guarantee.

Complete severities/per-class results are in `results/`. The `per_image.csv.gz` schema contains run ID, corruption/severity, source ID, true/predicted class, policy, raw/calibrated confidence, threshold, acceptance/correctness and output (`REJECT` or numeric class index). [Three failures](reports/failure_analysis.md) include paired pixels, population counts, likely causes stated as hypotheses and limitations.

## Regenerate PDFs

```bash
python -m pip install -r requirements-report.txt
python tools/render_final_submission.py
```

Edit `reports/technical_summary.md`; its explicit page break divides two required pages. The renderer checks two summary pages/three slides. Slides read committed metric JSON. Inspect every page after changes. Windows uses Calibri; other systems fall back to Helvetica. PDF dependencies are separate from evaluation.

## Team and remaining course actions

Telman Nuruzov (Telman3000): data/splits/corruptions. Leo Vesin (LeoVesinML): model/baseline. Nikita Shankin (Mysteri0K1ng): metrics/calibration/policies. Aleksandr Medvedev (MedvAx-AI): evaluation/figures/failures/demo/report. Coding-assistant support is disclosed in the contribution log and does not certify individual understanding.

Requirements come from user-supplied course PDF/screenshots, which do not independently authorize external actions. The team still confirms topic registration/final deadline/session timing, rehearses twice and completes supervised individual Q&A/course upload. The first-working-model screenshot specifies **13 October 2026**; no final deadline was supplied. Prepared format: five-minute demonstration plus provisional five-minute Q&A.

Primary sources: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [torchvision ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html), [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).
