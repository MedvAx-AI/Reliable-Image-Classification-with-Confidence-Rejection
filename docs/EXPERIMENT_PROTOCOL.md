# Experimental protocol

All numeric settings here are **proposed team defaults**, not course requirements or measured results. Freeze them at M2 before final test evaluation. Record and justify any earlier changes. Never tune on final test results.

## 1. Question, hypothesis, and success criteria

Question: for a fixed image classifier, how does confidence rejection change accepted error versus coverage, and how does temperature calibration affect confidence quality on clean, blurred, noisy, and compressed images?

Hypotheses to test: rejection may reduce error among accepted images at the cost of coverage; temperature scaling may improve clean calibration; corruption may weaken either effect. None is assumed true. Success means a reproducible, controlled answer with quantitative evidence and explained failures, not achieving a promised accuracy improvement.

## 2. Dataset and leakage prevention

Use [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), which contains 60,000 RGB 32 × 32 images in ten classes, with official 50,000 training and 10,000 test partitions. Cite the original dataset report in the final summary.

Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Use a fixed class-balanced subset for a small course experiment. Seed: **42**. Shuffle original indices within each class using a recorded NumPy RNG/version. Allocate disjoint consecutive blocks from the official training partition:

| Use | Total | Per class | Permitted decisions |
|---|---:|---:|---|
| Head training | 10,000 | 1,000 | Fit classifier head weights |
| Model validation | 2,000 | 200 | Choose head checkpoint and development settings |
| Temperature calibration | 1,500 | 150 | Fit temperature only |
| Policy selection | 1,500 | 150 | Select raw/calibrated rejection thresholds only |
| Final test, from official test partition | 2,000 | 200 | Final reporting only |

Thus 15,000 official training examples and 2,000 official test examples are used. Remaining images are unused; record them as such. Publish index manifests, class order, checksums, and actual sizes. Do not reshuffle until results look good. Development previews and integration checks use model-validation examples, not final test examples.

Use stable IDs such as `train:1234` and `test:5678`. Assert pairwise split disjointness and no source-ID overlap. Generate corruptions only after splitting; variants stay with their source partition. Never fit preprocessing statistics, checkpoint settings, temperature, or thresholds on final test data. Record external pretraining provenance separately; these checks prevent experiment-induced leakage but do not establish that external pretraining datasets are duplicate-free.

## 3. Fixed classifier and baseline

Proposed stack: Python, PyTorch/torchvision, NumPy, Pillow, pandas, matplotlib, and pytest for targeted correctness checks. LeoVesinML records a tested compatible version set rather than assuming that a dependency list is already runnable.

Use [torchvision ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18) with explicit `ResNet18_Weights.IMAGENET1K_V1`. Remove its original classifier, freeze the backbone parameters, and keep the backbone in evaluation mode so BatchNorm statistics are fixed. Cache its 512-dimensional features; train a new ten-class linear head. Do not treat ImageNet's original class outputs as CIFAR-10 labels.

Use that weight enum's deterministic transforms: bilinear resize to 256, center crop to 224, scale to [0,1], then normalize by mean [0.485,0.456,0.406] and standard deviation [0.229,0.224,0.225]. Serialize the actual transform configuration in the run manifest. Apply all corruptions to the native RGB image before the model's resizing/normalization. No random augmentation in the core experiment.

Proposed head training: cross-entropy, AdamW, learning rate 0.001, weight decay 0.0001, feature batch size 256, maximum 30 epochs, seed 42. Select the lowest model-validation NLL checkpoint, breaking ties by earliest epoch. Backbone extraction batch size starts at 64 and can be reduced for memory; log its final value. Tune only in development and freeze the final settings at M2.

First save the development **raw-confidence/no-rejection baseline**. Then freeze the chosen checkpoint for all comparisons. The second condition is a changed decision/calibration policy; training another network is unnecessary.

## 4. Confidence and calibration

For logits z, raw class probabilities are `softmax(z)`. Prediction is `argmax(z)` and confidence is the largest class probability.

Temperature scaling uses `softmax(z / T)` with a single positive scalar T. Optimize `log(T)` starting at 0 on the calibration split by minimizing NLL; record optimizer, convergence, temperature, and objective before/after. Keep model weights frozen. If fitting is numerically invalid or worsens the fitting objective beyond tolerance, flag the failure and use the explicit T=1 baseline rather than reporting a successful fit.

Temperature scaling is a lightweight post-processing method studied by [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html). Our experiment tests its effect here. A positive scalar temperature preserves class argmax, so full-set classification accuracy should match before/after scaling. It does not guarantee good calibration under corruption or better ordering of images by confidence.

## 5. Conditions and controlled comparisons

| ID | Confidence | Rejection | Purpose |
|---|---|---|---|
| A | Raw, T=1 | Disabled | Baseline, 100% coverage |
| B | Raw, T=1 | Enabled | Isolate effect of rejecting low confidence: compare A vs B |
| C | Temperature-scaled | Disabled | Isolate calibration: compare A vs C |
| D | Temperature-scaled | Enabled | Full method; compare C vs D and B vs D with coverage disclosed |

Use the identical checkpoint, source images, corruption bytes, preprocessing, and metric definitions. Compare B and D additionally at identical predeclared numeric thresholds and with descriptive risk–coverage curves; separately selected operating thresholds can have different coverage. Do not attribute a difference solely to calibration when the compared accepted populations differ.

## 6. Rejection thresholds and sensitivity

Accept if `confidence >= tau`; otherwise return **REJECT**. Always retain the latent predicted class for analysis, but the user-facing output for rejected samples is rejection. A rejection is not an eleventh ground-truth class.

Choose one raw and one calibrated operating threshold on the **clean policy-selection split only**. Use the same fixed grid `tau = 0.00, 0.01, ..., 1.00`. Target: empirical accepted accuracy at least **90%** and coverage at least **20%**. These are an illustrative operating requirement, not an instructor target or statistical guarantee.

Among feasible thresholds choose maximum coverage; break equal-coverage ties by the smallest tau. Save tau, accepted/correct counts, empirical accuracy, coverage, split hash, and selection rule. A and C always accept all images independently of this rule.

If there is no feasible threshold, mark the target **infeasible on the policy split**. Use predeclared `tau=0.80` only as a labeled fallback tradeoff/demo operating point; report its actual validation/test metrics and do not describe it as satisfying the target. Do not lower the target after viewing test outcomes.

Lock T and both thresholds before final test. Apply the same clean-fitted values to every corruption and severity with no adaptation. This measures transfer under image degradation.

**Required sensitivity experiment:** evaluate the fixed threshold grid on clean and corrupted test predictions and plot selective risk versus coverage. This is descriptive sensitivity analysis, not test-based policy selection. A secondary comparison at top 50%, 80%, and 100% confidence coverage may help compare ranking; label it a retrospective ranking diagnostic, not a deployable threshold. Break confidence ties deterministically by sample ID.

## 7. Blur, noise, and compression

These are custom controlled corruptions, not a claim to reproduce the CIFAR-10-C benchmark. Each test image has one clean version and nine corrupted versions. Apply each corruption independently; do not stack types or severities.

| Family | Implementation at native 32 × 32 resolution | Mild / medium / severe |
|---|---|---|
| Blur | Pillow `ImageFilter.GaussianBlur(radius=r)`; log Pillow version | r = 0.5 / 1.0 / 1.5 pixels |
| Noise | Independent Gaussian noise per RGB channel in [0,1], then clip and round to uint8 | std = 0.03 / 0.06 / 0.12 |
| JPEG | Encode and decode in memory, RGB, `subsampling=0`, `optimize=False`, `progressive=False` | quality = 70 / 40 / 15 |

Derive deterministic noise seeds from SHA-256 of `seed:split:source_id:family:severity`, using the first eight bytes as an unsigned integer; never use Python's randomized `hash()`. Log corruption parameters and output hashes. Freeze values after visual checks on development images.

The final matrix contains **4 policy conditions × 10 image conditions = 40 result rows**, each based on 2,000 images. Only **20,000 image forwards** are needed for the final test suite because A–D share logits. Head fitting, feature extraction for development subsets, and calibration are additional work. Avoid rerunning the backbone for each tau or policy.

## 8. Metrics and reporting conventions

Use fractions in stored data, percentages when labeled in presentation. All counts refer to the relevant image condition, not the number of classes.

| Metric | Definition | Interpretation |
|---|---|---|
| Full-set top-1 accuracy | Correct latent predictions / N | Classification quality before rejection |
| Coverage | Accepted / N | Fraction receiving a class prediction |
| Rejection rate | 1 − coverage | Fraction deferred |
| Accepted accuracy | Correct accepted / accepted | Accuracy on retained predictions; always report coverage and count |
| Selective risk | 1 − accepted accuracy | Error among retained predictions |
| NLL | Mean `−log p(true class)` on all N images | Probability quality, lower is better; use stable log-softmax |
| ECE | Sum over 15 fixed equal-width confidence bins of `(bin_count/N) × abs(bin_accuracy − bin_mean_confidence)` | Binned calibration gap, lower is better; bin-dependent |

For ECE, use bins `[0,1/15)`, ..., `[14/15,1]`; omit empty-bin contributions. Include confidence 1 in the last bin. Compute NLL and primary ECE over **all images**, including rejected images, to avoid presenting an apparently improved score caused solely by removing difficult cases. If accepted-only ECE is added, label it separately with counts.

If accepted count is zero, coverage=0 and accepted accuracy/risk are **null/undefined**, never 100% or 0% accuracy. For A/C, coverage=1 and accepted accuracy equals full-set accuracy. Include clean per-class counts/accuracy and inspect severe-corruption class differences so aggregate metrics do not conceal a class with near-zero coverage.

Save all 40 rows plus threshold-sweep results. Main report table: clean and severe blur/noise/JPEG, A–D, with full accuracy, ECE, accepted accuracy and coverage; NLL and complete severity results remain in the repository if they do not fit two pages. Add one compact risk–coverage panel if space permits. Clearly label any percentage-point difference.

Optional, after the core checklist is complete: paired bootstrap uncertainty using source-image IDs as the sampling unit, keeping variants together. Multiple corruptions of one image are not independent original samples. Do not expand to extra models before required work is finished.

## 9. Shared interfaces and artifacts

- Sample: `source_id`, `split`, `label` (0–9), `corruption`, `severity`, RGB image.
- Logit artifact: ordered `source_ids`, labels of shape `[N]`, finite logits `[N,10]`, class order, model/config/split hashes. The same ID order must align all arrays.
- Policy artifact: `temperature`, raw/calibrated thresholds, feasibility flags, target values, selection split hash, counts, fit settings, checkpoint hash.
- Per-image result: run ID, source ID, true/predicted class, corruption/severity, condition A–D, raw/calibrated confidence, active threshold, accepted, correct.
- Aggregate row: run/model/config identifiers, condition, corruption/severity, N, accepted count, correct accepted count, all named metrics, measured runtime reference.
- Runtime manifest: OS, Python and package versions, CPU/GPU, RAM/VRAM, accelerator/runtime versions if applicable, seeds, elapsed training/feature extraction/calibration/evaluation time, inference batch size and latency method.

Report end-to-end image inference separately from cached-logit policy evaluation. Warm up, synchronize GPU timing where applicable, and state whether decoding/transforms are included. Measure actual times; do not present planning estimates as measurements.

## 10. Failure analysis and verification

Inspect at least three real failures. Prefer: a confidently wrong accepted image, a correct latent prediction rejected unnecessarily, and an image whose clean prediction/decision becomes harmful after corruption. If a category is absent, state that and substitute another observed failure.

For each: show source ID and image/clean counterpart, true and predicted labels, raw/calibrated confidence, T, tau, accept/reject decision, corruption settings, likely cause, evidence, and a limitation or next test. Plausible causes include lost discriminative detail or ambiguity; do not assert a model's internal reasoning from an image alone. Record the selection method and counts to distinguish examples from prevalence.

Meaningful checks: disjoint split IDs; deterministic corruptions; aligned labels/logits; probability normalization; temperature preserves argmax; tiny hand-computed metric examples; exact threshold boundary; zero accepted samples; all-accepted equivalence; reproducible final metrics in a clean environment. Check invariants before the locked test run. If a bug forces a rerun, record what changed and why; do not silently optimize the policy on observed failures.
