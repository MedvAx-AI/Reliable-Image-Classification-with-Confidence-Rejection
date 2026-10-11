# Method and runtime

Owner: LeoVesinML. This is the method/runtime record for the development baseline. It is not the final two-page summary, and it does not report final-test numbers. Calibration, rejection thresholds, the locked test matrix, figures, and the demo are later packages.

## Classifier

The backbone is torchvision `resnet18` with `ResNet18_Weights.IMAGENET1K_V1`.

- Weight file retrieved by torchvision: [resnet18-f37072fd.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth)
- The original 1000-way layer is removed. ImageNet class scores are not used as CIFAR-10 labels.
- Backbone parameters are frozen. Every BatchNorm layer stays in evaluation mode, including if `train()` is called on the wrapper, so running statistics do not update.
- The new head is `Linear(512, 10)`.
- Transforms are the weight enum's own `ImageClassification` transform: bilinear resize to 256, center crop to 224, scale to [0, 1], then normalize with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Corruptions, when used later, stay on the native 32×32 image and happen before this transform.

Settings are frozen in `configs/model.yaml` and were not searched. Head training uses cross-entropy, AdamW, learning rate `0.001`, weight decay `0.0001`, feature batch size 256, at most 30 epochs, seed 42. The checkpoint is the lowest model-validation NLL, with the earliest epoch kept on a tie.

Weight updates use the `training` split only (10,000 images). `model_validation` is used to score checkpoints. `calibration` and `policy_selection` are exported as logits and are not fit. `final_test` is refused by the training script.

## Development baseline (condition A)

Saved at `artifacts/baseline/development_baseline.json` on 11 October 2026, before any rejection or temperature policy. Split: clean `model_validation`, 2,000 images, 200 per class. Coverage is 1 because rejection is off.

| Quantity | Value |
|---|---:|
| Selected epoch | 30 of 30 |
| Full-set accuracy | 0.8645 (1729 / 2000) |
| NLL | 0.4245 |
| ECE (15 equal-width bins) | 0.0246 |
| Coverage | 1 |
| Accepted accuracy | 0.8645, same as full-set accuracy |

Validation NLL fell from 0.983 at epoch 1 to 0.424 at epoch 30. Epoch 30 was the minimum, so the cap in the protocol stopped training. The cap was not raised after seeing that curve.

Per-class accuracy on this development split ranges from bird at 0.775 to truck at 0.945. Counts are 200 for every class. These figures choose nothing about the final test.

Checkpoint SHA-256 of `artifacts/model/linear_head.npz`:

`9a02afa9f5595437a35dcac020ccdf9fafc1944088948bdd781cb03f92eb107c`

Logits for `model_validation`, `calibration`, and `policy_selection` are in `artifacts/logits/`. A new process that reloads this head and reruns the backbone on `model_validation` matched the saved logits with max absolute difference `1.14e-5` and identical argmaxes. The declared same-device tolerance is `1e-4`.

BatchNorm buffer hashes were identical before and after feature extraction, and again during the fresh-load inference.

## Calibration isolation

Temperature scaling is not fit here. A positive temperature is a scalar on logits after the head is frozen. On the exported calibration logits, `softmax(z / 1.5)` kept the same argmax as `softmax(z)`, and the head file hash did not change. Mysteri0K1ng's fit must not write this checkpoint. If that fit is numerically invalid, the protocol says to keep T = 1 rather than report a successful temperature.

## Runtime

Measured on macOS 26.5.2 arm64, Apple M4 Pro, 48 GB RAM, Python 3.12.13, torch 2.14.1, torchvision 0.29.1, MPS. Exact fields are in `artifacts/baseline/runtime_manifest.json`.

| Stage | What the clock includes | Measured |
|---|---|---|
| Feature extraction | in-memory CIFAR fetch, resize/crop/normalize, frozen backbone. Batch size 64. | 25.5 s for 15,000 clean development images (10,000 train in 16.7 s, 2,000 validation in 3.2 s, 1,500 calibration in 2.7 s, 1,500 policy in 2.8 s) |
| Head training | AdamW on cached 512-d features, CPU, 30 epochs. No backbone. | 0.51 s |
| Full image inference | same fetch and transforms, backbone, and head, after one warmup batch of 64 | 3.03 s for the remaining 1,936 validation images (639 images/s). A later standalone `--verify-reload-only` pass measured 2.94 s |
| Cached logit evaluation | log-softmax, argmax, and NLL on the 2,000 saved validation logits. No backbone. | 0.10 ms per pass |

A 32-image CPU copy of the same backbone differed from MPS features by at most `3.24e-5`. Refitting the head from the saved feature cache on this machine reproduced the committed weights exactly. A new extraction on another device can move features by about that gap, so the reported baseline should be reloaded from `artifacts/model/linear_head.npz`, not assumed bit-identical after a from-scratch retrain.

## Reproduction

Commands that were run are in the repository README. `--verify-reload-only` reloads the committed head and checks validation logits. It does not train and it does not touch `final_test`.

`python -m pytest tests/test_model.py` passed (12 tests): protocol defaults, refusal of `final_test`, checkpoint tie-break, refusal to fit a held-out split, a real ten-class update, NLL, ECE bins, condition-A coverage, temperature argmax, logit round-trip, and frozen BatchNorm.

The split ID hash `0a731713a64cd147594f806f19eff3e0ceaf7c9db1e1c0f8c116cf25787df2be` is unchanged. The per-file SHA-256 values stored beside those IDs did not match the committed JSON bytes; those five file hashes were corrected so `scripts/prepare_data.py --verify-only` passes. IDs and counts were not regenerated.

## Q&A notes for this package

1. Why this backbone and a new head? CIFAR-10 has ten classes. The ImageNet head has a thousand, so its scores are the wrong label space. Freezing ResNet-18 and training a linear layer keeps the experiment small and makes the checkpoint one shared object for every later condition.
2. What is model validation for? It picks the head checkpoint by NLL. It does not fit temperature and it does not pick the rejection threshold. Those two decisions have their own splits. Final test is unused until those decisions are frozen.
3. What did temperature change in this package? Nothing in the weights. Dividing logits by a positive T cannot change `argmax`. Accuracy of condition A and a later no-rejection calibrated condition should match if the checkpoint is the same.
4. What do the clocks mean? The 3.03 s figure includes the backbone. The 0.10 ms figure does not. Reporting only the cached pass would hide the cost of the network.
5. What is the 0.8645 accuracy? Clean model-validation accuracy of the raw model with no rejection, on 2,000 images. It is the baseline that later conditions must cite. It is not a final-test result and it is not an accepted-accuracy claim under a threshold.
