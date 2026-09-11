# Dataset retrieval and split plan

Owner: **Telman3000**. This page contains a download starting point and the planned data contract. The final preparation script and environment are not yet implemented.

Use [official CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). The official Python archive is linked there; the source page supplies its MD5 checksum `c58f30108f718f92721af3b95e74349a`. Confirm the source/checksum when implementing retrieval and record a SHA-256 of the actual downloaded archive. Retain the dataset citation and upstream usage information.

After installing the project's tested torchvision dependency, the planned download uses:

```python
from torchvision.datasets import CIFAR10

train_source = CIFAR10(root="data/raw", train=True, download=True)
test_source = CIFAR10(root="data/raw", train=False, download=True)
```

This is an API example, not a completed end-to-end preparation command. Telman3000 must replace/add exact executable commands once `src/data.py` exists and verify them in a fresh environment.

Follow the five-way, seed-42 stratified allocation in [EXPERIMENT_PROTOCOL.md](../docs/EXPERIMENT_PROTOCOL.md#2-dataset-and-leakage-prevention). Use original training IDs to allocate 10,000 head-training, 2,000 model-validation, 1,500 calibration and 1,500 policy-selection images. Select 2,000 final-test images from the official test partition independently. Preserve the remaining images as unused.

Final documentation must include the precise RNG and class order, download directory, file layout, verification commands, generated manifest paths and hashes, actual class/split counts, preprocessing and corruption order. Check that no source ID appears in two partitions and that each corruption retains its source label and partition.

Keep raw datasets, generated image caches, feature caches and large checkpoints out of Git. Commit small split manifests, configuration, source attribution and checksums. Document retrieval/retraining for every externally stored file needed to reproduce the project.
