"""Run the frozen 4 x 10 final-test experiment or regenerate checked cached results."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.corruptions import corruption_specs, image_sha256
from src.data import Cifar10SplitDataset, download_cifar10, load_config, sha256_file, sha256_json, sha256_text_file
from src.evaluate import (condition_key, evaluate_matrix, write_csv, METRIC_COLUMNS,
                          PER_IMAGE_COLUMNS, SWEEP_COLUMNS)
from src.evaluation_provenance import inference_code_hashes, validate_inference_provenance
from src.model import build_frozen_resnet18, choose_device, load_head_npz, save_array_artifact
from src.plots import plot_final_results
from src.policy_artifacts import current_provenance, load_checked_logits, load_frozen_policy, read_json, write_json
from src.predict import infer_split_logits


def environment(device: str) -> dict:
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'numpy', 'pillow', 'PyYAML', 'pandas', 'matplotlib', 'pytest')}
    return {'os': platform.platform(), 'python': platform.python_version(), 'packages': packages,
            'device': device, 'torch_threads': torch.get_num_threads(), 'processor': platform.processor()}


class AuditedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.image_hashes = [None] * len(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        self.image_hashes[index] = image_sha256(sample.image)
        return sample


def run(*, output_dir: Path, from_cache: bool, device_name: str, batch_size: int) -> dict:
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    # Prove the lock and fitting provenance before reading any final-test image/logit.
    policy = load_frozen_policy()
    provenance, manifest = current_provenance()
    frozen_files = [ROOT / 'artifacts/policy.json', ROOT / 'artifacts/temperature.json',
                    ROOT / 'artifacts/model/linear_head.npz']
    frozen_before = [sha256_file(p) for p in frozen_files]
    policy_hash = sha256_text_file(frozen_files[0])
    run_id = sha256_json({'provenance': provenance, 'policy': policy_hash, 'split': 'final_test',
                         'corruptions': corruption_specs()})[:16]
    identity = {'run_id': run_id, **provenance, 'policy_sha256': policy_hash}
    output_dir.mkdir(parents=True, exist_ok=True)
    logit_dir = ROOT / 'results/logits' if from_cache else output_dir / 'logits'
    setup_started = time.perf_counter()
    model, transform, device, bn_before = None, None, None, None
    cfg = load_config()
    if not from_cache:
        download_cifar10(ROOT / cfg['dataset']['root'], download=True)
        model, transform, model_provenance = build_frozen_resnet18()
        weight, bias = load_head_npz(frozen_files[2])
        model.load_head_arrays(weight, bias)
        device = choose_device(device_name)
        model.to(device).eval()
        bn_before = model.batchnorm_buffer_sha256()
    setup_seconds = time.perf_counter() - setup_started
    bundles, clocks = [], []
    code_hashes = inference_code_hashes(ROOT)
    original_records = read_json(ROOT / 'results/run_manifest.json')['inference_records'] if from_cache else None
    for spec in corruption_specs(cfg):
        key = condition_key(spec)
        path = logit_dir / f'final_test_{key}.npz'
        if from_cache:
            original = next(r for r in original_records if r['condition'] == key)
            if sha256_file(path) != original['logit_sha256'] or sha256_text_file(path.with_suffix('.json')) != original['sidecar_sha256']:
                raise ValueError('cached inference artifacts changed since the recorded run')
            data = load_checked_logits(path, expected_split='final_test', manifest=manifest, provenance=provenance)
            meta = data['metadata']
            if (meta['corruption'], meta['severity']) != (spec['corruption'], spec['severity']):
                raise ValueError('cached condition does not match its filename')
            validate_inference_provenance(meta, provenance, policy_hash, code_hashes)
        else:
            print(f"Inferring {key}: 2000 final-test images ...", flush=True)
            dataset = AuditedDataset(Cifar10SplitDataset('final_test', config=cfg, manifest=manifest,
                                    corruption=spec['corruption'], severity=spec['severity'], param_value=spec['param_value']))
            started = time.perf_counter()
            data = infer_split_logits(model, dataset, transform, device, batch_size=batch_size,
                                      time_it=True, warmup_batches=1)
            if data['source_ids'] != manifest['split_ids']['final_test']:
                raise ValueError('inference reordered source IDs')
            expected_labels = [r['label'] for r in manifest['splits']['final_test']]
            if data['labels'].tolist() != expected_labels:
                raise ValueError('inference labels differ from the manifest')
            meta = {**provenance, 'policy_sha256': policy_hash, 'code_hashes': code_hashes,
                    'split': 'final_test', 'final_test': True, 'n': len(dataset), **spec,
                    'batch_size': batch_size, 'warmup_batches_excluded': 1,
                    'timed_seconds': data['timed_seconds'], 'timed_images': data['timed_images'],
                    'condition_wall_seconds': time.perf_counter() - started,
                    'image_sha256': dataset.image_hashes, 'image_hashes_sha256': sha256_json(dataset.image_hashes),
                    'environment': environment(str(device)), 'model_provenance': model_provenance,
                    'saved_utc': datetime.now(timezone.utc).isoformat(),
                    'timing_includes': data['includes'] + ['native corruption and image hashing'],
                    'timing_excludes': ['archive loading', 'dataset download', 'weight loading', 'one warmup batch']}
            save_array_artifact(path, arrays={'logits': data['logits'], 'labels': data['labels'],
                                             'source_ids': np.asarray(data['source_ids'])}, metadata=meta)
            # Re-read actual bytes through the same checked handoff consumed in reproduction.
            data = load_checked_logits(path, expected_split='final_test', manifest=manifest, provenance=provenance)
        bundles.append({**spec, **data})
        clocks.append({'condition': key, 'seconds': meta['timed_seconds'], 'images': meta['timed_images'],
                       'environment': meta['environment'], 'logit_sha256': sha256_file(path),
                       'sidecar_sha256': sha256_text_file(path.with_suffix('.json'))})
    if model is not None and model.batchnorm_buffer_sha256() != bn_before:
        raise RuntimeError('frozen BatchNorm changed during final inference')
    started = time.perf_counter()
    result = evaluate_matrix(bundles, temperature=policy['temperature'], raw_threshold=policy['raw']['threshold'],
                             calibrated_threshold=policy['calibrated']['threshold'], class_order=policy['class_order'])
    policy_seconds = time.perf_counter() - started
    write_json(output_dir / 'final_metrics.json', {**identity, 'metrics': result['metrics']})
    write_json(output_dir / 'sensitivity.json', {**identity, 'conditions': result['sensitivity']})
    write_csv(output_dir / 'final_metrics.csv', result['metrics'], METRIC_COLUMNS, identity)
    write_csv(output_dir / 'per_image.csv.gz', result['per_image'], PER_IMAGE_COLUMNS, identity)
    write_csv(output_dir / 'threshold_sweep.csv', result['threshold_sweep'], SWEEP_COLUMNS, identity)
    plot_final_results(result, output_dir / 'figures')
    if [sha256_file(p) for p in frozen_files] != frozen_before:
        raise RuntimeError('frozen model or policy artifacts changed')
    paths = ['final_metrics.json', 'final_metrics.csv', 'sensitivity.json', 'per_image.csv.gz',
             'threshold_sweep.csv', 'figures/reliability.png', 'figures/risk_coverage.png', 'figures/accuracy_coverage.png']
    record = {**identity, 'split': 'final_test', 'n_source_images': 2000,
              'n_image_conditions': 10, 'n_policy_conditions': 4, 'aggregate_rows': 40,
              'per_image_rows': 80000, 'actual_image_forwards_this_run': 0 if from_cache else 20000,
              'from_checked_cache': from_cache, 'setup_seconds_this_run': setup_seconds,
              'cached_policy_evaluation_seconds': policy_seconds,
              'reproduction_environment': environment('cpu' if from_cache else str(device)),
              'inference_records': clocks, 'total_timed_image_seconds': sum(r['seconds'] for r in clocks),
              'total_timed_images': sum(r['images'] for r in clocks),
              'batchnorm_unchanged': True, 'frozen_files_unchanged': True,
              'no_policy_refitting': True, 'sensitivity_is_descriptive': True,
              'artifacts_sha256': {p: sha256_file(output_dir / p) for p in paths},
              'saved_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output_dir / 'run_manifest.json', record)
    print(f"Saved {output_dir}: 40 rows, 80000 policy decisions; no policy refitting", flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-cache', action='store_true', help='regenerate only from committed checked final logits')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    run(output_dir=args.output_dir, from_cache=args.from_cache, device_name=args.device, batch_size=args.batch_size)
