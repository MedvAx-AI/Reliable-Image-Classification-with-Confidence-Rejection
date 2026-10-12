"""Evaluate the fixed ten image conditions with four frozen policies."""
from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path
from typing import Any

import numpy as np

from src.corruptions import corruption_specs
from src.rejection import evaluate_policies, sensitivity_analysis


def condition_key(item: dict) -> str:
    return f"{item['corruption']}_{item['severity']}"


def validate_matrix(bundles: list[dict]) -> None:
    expected = [condition_key(s) for s in corruption_specs()]
    keys = [condition_key(b) for b in bundles]
    if keys != expected:
        raise ValueError('matrix must contain all ten unique conditions in protocol order')
    ids, labels = bundles[0]['source_ids'], np.asarray(bundles[0]['labels'])
    if not ids or len(set(ids)) != len(ids) or labels.shape != (len(ids),):
        raise ValueError('source IDs and labels must be nonempty, unique and aligned')
    for bundle in bundles:
        if bundle['source_ids'] != ids or not np.array_equal(bundle['labels'], labels):
            raise ValueError('all variants must have the identical ordered source IDs and labels')


def evaluate_matrix(bundles: list[dict], *, temperature: float, raw_threshold: float,
                    calibrated_threshold: float, class_order: list[str]) -> dict[str, Any]:
    validate_matrix(bundles)
    rows, per_image, sweeps, diagnostics = [], [], [], {}
    for bundle in bundles:
        identity = {k: bundle[k] for k in ('corruption', 'severity')}
        comparison = evaluate_policies(bundle['logits'], bundle['labels'], temperature=temperature,
                                       raw_threshold=raw_threshold, calibrated_threshold=calibrated_threshold,
                                       class_order=class_order)
        rows.extend({**identity, **row} for row in comparison['metrics'])
        sensitivity = sensitivity_analysis(bundle['logits'], bundle['labels'], bundle['source_ids'],
                                           temperature=temperature)
        diagnostics[condition_key(bundle)] = sensitivity
        for name in ('raw', 'calibrated'):
            sweeps.extend({**identity, 'confidence': name, **row}
                          for row in sensitivity[name]['threshold_sweep'])
        for i, source_id in enumerate(bundle['source_ids']):
            label = int(bundle['labels'][i])
            for condition, decision in comparison['decisions'].items():
                prediction = int(decision['latent_predictions'][i])
                per_image.append({**identity, 'source_id': source_id, 'true_class': label,
                                  'predicted_class': prediction, 'condition': condition,
                                  'raw_confidence': float(decision['raw_confidence'][i]),
                                  'calibrated_confidence': float(decision['calibrated_confidence'][i]),
                                  'active_threshold': decision['threshold'],
                                  'accepted': bool(decision['accepted'][i]), 'correct': prediction == label,
                                  'output': decision['output'][i]})
    return {'metrics': rows, 'per_image': per_image, 'threshold_sweep': sweeps,
            'sensitivity': diagnostics}


def write_csv(path: Path, rows: list[dict], columns: list[str], identity: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        raw = path.open('wb')
        stream = io.TextIOWrapper(gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0),
                                  encoding='utf-8', newline='')
    else:
        raw = None
        stream = path.open('w', encoding='utf-8', newline='')
    try:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows({**(identity or {}), **row} for row in rows)
    finally:
        stream.close()
        if raw is not None:
            raw.close()


METRIC_COLUMNS = ['run_id', 'checkpoint_sha256', 'policy_sha256', 'corruption', 'severity', 'condition',
                  'temperature', 'threshold', 'n', 'correct', 'full_set_accuracy', 'accepted_count',
                  'correct_accepted_count', 'coverage', 'rejection_rate', 'accepted_accuracy',
                  'selective_risk', 'nll', 'ece', 'ece_bins']
PER_IMAGE_COLUMNS = ['run_id', 'corruption', 'severity', 'source_id', 'true_class', 'predicted_class',
                     'condition', 'raw_confidence', 'calibrated_confidence', 'active_threshold',
                     'accepted', 'correct', 'output']
SWEEP_COLUMNS = ['run_id', 'corruption', 'severity', 'confidence', 'threshold', 'n', 'correct',
                 'full_set_accuracy', 'accepted_count', 'correct_accepted_count', 'coverage',
                 'rejection_rate', 'accepted_accuracy', 'selective_risk']
