"""Catch condition/row drift and incorrect policy population accounting."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluator():
    assert importlib.util.find_spec('src.evaluate'), 'Frozen evaluation is not implemented'
    return importlib.import_module('src.evaluate')


def bundles():
    # Two examples: first confidently correct, second confidently wrong.
    from src.corruptions import corruption_specs
    return [{**spec, 'source_ids': ['test:1', 'test:2'], 'labels': np.array([0, 0]),
             'logits': np.log([[0.95, 0.05], [0.20, 0.80]])}
            for spec in corruption_specs()]


def test_matrix_preserves_denominators_and_rejection_counts():
    ev = evaluator()
    result = ev.evaluate_matrix(bundles(), temperature=1, raw_threshold=.9,
                                calibrated_threshold=.9, class_order=['first', 'second'])
    assert len(result['metrics']) == 40
    assert len(result['per_image']) == 80
    a, b, c, d = result['metrics'][:4]
    assert a['n'] == 2 and a['full_set_accuracy'] == .5 and a['coverage'] == 1
    assert b['accepted_count'] == 1 and b['correct_accepted_count'] == 1
    assert b['coverage'] == .5 and b['accepted_accuracy'] == 1
    assert b['nll'] == a['nll'] and b['ece'] == a['ece']
    assert c['full_set_accuracy'] == a['full_set_accuracy']
    rejected = next(r for r in result['per_image'] if r['source_id'] == 'test:2'
                    and r['condition'] == 'B' and r['corruption'] == 'clean')
    assert rejected['output'] == 'REJECT'
    assert d['coverage'] == .5


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'ids', 'labels'])
def test_matrix_rejects_misaligned_or_incomplete_conditions(mutation):
    ev = evaluator()
    data = bundles()
    if mutation == 'missing':
        data.pop()
    elif mutation == 'duplicate':
        data[-1] = data[0]
    elif mutation == 'ids':
        data[-1]['source_ids'] = ['test:2', 'test:1']
    else:
        data[-1]['labels'] = np.array([0, 1])
    with pytest.raises(ValueError):
        ev.evaluate_matrix(data, temperature=1, raw_threshold=.9,
                           calibrated_threshold=.9, class_order=['first', 'second'])


def test_matrix_zero_acceptance_is_null():
    ev = evaluator()
    result = ev.evaluate_matrix(bundles(), temperature=1, raw_threshold=1,
                                calibrated_threshold=1, class_order=['first', 'second'])
    assert result['metrics'][1]['coverage'] == 0
    assert result['metrics'][1]['accepted_accuracy'] is None
    assert result['metrics'][1]['selective_risk'] is None
