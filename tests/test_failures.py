"""Failure examples must be real paired mistakes, not selected by row position."""
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_failure_selection_catches_three_distinct_failure_types():
    assert importlib.util.find_spec('src.failures'), 'Failure analysis is not implemented'
    from src.failures import select_failures
    common = dict(condition='D', corruption='clean', severity='none', true_class=0, active_threshold=.57)
    rows = [{**common, 'source_id':'test:1', 'predicted_class':1, 'accepted':True,
             'correct':False, 'calibrated_confidence':.99},
            {**common, 'source_id':'test:2', 'predicted_class':0, 'accepted':False,
             'correct':True, 'calibrated_confidence':.56},
            {**common, 'source_id':'test:3', 'predicted_class':0, 'accepted':True,
             'correct':True, 'calibrated_confidence':.92},
            {**common, 'source_id':'test:3', 'corruption':'gaussian_blur', 'severity':'severe',
             'predicted_class':1, 'accepted':True, 'correct':False, 'calibrated_confidence':.97}]
    result = select_failures(rows)
    assert [c['row']['source_id'] for c in result['cases']] == ['test:1','test:2','test:3']
    assert result['cases'][2]['clean']['predicted_class'] == 0
    assert result['cases'][2]['row']['predicted_class'] == 1
    assert not result['missing_categories']
    assert select_failures(list(reversed(rows))) == result


def test_absent_failure_category_is_disclosed():
    assert importlib.util.find_spec('src.failures'), 'Failure analysis is not implemented'
    from src.failures import select_failures
    result = select_failures([dict(condition='D',corruption='clean',severity='none',source_id='test:1',
                                  correct=True,accepted=True,calibrated_confidence=.95)])
    assert result['cases'] == []
    assert len(result['missing_categories']) == 3
