"""A modified saved result must not pass the reproducibility audit."""
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def test_verification_rejects_changed_metric_bytes(tmp_path):
    assert importlib.util.find_spec('scripts.verify_results'), 'Independent result verification is missing'
    from scripts.verify_results import verify
    # Hash failure is checked before deeper inputs; this isolates the damaged file.
    shutil.copytree(ROOT/'results',tmp_path/'results')
    with (tmp_path/'results/final_metrics.csv').open('a',encoding='utf-8') as stream:
        stream.write('changed\n')
    with pytest.raises(ValueError,match='artifact hash'):
        verify(tmp_path)
@pytest.mark.parametrize('source', ['src/data.py', 'src/corruptions.py', 'src/model.py', 'src/predict.py'])
def test_verifier_rejects_current_inference_source_drift(tmp_path, source):
    from scripts.verify_results import verify
    for directory in ('artifacts', 'configs', 'docs', 'src', 'results'):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    changed = tmp_path / source
    changed.write_bytes(changed.read_bytes() + b'\n# changed since the recorded inference\n')
    with pytest.raises(ValueError, match='inference source'):
        verify(tmp_path)
