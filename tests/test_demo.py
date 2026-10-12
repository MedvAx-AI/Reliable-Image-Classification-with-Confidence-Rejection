"""Demo must show frozen policies and actual live model predictions."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_demo_inference_preserves_raw_baseline_and_rejection():
    assert importlib.util.find_spec('demo'), 'Live demo is not implemented'
    from demo import predict_image

    class SmallModel(torch.nn.Module):
        def forward(self, image):
            # Actual input is consumed; a white image yields known two-class logits.
            score = image.mean(dim=(1,2,3))
            return torch.stack((score, torch.zeros_like(score)), dim=1)

    policy = dict(temperature=2, raw={'threshold':.8}, calibrated={'threshold':.8}, class_order=['white','other'])
    transform = lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
    result = predict_image(Image.new('RGB',(32,32),'white'), SmallModel(), transform, torch.device('cpu'), policy)
    assert result['prediction'] == 'white'
    assert result['live_inference'] is True
    assert result['policies']['A']['output'] == 'white'
    assert result['policies']['B']['output'] == 'REJECT'
    assert result['policies']['C']['output'] == 'white'
    assert result['policies']['D']['output'] == 'REJECT'
    assert result['policies']['D']['threshold'] == .8
    assert abs(result['raw_confidence']-.7310586)<1e-6
    assert abs(result['calibrated_confidence']-.6224593)<1e-6
