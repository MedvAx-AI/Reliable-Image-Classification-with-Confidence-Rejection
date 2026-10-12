"""Local live-inference demonstration using the committed head and frozen policies."""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from src.corruptions import corruption_specs
from src.data import Cifar10SplitDataset, download_cifar10, sha256_text_file
from src.model import build_frozen_resnet18, choose_device, load_head_npz
from src.policy_artifacts import load_frozen_policy, read_json
from src.rejection import apply_policy

ROOT = Path(__file__).resolve().parent


def predict_image(image: Image.Image, model, transform, device, policy: dict) -> dict:
    tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor).detach().cpu().numpy()
    decisions = {name: apply_policy(logits, condition=name, temperature=policy['temperature'],
                                   raw_threshold=policy['raw']['threshold'],
                                   calibrated_threshold=policy['calibrated']['threshold']) for name in 'ABCD'}
    predicted = policy['class_order'][int(logits.argmax(1)[0])]
    return {'prediction': predicted, 'logits': logits[0].tolist(),
            'raw_confidence': float(decisions['A']['confidence'][0]),
            'calibrated_confidence': float(decisions['C']['confidence'][0]),
            'temperature': policy['temperature'], 'live_inference': True,
            'policies': {name: {'accepted': bool(d['accepted'][0]), 'threshold': d['threshold'],
                                'confidence': float(d['confidence'][0]),
                                'output': predicted if d['accepted'][0] else 'REJECT'}
                         for name, d in decisions.items()}}


class DemoEngine:
    def __init__(self, device_name: str):
        self.policy = load_frozen_policy()
        cases = read_json(ROOT / 'results/failures/cases.json')
        if cases['policy_sha256'] != sha256_text_file(ROOT / 'artifacts/policy.json'):
            raise ValueError('saved demo examples do not match the frozen policy')
        self.cases = cases['cases']
        download_cifar10(ROOT / 'data/raw')
        self.model, self.transform, _ = build_frozen_resnet18()
        self.model.load_head_arrays(*load_head_npz(ROOT / 'artifacts/model/linear_head.npz'))
        self.device = choose_device(device_name)
        self.model.to(self.device).eval()
        self.bn_hash = self.model.batchnorm_buffer_sha256()
        self.specs = {(s['corruption'],s['severity']): s for s in corruption_specs()}
        self.datasets = {}

    def predict_sample(self, source_id: str, corruption: str, severity: str) -> dict:
        spec = self.specs.get((corruption, severity))
        if spec is None:
            raise ValueError('choose a configured corruption and severity')
        key = corruption, severity
        if key not in self.datasets:
            self.datasets[key] = Cifar10SplitDataset('final_test',corruption=corruption,severity=severity,
                                                    param_value=spec['param_value'])
        try:
            sample = self.datasets[key].get_by_source_id(source_id)
        except KeyError as exc:
            raise ValueError('source ID must belong to the committed final-test subset') from exc
        image = Image.fromarray(sample.image)
        result = predict_image(image,self.model,self.transform,self.device,self.policy)
        if self.model.batchnorm_buffer_sha256() != self.bn_hash:
            raise RuntimeError('frozen model state changed')
        buffer = io.BytesIO()
        image.save(buffer,format='PNG')
        return {**result, 'source_id':source_id, 'corruption':corruption,'severity':severity,
                'true_class':self.policy['class_order'][sample.label],
                'image':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')}


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == '/':
                    data = (ROOT / 'demo/index.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                elif parsed.path == '/api/examples':
                    data = json.dumps(engine.cases,allow_nan=False).encode('utf-8')
                    content_type = 'application/json'
                elif parsed.path == '/api/predict':
                    query = parse_qs(parsed.query)
                    payload = engine.predict_sample(query.get('source_id',[''])[0],
                                                    query.get('corruption',['clean'])[0],
                                                    query.get('severity',['none'])[0])
                    data = json.dumps(payload,allow_nan=False).encode('utf-8')
                    content_type = 'application/json'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store')
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError) as exc:
                data = json.dumps({'error':str(exc)}).encode('utf-8')
                self.send_response(400)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, format, *args):
            return
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--device',default='auto')
    args = parser.parse_args()
    engine = DemoEngine(args.device)
    server = HTTPServer(('127.0.0.1',args.port),make_handler(engine))
    print(f'Live demo ready: http://127.0.0.1:{args.port} (local only)',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
