"""Shared source/policy checks for recorded final inference."""
from pathlib import Path

from src.data import sha256_text_file

INFERENCE_SOURCES = ('src/data.py', 'src/corruptions.py', 'src/model.py', 'src/predict.py')


def inference_code_hashes(root: Path) -> dict:
    return {path: sha256_text_file(root / path) for path in INFERENCE_SOURCES}


def validate_inference_provenance(metadata: dict, provenance: dict, policy_hash: str,
                                  code_hashes: dict) -> None:
    if metadata.get('code_hashes') != code_hashes:
        raise ValueError('recorded inference source implementation changed')
    if metadata.get('data_config_sha256') != provenance['data_config_sha256']:
        raise ValueError('recorded inference data/preprocessing settings changed')
    if metadata.get('policy_sha256') != policy_hash:
        raise ValueError('policy changed since recorded final inference')
