"""Split-file hashes survive a Windows CR LF checkout."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import sha256_file, sha256_text_file, verify_manifest_directory


def _copy_splits(tmp_path: Path) -> Path:
    dst = tmp_path / "splits"
    shutil.copytree(REPO_ROOT / "artifacts" / "splits", dst)
    return dst


def test_crlf_checkout_matches_stored_lf_hash(tmp_path):
    dst = _copy_splits(tmp_path)
    train = dst / "training.json"
    original = train.read_bytes()
    assert b"\r" not in original
    train.write_bytes(original.replace(b"\n", b"\r\n"))
    assert sha256_text_file(train) != sha256_file(train)
    manifest = verify_manifest_directory(dst)
    assert manifest["counts"]["training"]["total"] == 10000


def test_crlf_does_not_hide_an_edited_manifest(tmp_path):
    dst = _copy_splits(tmp_path)
    train = dst / "training.json"
    edited = train.read_bytes().replace(b"\n", b"\r\n").replace(b'"count": 10000', b'"count": 10001', 1)
    train.write_bytes(edited)
    with pytest.raises(AssertionError, match="hash mismatch"):
        verify_manifest_directory(dst)
