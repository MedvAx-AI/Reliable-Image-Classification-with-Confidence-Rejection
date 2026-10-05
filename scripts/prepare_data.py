#!/usr/bin/env python3
"""Prepare CIFAR-10 splits and corruption QA panel (Telman3000 / Issue #1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_config, load_manifest, manifests_dir_from_config, prepare_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override manifests output directory (default: config paths.manifests_dir)",
    )
    parser.add_argument(
        "--no-panel",
        action="store_true",
        help="Skip writing the development corruption sample panel",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-check an existing manifest directory (records + file hashes)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifests_dir = args.out_dir or manifests_dir_from_config(cfg)

    if args.verify_only:
        manifest = load_manifest(manifests_dir=manifests_dir, config=cfg, verify=True)
        print(
            json.dumps(
                {
                    "ok": True,
                    "manifests_dir": str(manifests_dir),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "counts": {k: v["total"] for k, v in manifest["counts"].items()},
                },
                indent=2,
            )
        )
        return 0

    result = prepare_all(
        config=cfg,
        out_dir=args.out_dir,
        write_panel=not args.no_panel,
    )
    print(json.dumps(result, indent=2))
    print(f"\nDone. Manifests are in {result['manifests_dir']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
