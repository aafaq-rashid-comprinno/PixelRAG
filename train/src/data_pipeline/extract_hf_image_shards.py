#!/usr/bin/env python3
"""Extract tar-sharded HF dataset images into an `images/` directory."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--shards-dir",
        type=Path,
        default=None,
        help="Directory containing .tar image shards (defaults to <dataset-dir>/image_shards).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Extraction target for images (defaults to <dataset-dir>/images).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    shards_dir = (
        args.shards_dir.resolve() if args.shards_dir else dataset_dir / "image_shards"
    )
    output_dir = (
        args.output_dir.resolve() if args.output_dir else dataset_dir / "images"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(shards_dir.glob("*.tar"))
    for shard_path in shard_paths:
        with tarfile.open(shard_path, mode="r") as tar:
            tar.extractall(output_dir)
        print(f"Extracted {shard_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
