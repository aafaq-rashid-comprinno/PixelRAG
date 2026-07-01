#!/usr/bin/env python3
"""Pre-validate images in a JSONL file and write a clean version.

Usage:
    python validate_images.py input.jsonl output_clean.jsonl [--workers 8]

Only rows with valid positive images (exists + opens correctly) are kept.
Hard negative paths are checked for existence only (no image verify).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def validate_row(args):
    """Validate a single JSONL row. Returns (line, is_valid)."""
    line, jsonl_dir = args
    item = json.loads(line)

    # Resolve positive path
    path = Path(item["chunk_path"])
    if not path.is_absolute():
        path = (jsonl_dir / path).resolve()
    pos_path = str(path)

    if not os.path.exists(pos_path):
        return line, False, "missing"
    try:
        with Image.open(pos_path) as im:
            im.convert("RGB").verify()
    except Exception:
        return line, False, "corrupt"

    # Check neg paths exist (no image verify — too slow and rarely corrupt)
    if "neg_chunk_paths" in item:
        valid_negs = []
        for np_ in item["neg_chunk_paths"]:
            np_path = Path(np_)
            if not np_path.is_absolute():
                np_path = (jsonl_dir / np_path).resolve()
            if os.path.exists(str(np_path)):
                valid_negs.append(np_)
        item["neg_chunk_paths"] = valid_negs
        return json.dumps(item, ensure_ascii=False) + "\n", True, "ok"

    return line, True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output clean JSONL file")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    jsonl_dir = Path(args.input).resolve().parent
    with open(args.input) as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Validating {total} rows from {args.input} with {args.workers} workers...")

    valid_lines = []
    missing = corrupt = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(validate_row, (line, jsonl_dir)) for line in lines]
        for i, fut in enumerate(as_completed(futures)):
            result_line, is_valid, reason = fut.result()
            if is_valid:
                valid_lines.append(result_line)
            elif reason == "missing":
                missing += 1
            else:
                corrupt += 1
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{total} checked, {len(valid_lines)} valid")

    with open(args.output, "w") as f:
        f.writelines(valid_lines)

    passed = len(valid_lines)
    failed = total - passed
    print(
        f"\nDone: {passed}/{total} valid ({failed} failed: {missing} missing, {corrupt} corrupt)"
    )
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
