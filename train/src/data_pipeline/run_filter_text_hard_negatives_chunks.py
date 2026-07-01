#!/usr/bin/env python3
"""Run text false-negative filtering in fixed-size chunks.

Each chunk is written to its own folder so long runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input JSONL with retrieve_top20"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Base directory for chunk outputs"
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Rows per chunk")
    parser.add_argument(
        "--start-offset", type=int, default=0, help="Initial row offset"
    )
    parser.add_argument(
        "--limit-total",
        type=int,
        default=0,
        help="Optional max rows to process overall",
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--num-hard-negatives", type=int, default=7)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument(
        "--gemini", action="store_true", help="Alias for --provider gemini."
    )
    parser.add_argument("--gemini-project", default="wise-coyote-478119-h0")
    parser.add_argument("--gemini-location", default="global")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip chunk folders that already contain summary.json",
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def chunk_dir(base: Path, start: int, end: int) -> Path:
    return base / f"chunk_{start:06d}_{end:06d}"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = count_lines(input_path)
    total_target = total_rows - args.start_offset
    if args.limit_total > 0:
        total_target = min(total_target, args.limit_total)

    chunks = (
        math.ceil(max(total_target, 0) / args.chunk_size) if total_target > 0 else 0
    )
    print(
        f"Input rows={total_rows} start_offset={args.start_offset} "
        f"target_rows={total_target} chunk_size={args.chunk_size} chunks={chunks}",
        flush=True,
    )

    processed = 0
    chunk_index = 0
    while processed < total_target:
        start = args.start_offset + processed
        size = min(args.chunk_size, total_target - processed)
        end = start + size - 1
        cur_dir = chunk_dir(output_dir, start, end)
        summary_path = cur_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            print(f"Skipping existing chunk {cur_dir.name}", flush=True)
            processed += size
            chunk_index += 1
            continue

        cur_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "filter_text_hard_negatives_llm.py",
            "--input",
            str(input_path),
            "--output",
            str(cur_dir / "filtered_hn.jsonl"),
            "--reviews-output",
            str(cur_dir / "candidate_reviews.jsonl"),
            "--summary-output",
            str(summary_path),
            "--offset",
            str(start),
            "--limit",
            str(size),
            "--candidate-k",
            str(args.candidate_k),
            "--num-hard-negatives",
            str(args.num_hard_negatives),
            "--model",
            args.model,
            "--provider",
            "gemini" if args.gemini else args.provider,
            "--max-retries",
            str(args.max_retries),
            "--sleep-seconds",
            str(args.sleep_seconds),
            "--concurrency",
            str(args.concurrency),
            "--gemini-project",
            args.gemini_project,
            "--gemini-location",
            args.gemini_location,
        ]
        if args.gemini:
            cmd.append("--gemini")

        print(
            f"[chunk {chunk_index + 1}/{chunks}] rows {start}-{end} -> {cur_dir}",
            flush=True,
        )
        subprocess.run(cmd, check=True, env=os.environ.copy())
        processed += size
        chunk_index += 1

    print("All chunks complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
