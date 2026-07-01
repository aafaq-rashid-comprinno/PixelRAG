#!/usr/bin/env python3
"""Export rows passing SimpleQA-style query filters in original HN format."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        default="training/data/lite-query-v2-full-filtered-hn-v2-chunks/simpleqa_style_cleaned_50k.reviews.jsonl",
        help="Review JSONL with naturalness/style scores.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/data/natrual_filtered_v2",
        help="Directory for exported filtered dataset.",
    )
    parser.add_argument(
        "--output-name",
        default="lite-query-v2-full-filtered-hn.jsonl",
        help="Output JSONL filename.",
    )
    parser.add_argument("--min-naturalness", type=int, default=4)
    parser.add_argument("--min-style-fit", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reviews_path = Path(args.reviews)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    summary_path = output_dir / "summary.json"

    selected_lines_by_file: dict[str, set[int]] = defaultdict(set)
    reviewed_rows = 0
    selected_rows = 0

    with reviews_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reviewed_rows += 1
            item = json.loads(line)
            naturalness = int(item.get("naturalness", 0) or 0)
            style_fit = int(item.get("simpleqa_style_fit", 0) or 0)
            if naturalness >= args.min_naturalness and style_fit >= args.min_style_fit:
                selected_lines_by_file[item["source_file"]].add(
                    int(item["source_line"])
                )
                selected_rows += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    written_rows = 0
    source_files = sorted(selected_lines_by_file)
    with output_path.open("w") as out:
        for source_file in source_files:
            selected_lines = selected_lines_by_file[source_file]
            with Path(source_file).open() as f:
                for line_no, line in enumerate(f, 1):
                    if line_no in selected_lines:
                        out.write(line)
                        written_rows += 1

    summary = {
        "reviews_path": str(reviews_path),
        "output_path": str(output_path),
        "reviewed_rows": reviewed_rows,
        "selected_rows": selected_rows,
        "written_rows": written_rows,
        "min_naturalness": args.min_naturalness,
        "min_style_fit": args.min_style_fit,
        "source_files": len(source_files),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
