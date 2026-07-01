#!/usr/bin/env python3
"""Build test_miniv8.json from SimpleQA dataset + v8 meta.json golden_mapping.

Mirrors the format of test_miniv6.json:
{
  "description": "...",
  "tiles_dir": "...",
  "questions": [{"id": ..., "problem": ..., "answer": ...}, ...],
  "golden_mapping": {"example_id": ["article_id", ...], ...}
}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    meta_path = Path("/home/user/Vis-RAG/agent/tiles-hard-mini-v8.meta.json")
    output_path = Path("training/data/test_miniv8.json")

    # Load v8 meta
    with open(meta_path) as f:
        meta = json.load(f)
    golden_mapping = meta["golden_mapping"]
    print(f"v8 golden_mapping: {len(golden_mapping)} example IDs")

    # Load SimpleQA CSV
    csv_path = Path(
        "/home/user/Vis-RAG/agent/evaluation/simple_qa_eval/data/simple_qa_test_set.csv"
    )
    if not csv_path.exists():
        print(f"SimpleQA CSV not found at {csv_path}, trying download...")
        url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(csv_path)

    df = df.reset_index(drop=True)
    df["id"] = df["problem"].apply(lambda p: hashlib.md5(p.encode()).hexdigest())

    # Filter to examples in golden_mapping
    questions = []
    for _, row in df.iterrows():
        if row["id"] in golden_mapping:
            questions.append(
                {
                    "id": row["id"],
                    "problem": row["problem"],
                    "answer": row["answer"],
                }
            )

    print(f"Matched {len(questions)} questions out of {len(golden_mapping)} golden IDs")

    # Check for any golden IDs not found in SimpleQA
    found_ids = {q["id"] for q in questions}
    missing = set(golden_mapping.keys()) - found_ids
    if missing:
        print(
            f"WARNING: {len(missing)} golden IDs not found in SimpleQA CSV: {list(missing)[:5]}..."
        )

    # Build output
    output = {
        "description": meta["description"],
        "tiles_dir": "/home/user/Vis-RAG/agent/tiles-hard-mini-v8",
        "questions": questions,
        "golden_mapping": golden_mapping,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(
        f"Written {output_path} ({len(questions)} questions, {len(golden_mapping)} golden mappings)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
