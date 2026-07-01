#!/usr/bin/env python3
"""Convert training data JSONL from train_contrastors.py format to ms-swift embedding format.

Input format (train_contrastors.py):
    {"query": "...", "chunk_path": "/path/to/pos.png"}
    {"query": "...", "chunk_path": "/path/to/pos.png", "neg_chunk_paths": ["/path/to/neg1.png", ...]}

Output format (ms-swift embedding):
    {
      "messages": [
        {"role": "system", "content": "Retrieve images or text relevant to the user's query."},
        {"role": "user", "content": "<query>"}
      ],
      "positive_messages": [[
        {"role": "system", "content": "Represent the user's input."},
        {"role": "user", "content": "<image>"}
      ]],
      "positive_images": [["/path/to/pos.png"]],
      "negative_messages": [
        [{"role": "system", "content": "Represent the user's input."}, {"role": "user", "content": "<image>"}],
        ...
      ],
      "negative_images": [["/path/to/neg1.png"], ...]
    }

Usage:
    uv run python convert_data_for_swift.py \
        --input data/train_hn.jsonl --output data/train_hn_swift.jsonl
    uv run python convert_data_for_swift.py \
        --input data/eval.jsonl --output data/eval_swift.jsonl
"""

import argparse
import json
import os

# Match train_contrastors.py instructions exactly
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOC_INSTRUCTION = "Represent the user's input."


def convert_line(item):
    """Convert one data item from contrastors format to swift format."""
    query = item["query"]
    pos_path = item["chunk_path"]

    if not os.path.exists(pos_path):
        return None

    doc_messages = [
        {"role": "system", "content": DOC_INSTRUCTION},
        {"role": "user", "content": "<image>"},
    ]

    out = {
        "messages": [
            {"role": "system", "content": QUERY_INSTRUCTION},
            {"role": "user", "content": query},
        ],
        "positive_messages": [doc_messages],
        "positive_images": [[pos_path]],
    }

    neg_paths = item.get("neg_chunk_paths", [])
    if neg_paths:
        neg_messages = []
        neg_images = []
        for np_ in neg_paths:
            if np_ and os.path.exists(np_):
                neg_messages.append(doc_messages)
                neg_images.append([np_])
        if neg_messages:
            out["negative_messages"] = neg_messages
            out["negative_images"] = neg_images

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input JSONL (contrastors format)"
    )
    parser.add_argument("--output", required=True, help="Output JSONL (swift format)")
    args = parser.parse_args()

    converted = 0
    skipped = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            item = json.loads(line.strip())
            out = convert_line(item)
            if out is None:
                skipped += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted {converted} samples, skipped {skipped} → {args.output}")


if __name__ == "__main__":
    main()
