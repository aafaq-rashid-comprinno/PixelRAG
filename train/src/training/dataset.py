#!/usr/bin/env python3
"""Dataset and DataLoader for query-chunk pair training."""

import json
import logging

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Task-specific instructions (Qwen3-VL-Embedding style)
# Query instruction describes the retrieval task; document instruction is generic.
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOC_INSTRUCTION = "Represent the user's input."


class QueryChunkDataset(Dataset):
    """Dataset of (query, chunk_path) pairs from JSONL."""

    def __init__(self, jsonl_path: str):
        self.pairs = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                self.pairs.append((item["query"], item["chunk_path"]))
        logger.info(f"Loaded {len(self.pairs)} pairs from {jsonl_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_collate_fn(processor, device="cuda"):
    """Create a collate function that preprocesses queries and images.

    Returns None for batches where all images fail to load.
    Uses the same chat template as the production embedding pipeline.
    """

    def collate(batch):
        valid = []
        for query, path in batch:
            try:
                img = Image.open(path).convert("RGB")
                valid.append((query, path, img))
            except Exception as e:
                logger.warning(f"Skipping bad image {path}: {e}")
        if not valid:
            return None

        queries, paths, images = zip(*valid)

        # Build query inputs using chat template (text-only)
        q_messages_batch = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": QUERY_INSTRUCTION}],
                },
                {"role": "user", "content": [{"type": "text", "text": q}]},
            ]
            for q in queries
        ]
        q_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in q_messages_batch
        ]
        q_inputs = processor(text=q_texts, return_tensors="pt", padding=True)
        q_inputs = {
            k: v.to(device) if hasattr(v, "to") else v for k, v in q_inputs.items()
        }

        # Build image inputs using chat template (image)
        i_messages_batch = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": DOC_INSTRUCTION}],
                },
                {"role": "user", "content": [{"type": "image", "image": img}]},
            ]
            for img in images
        ]
        i_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in i_messages_batch
        ]
        i_inputs = processor(
            text=i_texts,
            images=list(images),
            return_tensors="pt",
            padding=True,
            device=device,
        )
        i_inputs = {
            k: v.to(device) if hasattr(v, "to") else v for k, v in i_inputs.items()
        }

        return q_inputs, i_inputs

    return collate
