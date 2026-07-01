#!/usr/bin/env python3
"""Verify fine-tuned embeddings: positives should be closer than negatives.

Compares base model vs fine-tuned (LoRA) model on eval pairs.
For each query, computes cosine similarity to its positive image and to
all other images (negatives). Reports mean positive/negative similarity
and Recall@1/5/10.

Usage:
    python training/verify_embeddings.py --adapter training/output_test
    python training/verify_embeddings.py --adapter training/output_test --max-pairs 50
"""

import argparse
import json
import logging
import sys

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_model_and_processor(model_name, adapter_path=None, max_visual_tokens=256):
    from models.biqwen3 import BiQwen3
    from transformers import AutoProcessor

    model = BiQwen3.from_pretrained(model_name, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_name)
    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size
    tile = patch_size * merge_size
    processor.image_processor.max_pixels = max_visual_tokens * tile * tile
    processor.image_processor.size["longest_edge"] = (
        processor.image_processor.max_pixels
    )
    processor.tokenizer.padding_side = "left"

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"Loaded LoRA adapter from {adapter_path}")

    model.eval()
    model.cuda()
    return model, processor


# Task-specific instructions (Qwen3-VL-Embedding style)
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOC_INSTRUCTION = "Represent the user's input."


def _process_queries(processor, queries):
    messages_batch = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": QUERY_INSTRUCTION}],
            },
            {"role": "user", "content": [{"type": "text", "text": q}]},
        ]
        for q in queries
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    return processor(text=texts, return_tensors="pt", padding="longest")


def _process_doc_images(processor, images):
    messages_batch = [
        [
            {"role": "system", "content": [{"type": "text", "text": DOC_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "image", "image": img}]},
        ]
        for img in images
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    return processor(text=texts, images=images, return_tensors="pt", padding="longest")


def embed_queries(model, processor, queries, batch_size=16):
    all_embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        inputs = _process_queries(processor, batch)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            embs = model(**inputs)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


def embed_images(model, processor, images, batch_size=8):
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = _process_doc_images(processor, batch)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            embs = model(**inputs)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


def compute_metrics(q_embs, i_embs):
    """Compute retrieval metrics. Each query[i] matches image[i]."""
    # Cosine similarity matrix (already L2 normalized by model)
    sims = q_embs @ i_embs.T  # (N, N)
    n = sims.shape[0]

    # Per-query: positive sim vs mean negative sim
    pos_sims = np.diag(sims)
    # Mask out diagonal for negative sims
    mask = ~np.eye(n, dtype=bool)
    neg_sims = sims[mask].reshape(n, n - 1)
    mean_neg_sims = neg_sims.mean(axis=1)
    max_neg_sims = neg_sims.max(axis=1)

    # Recall@K
    rankings = (-sims).argsort(axis=1)
    correct = np.arange(n)
    ranks = np.array([np.where(rankings[i] == correct[i])[0][0] for i in range(n)])

    recall_1 = (ranks < 1).mean()
    recall_5 = (ranks < 5).mean()
    recall_10 = (ranks < 10).mean()
    mrr = (1.0 / (ranks + 1)).mean()

    return {
        "mean_pos_sim": float(pos_sims.mean()),
        "mean_neg_sim": float(mean_neg_sims.mean()),
        "mean_max_neg_sim": float(max_neg_sims.mean()),
        "margin": float((pos_sims - mean_neg_sims).mean()),
        "recall@1": float(recall_1),
        "recall@5": float(recall_5),
        "recall@10": float(recall_10),
        "mrr": float(mrr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument(
        "--adapter", type=str, required=True, help="Path to LoRA adapter dir"
    )
    parser.add_argument("--eval-jsonl", default="training/data/eval.jsonl")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    args = parser.parse_args()

    # Load eval data
    pairs = []
    with open(args.eval_jsonl) as f:
        for line in f:
            item = json.loads(line)
            try:
                img = Image.open(item["chunk_path"]).convert("RGB")
                pairs.append((item["query"], img))
            except Exception:
                continue
    pairs = pairs[: args.max_pairs]
    queries = [p[0] for p in pairs]
    images = [p[1] for p in pairs]
    logger.info(f"Loaded {len(pairs)} eval pairs")

    results = {}

    # 1. Base model (no LoRA)
    logger.info("=== Base model (no fine-tuning) ===")
    model, processor = load_model_and_processor(
        args.model, adapter_path=None, max_visual_tokens=args.max_visual_tokens
    )
    q_embs = embed_queries(model, processor, queries)
    i_embs = embed_images(model, processor, images)
    base_metrics = compute_metrics(q_embs, i_embs)
    results["base"] = base_metrics
    for k, v in base_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    del model
    torch.cuda.empty_cache()

    # 2. Fine-tuned model (with LoRA)
    logger.info(f"=== Fine-tuned model ({args.adapter}) ===")
    model, processor = load_model_and_processor(
        args.model, adapter_path=args.adapter, max_visual_tokens=args.max_visual_tokens
    )
    q_embs_ft = embed_queries(model, processor, queries)
    i_embs_ft = embed_images(model, processor, images)
    ft_metrics = compute_metrics(q_embs_ft, i_embs_ft)
    results["finetuned"] = ft_metrics
    for k, v in ft_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    del model
    torch.cuda.empty_cache()

    # 3. Comparison
    logger.info("=== Comparison ===")
    for k in base_metrics:
        diff = ft_metrics[k] - base_metrics[k]
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        logger.info(
            f"  {k}: {base_metrics[k]:.4f} → {ft_metrics[k]:.4f} ({arrow}{abs(diff):.4f})"
        )


if __name__ == "__main__":
    main()
