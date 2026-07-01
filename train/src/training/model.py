#!/usr/bin/env python3
"""Model loading: Qwen3-VL + LoRA for embedding fine-tuning."""

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_model_for_training(
    model_path: str,
    gpu_id: int,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Load Qwen3-VL with LoRA adapters for fine-tuning.

    Uses Qwen3VLForConditionalGeneration (NOT AutoModel, which loads
    Qwen3VLModel with random language_model weights).
    """
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    device = f"cuda:{gpu_id}"
    model = model.to(device)
    return model


def load_processor(
    model_path: str, min_pixels: int = 128 * 28 * 28, max_pixels: int = 256 * 28 * 28
):
    """Load the processor with configurable image resolution.

    Default resolution is reduced from the model default (~1.3M pixels) to
    ~100K-200K pixels for training speed. Full resolution can be restored
    for inference by setting max_pixels=1280*28*28.
    """
    return AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def pool_and_normalize(hidden_states, attention_mask):
    """Last-token pooling + L2 normalization.

    Matches the production embedding pipeline (embed_tiles.py direct_gpu backend).
    """
    last_idx = attention_mask.sum(dim=1) - 1
    pooled = hidden_states[
        torch.arange(hidden_states.size(0), device=hidden_states.device),
        last_idx,
    ]
    return F.normalize(pooled, p=2, dim=-1)
