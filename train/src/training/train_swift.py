#!/usr/bin/env python3
"""Fine-tune Qwen3-VL-Embedding with ms-swift — equivalent to train_contrastors.py.

Uses ms-swift's embedding training pipeline (InfoNCE loss, cross-GPU negative sharing)
instead of our custom GradCache training loop.

Equivalence notes vs train_contrastors.py:
  - Same model (Qwen3-VL-Embedding-2B), same LoRA targets (q/k/v/o_proj)
  - Same loss (InfoNCE with in-batch + hard negatives)
  - Same instructions (QUERY_INSTRUCTION / DOC_INSTRUCTION in data JSONL)
  - Temperature: FIXED at 0.07 (swift has no learnable LogitScale)
  - No GradCache: memory bounded by batch size (use DeepSpeed ZeRO-2 to compensate)
  - No custom retrieval eval (R@1/5/10): swift only does loss-based eval
    → Run retrieval eval separately after training

Data format: use convert_data_for_swift.py to convert from contrastors format.

Single GPU:
    CUDA_VISIBLE_DEVICES=3 uv run python train_swift.py

Multi-GPU:
    CUDA_VISIBLE_DEVICES=1,2 uv run python train_swift.py --nproc-per-node 2

Resume:
    uv run python train_swift.py --resume training/output_swift/vX-XXX/checkpoint-50

Best config (matching train_contrastors.py defaults):
    CUDA_VISIBLE_DEVICES=1,2 uv run python train_swift.py \\
        --train-jsonl data/train_hn_swift.jsonl \\
        --eval-jsonl data/eval_swift.jsonl \\
        --num-hard-negatives 5 \\
        --batch-size 16 \\
        --lr 1e-5 \\
        --max-steps 50 \\
        --warmup-steps 20 \\
        --eval-steps 25 \\
        --save-steps 50 \\
        --nproc-per-node 2
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL-Embedding with ms-swift (InfoNCE)"
    )

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")

    # Data (swift format — use convert_data_for_swift.py first)
    parser.add_argument("--train-jsonl", default="data/train_hn_swift.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval_swift.jsonl")

    # Training
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--scheduler", choices=["cosine", "constant"], default="cosine")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Fixed InfoNCE temperature (not learnable in swift)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=0,
        help="Hard negatives per query (requires swift-format data with negative_messages)",
    )

    # LoRA (match train_contrastors.py defaults)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Resolution
    parser.add_argument(
        "--max-num-visual-tokens",
        type=int,
        default=4096,
        help="Max visual tokens → converted to max_pixels for processor",
    )

    # Eval / Save
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=5)

    # Output
    parser.add_argument("--output-dir", default="training/output_swift")

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )

    # Distributed
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of GPUs (sets NPROC_PER_NODE for swift)",
    )
    parser.add_argument(
        "--deepspeed",
        default=None,
        help="DeepSpeed config: 'zero2', 'zero3', or path to JSON",
    )

    # Wandb
    parser.add_argument("--wandb-project", default="wiki-screenshot-training")
    parser.add_argument("--no-wandb", action="store_true")

    # Freeze
    parser.add_argument(
        "--freeze-vit",
        action="store_true",
        default=True,
        help="Freeze vision encoder (default: True)",
    )
    parser.add_argument("--no-freeze-vit", dest="freeze_vit", action="store_false")

    args = parser.parse_args()

    # --- Environment variables for swift InfoNCE ---
    os.environ["INFONCE_TEMPERATURE"] = str(args.temperature)
    os.environ["INFONCE_USE_BATCH"] = (
        "True"  # in-batch negatives (like train_contrastors.py)
    )
    if args.num_hard_negatives > 0:
        os.environ["INFONCE_HARD_NEGATIVES"] = str(args.num_hard_negatives)

    if args.nproc_per_node > 1:
        os.environ["NPROC_PER_NODE"] = str(args.nproc_per_node)

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Convert max_num_visual_tokens → max_pixels
    # Qwen3-VL: each visual token covers (patch_size * spatial_merge_size)^2 = 28^2 = 784 pixels
    pixels_per_token = 28 * 28  # patch_size=14, spatial_merge_size=2
    max_pixels = args.max_num_visual_tokens * pixels_per_token

    # --- Build SftArguments ---
    # Import after env vars are set so swift picks them up
    from swift import SftArguments, sft_main

    # Map scheduler name
    lr_scheduler_type = (
        "cosine" if args.scheduler == "cosine" else "constant_with_warmup"
    )

    sft_args = SftArguments(
        # Model
        model=args.model,
        task_type="embedding",
        # LoRA — match train_contrastors.py: q_proj, k_proj, v_proj, o_proj
        tuner_type="lora",
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        freeze_vit=args.freeze_vit,
        # Loss
        loss_type="infonce",
        # Data
        dataset=[args.train_jsonl],
        val_dataset=[args.eval_jsonl],
        split_dataset_ratio=0.0,  # We provide val_dataset explicitly
        # Training hyperparams
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        # Precision
        torch_dtype="bfloat16",
        # Resolution
        max_pixels=max_pixels,
        # Eval / Save / Log
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        dataloader_drop_last=True,
        # Output
        output_dir=args.output_dir,
        # Resume
        resume_from_checkpoint=args.resume,
        # DeepSpeed
        deepspeed=args.deepspeed,
        # Misc
        dataloader_num_workers=4,
    )

    # --- Run training ---
    result = sft_main(sft_args)

    # Print results
    if result:
        print(f"\n{'=' * 60}")
        print("Training complete!")
        if hasattr(result, "last_model_checkpoint") and result.last_model_checkpoint:
            print(f"Last checkpoint: {result.last_model_checkpoint}")
        if hasattr(result, "best_model_checkpoint") and result.best_model_checkpoint:
            print(f"Best checkpoint: {result.best_model_checkpoint}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
