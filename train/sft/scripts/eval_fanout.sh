#!/usr/bin/env bash
# Fan out eval_baseline.py across free GPUs for all checkpoints of a run.
# Usage:
#   bash sft/eval_fanout.sh <run_name> <compress_ratio> <gpu_list> [n_examples]
# Example:
#   bash sft/eval_fanout.sh qwen3vl_3x 3x "0,1,2,3" 500
#
# Evaluates every checkpoint-* subdir under /scratch/.../sft_output/<run_name>/,
# spreading across the given comma-separated GPU list. Each eval runs on 1 GPU.

set -euo pipefail

REPO=/scratch/users/zwcolin/cxr_embeds/cxr_embedding
cd "$REPO"

RUN_NAME="${1:?run_name required (e.g. qwen3vl_3x)}"
RATIO="${2:?compress ratio required (e.g. 3x, 5x, 9x, 0x)}"
GPU_LIST="${3:?gpu list required (e.g. 0,1,2,3)}"
N="${4:-500}"

RUN_DIR="/scratch/users/zwcolin/cxr_embeds/sft_output/${RUN_NAME}"
if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: $RUN_DIR not found" >&2
    exit 1
fi

# Discover checkpoints, sorted by step
mapfile -t CKPTS < <(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "ERROR: no checkpoint-* in $RUN_DIR" >&2
    exit 1
fi
echo "Found ${#CKPTS[@]} checkpoints for $RUN_NAME:"
printf '  %s\n' "${CKPTS[@]}"

# Parse GPU list
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPUS=${#GPUS[@]}
echo "Using $NGPUS GPUs: ${GPUS[*]}"

# Resolve images-root for given ratio
if [[ "$RATIO" == "0x" ]]; then
    IMAGES_ROOT=""  # uncompressed; default = dataset-dir in eval_baseline.py
else
    IMAGES_ROOT="/scratch/users/zwcolin/cxr_embeds/sft_data/compressed_${RATIO}/images"
fi

mkdir -p logs/ckpt_eval
source .env

# Dispatch in parallel, queuing once NGPUS in flight
declare -A PIDS=()
for i in "${!CKPTS[@]}"; do
    CKPT="${CKPTS[$i]}"
    STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
    GPU=${GPUS[$((i % NGPUS))]}
    TAG="sft_${RATIO}_step${STEP}"
    LOG="logs/ckpt_eval/${TAG}.log"

    # Wait if this GPU slot is busy
    if [[ -n "${PIDS[$GPU]:-}" ]] && kill -0 "${PIDS[$GPU]}" 2>/dev/null; then
        wait "${PIDS[$GPU]}" || true
    fi

    IMG_ARG=""
    [[ -n "$IMAGES_ROOT" ]] && IMG_ARG="--images-root $IMAGES_ROOT"

    echo "[launch] GPU=$GPU $TAG  ($CKPT)"
    CUDA_VISIBLE_DEVICES="$GPU" \
        .venv/bin/python sft/eval_baseline.py \
            --adapter "$CKPT" \
            --n-examples "$N" \
            --tag "$TAG" \
            --device cuda:0 \
            $IMG_ARG \
        > "$LOG" 2>&1 &
    PIDS[$GPU]=$!
done

# Wait for all
for gpu in "${!PIDS[@]}"; do
    pid="${PIDS[$gpu]}"
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true
    fi
done

echo "=== Summary ==="
for CKPT in "${CKPTS[@]}"; do
    STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
    TAG="sft_${RATIO}_step${STEP}"
    LOG="logs/ckpt_eval/${TAG}.log"
    JUDGE=$(grep -oE 'llm_judge:[[:space:]]+[0-9.]+' "$LOG" | awk '{print $2}' | head -1)
    EM=$(grep -oE 'exact_match:[[:space:]]+[0-9.]+' "$LOG" | awk '{print $2}' | head -1)
    printf '  step %-5s  EM=%-8s  LLM-judge=%s\n' "$STEP" "${EM:-?}" "${JUDGE:-?}"
done
