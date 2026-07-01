#!/usr/bin/env bash
# Run 8-way parallel eval: base+thinking and adapter+thinking across all 4 compressions.
# Assumes mixed-think adapter is at checkpoint-FINAL (pass as $1) and base Qwen used for baseline.
# Prints a comparison table at the end.

set -euo pipefail

REPO=/scratch/users/zwcolin/cxr_embeds/cxr_embedding
cd "$REPO"
source .env
mkdir -p logs/think_eval

ADAPTER_CKPT="${1:?adapter checkpoint dir required}"
N="${2:-500}"

declare -A BASE_JUDGE=()
declare -A ADAP_JUDGE=()

echo "Launching 8 evals in parallel (base+thinking on 0-3, adapter+thinking on 4-7)..."
# Base Qwen with thinking, 4 compressions on GPU 0-3
for idx in 0 1 2 3; do
  case $idx in 0) R=2x ;; 1) R=3x ;; 2) R=5x ;; 3) R=9x ;; esac
  CUDA_VISIBLE_DEVICES=$idx .venv/bin/python sft/eval_baseline.py \
    --n-examples $N --tag base_think_$R --device cuda:0 \
    --thinking \
    --images-root /scratch/users/zwcolin/cxr_embeds/sft_data/compressed_$R/images \
    > logs/think_eval/base_think_$R.log 2>&1 &
done
# Adapter with thinking, 4 compressions on GPU 4-7
for idx in 0 1 2 3; do
  case $idx in 0) R=2x ;; 1) R=3x ;; 2) R=5x ;; 3) R=9x ;; esac
  GPU=$((idx + 4))
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python sft/eval_baseline.py \
    --adapter "$ADAPTER_CKPT" \
    --n-examples $N --tag mixed_think_v1_$R --device cuda:0 \
    --thinking \
    --images-root /scratch/users/zwcolin/cxr_embeds/sft_data/compressed_$R/images \
    > logs/think_eval/mixed_think_v1_$R.log 2>&1 &
done
wait
echo ""
echo "=== Think-enabled comparison ==="
printf '%-6s %-18s %-20s %-10s\n' "ratio" "base+thinking" "adapter+thinking" "Δ"
for R in 2x 3x 5x 9x; do
  B=$(grep -oE "llm_judge:[[:space:]]+[0-9.]+" logs/think_eval/base_think_$R.log 2>/dev/null | awk '{print $2}' | head -1)
  A=$(grep -oE "llm_judge:[[:space:]]+[0-9.]+" logs/think_eval/mixed_think_v1_$R.log 2>/dev/null | awk '{print $2}' | head -1)
  DELTA=$(awk -v a="${A:-0}" -v b="${B:-0}" 'BEGIN{printf "%+.3f", a-b}')
  printf '%-6s %-18s %-20s %-10s\n' "$R" "${B:-?}" "${A:-?}" "$DELTA"
done
