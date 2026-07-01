#!/bin/bash
# Run the full LLM-judge matrix in parallel across 8 GPUs.
# 4 adapters (base, 2x_v2, 3x_v2, 4x_v2) × 4 k values (1,2,3,4) = 16 cells
# minus: base@k=1 (0.958), base@k=3 (0.892), 2x_v2@k=3 (0.930) already done = 13 new cells.
set -e

cd /scratch/users/zwcolin/cxr_embeds/cxr_embedding
source ~/.zshrc

DATA=/scratch/users/zwcolin/cxr_embeds/sft_data
OUT=/scratch/users/zwcolin/cxr_embeds/sft_output
LOGS=/scratch/users/zwcolin/cxr_embeds/cxr_embedding/sft/eval_out

launch() {
  local gpu=$1 tag=$2 test_json=$3 adapter_arg=$4
  CUDA_VISIBLE_DEVICES=$gpu uv run python sft/eval_multiimage.py \
      --test-json "$test_json" $adapter_arg \
      --tag "$tag" --n-examples 500 \
      > "$LOGS/eval_${tag}.log" 2>&1 &
}

# ROUND 1: 8 cells
launch 0 "base_0x_k2"   "$DATA/compressed_top6_0x/test_k2.json" ""
launch 1 "base_0x_k4"   "$DATA/compressed_top6_0x/test_k4.json" ""
launch 2 "2x_v2_k1"     "$DATA/compressed_top6_2x/test_k1.json" "--adapter $OUT/qwen3vl_top6_2x_v2"
launch 3 "2x_v2_k2"     "$DATA/compressed_top6_2x/test_k2.json" "--adapter $OUT/qwen3vl_top6_2x_v2"
launch 4 "2x_v2_k4"     "$DATA/compressed_top6_2x/test_k4.json" "--adapter $OUT/qwen3vl_top6_2x_v2"
launch 5 "3x_v2_k1"     "$DATA/compressed_top6_3x/test_k1.json" "--adapter $OUT/qwen3vl_top6_3x_v2"
launch 6 "3x_v2_k2"     "$DATA/compressed_top6_3x/test_k2.json" "--adapter $OUT/qwen3vl_top6_3x_v2"
launch 7 "3x_v2_k3"     "$DATA/compressed_top6_3x/test.json"    "--adapter $OUT/qwen3vl_top6_3x_v2"
wait
echo "=== ROUND 1 DONE ==="

# ROUND 2: 5 cells
launch 0 "3x_v2_k4"     "$DATA/compressed_top6_3x/test_k4.json" "--adapter $OUT/qwen3vl_top6_3x_v2"
launch 1 "4x_v2_k1"     "$DATA/compressed_top6_4x/test_k1.json" "--adapter $OUT/qwen3vl_top6_4x_v2"
launch 2 "4x_v2_k2"     "$DATA/compressed_top6_4x/test_k2.json" "--adapter $OUT/qwen3vl_top6_4x_v2"
launch 3 "4x_v2_k3"     "$DATA/compressed_top6_4x/test.json"    "--adapter $OUT/qwen3vl_top6_4x_v2"
launch 4 "4x_v2_k4"     "$DATA/compressed_top6_4x/test_k4.json" "--adapter $OUT/qwen3vl_top6_4x_v2"
wait
echo "=== ROUND 2 DONE ==="

# Summary
echo ""
echo "=== FINAL LLM-JUDGE ==="
for tag in base_0x_k2 base_0x_k4 \
           2x_v2_k1 2x_v2_k2 2x_v2_k4 \
           3x_v2_k1 3x_v2_k2 3x_v2_k3 3x_v2_k4 \
           4x_v2_k1 4x_v2_k2 4x_v2_k3 4x_v2_k4; do
  score=$(grep "llm_judge:" "$LOGS/eval_${tag}.log" 2>/dev/null | tail -1)
  echo "  $tag: $score"
done
