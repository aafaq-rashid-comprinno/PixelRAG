#!/bin/bash
set -e
cd /home/user/wiki-screenshot-training
source ~/.zshrc

CKPT_DIR="training/output_standard_lr5e6_wu10_cosine_1000step"
TEST_DATA="training/data/test_miniv7.json"
VLLM_URL="http://localhost:8201/v1"
VLLM_MODEL="Qwen/Qwen3-VL-4B-Instruct"
RESULTS_DIR="training/eval_results"
GPU=7

eval_one() {
    local ckpt=$1
    local label=$2
    echo "[$(date)] Starting eval: $label on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU uv run python eval_checkpoint.py "$ckpt" \
        --test-data "$TEST_DATA" \
        --vllm-url "$VLLM_URL" \
        --vllm-model "$VLLM_MODEL" \
        --batch-size 8
    echo "[$(date)] Finished eval: $label"
}

# Re-run failed/incomplete evals sequentially on GPU 7
for step in 200 600 800 1000; do
    result_file="${RESULTS_DIR}/${CKPT_DIR##*/}_checkpoint-${step}_test_miniv7.jsonl"
    if [ -f "$result_file" ] && grep -q '"grade"' "$result_file" 2>/dev/null; then
        echo "Skipping checkpoint-${step}: already has complete QA results"
        continue
    fi
    # Remove incomplete result
    rm -f "$result_file"
    eval_one "${CKPT_DIR}/checkpoint-${step}" "checkpoint-${step}"
done

echo ""
echo "============================================"
echo "All retries complete! Final summary:"
echo "============================================"

python3 -c "
import json, glob, os

results_dir = '${RESULTS_DIR}'
files = sorted(glob.glob(os.path.join(results_dir, '*_test_miniv7.jsonl')))

print(f'\n{\"Checkpoint\":<15} {\"Recall@1\":>10} {\"Recall@3\":>10} {\"QA Acc\":>10} {\"N\":>5}')
print('-' * 55)

for f in files:
    lines = [json.loads(l) for l in open(f) if l.strip()]
    if not lines:
        continue
    n = len(lines)
    r1 = sum(1 for l in lines if l.get('hit@1', False)) / n
    r3 = sum(1 for l in lines if l.get('hit@3', False)) / n
    qa = sum(1 for l in lines if l.get('correct', False)) / n if 'correct' in lines[0] else float('nan')
    name = os.path.basename(f).replace('_test_miniv7.jsonl', '').replace('output_standard_lr5e6_wu10_cosine_1000step_', '')
    qa_str = f'{qa:.4f}' if qa == qa else 'N/A'
    print(f'{name:<15} {r1:>10.4f} {r3:>10.4f} {qa_str:>10} {n:>5}')
"
