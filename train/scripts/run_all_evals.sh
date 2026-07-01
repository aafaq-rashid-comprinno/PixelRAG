#!/bin/bash
# Evaluate all checkpoints on mini-v7, using GPU 6 and 7 in parallel.
# Waits for any currently running evals to finish first.

set -e
cd /home/user/wiki-screenshot-training

CKPT_DIR="training/output_standard_lr5e6_wu10_cosine_1000step"
TEST_DATA="training/data/test_miniv7.json"
VLLM_URL="http://localhost:8201/v1"
VLLM_MODEL="Qwen/Qwen3-VL-4B-Instruct"
RESULTS_DIR="training/eval_results"

eval_one() {
    local gpu=$1
    local ckpt=$2
    local label=$3
    echo "[$(date)] Starting eval: $label on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run python eval_checkpoint.py "$ckpt" \
        --test-data "$TEST_DATA" \
        --vllm-url "$VLLM_URL" \
        --vllm-model "$VLLM_MODEL" \
        --batch-size 8 \
        2>&1 | tee "${RESULTS_DIR}/${label}.log"
    echo "[$(date)] Finished eval: $label on GPU $gpu"
}

# Wait for currently running evals on GPU 6 and 7
echo "[$(date)] Waiting for current GPU 6/7 eval processes to finish..."
for pid in $(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "  Waiting for PID $pid..."
        tail --pid="$pid" -f /dev/null 2>/dev/null || true
    fi
done
echo "[$(date)] GPUs 6 and 7 are free."

# Checkpoints to eval (skip those already done with QA grading)
declare -a NEED_EVAL=()
for step in 200 400 600 800 1000; do
    result_file="${RESULTS_DIR}/${CKPT_DIR##*/}_checkpoint-${step}_test_miniv7.jsonl"
    if [ -f "$result_file" ] && grep -q '"grade"' "$result_file" 2>/dev/null; then
        echo "Skipping checkpoint-${step}: already has QA results"
    else
        NEED_EVAL+=("$step")
    fi
done

echo "Checkpoints to evaluate: ${NEED_EVAL[*]}"

# Run evals in pairs on GPU 6 and 7
i=0
while [ $i -lt ${#NEED_EVAL[@]} ]; do
    step1=${NEED_EVAL[$i]}
    step2=${NEED_EVAL[$i+1]:-}

    # GPU 6: step1
    eval_one 6 "${CKPT_DIR}/checkpoint-${step1}" \
        "${CKPT_DIR##*/}_checkpoint-${step1}_test_miniv7" &
    pid1=$!

    # GPU 7: step2 (if exists)
    if [ -n "$step2" ]; then
        eval_one 7 "${CKPT_DIR}/checkpoint-${step2}" \
            "${CKPT_DIR##*/}_checkpoint-${step2}_test_miniv7" &
        pid2=$!
        wait $pid1 $pid2
    else
        wait $pid1
    fi

    i=$((i + 2))
done

echo ""
echo "============================================"
echo "All evals complete! Summarizing results..."
echo "============================================"

# Summary
python3 -c "
import json, glob, os

results_dir = '${RESULTS_DIR}'
pattern = '*_test_miniv7.jsonl'
files = sorted(glob.glob(os.path.join(results_dir, pattern)))

print(f'\n{\"Checkpoint\":<50} {\"Recall@1\":>10} {\"Recall@3\":>10} {\"QA Acc\":>10} {\"N\":>5}')
print('-' * 90)

for f in files:
    lines = [json.loads(l) for l in open(f) if l.strip()]
    if not lines:
        continue
    n = len(lines)
    r1 = sum(1 for l in lines if l.get('hit@1', False)) / n
    r3 = sum(1 for l in lines if l.get('hit@3', False)) / n
    qa = sum(1 for l in lines if l.get('correct', False)) / n if 'correct' in lines[0] else float('nan')
    name = os.path.basename(f).replace('_test_miniv7.jsonl', '')
    qa_str = f'{qa:.4f}' if qa == qa else 'N/A'
    print(f'{name:<50} {r1:>10.4f} {r3:>10.4f} {qa_str:>10} {n:>5}')
"
