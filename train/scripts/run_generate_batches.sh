#!/bin/bash
# Batch runner for generate_query_pairs.py
#
# Divides the page pool into non-overlapping batches and runs one at a time,
# skipping any batch whose output file already exists.
#
# Usage:
#   bash run_generate_batches.sh <tiles_dir> <start> <end> [model] [extra args...]
#
# Examples:
#   # Generate batches 0–99 with default Gemini Pro:
#   bash run_generate_batches.sh /opt/dlami/nvme/kiwix_tiles 0 99
#
#   # Use Flash Lite for cheaper/faster generation:
#   bash run_generate_batches.sh /opt/dlami/nvme/kiwix_tiles 0 99 gemini-3.1-flash-lite-preview
#
#   # Filter by page chunk count:
#   bash run_generate_batches.sh /opt/dlami/nvme/kiwix_tiles 0 50 gemini-3.1-flash-lite-preview \
#       --postfilter-min-page-chunks 21 --postfilter-max-page-chunks 50

set -euo pipefail

TILES_DIR=${1:?Usage: run_generate_batches.sh <tiles_dir> <start> <end> [model] [extra args...]}
START=${2:-0}
END=${3:-9}
MODEL=${4:-gemini-3.1-pro-preview}
if [ "$#" -ge 4 ]; then
    shift 4
else
    shift "$#"
fi
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/batches"
mkdir -p "$OUTPUT_DIR"

TOTAL_BATCHES=1000
PAGES_PER_BATCH=2000

echo "Tiles dir:     $TILES_DIR"
echo "Batches:       $START to $END (of $TOTAL_BATCHES total)"
echo "Model:         $MODEL"
echo "Extra args:    ${EXTRA_ARGS[*]:-none}"
echo "Output dir:    $OUTPUT_DIR"
echo "Started:       $(date)"
echo ""

for i in $(seq "$START" "$END"); do
    BATCH=$(printf "%03d" "$i")
    OUT="$OUTPUT_DIR/batch_${BATCH}.jsonl"
    LOG="$OUTPUT_DIR/batch_${BATCH}.log"

    if [ -f "$OUT" ]; then
        echo "[$BATCH] exists ($(wc -l < "$OUT") pairs), skipping."
        continue
    fi

    echo "[$BATCH] start $(date)"
    uv run python "$SCRIPT_DIR/generate_query_pairs.py" \
        --tiles-dir "$TILES_DIR" \
        --batch-index "$i" \
        --total-batches "$TOTAL_BATCHES" \
        --num-pages "$PAGES_PER_BATCH" \
        --output "$OUT" \
        --seed "$i" \
        --model "$MODEL" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG"

    echo "[$BATCH] done $(wc -l < "$OUT" 2>/dev/null || echo 0) pairs | $(date)"
    echo ""
done

echo "All done | $(date)"

# Merge all batch files
MERGED="$OUTPUT_DIR/all_pairs.jsonl"
cat "$OUTPUT_DIR"/batch_*.jsonl > "$MERGED" 2>/dev/null || true
echo "Merged: $(wc -l < "$MERGED" 2>/dev/null || echo 0) pairs total → $MERGED"
