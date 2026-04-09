#!/bin/bash
set -e

echo "========================================================"
echo "Micro-Batch Sweep: Llama-3.3-70B-Instruct on 4 GPUs"
echo "Testing batch sizes: 1, 2, 4, 8"
echo "Testing modes: superoffload, zerooffload, zeroinfinity"
echo "========================================================"

SCRIPT_DIR=$(dirname "$0")
BATCH_SIZES=(1 2 4 8)
MODES=("superoffload" "zerooffload" "zeroinfinity")

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for MODE in "${MODES[@]}"; do
        echo ""
        echo "========================================================"
        echo "[SWEEP] Mode=$MODE  Batch=$BATCH_SIZE"
        echo "========================================================"
        bash "${SCRIPT_DIR}/finetune_llama-70b_4gpu.sh" "$MODE" "$BATCH_SIZE" || {
            echo "[SWEEP] FAILED: Mode=$MODE Batch=$BATCH_SIZE (likely OOM)"
            continue
        }
        echo "[SWEEP] DONE: Mode=$MODE Batch=$BATCH_SIZE"
    done
done

echo ""
echo "========================================================"
echo "Micro-batch sweep completed."
echo "========================================================"
