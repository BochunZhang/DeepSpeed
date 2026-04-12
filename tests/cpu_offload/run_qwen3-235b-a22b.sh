#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Full benchmark sweep for Qwen3-235B-A22B-Instruct-2507 (MoE) on 4 GPUs.
#
# Modes tested:
#   superoffload           (optimizer → CPU, super_offload=true)
#   zerooffload            (optimizer → CPU)
#   zeroinfinity           (optimizer + param → CPU)
#   zeroinfinity-superoffload (optimizer + param → CPU, super_offload=true)
#
# GBS tested : 64, 256
# MBS tested : 1, 2, 4, 8
# Combinations: 4 modes × 2 GBS × 4 MBS = 32 runs
#
# Usage (from DeepSpeed-v0.18.9/):
#   bash tests/cpu_offload/run_qwen3-235b-a22b.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FINETUNE="${SCRIPT_DIR}/finetune_qwen3-235b-a22b_4gpu.sh"
CPU_RATIO=0.90   # used by superoffload and zeroinfinity-superoffload

echo "========================================================"
echo "Qwen3-235B-A22B-Instruct-2507 Full Benchmark Sweep (32 runs)"
echo "========================================================"

run() {
    local mode="$1" gbs="$2" mbs="$3"
    echo ""
    echo "[SWEEP] mode=${mode} gbs=${gbs} mbs=${mbs}"
    bash "${FINETUNE}" --mode "${mode}" --gbs "${gbs}" --mbs "${mbs}" \
        --cpu_ratio "${CPU_RATIO}" || {
        echo "[SWEEP] FAILED: mode=${mode} gbs=${gbs} mbs=${mbs}"
        # continue sweep even on failure
    }
}

for MODE in superoffload zerooffload zeroinfinity zeroinfinity-superoffload; do
    for GBS in 64 256; do
        for MBS in 1 2 4 8; do
            run "${MODE}" "${GBS}" "${MBS}"
        done
    done
done

echo ""
echo "========================================================"
echo "Sweep completed. Run summarize_results.py to generate Excel."
echo "========================================================"
