#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Full benchmark sweep for Qwen2.5-72B-Instruct on 4 GPUs.
#
# Runs (18 total):
#   ZeRO-Infinity × mbs {1, 2, 4}                                     =  3 runs
#   SuperOffload  × cpu_ratio {1.0, 0.9, 0.5, 0.0} × mbs {1, 2, 4}  = 12 runs
#   Pipeline Parallel × mbs {1, 2, 4}                                 =  3 runs
#
# Usage (from DeepSpeed-v0.18.9/):
#   bash tests/cpu_offload/run_qwen2.5-72b.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================================"
echo "Qwen2.5-72B-Instruct Full Benchmark Sweep"
echo "========================================================"

# ── ZeRO-Infinity: 3 mbs ─────────────────────────────────────────────────────
for MBS in 1 2 4; do
    GAS=$((256 / MBS / 4))
    echo ""
    echo "[SWEEP] zeroinfinity mbs=${MBS} gas=${GAS}"
    bash "${SCRIPT_DIR}/finetune_qwen2.5-72b_4gpu.sh" zeroinfinity "${MBS}" 0 "${GAS}" || {
        echo "[SWEEP] FAILED: zeroinfinity mbs=${MBS}"
        continue
    }
done

# ── SuperOffload: 4 cpu_ratio × 3 mbs ────────────────────────────────────────
for CPU_RATIO in 1.0 0.9 0.5 0.0; do
    for MBS in 1 2 4; do
        GAS=$((256 / MBS / 4))
        echo ""
        echo "[SWEEP] superoffload ratio=${CPU_RATIO} mbs=${MBS} gas=${GAS}"
        bash "${SCRIPT_DIR}/finetune_qwen2.5-72b_4gpu.sh" superoffload "${MBS}" "${CPU_RATIO}" "${GAS}" || {
            echo "[SWEEP] FAILED: superoffload ratio=${CPU_RATIO} mbs=${MBS}"
            continue
        }
    done
done

# ── Pipeline Parallel: 3 mbs ──────────────────────────────────────────────────
for MBS in 1 2 4; do
    echo ""
    echo "[SWEEP] pp mbs=${MBS}"
    bash "${SCRIPT_DIR}/finetune_qwen2.5-72b_4gpu_pp.sh" "${MBS}" || {
        echo "[SWEEP] FAILED: pp mbs=${MBS}"
        continue
    }
done

echo ""
echo "========================================================"
echo "Sweep completed. Run summarize_results.py to generate Excel."
echo "========================================================"
