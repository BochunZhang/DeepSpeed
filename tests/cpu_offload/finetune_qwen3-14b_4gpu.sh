#!/bin/bash
set -eo pipefail

echo "================================================"
echo "Qwen3-14B Fine-tuning with DeepSpeed on 4 GPU"
echo "================================================"

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE=superoffload
BATCH_SIZE=4
CPU_RATIO=0.90
PROFILE=false

usage() {
    echo "Usage: $0 [--mode MODE] [--batch_size N] [--cpu_ratio R] [--profile]"
    echo "  --mode       superoffload (default) | zerooffload | zeroinfinity"
    echo "  --batch_size micro-batch size per GPU (default: 4)"
    echo "  --cpu_ratio  CPU offload ratio for superoffload (default: 0.90)"
    echo "  --profile    enable nsys profiling (default: off)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";       shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --cpu_ratio)  CPU_RATIO="$2";  shift 2 ;;
        --profile)    PROFILE=true;    shift   ;;
        --help|-h)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ── Validate mode ─────────────────────────────────────────────────────────────
if [ "$MODE" = "superoffload" ]; then
    MODE_LABEL="super-offload"
    CONFIG_LABEL="mbs${BATCH_SIZE}-cpu${CPU_RATIO}"
elif [ "$MODE" = "zerooffload" ]; then
    MODE_LABEL="zero-offload"
    CONFIG_LABEL="mbs${BATCH_SIZE}"
elif [ "$MODE" = "zeroinfinity" ]; then
    MODE_LABEL="zero-infinity"
    CONFIG_LABEL="mbs${BATCH_SIZE}"
else
    echo "Error: Unknown mode '$MODE'. Use: superoffload | zerooffload | zeroinfinity"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_NAME="Qwen/Qwen3-14B"
OUTPUT_DIR="${DS_ROOT}/results/cpu_offload/qwen3-14b/${MODE_LABEL}/${CONFIG_LABEL}"
DS_CONFIG_JSON="${OUTPUT_DIR}/ds_config.json"

mkdir -p "${OUTPUT_DIR}"

# ── Script argument parameters ────────────────────────────────────────────────
ACTIVATION_CHECKPOINTING=true
SAVE_CHECKPOINT=false
MAX_LENGTH=4096
LOG_INTERVAL=1
DATASET_NAME="tatsu-lab/alpaca"
DATASET_PERCENTAGE=10.0
USE_WANDB=false
WANDB_PROJECT="qwen3-14b"
WANDB_RUN_NAME="qwen3-14b-${MODE}"
DETERMINISTIC=false
BENCH_STEPS=10
WARMUP_STEPS=20

EPOCHS=1
LR=1e-5
WARMUP=0.05
WEIGHT_DECAY=0.01
SEED=42

ACTIVATION_CHECKPOINTING_FLAG=""
if [ "$ACTIVATION_CHECKPOINTING" = "true" ]; then
    ACTIVATION_CHECKPOINTING_FLAG="--activation_checkpointing"
fi

SAVE_CHECKPOINT_ARG=""
if [ "$SAVE_CHECKPOINT" = "true" ]; then
    SAVE_CHECKPOINT_ARG="--save_checkpoint"
fi

WANDB_FLAG=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

DETERMINISTIC_FLAG=""
if [ "$DETERMINISTIC" = "true" ]; then
    DETERMINISTIC_FLAG="--deterministic"
fi

# ── DeepSpeed configuration ───────────────────────────────────────────────────
if [ "$MODE" = "superoffload" ]; then
cat > "${DS_CONFIG_JSON}" << EOF
{
    "train_batch_size": ${BATCH_SIZE},
    "gradient_accumulation_steps": 1,
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "ratio": ${CPU_RATIO},
            "super_offload": true,
            "cpuadam_cores_perc": 0.90
        }
    },
    "wall_clock_breakdown": true
}
EOF

elif [ "$MODE" = "zerooffload" ]; then
cat > "${DS_CONFIG_JSON}" << EOF
{
    "train_batch_size": ${BATCH_SIZE},
    "gradient_accumulation_steps": 1,
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "wall_clock_breakdown": true
}
EOF

elif [ "$MODE" = "zeroinfinity" ]; then
cat > "${DS_CONFIG_JSON}" << EOF
{
    "train_batch_size": ${BATCH_SIZE},
    "gradient_accumulation_steps": 1,
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "wall_clock_breakdown": true
}
EOF
fi

# ── numarun ───────────────────────────────────────────────────────────────────
if command -v numarun &>/dev/null; then
    NUMARUN="numarun"
    echo "[INFO] numarun found: enabling per-rank CPU affinity"
else
    NUMARUN=""
    echo "[INFO] numarun not found: skipping CPU affinity binding"
fi

# ── nsys profiling ────────────────────────────────────────────────────────────
# Capture rounds 6 and 7 out of 10 total steps.
# profile_start: global_step value BEFORE the 6th step's forward pass (= 5)
# profile_end:   global_step value AFTER  the 7th step completes        (= 7)
PROFILE_START=5
PROFILE_END=7
NSYS_OUT="${OUTPUT_DIR}/profile"

PROFILE_FLAG=""
if [ "$PROFILE" = "true" ]; then
    PROFILE_FLAG="--profile --profile_start ${PROFILE_START} --profile_end ${PROFILE_END}"
fi

GPUS_PER_NODE=4

echo "Qwen3-14B | mode=${MODE} batch_size=${BATCH_SIZE} cpu_ratio=${CPU_RATIO} profile=${PROFILE}"
echo "Output: ${OUTPUT_DIR}"
echo "================================================"

DEEPSPEED_CMD="${NUMARUN} deepspeed --num_gpus=${GPUS_PER_NODE} ${SCRIPT_DIR}/finetune_zero3.py \
    --deepspeed_config=${DS_CONFIG_JSON} \
    --model_name ${MODEL_NAME} \
    --num_train_epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --weight_decay ${WEIGHT_DECAY} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --max_length ${MAX_LENGTH} \
    --log_interval ${LOG_INTERVAL} \
    --dataset_name ${DATASET_NAME} \
    --dataset_percentage ${DATASET_PERCENTAGE} \
    --bench_steps ${BENCH_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    ${ACTIVATION_CHECKPOINTING_FLAG} \
    ${SAVE_CHECKPOINT_ARG} \
    ${WANDB_FLAG} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    ${DETERMINISTIC_FLAG} \
    ${PROFILE_FLAG}"

if [ "$PROFILE" = "true" ]; then
    nsys profile \
        --capture-range=cudaProfilerApi \
        --force-overwrite=true \
        -o "${NSYS_OUT}" \
        ${DEEPSPEED_CMD}
else
    eval ${DEEPSPEED_CMD}
fi
