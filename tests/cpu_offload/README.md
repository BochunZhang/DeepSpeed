# CPU Offload Benchmark Suite

Benchmark suite for comparing CPU offloading strategies in DeepSpeed using HuggingFace Transformers models.

## Overview

This suite tests three CPU offloading strategies for large model fine-tuning:

| Strategy | Config Key | What Gets Offloaded | Description |
|----------|-----------|---------------------|-------------|
| **ZeRO-Offload** | `zerooffload` | Optimizer states | ZeRO Stage 3 + optimizer offload to CPU |
| **ZeRO-Infinity** | `zeroinfinity` | Optimizer states + Parameters | ZeRO Stage 3 + both optimizer and parameter offload to CPU |
| **SuperOffload** | `superoffload` | Optimizer states (async) | ZeRO Stage 3 + async CPU optimizer in a separate process (Speculation-then-Validation) |

## Architecture: How Transformers and DeepSpeed Work Together

```
+---------------------------------------------------------------+
|  Shell Script (.sh)                                           |
|  - Generates DeepSpeed JSON config (offload mode selection)   |
|  - Launches: deepspeed finetune_zero3.py --deepspeed_config   |
+---------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------+
|  finetune_zero3.py (Training Script)                          |
|                                                               |
|  1. Model Loading (HuggingFace Transformers)                  |
|     model = AutoModelForCausalLM.from_pretrained(model_name)  |
|     - Model architecture defined by model provider            |
|       (Meta for Llama, Qwen team for Qwen, etc.)              |
|     - Downloaded from HuggingFace Hub                         |
|     - Returns a standard torch.nn.Module                      |
|                                                               |
|  2. Optimizer Creation                                        |
|     optimizer = DeepSpeedCPUAdam(model.parameters())          |
|     - C++ CPU-optimized Adam, works with pinned memory        |
|                                                               |
|  3. DeepSpeed Engine Initialization                           |
|     model_engine, optimizer, dataloader, _ =                  |
|         deepspeed.initialize(model=model, optimizer=optimizer) |
|     - Wraps model in DeepSpeedEngine                          |
|     - Applies ZeRO Stage 3 parameter partitioning             |
|     - Sets up CPU/NVMe offloading per JSON config             |
|     - If super_offload=true, uses SuperOffloadOptimizer_Stage3|
|       (spawns separate CPU Adam worker process)               |
|                                                               |
|  4. Training Loop (standard PyTorch pattern)                  |
|     outputs = model_engine(**batch)    # forward              |
|     model_engine.backward(loss)        # backward             |
|     model_engine.step()                # optimizer step       |
+---------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------+
|  DeepSpeed Runtime (deepspeed/runtime/)                       |
|                                                               |
|  engine.py:                                                   |
|    - Selects optimizer based on config:                       |
|      if super_offload: SuperOffloadOptimizer_Stage3           |
|      else:             DeepSpeedZeroOptimizer_Stage3          |
|                                                               |
|  zero/stage3.py (ZeRO-Offload / ZeRO-Infinity):              |
|    - Parameter partitioning across GPUs                       |
|    - Forward: all_gather params from CPU -> GPU               |
|    - Backward: reduce_scatter gradients, release params       |
|    - Optimizer step: DeepSpeedCPUAdam on pinned memory        |
|                                                               |
|  superoffload/superoffload_stage3.py (SuperOffload):          |
|    - Inherits from ZeRO Stage 3                               |
|    - Spawns separate CPU Adam worker process                  |
|    - Async optimizer step overlaps with backward pass         |
|    - CPU core isolation via cpuadam_cores_perc                |
|    - Rollback mechanism for gradient overflow/clipping        |
+---------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------+
|  PyTorch / NCCL                                               |
|  - Autograd engine handles backward pass                      |
|  - NCCL handles GPU-to-GPU communication                      |
|  - Pinned memory for efficient CPU-GPU data transfer          |
+---------------------------------------------------------------+
```

### Key Design Insight

The training script (`finetune_zero3.py`) is **identical** across all three offloading strategies. The offload mode is selected entirely through the DeepSpeed JSON config -- no code changes required. This means the same script works for any HuggingFace model by just changing `--model_name`.

## Dependencies

See `requirements.txt`. Key dependencies:

- `torch >= 2.5.1`
- `deepspeed >= 0.17.0` (SuperOffload requires >= 0.17.0)
- `transformers >= 4.56.1`
- `flash-attn >= 2.0.0` (for FlashAttention-2)
- `datasets` (for Alpaca dataset)

Install:
```bash
pip install -r requirements.txt
```

## Quick Start

### Single Model Tests

```bash
# Llama-3.1-8B on 1 GPU
bash finetune_llama-8b_1gpu.sh superoffload     # SuperOffload
bash finetune_llama-8b_1gpu.sh zerooffload      # ZeRO-Offload
bash finetune_llama-8b_1gpu.sh zeroinfinity     # ZeRO-Infinity

# Llama-3.3-70B-Instruct on 4 GPUs
bash finetune_llama-70b_4gpu.sh superoffload
bash finetune_llama-70b_4gpu.sh zerooffload
bash finetune_llama-70b_4gpu.sh zeroinfinity
```

### Custom Batch Size

```bash
bash finetune_llama-8b_1gpu.sh superoffload 2   # batch_size=2
bash finetune_llama-70b_4gpu.sh zeroinfinity 1  # batch_size=1
```

### Micro-Batch Sweep

Run all 3 modes with batch sizes 1, 2, 4, 8:

```bash
# Llama-8B (1 GPU) -- fast iteration
bash sweep_microbatch_llama-8b_1gpu.sh

# Llama-70B (4 GPUs) -- full scale
bash sweep_microbatch_llama-70b_4gpu.sh
```

Each sweep script runs 12 experiments (4 batch sizes x 3 modes). Failed runs (e.g., OOM) are logged and skipped.

## Configuration Reference

### ZeRO-Offload (baseline)

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### ZeRO-Infinity (offload both optimizer and parameters)

```json
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### SuperOffload (async CPU optimizer)

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "ratio": 0.90,
            "super_offload": true,
            "cpuadam_cores_perc": 0.90
        }
    }
}
```

Key SuperOffload-specific fields:
- `super_offload`: Enable async CPU optimizer in a separate process
- `ratio`: Fraction of optimizer work offloaded to CPU (0.0-1.0)
- `cpuadam_cores_perc`: Fraction of CPU cores allocated to the CPU Adam worker process

## finetune_zero3.py Analysis

### Code Flow

1. **Lines 137-147**: Load HuggingFace model via `AutoModelForCausalLM.from_pretrained()`
2. **Lines 150-158**: Enable activation checkpointing (gradient checkpointing) to save GPU memory
3. **Lines 161-168**: Create `DeepSpeedCPUAdam` optimizer
4. **Lines 253-259**: `deepspeed.initialize()` wraps model into engine, applies ZeRO-3 partitioning
5. **Lines 289-360**: Training loop -- standard forward/backward/step

### Known Issues

- **`--warmup` argument is parsed but never used** (line 412). No learning rate scheduler is implemented; training uses a constant learning rate.
- **Optimizer LR mismatch**: `create_optimizer()` uses a hardcoded `lr=0.001` (line 165), but this is overridden by the CLI `--lr` argument during `deepspeed.initialize()`.
- **DataLoader batch_size=1** (line 201): This is intentional -- DeepSpeed ZeRO Stage 3 controls the effective batch size via `train_batch_size` in the JSON config.

## File Structure

```
tests/cpu_offload/
    finetune_zero3.py                    # Training script (from DeepSpeedExamples)
    requirements.txt                     # Python dependencies
    finetune_llama-8b_1gpu.sh            # Llama-8B, 1 GPU, 3 modes
    finetune_llama-70b_4gpu.sh           # Llama-70B, 4 GPUs, 3 modes
    sweep_microbatch_llama-8b_1gpu.sh    # Batch sweep for Llama-8B
    sweep_microbatch_llama-70b_4gpu.sh   # Batch sweep for Llama-70B
    README.md                            # This file
```
