#!/bin/bash
# Train with HF backend (HFEngine + HFTargetModel)
#
# GPU allocation (default: 3 GPUs):
#   - 1 GPU for inference (HFEngine with HFTargetModel)
#   - 2 GPUs for training (FSDP/DP: draft model sharded across 2 GPUs)
#
# Usage:
#   ./examples/hf-quickstart/run.sh [EXTRA_ARGS...]
#
# Examples:
#   ./examples/hf-quickstart/run.sh
#   ./examples/hf-quickstart/run.sh training.num_train_steps=20

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export TORCHSPEC_LOG_LEVEL=INFO

CONFIG_FILE="$ROOT_DIR/configs/hf_qwen3_8b.yaml"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

TRAIN_GPUS=2
INFERENCE_GPUS=1


echo "=============================================="
echo "HF Backend Training (Eagle3 Distillation)"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Total GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training GPUs: $TRAIN_GPUS (FSDP/DP)"
echo "  - Inference GPUs: $INFERENCE_GPUS (HFTargetModel)"
echo "Extra args: $*"
echo "=============================================="

python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_engine_type="hf" \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_engine=1 \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
