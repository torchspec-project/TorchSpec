#!/bin/bash
# Launch torchspec.train_entry for Kimi-K2.5 BF16 Eagle3 2-node training
#
# Run this on the head node AFTER the Ray cluster is fully ready.
# See examples/kimi-k25-2node-h200/setup_ray_cluster.sh to set up the cluster first.
#
# Node layout:
#   Head node (this node): 8 GPUs (GPU 0-7) — FSDP training
#   Worker node:           8 GPUs (GPU 0-7) — SglEngine inference (TP=8)
#
# Usage:
#   bash examples/kimi-k25-2node-h200/run.sh [extra_overrides...]
#
# Environment variables:
#   TRAIN_GPUS          — GPUs for training (default: 8)
#   INFERENCE_GPUS      — Total inference GPUs (default: 8)
#   CONFIG_FILE         — Override config file path

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
# export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCHSPEC_LOG_LEVEL=INFO

TRAIN_GPUS="${TRAIN_GPUS:-8}"
INFERENCE_GPUS="${INFERENCE_GPUS:-8}"

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/sglang_kimi_k25_2node.yaml}"

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/kimi25_2node_train_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "=============================================="
echo "Kimi-K2.5 BF16 2-Node Training"
echo "=============================================="
echo "Config:                $CONFIG_FILE"
echo "  Training GPUs:  $TRAIN_GPUS (head node)"
echo "  Inference GPUs: $INFERENCE_GPUS (worker node, TP=$INFERENCE_GPUS)"
echo "  dist_init_addr: (auto-negotiated via Ray)"
echo "=============================================="

if ! ray status &>/dev/null; then
  echo "ERROR: Cannot connect to Ray cluster (is Ray running on this node?)"
  echo "Start the cluster first:"
  echo "  NODE_ROLE=head   bash examples/kimi-k25-2node-h200/setup_ray_cluster.sh"
  echo "  HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/kimi-k25-2node-h200/setup_ray_cluster.sh"
  exit 1
fi

echo "=== Launching training ==="
python3 -m torchspec.train_entry \
  --config "$CONFIG_FILE" \
  training.training_num_gpus_per_node="$TRAIN_GPUS" \
  inference.inference_engine_type="sgl" \
  inference.inference_num_gpus="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_engine="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_node="$INFERENCE_GPUS" \
  inference.sglang.tp_size="$INFERENCE_GPUS" \
  model.draft_model_config="$ROOT_DIR/configs/draft_models/kimi_k25_eagle3.json" \
  dataset.last_turn_loss_only=true \
  "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
