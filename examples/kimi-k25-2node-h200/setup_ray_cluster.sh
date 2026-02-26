#!/bin/bash
# Ray cluster setup for Kimi-K2.5 BF16 Eagle3 2-node training
#
# Node layout:
#   Node 0 (head):   Ray head + training, 8 GPUs (GPU 0-7)
#   Node 1 (worker): inference, 8 GPUs (TP=8)
#
# Usage — run on each node with the appropriate NODE_ROLE:
#
#   # Node 0 (Ray head + training):
#   NODE_ROLE=head bash examples/kimi-k25-2node-h200/setup_ray_cluster.sh
#
#   # Node 1 (inference worker):
#   HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/kimi-k25-2node-h200/setup_ray_cluster.sh
#
# Environment variables:
#   HEAD_IP      — IP of node 0 (Ray head). Required only for worker nodes.
#   NODE_ROLE    — "head" | "worker"
#   RAY_PORT     — Ray GCS port (default: 6380)

set -euo pipefail
set -x

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export SGLANG_DISABLE_CUDNN_CHECK=1
export TORCHSPEC_LOG_LEVEL=INFO

NODE_ROLE="${NODE_ROLE:?NODE_ROLE must be set to head or worker}"
RAY_PORT="${RAY_PORT:-6380}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_torchspec_kimi25_$(id -u)}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/kimi25_2node_ray_${NODE_ROLE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

echo "=============================================="
echo "Kimi-K2.5 BF16 2-Node Ray Cluster"
echo "NODE_ROLE: $NODE_ROLE  LOCAL_IP: $LOCAL_IP"
echo "=============================================="

case "$NODE_ROLE" in
  head)
    echo "=== Starting Ray HEAD node (training, 8 GPUs, GPU 0-7) ==="
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ray stop --force 2>/dev/null || true
    ray start --head \
      --port "$RAY_PORT" \
      --temp-dir "$RAY_TEMP_DIR" \
      --num-gpus 8 \
      --disable-usage-stats
    echo "Ray head started at $LOCAL_IP:$RAY_PORT"
    echo "Next steps:"
    echo "  Worker node: HEAD_IP=$LOCAL_IP NODE_ROLE=worker bash examples/kimi-k25-2node-h200/setup_ray_cluster.sh"
    echo "  Training:    bash examples/kimi-k25-2node-h200/run.sh"
    ;;

  worker)
    HEAD_IP="${HEAD_IP:?HEAD_IP must be set to node 0 IP address}"
    RAY_ADDR="${HEAD_IP}:${RAY_PORT}"
    export RAY_ADDRESS="$RAY_ADDR"
    echo "=== Joining Ray cluster as WORKER (inference, 8 GPUs) ==="
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ray stop --force 2>/dev/null || true
    ray start \
      --address "$RAY_ADDR" \
      --num-gpus 8 \
      --disable-usage-stats
    echo "Worker joined Ray cluster at $RAY_ADDR"
    ;;

  *)
    echo "ERROR: NODE_ROLE must be 'head' or 'worker'"
    echo "  head   — Ray head + training node (8 GPUs, GPU 0-7)"
    echo "  worker — inference worker node (8 GPUs)"
    exit 1
    ;;
esac
