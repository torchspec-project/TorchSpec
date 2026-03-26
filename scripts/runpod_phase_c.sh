#!/bin/bash
set -euo pipefail

# Phase C: Full DFlash training on 4x H100
# Prerequisites: Run scripts/runpod_setup.sh first
#
# Usage:
#   export HF_HOME=/workspace/.cache/huggingface
#   export PYTORCH_ALLOC_CONF=expandable_segments:True
#   nohup bash scripts/runpod_phase_c.sh > /tmp/phase_c.log 2>&1 &

cd /workspace/TorchSpec

echo "=== Phase C: Full DFlash Training ==="
echo "Timestamp: $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log -1 --format='%h %s')"

# ── Dataset ──
DATA_FILE="/workspace/data/perfectblend_50k.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Dataset not found at $DATA_FILE"
    echo "Run scripts/runpod_setup.sh first."
    exit 1
fi
echo "Dataset: $DATA_FILE ($(wc -l < "$DATA_FILE") samples)"

# ── Output dir (matches YAML default) ──
OUTPUT_DIR="./outputs/qwen3-8b-dflash"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR"

# ── Auto-resume from checkpoint ──
RESUME_ARGS=""
if [ -d "$CHECKPOINT_DIR" ] && ls "$CHECKPOINT_DIR"/iter_* >/dev/null 2>&1; then
    LATEST_ITER=$(cat "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo "unknown")
    echo "Resuming from checkpoint: iter_$LATEST_ITER"
    RESUME_ARGS="training.load_path=$CHECKPOINT_DIR"
fi

# ── Print training config summary ──
echo ""
echo "=== Training Configuration ==="
echo "Config YAML: configs/sglang_qwen3_8b_dflash.yaml"
echo "  micro_batch_size: 1"
echo "  accumulation_steps: 4"
echo "  global_batch_size: 1 x 3(dp) x 4(accum) = 12"
echo "  max_seq_length: 2048"
echo "  num_epochs: 2"
echo "  learning_rate: 6e-4"
echo "  warmup_ratio: 0.04"
echo "  fsdp_strategy: FULL_SHARD (3 GPUs)"
echo "  prefetch_depth: 8"
echo "  save_interval: 1000 (keep latest 1)"
echo "  num_anchors: 512, block_size: 16"
echo "  target_layers: 5"
echo "  precision: bf16"
echo ""

# ── Environment (source env file from setup, or use defaults) ──
if [ -f "$PWD/.env.runpod" ]; then
    source "$PWD/.env.runpod"
else
    export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
    export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
    export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS:-ATEN,TRITON}
fi

# ── Launch training ──
# All hyperparams are in the YAML. Only override paths.
python3 -m torchspec.train_entry \
    --config configs/sglang_qwen3_8b_dflash.yaml \
    dataset.train_data_path="$DATA_FILE" \
    dataset.eval_data_path=null \
    dataset.eval_interval=0 \
    output_dir="$OUTPUT_DIR" \
    $RESUME_ARGS \
    2>&1

echo ""
echo "=== Training Complete ==="
echo "Timestamp: $(date)"
echo "Output: $OUTPUT_DIR"
ls -la "$CHECKPOINT_DIR/" 2>/dev/null || echo "No checkpoints found"
