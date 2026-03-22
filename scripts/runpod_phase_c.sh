#!/bin/bash
set -euo pipefail

# Phase C: Full DFlash training on 4x H100
# Usage: deploy via base64 to RunPod, run in tmux

cd /workspace/TorchSpec

echo "=== Phase C: Full DFlash Training ==="
echo "Timestamp: $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Pull latest code
echo ""
echo "=== Step 1: Pull latest code ==="
git fetch --all
git checkout feature/dflash-training
git pull fork feature/dflash-training || git reset --hard fork/feature/dflash-training
pip3 install -e ".[dev]" 2>&1 | tail -3
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log -1 --format='%h %s')"

# Step 2: Prepare PerfectBlend dataset (50K samples)
echo ""
echo "=== Step 2: Prepare PerfectBlend dataset ==="
DATA_DIR="/workspace/data"
DATA_FILE="$DATA_DIR/perfectblend_50k.jsonl"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_FILE" ]; then
    LINES=$(wc -l < "$DATA_FILE")
    echo "Dataset already exists: $DATA_FILE ($LINES lines)"
else
    echo "Downloading and preparing PerfectBlend (50K samples)..."
    python3 scripts/prepare_perfectblend.py \
        --output "$DATA_FILE" \
        --sample-size 50000 \
        --seed 42 \
        2>&1
fi

echo "Dataset ready: $(wc -l < "$DATA_FILE") samples"

# Step 3: Verify SGLang patch
echo ""
echo "=== Step 3: Verify SGLang patch ==="
SGLANG_DIR="/workspace/TorchSpec/_sglang"
if [ -d "$SGLANG_DIR" ]; then
    cd "$SGLANG_DIR"
    # Re-apply patch in case code was updated
    git apply /workspace/TorchSpec/patches/sglang/v0.5.8.post1/sglang.patch 2>/dev/null \
        && echo "Patch applied." \
        || echo "Patch already applied."
    cd /workspace/TorchSpec
else
    echo "WARNING: SGLang not found at $SGLANG_DIR"
fi

# Step 4: Launch training
echo ""
echo "=== Step 4: Launch DFlash training ==="
echo "Config: 4x H100, 1 inference + 3 training (FULL_SHARD)"
echo "Dataset: $DATA_FILE (50K samples)"
echo "Epochs: 2, LR: 6e-4, block_size: 16"
echo ""

OUTPUT_DIR="./outputs/qwen3-8b-dflash-phase-c"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR"

# Build resume args if checkpoint exists
RESUME_ARGS=""
if [ -d "$CHECKPOINT_DIR" ] && ls "$CHECKPOINT_DIR"/iter_* >/dev/null 2>&1; then
    echo "Resuming from checkpoint: $(ls -d "$CHECKPOINT_DIR"/iter_* | tail -1)"
    RESUME_ARGS="training.load_path=$CHECKPOINT_DIR"
fi

# All settings in YAML — only override dataset path and output_dir
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
