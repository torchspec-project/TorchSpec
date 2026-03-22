#!/bin/bash
# =============================================================================
# RunPod Fresh Pod Setup Script for DFlash Training
# =============================================================================
#
# Consolidates all fixes from docs/inference/dflash/dflash_issues.md
# and commit history (feature/dflash-training branch):
#
#   Issue 1:  RunPod SSH PTY requirement (documented in usage below)
#   Issue 2:  PyTorch 2.4.1 -> 2.9.1 via SGLang [all] (Step 5)
#   Issue 3:  Missing RDMA libs for Mooncake (Step 3)
#   Issue 4:  SGLang/vLLM lazy imports (commit ac4f3ad, already in code)
#   Issue 5:  Container disk 100GB minimum (Step 1 validation)
#   Issue 10: FlexAttention inductor backend (commit fee3156, already in code)
#   Issue 15: Volume disk quota — clear stale caches (Step 12)
#   Issue 19: HF cache miss in Ray workers (Step 9)
#   Issue 23: PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF (Step 8)
#   Issue 24: Don't install flashinfer separately (Step 5 note)
#   Issue 25: factory.py timeout=30 -> 120 (commit cedef38, already in code)
#   Issue 26: TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS (commit 447705a + train_entry.py)
#   Issue 30: FlexAttention recompile overflow (commit 2e105c4, already in code)
#   Issue 31: Mooncake buffer use-after-free (commit cdd18cf, already in code)
#   Issue 32: clone() pinned memory regression (commit 2a6a3d9, already in code)
#
# Also references:
#   - docker/sglang/v0.5.8.post1/Dockerfile.runpod (commit 7813f79)
#   - scripts/runpod_dflash_train.sh (commit b28f06c) — older comprehensive script
#   - docs/inference/dflash/dflash_runpod_guide.md — step-by-step guide
#
# Usage (from local machine):
#   # 1. SSH into pod (Issue 1 — RunPod requires PTY):
#   expect -c '
#     set timeout 60
#     spawn ssh -o StrictHostKeyChecking=no -o RequestTTY=force \
#       -i ~/.ssh/id_ed25519 USER@ssh.runpod.io
#     expect -re {[#\$] }
#     interact
#   '
#
#   # 2. Clone repo (if fresh pod without volume):
#   cd /workspace
#   git clone https://github.com/zhubohao911/TorchSpec.git
#   cd TorchSpec && git checkout feature/dflash-training
#
#   # 3. Run this script:
#   bash scripts/runpod_setup.sh
#
#   # Re-run after pod restart (skip pip installs, just restore + verify):
#   SKIP_INSTALL=1 bash scripts/runpod_setup.sh
#
# Requirements (Issue 5):
#   - RunPod pod with 4x H100 80GB
#   - Container disk: 100 GB minimum (training needs ~30-40 GB)
#   - Base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# =============================================================================

set -euo pipefail

# ── Configuration ──
SGLANG_COMMIT="0f2df9370a1de1b4fb11b071d39ab3ce2287a350"
DATASET_SIZE=50000
DATASET_SEED=42
HF_CHECKPOINT_REPO="Xingh3/dflash-qwen3-8b-1k"
MIN_CONTAINER_DISK_GB=50   # Minimum free space on container disk
SKIP_INSTALL="${SKIP_INSTALL:-0}"  # Set to 1 to skip pip installs on pod restart

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $1${NC}"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"; }
step() { echo -e "\n${GREEN}========== Step $1: $2 ==========${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "============================================="
echo " DFlash RunPod Setup"
echo " Repo: $REPO_DIR"
echo " Date: $(date)"
echo "============================================="

START_TIME=$SECONDS

# ─────────────────────────────────────────────────
# Step 1: Pre-flight checks
# ─────────────────────────────────────────────────
step 1 "Pre-flight checks"

# GPU check
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPUs: ${GPU_COUNT}x ${GPU_NAME}"
if [ "$GPU_COUNT" -lt 4 ]; then
    warn "Only $GPU_COUNT GPUs detected (need 4 for 1-inference + 3-training)"
fi

# Container disk check (Issue 5: need 100GB container disk)
CONTAINER_FREE_GB=$(df -BG / 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
echo "Container disk free: ${CONTAINER_FREE_GB}GB"
if [ "${CONTAINER_FREE_GB:-0}" -lt "$MIN_CONTAINER_DISK_GB" ]; then
    err "Container disk too small: ${CONTAINER_FREE_GB}GB free (need ${MIN_CONTAINER_DISK_GB}GB)"
    err "Provision pod with 100GB container disk (Issue 5)"
    exit 1
fi
log "Pre-flight checks passed"

# ─────────────────────────────────────────────────
# Step 2: Restore git files (pod restart can delete tracked files)
# ─────────────────────────────────────────────────
step 2 "Restore git-tracked files"
if git diff --stat HEAD 2>/dev/null | grep -q .; then
    warn "Modified files detected — restoring git-tracked files"
    git restore .
    log "Git restore done"
else
    log "No git restore needed"
fi
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log -1 --format='%h %s')"

# ─────────────────────────────────────────────────
# Step 3: System libraries (Issue 3: RDMA libs for Mooncake)
# ─────────────────────────────────────────────────
step 3 "Install system libraries (Issue 3)"
if ldconfig -p 2>/dev/null | grep -q libibverbs; then
    log "RDMA libraries already installed"
else
    apt-get update -qq
    apt-get install -y -qq libibverbs-dev librdmacm-dev libnuma-dev > /dev/null 2>&1
    ldconfig
    log "Installed libibverbs-dev, librdmacm-dev, libnuma-dev"
fi

if [ "$SKIP_INSTALL" = "1" ]; then
    log "SKIP_INSTALL=1 — skipping Steps 4-7 (pip, SGLang, patches, TorchSpec)"
else

# ─────────────────────────────────────────────────
# Step 4: Upgrade pip
# ─────────────────────────────────────────────────
step 4 "Upgrade pip"
pip3 install --upgrade pip > /dev/null 2>&1
log "pip upgraded to $(pip3 --version | awk '{print $2}')"

# ─────────────────────────────────────────────────
# Step 5: Install SGLang from source (Issue 2 + Issue 24)
#
# IMPORTANT (Issue 2):  SGLang [all] pulls PyTorch 2.9.1 automatically.
#                       Do NOT install PyTorch separately.
# IMPORTANT (Issue 24): SGLang [all] includes flashinfer_python + flashinfer_cubin.
#                       Do NOT install flashinfer separately.
# ─────────────────────────────────────────────────
step 5 "Install SGLang 0.5.9 from source (Issues 2, 24)"

SGLANG_DIR="$REPO_DIR/_sglang"

if [ ! -d "$SGLANG_DIR/.git" ]; then
    log "Cloning SGLang..."
    git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR" 2>&1 | tail -3
fi

cd "$SGLANG_DIR"
git fetch --all -q 2>/dev/null || true
git checkout "$SGLANG_COMMIT" 2>/dev/null
git reset --hard HEAD 2>/dev/null
cd "$REPO_DIR"
log "SGLang checked out at ${SGLANG_COMMIT:0:12}"

log "Installing SGLang with all extras (this takes 3-5 minutes)..."
pip3 install -e "$SGLANG_DIR/python[all]" 2>&1 | tail -5
log "SGLang installed"

# ─────────────────────────────────────────────────
# Step 6: Apply SGLang patches for speculative training
#
# Order matters (from dflash_runpod_guide.md Step 3):
#   1. pip install SGLang (done above)
#   2. Remove spec_training_info.py (conflicts with patch)
#   3. git apply sglang.patch
# ─────────────────────────────────────────────────
step 6 "Apply SGLang patches"

PATCH_DIR="$REPO_DIR/patches/sglang/v0.5.8.post1"
SPEC_TRAINING_FILE="$SGLANG_DIR/python/sglang/srt/speculative/spec_training_info.py"

# Remove conflicting file AFTER install, BEFORE patch (per guide)
if [ -f "$SPEC_TRAINING_FILE" ]; then
    rm -f "$SPEC_TRAINING_FILE"
    log "Removed conflicting spec_training_info.py"
fi

cd "$SGLANG_DIR"

# Apply main SGLang patch (required for speculative training)
if git apply --check "$PATCH_DIR/sglang.patch" 2>/dev/null; then
    git apply "$PATCH_DIR/sglang.patch"
    log "Applied sglang.patch"
else
    warn "sglang.patch already applied or conflicts — skipping"
fi

# Note: sglang_decode.patch is NOT applied per dflash_runpod_guide.md
# It contains additional decode-related changes that are optional.

cd "$REPO_DIR"

# ─────────────────────────────────────────────────
# Step 7: Install TorchSpec
# ─────────────────────────────────────────────────
step 7 "Install TorchSpec"
pip3 install -e ".[dev]" 2>&1 | tail -3
log "TorchSpec installed"

fi  # end SKIP_INSTALL

# ─────────────────────────────────────────────────
# Step 8: Environment variables
#
# Issue 23: PYTORCH_CUDA_ALLOC_CONF deprecated in 2.9+, use PYTORCH_ALLOC_CONF
# Issue 26: TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS already set in train_entry.py
#           via os.environ.setdefault(), but also export here for any manual usage
# ─────────────────────────────────────────────────
step 8 "Configure environment (Issues 23, 26)"

# Write env vars to a sourceable file
ENV_FILE="$REPO_DIR/.env.runpod"
cat > "$ENV_FILE" << 'ENVEOF'
# RunPod environment for DFlash training
# Source this: source .env.runpod

# Issue 19: HF cache path for Ray workers
export HF_HOME=/workspace/.cache/huggingface

# Issue 23: PyTorch 2.9+ renamed this env var
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Issue 26: Fix PyTorch 2.9+ TorchInductor GEMM backend regression (3x slower without this)
# Also set in train_entry.py via os.environ.setdefault(), but export here for safety
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
ENVEOF

# Source it now
source "$ENV_FILE"
log "Environment configured (saved to .env.runpod)"
echo "  HF_HOME=$HF_HOME"
echo "  PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
echo "  TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=$TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"

# ─────────────────────────────────────────────────
# Step 9: HuggingFace cache symlink (Issue 19)
#
# Ray workers run as root and look at /root/.cache/huggingface/
# but models are cached at /workspace/.cache/huggingface/
# ─────────────────────────────────────────────────
step 9 "Setup HuggingFace cache (Issue 19)"
mkdir -p "$HF_HOME"

if [ -L /root/.cache/huggingface ]; then
    log "HF cache symlink already exists"
elif [ -d /root/.cache/huggingface ]; then
    # Real directory exists — merge into workspace cache
    warn "/root/.cache/huggingface is a real directory — moving to workspace"
    cp -rn /root/.cache/huggingface/* "$HF_HOME/" 2>/dev/null || true
    rm -rf /root/.cache/huggingface
    ln -s "$HF_HOME" /root/.cache/huggingface
    log "Moved existing cache and created symlink"
else
    mkdir -p /root/.cache
    ln -s "$HF_HOME" /root/.cache/huggingface
    log "Symlinked /root/.cache/huggingface -> $HF_HOME"
fi

# ─────────────────────────────────────────────────
# Step 10: Prepare PerfectBlend dataset
# ─────────────────────────────────────────────────
step 10 "Prepare PerfectBlend dataset"
DATA_DIR="/workspace/data"
DATA_FILE="$DATA_DIR/perfectblend_50k.jsonl"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_FILE" ]; then
    LINES=$(wc -l < "$DATA_FILE")
    log "Dataset already exists: $DATA_FILE ($LINES lines)"
else
    log "Downloading and preparing PerfectBlend (${DATASET_SIZE} samples)..."
    python3 scripts/prepare_perfectblend.py \
        --output "$DATA_FILE" \
        --sample-size "$DATASET_SIZE" \
        --seed "$DATASET_SEED" \
        2>&1 | tail -5
    LINES=$(wc -l < "$DATA_FILE")
    log "Dataset ready: $LINES samples"
fi

# ─────────────────────────────────────────────────
# Step 11: Download checkpoint from HF (if not present)
# ─────────────────────────────────────────────────
step 11 "Check/download training checkpoint"
CHECKPOINT_DIR="$REPO_DIR/outputs/qwen3-8b-dflash/checkpoints"

if [ -d "$CHECKPOINT_DIR" ] && ls "$CHECKPOINT_DIR"/iter_* > /dev/null 2>&1; then
    LATEST=$(cat "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo "unknown")
    log "Checkpoint exists: iter_$LATEST"
else
    log "Downloading checkpoint from HF: $HF_CHECKPOINT_REPO"
    mkdir -p "$CHECKPOINT_DIR"
    huggingface-cli download "$HF_CHECKPOINT_REPO" \
        --local-dir "$CHECKPOINT_DIR/iter_0001001" \
        --repo-type model 2>&1 | tail -5
    echo "1001" > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
    log "Checkpoint downloaded to $CHECKPOINT_DIR/iter_0001001"
fi

# ─────────────────────────────────────────────────
# Step 12: Fix Mooncake binary permissions
#
# From Dockerfile.runpod (commit 7813f79):
#   mooncake_master binary needs execute permission
#   Also prevent mooncake from modifying binary permissions at runtime
# ─────────────────────────────────────────────────
step 12 "Fix Mooncake permissions (from Dockerfile)"

MOONCAKE_PKG_DIR=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))" 2>/dev/null || echo "")
if [ -n "$MOONCAKE_PKG_DIR" ]; then
    MOONCAKE_BIN="$MOONCAKE_PKG_DIR/mooncake_master"
    if [ -f "$MOONCAKE_BIN" ]; then
        chmod 755 "$MOONCAKE_BIN"
        log "Fixed mooncake_master binary permissions"
    fi
    # Prevent mooncake from trying to chmod at runtime (may fail in containers)
    MOONCAKE_CLI="$MOONCAKE_PKG_DIR/cli.py"
    if [ -f "$MOONCAKE_CLI" ] && grep -q "os.chmod(bin_path, 0o755)" "$MOONCAKE_CLI"; then
        sed -i 's/os.chmod(bin_path, 0o755)/pass/' "$MOONCAKE_CLI"
        log "Patched mooncake cli.py to skip runtime chmod"
    fi
else
    warn "Could not locate mooncake package directory"
fi

# ─────────────────────────────────────────────────
# Step 13: Clear stale caches
#
# Issue 15: Volume disk quota can fill up. Use container disk for temp files.
# Stale tokenizer/inductor caches from previous runs can cause issues.
# ─────────────────────────────────────────────────
step 13 "Clear stale caches"

# Clear tokenized dataset cache (may be stale if seq_length changed)
CACHE_DIR="$REPO_DIR/cache"
if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
    log "Cleared tokenized dataset cache"
else
    log "No stale cache found"
fi

# Clear torch inductor cache (recompiles on fresh pod anyway)
if [ -d /tmp/torchinductor_root ]; then
    rm -rf /tmp/torchinductor_root
    log "Cleared torch inductor cache"
fi

# ─────────────────────────────────────────────────
# Step 13: Verify installation
# ─────────────────────────────────────────────────
step 14 "Verify installation"

ERRORS=0

# PyTorch (Issue 2: need 2.6+ for FlexAttention)
TORCH_VER=$(python3 -c 'import torch; print(torch.__version__)' 2>&1) || true
CUDA_OK=$(python3 -c 'import torch; print(torch.cuda.is_available())' 2>&1) || true
if [[ "$CUDA_OK" == "True" ]]; then
    log "PyTorch $TORCH_VER with CUDA"
else
    err "PyTorch CUDA not available: $TORCH_VER / $CUDA_OK"
    ERRORS=$((ERRORS + 1))
fi

# SGLang
if python3 -c 'import sglang; print("OK")' 2>/dev/null | grep -q OK; then
    log "SGLang OK"
else
    err "SGLang import failed"
    ERRORS=$((ERRORS + 1))
fi

# TorchSpec
if python3 -c 'import torchspec; print("OK")' 2>/dev/null | grep -q OK; then
    log "TorchSpec OK"
else
    err "TorchSpec import failed"
    ERRORS=$((ERRORS + 1))
fi

# Mooncake (Issue 3: needs RDMA libs)
if python3 -c 'import mooncake; print("OK")' 2>/dev/null | grep -q OK; then
    log "Mooncake OK"
else
    err "Mooncake import failed — check RDMA libs (Issue 3)"
    ERRORS=$((ERRORS + 1))
fi

# FlexAttention (Issue 2: needs PyTorch 2.6+)
if python3 -c 'from torch.nn.attention.flex_attention import flex_attention; print("OK")' 2>/dev/null | grep -q OK; then
    log "FlexAttention OK"
else
    err "FlexAttention not available — need PyTorch 2.6+ (Issue 2)"
    ERRORS=$((ERRORS + 1))
fi

# DFlash model imports (from runpod_dflash_train.sh verification)
if python3 -c 'from torchspec import DFlashModel, DFlashConfig, DFlashDraftModel; print("OK")' 2>/dev/null | grep -q OK; then
    log "DFlash model imports OK"
else
    err "DFlash model imports failed"
    ERRORS=$((ERRORS + 1))
fi

# Dataset
if [ -f "$DATA_FILE" ]; then
    log "Dataset: $(wc -l < "$DATA_FILE") samples at $DATA_FILE"
else
    err "Dataset not found at $DATA_FILE"
    ERRORS=$((ERRORS + 1))
fi

# Checkpoint
if [ -d "$CHECKPOINT_DIR" ] && ls "$CHECKPOINT_DIR"/iter_* > /dev/null 2>&1; then
    log "Checkpoint ready for resume"
else
    warn "No checkpoint — will train from scratch"
fi

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "============================================="
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN} Setup complete in ${ELAPSED}s — all checks passed${NC}"
else
    echo -e "${RED} Setup complete in ${ELAPSED}s — $ERRORS error(s)${NC}"
fi
echo "============================================="
echo ""
echo "To start training:"
echo "  source .env.runpod"
echo "  nohup bash scripts/runpod_phase_c.sh > /tmp/phase_c.log 2>&1 &"
echo "  tail -f /tmp/phase_c.log"
echo ""
echo "To monitor:"
echo "  tail -f /tmp/phase_c.log"
echo "  grep 'TIMING step=' /tmp/phase_c.log | tail -5"
echo ""
