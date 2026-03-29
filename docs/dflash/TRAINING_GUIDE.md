# DFlash Training Guide

Train a DFlash block-diffusion draft model for speculative decoding with Qwen3-8B.

DFlash predicts 16-token blocks in parallel using dual-source KV attention from the target model's hidden states. The training architecture is disaggregated — inference and training run on separate GPUs, communicating via the Mooncake transfer engine.

---

## Prerequisites

### Hardware

| Setup | GPUs | VRAM | Notes |
|-------|------|------|-------|
| **Multi-GPU (recommended)** | 4x H100 80GB | 320 GB total | 1 inference + 3 training (FSDP) |
| **Single-GPU** | 1x H100/A100 80GB | 80 GB | Colocate mode, no FSDP |

- **Disk**: ~100 GB free (model weights ~16 GB, dataset ~1 GB, checkpoints ~30 GB)
- **CUDA**: 12.4+
- **Python**: 3.11+

### System Libraries

Required for Mooncake transfer engine (RDMA support):

```bash
# Ubuntu/Debian
sudo apt-get install -y libibverbs-dev librdmacm-dev libnuma-dev

# RHEL/CentOS
sudo yum install -y libibverbs-devel librdmacm-devel numactl-devel
```

---

## Installation

There are two ways to set up the environment: **Docker image** (recommended) or **manual installation**.

### Option A: Docker Image (Recommended)

We provide pre-built Docker images with all dependencies (SGLang, PyTorch 2.9.1, Mooncake, TorchSpec) already installed. This avoids 20+ minutes of manual setup.

**Two image variants:**

| Image | Size | Description |
|-------|------|-------------|
| `xingh3/torchspec-dflash:latest` | ~20 GB | Full — includes pre-downloaded Qwen3-8B weights |
| `xingh3/torchspec-dflash:slim` | ~4 GB | Slim — model downloads at runtime (~5 min) |

**Use the pre-built image directly:**

```bash
docker pull xingh3/torchspec-dflash:latest
docker run --gpus all -it --shm-size=16g \
  -v /path/to/your/data:/workspace/data \
  -v /path/to/your/outputs:/workspace/outputs \
  xingh3/torchspec-dflash:latest bash
```

Inside the container, the repo is at `/root/torchspec` with everything pre-installed. Environment variables (`HF_HOME`, `PYTORCH_ALLOC_CONF`, `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`) are already set.

**Or build the image yourself:**

```bash
cd TorchSpec

# Full image (~20GB) — Qwen3-8B pre-downloaded, instant start:
docker build -t your-registry/torchspec-dflash:latest \
  -f docker/sglang/v0.5.8.post1/Dockerfile.runpod .

# Slim image (~4GB) — model downloads at runtime:
docker build --build-arg INCLUDE_MODEL=0 \
  -t your-registry/torchspec-dflash:slim \
  -f docker/sglang/v0.5.8.post1/Dockerfile.runpod .
```

The Dockerfile handles: SGLang 0.5.9 installation, SGLang patch application, system libraries (libibverbs, rdmacm, libnuma), TorchSpec installation, Mooncake permissions, and all environment variables.

After launching with Docker, skip to [Prepare Training Data](#prepare-training-data).

### Option B: Manual Installation

If you prefer to install without Docker (e.g., bare-metal GPU servers, Slurm clusters):

#### Step 1: Clone and checkout

```bash
git clone https://github.com/zhubohao911/TorchSpec.git
cd TorchSpec
git checkout feature/dflash-training
```

#### Step 2: Install SGLang 0.5.9 (multi-GPU mode only)

SGLang provides the inference engine and pulls in PyTorch 2.9.1:

```bash
pip install "sglang[all]==0.5.9" --find-links https://flashinfer.ai/whl/cu124/torch2.9/flashinfer-python
```

> **Single-GPU mode** uses the HuggingFace engine instead — SGLang is not required.

#### Step 3: Apply SGLang patch

The patch adds speculative training hooks (`enable_aux_hidden_states`, etc.):

```bash
SGLANG_DIR=$(python -c "import sglang; print(sglang.__path__[0])")
cd "$(dirname "$SGLANG_DIR")"
git apply /path/to/TorchSpec/patches/sglang/v0.5.8.post1/sglang.patch
cd /path/to/TorchSpec
```

#### Step 4: Install TorchSpec

```bash
pip install -e ".[dev]"
```

#### Step 5: Set environment variables

These are **required** for correct training behavior:

```bash
export HF_HOME=/path/to/your/huggingface/cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
```

You may want to add these to your shell profile or a `.env` file.

#### Step 6: Verify installation

```bash
python -c "
import torchspec
import torch
print(f'TorchSpec installed')
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```

---

## Prepare Training Data

DFlash expects a JSONL file with conversation data:

```json
{"id": "conv_001", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"id": "conv_002", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Recommended dataset** — PerfectBlend 50K (good mix of instruction-following data):

```bash
python scripts/tools/prepare_perfectblend.py \
  --output /path/to/data/perfectblend_50k.jsonl \
  --max_samples 50000
```

Or bring your own JSONL in the same format. Sequences shorter than 32 tokens of supervised content are automatically filtered.

---

## Training

### Option A: Multi-GPU (4x H100 — recommended)

Uses SGLang for inference on 1 GPU, FSDP training on 3 GPUs:

```bash
python -m torchspec.train_entry \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  dataset.train_data_path=/path/to/data/perfectblend_50k.jsonl \
  output_dir=./outputs/qwen3-8b-dflash
```

Key hyperparameters in `configs/sglang_qwen3_8b_dflash.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `micro_batch_size` | 1 | Per-GPU batch size |
| `draft_accumulation_steps` | 4 | Gradient accumulation |
| `global_batch_size` | 12 | 1 x 3 GPUs x 4 accum |
| `max_seq_length` | 2048 | Max tokens per sequence |
| `learning_rate` | 6e-4 | AdamW learning rate |
| `warmup_ratio` | 0.04 | Warmup fraction |
| `num_epochs` | 3 | Training epochs |
| `fsdp_strategy` | FULL_SHARD | Shards params + optimizer |
| `dflash_block_size` | 16 | Block-parallel prediction size |
| `dflash_num_anchors` | 512 | Anchor positions per sequence |
| `save_interval` | 1000 | Checkpoint every N steps |

Expected throughput: **~5 step/s** (~15 samples/sec).

### Option B: Single-GPU (1x H100/A100 80GB)

Uses HuggingFace engine for inference, training on the same GPU (colocate mode):

```bash
python -m torchspec.train_entry \
  --config configs/hf_qwen3_8b_dflash_1gpu.yaml \
  dataset.train_data_path=/path/to/data/perfectblend_50k.jsonl \
  output_dir=./outputs/qwen3-8b-dflash-1gpu
```

Single-GPU differences: no FSDP, lower accumulation steps, HF inference engine.

### Resume from Checkpoint

Training auto-resumes when a checkpoint directory is specified:

```bash
python -m torchspec.train_entry \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  dataset.train_data_path=/path/to/data/perfectblend_50k.jsonl \
  training.load_path=./outputs/qwen3-8b-dflash/checkpoints
```

The trainer reads `latest_checkpointed_iteration.txt` in the checkpoint directory and resumes from that step, restoring model weights, optimizer state, LR scheduler, and RNG state.

---

## Monitoring

### Console Output

| Metric | Healthy | Problem |
|--------|---------|---------|
| step time | ~1.0 s/step | >2 s/step |
| loss | Decreasing (10 -> 3 over epoch 1) | Flat or increasing |
| accuracy | Increasing (0 -> 0.25 over epoch 1) | Stuck at 0 |
| data_time | <0.7 s | >1 s consistently |

### Useful Log Filters

```bash
grep 'TIMING step=' training.log | tail -5
grep 'Training:' training.log | tail -3
```

### Weights & Biases

WandB logging is supported out of the box. Set `WANDB_API_KEY` and `WANDB_PROJECT` before launching.

---

## After Training

### 1. Extract Checkpoint

Convert the FSDP distributed checkpoint to a single file:

```bash
python scripts/tools/extract_dflash_checkpoint.py \
  --checkpoint_dir outputs/qwen3-8b-dflash/checkpoints/iter_NNNNNNN \
  --output /tmp/dflash_draft.pt
```

### 2. Benchmark with SGLang

Run the SGLang inference benchmark on Modal:

```bash
modal run --env sandbox scripts/modal/modal_dflash_benchmark_sglang.py
```

**Targets**: tau >= 3.0, speedup >= 1.5x over autoregressive baseline.

### 3. Upload Checkpoint (optional)

```bash
huggingface-cli upload YOUR_ORG/dflash-qwen3-8b \
  outputs/qwen3-8b-dflash/checkpoints/iter_NNNNNNN/ \
  --repo-type model
```

---

## Configuration Reference

### Customizing GPU Allocation

In the YAML config:

```yaml
training:
  training_num_gpus_per_node: 3   # Number of GPUs for training
  training_num_nodes: 1

inference:
  inference_num_gpus: 1            # Number of GPUs for inference
  inference_num_gpus_per_node: 1
```

Total GPUs used = `training_num_gpus_per_node` + `inference_num_gpus`. Adjust based on your cluster.

### DFlash-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dflash_block_size` | 16 | Tokens predicted per block |
| `dflash_num_anchors` | 512 | Anchor positions per sequence (biggest speed lever — halving this halves FlexAttention cost) |
| `dflash_loss_decay_gamma` | 7.0 | Exponential decay rate for position-wise loss weighting |
| `dflash_num_target_layers` | 5 | Number of target model layers to project from |

### Draft Model Architecture

Defined in `torchspec/config/dflash_draft_config.json`:

- 5-layer transformer (~1.05B trainable parameters)
- Frozen embedding + LM head shared from Qwen3-8B (~622M frozen)
- Hidden size: 4096, 32 attention heads, 8 KV heads (GQA)
- Target layer IDs: [1, 9, 17, 25, 33] (evenly spaced across Qwen3-8B's 36 layers)

---

## Platform: Modal

Training and benchmarking on Modal uses the scripts in `scripts/modal/`:

| Script | Purpose |
|--------|---------|
| `scripts/modal/modal_dflash_train.py` | Launch DFlash training on Modal (8x H100) |
| `scripts/modal/modal_dflash_benchmark_sglang.py` | SGLang inference benchmark (tau, speedup) |
| `scripts/modal/setup_modal_secrets.sh` | One-time Modal secrets configuration |

```bash
# Setup secrets (one-time)
bash scripts/modal/setup_modal_secrets.sh

# Launch training
modal run --env sandbox scripts/modal/modal_dflash_train.py

# Benchmark inference
modal run --env sandbox scripts/modal/modal_dflash_benchmark_sglang.py
```

---

## Platform: RunPod

### Quick Start

**Option A: Custom Docker Image (Recommended — <1 min setup)**

Create a RunPod pod with:

| Setting | Value |
|---------|-------|
| Container Image | `xingh3/torchspec-dflash:latest` (or `:slim`) |
| GPUs | 4x H100 80GB |
| Container Disk | **100 GB** |
| Volume Disk | Optional (checkpoint persistence) |

On the pod:

```bash
cd /workspace
git clone https://github.com/zhubohao911/TorchSpec.git
cd TorchSpec && git checkout feature/dflash-training

bash scripts/runpod/runpod_setup.sh

source .env.runpod
tmux new -s train
bash scripts/runpod/runpod_phase_c.sh 2>&1 | tee /tmp/phase_c.log
# Detach: Ctrl+B, then D
```

**Option B: Manual Setup (RunPod stock image — ~20 min setup)**

Use container image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with the same pod settings. The `runpod_setup.sh` script handles full dependency installation.

### What Survives Pod Restart

| Survives (`/workspace/`) | Lost (container disk) |
|---|---|
| Git repo, checkpoints, training data | All pip packages (unless custom image) |
| HF model cache | System packages (libibverbs, etc.) |

> **After restart**: Run `bash scripts/runpod/runpod_setup.sh` — it auto-detects what's missing.

### SSH Access

RunPod requires a pseudo-terminal. Use `expect`:

```bash
expect -c '
set timeout 60
spawn ssh -o StrictHostKeyChecking=no -o RequestTTY=force \
    -i ~/.ssh/id_ed25519 USER@ssh.runpod.io
expect -re {[#\$] }
interact
'
```

Or use the **Jupyter terminal** in the RunPod web UI (no SSH needed).

### Scripts Reference

| Script | Purpose | When to run |
|--------|---------|-------------|
| `scripts/runpod/runpod_setup.sh` | Install deps, dataset, checkpoint | Once per new pod |
| `scripts/runpod/runpod_phase_c.sh` | Launch/resume training | After setup |
| `scripts/runpod/runpod_dflash_train.sh` | Alternative training launcher | Manual runs |
| `scripts/runpod/runpod_dflash_test.sh` | Validation/test script | After training |
| `scripts/runpod/runpod_ssh.sh` | SSH helper (expect wrapper) | For remote access |

#### `runpod_setup.sh` — What it does

1. Pre-flight checks (GPU count, disk space)
2. Restore git-tracked files (if pod was restarted)
3. Install system libs: libibverbs, rdmacm, libnuma
4. Install SGLang 0.5.9 from source with `[all]` extras
5. Apply SGLang speculative training patch
6. Install TorchSpec
7. Configure environment variables
8. Setup HF cache symlink for Ray workers
9. Download PerfectBlend 50K dataset
10. Download checkpoint from HF (for resume)
11. Fix Mooncake binary permissions
12. Clear stale caches
13. Verify all imports

> Auto-detects custom Docker image and skips Steps 4-7 (`SKIP_INSTALL=1`).

#### `runpod_phase_c.sh` — What it does

1. Validates dataset exists
2. Auto-detects checkpoint for resume (`training.load_path`)
3. Sources `.env.runpod` for environment variables
4. Launches training with YAML config

### Monitoring on RunPod

```bash
tmux attach -t train                            # reattach to session
tail -f /tmp/phase_c.log                        # live output
grep 'TIMING step=' /tmp/phase_c.log | tail -5  # step timing
grep 'Training:' /tmp/phase_c.log | tail -3     # progress bar
```

#### tmux Cheat Sheet

| Action | Keys |
|--------|------|
| Detach (leave running) | `Ctrl+B`, then `D` |
| New window | `Ctrl+B`, then `C` |
| Switch windows | `Ctrl+B`, then `0`/`1`/`2` |
| Reattach | `tmux attach -t train` |

---

## Troubleshooting

| Issue | Symptom | Fix |
|-------|---------|-----|
| Missing RDMA libs | `ImportError: libibverbs.so.1` | Install system libs (see Prerequisites) |
| SGLang patch not applied | `unexpected keyword argument 'enable_aux_hidden_states'` | Re-apply the SGLang patch (Step 3) |
| Slow training (3x expected) | step time >3 s | Set `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` |
| OOM on A100 40GB | CUDA out of memory | Use H100/A100 80GB, or reduce `max_seq_length` / `dflash_num_anchors` |
| Loss stuck at ~10 | No convergence after 500 steps | Check data format, ensure conversations have assistant turns |
| `No module named 'sglang'` | Import error | Install SGLang (Step 2) or run `runpod_setup.sh` |
| HF cache miss in Ray workers | Model re-downloads on each step | Set `HF_HOME` env var before launching |
| Git files deleted after restart | `No module named torchspec` | `git restore .` (RunPod setup script does this) |
| Standard SSH fails (RunPod) | `Your SSH client doesn't support PTY` | Use `expect` wrapper or Jupyter terminal |
| Ray actor timeout (RunPod) | `GetTimeoutError` during engine init | Setup script patches factory.py |
| Container disk too small | `No space left on device` | Use 100GB container disk |
| Training starts from step 0 | Loss ~10, accuracy 0 | Check checkpoint download in setup |
