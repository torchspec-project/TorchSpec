# DFlash RunPod Setup & Training Guide

> Updated 2026-03-22. Covers both custom Docker image and manual setup.

## Quick Start (Two Options)

### Option A: Custom Docker Image (Recommended — <1 min setup)

**1. Build & push image** (one-time, from your local machine):

```bash
cd /path/to/TorchSpec

# Full image (~20GB) — Qwen3-8B pre-downloaded, instant pod start:
docker build -t xingh3/torchspec-dflash:latest \
  -f docker/sglang/v0.5.8.post1/Dockerfile.runpod .

# Or slim image (~4GB) — model downloads at runtime (~5 min on pod):
docker build --build-arg INCLUDE_MODEL=0 \
  -t xingh3/torchspec-dflash:slim \
  -f docker/sglang/v0.5.8.post1/Dockerfile.runpod .

docker push xingh3/torchspec-dflash:latest
docker push xingh3/torchspec-dflash:slim
```

**2. Create RunPod pod:**


| Setting         | Value                                         |
| --------------- | --------------------------------------------- |
| Container Image | `xingh3/torchspec-dflash:latest` (or `:slim`) |
| GPUs            | 4x H100 80GB                                  |
| Container Disk  | **100 GB**                                    |
| Volume Disk     | Optional (checkpoint persistence)             |


**3. On the pod** (Jupyter terminal or SSH):

```bash
cd /workspace
git clone https://github.com/zhubohao911/TorchSpec.git
cd TorchSpec && git checkout feature/dflash-training

# Setup (auto-detects Docker image, skips pip installs):
bash scripts/runpod_setup.sh

# Launch training in tmux:
source .env.runpod
tmux new -s train
bash scripts/runpod_phase_c.sh 2>&1 | tee /tmp/phase_c.log
# Detach: Ctrl+B, then D
```

### Option B: Manual Setup (RunPod stock image — ~20 min setup)

**1. Create RunPod pod:**


| Setting         | Value                                                      |
| --------------- | ---------------------------------------------------------- |
| Container Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| GPUs            | 4x H100 80GB                                               |
| Container Disk  | **100 GB**                                                 |
| Volume Disk     | Optional                                                   |


**2. On the pod** (Jupyter terminal or SSH):

```bash
cd /workspace
git clone https://github.com/zhubohao911/TorchSpec.git
cd TorchSpec && git checkout feature/dflash-training

# Full setup (installs all deps — takes ~20 min):
bash scripts/runpod_setup.sh

# Launch training in tmux:
source .env.runpod
tmux new -s train
bash scripts/runpod_phase_c.sh 2>&1 | tee /tmp/phase_c.log
# Detach: Ctrl+B, then D
```

---

## Pod Requirements


| Spec           | Value                                         |
| -------------- | --------------------------------------------- |
| GPUs           | 4x H100 80GB (1 inference + 3 training)       |
| Container Disk | **100 GB** (training needs ~30-40 GB)         |
| Volume Disk    | Optional (checkpoint persistence across pods) |


## What Survives Pod Restart


| Survives (`/workspace/`)             | Lost (container disk)                  |
| ------------------------------------ | -------------------------------------- |
| Git repo, checkpoints, training data | All pip packages (unless custom image) |
| HF model cache                       | System packages (libibverbs, etc.)     |


> **After restart**: Run `bash scripts/runpod_setup.sh` — it auto-detects what's missing.

## SSH Access

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

---

## Scripts Reference


| Script                      | Purpose                           | When to run      |
| --------------------------- | --------------------------------- | ---------------- |
| `scripts/runpod_setup.sh`   | Install deps, dataset, checkpoint | Once per new pod |
| `scripts/runpod_phase_c.sh` | Launch/resume training            | After setup      |


### `runpod_setup.sh` — What it does

1. Pre-flight checks (GPU count, disk space)
2. Restore git-tracked files (if pod was restarted)
3. Install system libs: libibverbs, rdmacm, libnuma (Issue 3)
4. Install SGLang 0.5.9 from source with `[all]` extras (Issues 2, 24)
5. Apply SGLang speculative training patch
6. Install TorchSpec
7. Configure environment variables (Issues 23, 26)
8. Setup HF cache symlink for Ray workers (Issue 19)
9. Download PerfectBlend 50K dataset
10. Download checkpoint from HF (for resume)
11. Fix Mooncake binary permissions
12. Clear stale caches
13. Verify all imports

> Auto-detects custom Docker image and skips Steps 4-7 (`SKIP_INSTALL=1`).

### `runpod_phase_c.sh` — What it does

1. Validates dataset exists
2. Auto-detects checkpoint for resume (`training.load_path`)
3. Sources `.env.runpod` for environment variables
4. Launches training with YAML config (no CLI overrides for hyperparams)

---

## Training Configuration

All hyperparameters are in `configs/sglang_qwen3_8b_dflash.yaml`:


| Parameter          | Value                                        |
| ------------------ | -------------------------------------------- |
| Target model       | Qwen/Qwen3-8B                                |
| Draft model        | DFlash (1.05B trainable + 622M frozen embed) |
| GPUs               | 1 inference + 3 training (4x H100)           |
| FSDP               | FULL_SHARD, bf16 reduce                      |
| micro_batch_size   | 1                                            |
| accumulation_steps | 4                                            |
| global_batch_size  | 1 × 3(dp) × 4(accum) = 12                    |
| max_seq_length     | 2048                                         |
| num_epochs         | 2                                            |
| learning_rate      | 6e-4                                         |
| warmup_ratio       | 0.04                                         |
| prefetch_depth     | 8                                            |
| save_interval      | 1000 (keep latest 1)                         |
| num_anchors        | 512, block_size                              |
| target_layers      | 5                                            |
| precision          | bf16                                         |


---

## Monitoring Training

```bash
# Reattach to tmux session:
tmux attach -t train

# Or from another terminal/window:
tail -f /tmp/phase_c.log                        # live output
grep 'TIMING step=' /tmp/phase_c.log | tail -5  # step timing
grep 'Training:' /tmp/phase_c.log | tail -3     # progress bar
```

### tmux Cheat Sheet


| Action                 | Keys                       |
| ---------------------- | -------------------------- |
| Detach (leave running) | `Ctrl+B`, then `D`         |
| New window             | `Ctrl+B`, then `C`         |
| Switch windows         | `Ctrl+B`, then `0`/`1`/`2` |
| Reattach               | `tmux attach -t train`     |


### What to Look For


| Metric    | Healthy                          | Problem            |
| --------- | -------------------------------- | ------------------ |
| step time | ~1.0 s/step                      | >2 s/step          |
| loss      | Decreasing (10→3 over epoch 1)   | Flat or increasing |
| accuracy  | Increasing (0→0.25 over epoch 1) | Stuck at 0         |
| data_time | <0.7s                            | >1s consistently   |


---

## After Training Completes

**1. Extract checkpoint:**

```bash
python scripts/extract_dflash_checkpoint.py \
  --checkpoint_dir outputs/qwen3-8b-dflash/checkpoints/iter_NNNNNNN \
  --output /tmp/dflash_draft.pt
```

**2. Benchmark τ:**

```bash
python scripts/benchmark_dflash_inference.py \
  --target_model Qwen/Qwen3-8B \
  --draft_checkpoint /tmp/dflash_draft.pt \
  --num_prompts 20 --max_new_tokens 256
```

**3. Target**: τ ≥ 3.0, speedup ≥ 1.5x over baseline.

**4. Upload checkpoint to HF:**

```bash
huggingface-cli upload Xingh3/dflash-qwen3-8b-STEP \
  outputs/qwen3-8b-dflash/checkpoints/iter_NNNNNNN/ \
  --repo-type model
```

---

## Resume from Checkpoint

`runpod_phase_c.sh` handles this automatically. It checks for existing checkpoints in `outputs/qwen3-8b-dflash/checkpoints/` and sets `training.load_path`.

If starting a fresh pod, `runpod_setup.sh` downloads the latest checkpoint from `Xingh3/dflash-qwen3-8b-1k`.

Resume mechanism: `checkpoint.py` reads `training.load_path` → finds `latest_checkpointed_iteration.txt` → loads model/optimizer/lr_scheduler/rng → sets `global_step` → `loop.py` skips first N steps.

---

## Common Pitfalls


| Pitfall                         | Symptom                                                  | Fix                                      |
| ------------------------------- | -------------------------------------------------------- | ---------------------------------------- |
| Missing deps on stock image     | `No module named 'sglang'`                               | Run `bash scripts/runpod_setup.sh`       |
| Git files deleted after restart | `No module named torchspec`                              | `git restore .` (setup script does this) |
| Missing RDMA libs               | `ImportError: libibverbs.so.1`                           | Setup script installs these (Issue 3)    |
| SGLang patch not applied        | `unexpected keyword argument 'enable_aux_hidden_states'` | Re-run setup script                      |
| Standard SSH fails              | `Your SSH client doesn't support PTY`                    | Use `expect` or Jupyter terminal         |
| Ray actor timeout (30s)         | `GetTimeoutError` during engine init                     | Setup patches factory.py (Issue 25)      |
| Container disk too small        | `No space left on device`                                | Use 100GB container disk (Issue 5)       |
| Training starts from step 0     | Loss ~10, accuracy 0                                     | Check checkpoint download in setup       |
| Slow training (~1 s/step)       | Expected with Mooncake TCP                               | See Issue 29 in dflash_issues.md         |


