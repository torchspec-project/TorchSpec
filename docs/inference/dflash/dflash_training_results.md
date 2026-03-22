# DFlash Training Results

## Phase A: Single GPU (1x H100, 200 steps)

| Metric | DFlash | Eagle3 | Notes |
|--------|--------|--------|-------|
| Final accuracy | **0.894** (89.4%) | 0.646 (64.6%) | DFlash 38% higher |
| Final loss | 0.477 (CE) | 2.166 (KL) | Different scales, not comparable |
| Training time | **81.7s** | 202.4s | DFlash **2.5x faster** |
| Steps/sec | **~10** | ~1.8 | DFlash 5.5x more steps/sec |
| Forward passes/step | **1** | 7 | Block-parallel vs autoregressive |

Loss converges rapidly: 13.0 → 0.48 in 200 steps.

### Inference (200-step checkpoint)

| Method | Tokens/sec | τ | Speedup |
|--------|-----------|---|---------|
| Baseline (target-only) | 59.4 | N/A | 1.0x |
| DFlash speculative | 42.5 | 1.01 | 0.72x (slower) |

τ=1.01 expected — 200 steps on tiny sample data is insufficient for 899M parameter model.

---

## Phase B: 4-GPU SGLang Validation (200 steps)

| Metric | Baseline (target-only) | DFlash |
|--------|----------------------|--------|
| Throughput | 61.8 tok/s | 41.2 tok/s |
| τ | — | 1.03 |
| Speedup | 1.0x | 0.67x |

τ=1.03 expected — 200 steps on tiny data. Pipeline works end-to-end.

### SpecForge Cross-Check

**Intentional design divergences (not bugs):**

| Aspect | TorchSpec | SpecForge |
|--------|-----------|-----------|
| Architecture | Standalone PreTrainedModel | Extends Qwen3PreTrainedModel (model-specific) |
| RoPE | Computed on-the-fly per layer | Pre-computed, passed as embeddings |
| Context projection | In `extract_context_feature()` | In draft model `forward()` |
| KV cache | Not supported (recomputes) | Supported via past_key_values |
| Block mask | `if device.type == "cuda"` else None | Always created |

**Real issues fixed (training hyperparameters):**

| Parameter | Old | New | Source |
|-----------|-----|-----|--------|
| `learning_rate` | 1e-4 | **6e-4** | SpecForge default |
| `warmup_ratio` | 0.015 | **0.04** | SpecForge default |
| `max_grad_norm` | 0.5 | **1.0** | SpecForge default |
| `num_epochs` | 1 | **6** | SpecForge default |

### SpecForge Eagle3 Production Performance (Reference)

| Model | τ | Speedup |
|-------|---|---------|
| Llama-3.1-8B | 1.8–3.1 | 1.0–1.7x |
| Llama-3.3-70B | 1.4–3.2 | 1.1–2.0x |
| Qwen3-30B-A3B | 2.6–5.3 | 1.4–2.5x |
| Llama-4-Scout | 2.1–3.0 | 1.5–2.7x |

**DFlash target**: τ ≥ 3.0 with full training.

---

## Phase C: Full Training (4x H100)

### Speed Optimization

**Bottleneck**: FlexAttention is ~60% of per-step time (O(Q×KV) where Q = num_anchors × block_size).

**Key insight — `aot_eager` vs `inductor`**: Inductor generates fused Triton kernels that eliminate intermediate tensor materialization, giving **3x speedup and 20 GB less GPU memory**.

| Config | Speed | ETA | GPU Mem | Notes |
|--------|-------|-----|---------|-------|
| batch=1, accum=4, anchors=512, seq=4096 | 1.5 step/s | 6.6 hr | 40 GB | Baseline |
| batch=2, accum=2, anchors=512, seq=2048, epochs=4 | 1.4 step/s | 4.7 hr | 53 GB | batch=2 slower per step |
| batch=2, accum=2, anchors=**256**, seq=2048 | **1.88** step/s | 3.4 hr | 38 GB | **Big win from reducing anchors** |
| **batch=4, accum=1, anchors=256, seq=2048** | **2.1 step/s** | **3.1 hr** | 50 GB | **Final config** |

**Optimization insights**:
1. **`num_anchors` is the biggest lever**: 512→256 halves Q_LEN, gave largest single speedup
2. **`max_seq_length`** helps inference more than training
3. **`micro_batch_size`** has diminishing returns
4. anchors=256 + batch=2 has same FlexAttention cost as anchors=512 + batch=1, but processes 2x data per optimizer step

### Final Training Config

```yaml
micro_batch_size: 4
draft_accumulation_steps: 1
num_epochs: 4
max_seq_length: 2048
learning_rate: 6e-4
warmup_ratio: 0.04
max_grad_norm: 1.0
save_per_epoch: true
save_interval: 1000
max_checkpoints: 1
eval_data_path: null
eval_interval: 0
dflash_block_size: 16
dflash_num_anchors: 256
dflash_loss_decay_gamma: 7.0
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
```

### Inference Benchmark — Step 18,001

| Metric | Value |
|--------|-------|
| Acceptance length (τ) | **1.86** |
| Wall-clock speedup | **1.27x** |
| Target | τ ≥ 3.0 |

τ = 1.86 is significantly below target. Investigation identified two training quality bugs (see [Issues](dflash_issues.md#training-quality-bugs-found-in-session-12)).

### PyTorch 2.9.1 Speed Regression — RESOLVED

After PyTorch 2.6.0→2.9.1 migration: **2.1 step/s → 0.75 step/s** (3x slower).

**Fix**: `export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` — restored speed to ~5 step/s in colocate 1-GPU mode. See [Issue 26](dflash_issues.md#issue-26-pytorch-291-speed-regression-3x-slower--resolved).

### Checkpoint & Backup

- **Latest checkpoint**: `iter_0018001` (~15 GB per checkpoint)
- **HuggingFace backup**: [`Xingh3/qwen3-8b-dflash-checkpoint-phase-c`](https://huggingface.co/Xingh3/qwen3-8b-dflash-checkpoint-phase-c) (private)

| Location | Survives Pod Stop | Survives Pod Delete |
|----------|:-:|:-:|
| `/tmp/`, `/root/` (container disk) | No | No |
| `/workspace/` (volume) | Yes | No |
| HuggingFace Hub | Yes | Yes |

---

## Phase D: Bug Fixes & Smoke Test (2026-03-22)

### Environment

- torch 2.9.1 + sglang 0.5.9 + CUDA 12.4
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON`
- 1x H100 (colocate mode)
- Data: `perfectblend_50k.jsonl`

### Bug Fixes Applied

1. **Bug 1 — Zero-loss dummy** (commit `f3311e4`): Replaced silent zero-gradient fallback with `raise ValueError` in `dflash.py:130-134`
2. **`min_loss_tokens` pipeline** (commit `f3311e4`): Plumbed `min_loss_tokens` from config → data loader → preprocessing. Set `min_loss_tokens: 32` (2×block_size) in DFlash configs.
3. **SpecForge deep diff**: All 3 critical file pairs verified to match. No additional SpecForge fixes needed.

### Unit Tests

54/54 pass. 3 new tests added:
- `test_loss_decay_weights` — verifies `exp(-(k-1)/γ)` weighting
- `test_label_alignment_same_position` — verifies same-position prediction (not shifted)
- `test_anchor_loss_excluded` — verifies position 0 (anchor) has zero loss weight

### Smoke Test (30 steps, 1-GPU colocate)

| Step | Loss | Accuracy | Speed |
|------|------|----------|-------|
| 1 | 12.70 | 0.000 | warmup |
| 10 | 11.08 | 0.000 | ~1.2 step/s |
| 20 | 9.30 | 0.091 | ~3.6 step/s |
| 30 | 8.55 | 0.065 | ~5.0 step/s |

Loss starts ~12.7, drops monotonically. No NaN, no zero-loss steps. Speed ~5 step/s after warmup (exceeds Phase C baseline of 2.1 step/s on torch 2.6).

### Next Steps

- Fresh training from scratch with bug fixes, 6 epochs on 50K data (Phase 3)
- Measure τ at epoch boundaries to track progression
- Investigate pipeline overhead for speed improvements

---

## Phase E: Speed Benchmark (2026-03-22)

### Environment

- torch 2.9.1 + sglang 0.5.9 (patched) + CUDA 12.4
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON`
- 4x H100 80GB: 2 training + 1 inference + 1 idle
- Data: `perfectblend_50k.jsonl`, block_size=16, seq_len=2048, 50 steps each

### Results

| Config | batch | accum | anchors | step/s | Status |
|--------|-------|-------|---------|--------|--------|
| S1 | 1 | 4 | 512 | **1.05** | OK |
| S2 | 2 | 2 | 512 | **1.0** | OK |
| S3 | 4 | 1 | 512 | **OOM** | Backward alloc 9.27 GB |
| S3b | 2 | 2 | 256 | **1.0** | OK |

### Key Finding: Pipeline Bottleneck

**All configs converge to ~1.0 step/s regardless of DFlash hyperparameters.** The bottleneck is the inference→training pipeline (Mooncake KV transfer + Ray actor dispatch), not DFlash compute.

**Evidence:**
1. **Training compute is fast**: `T=6-15ms` per batch in training log, but step cycle is ~1000ms
2. **GPU utilization is near-zero**: nvidia-smi snapshot during training shows 0% on all GPUs — they idle between pipeline cycles
3. **Anchors don't matter**: 512→256 halves FlexAttention work but yields identical step/s
4. **Colocate mode is 5x faster**: 1-GPU colocate (Phase D) achieved ~5 step/s by avoiding Mooncake/Ray overhead entirely

### Pipeline Overhead Breakdown (estimated)

| Component | Time (ms) | % of Step |
|-----------|-----------|-----------|
| Mooncake KV transfer (hidden states GPU→GPU) | ~300-500 | 30-50% |
| Ray actor dispatch + result collection | ~200-300 | 20-30% |
| SGLang prefill (target model forward) | ~150-200 | 15-20% |
| DFlash training forward+backward | ~6-15 | <2% |
| Other overhead (scheduling, queueing) | ~100-200 | 10-20% |

### Recommendations for Speed Improvement

| Priority | Approach | Expected Impact |
|----------|----------|-----------------|
| **P0** | Bypass Mooncake — use shared GPU memory or colocate mode for KV transfer | 3-5x speedup |
| **P1** | Increase `max_concurrent_batches` to overlap inference and training | Up to 2x |
| **P1** | Increase inference throughput (batch_size, max_running_requests) | 1.5-2x if inference is sub-saturated |
| **P2** | Enable FSDP CPU offload to fit batch=4 + anchors=512 | Enables larger batches |
| **P2** | Profile Mooncake transfer vs Ray dispatch to isolate dominant overhead | Identifies optimization target |

### Training Time Estimate

With current 1.0 step/s:
- 50K samples × 6 epochs / (batch=2 × dp=2) = **37,500 optimizer steps → ~10.4 hours**
- With P0 fix (3-5x): **2-3.5 hours**

---

## Commits

| Hash | Description |
|------|-------------|
| `398138a` | Fix collator crash when loss_mask length differs from input_ids |
| `fee3156` | Switch FlexAttention from aot_eager to inductor backend for 3x speedup |
| `c7c6605` | Add checkpoint rotation to prevent disk quota exceeded during training |
| `ba3cd02` | Document Phase C crash debugging: disk quota, eval timeout, checkpoint rotation |
| `dddcd2a` | Document training speed optimization: 6.6hr → 3.1hr |
| `c5b71e8` | Optimize DFlash training speed: remove compile overhead and enable GQA |
