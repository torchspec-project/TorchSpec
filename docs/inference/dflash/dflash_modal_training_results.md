# DFlash Modal Training Results

## Environment

- **Platform**: Modal (serverless GPU cloud)
- **GPUs**: 8x NVIDIA H100 80GB HBM3
- **Interconnect**: NVLink (intra-node), no RDMA/InfiniBand
- **Software**: torch 2.11 + sglang 0.5.9 + CUDA 12.4
- **Mooncake**: TCP protocol, CPU prefetch (prefetch_depth=8)
- **Dataset**: PerfectBlend 50K (`perfectblend_50k.jsonl`, 47,484 samples)
- **Base config**: `configs/sglang_qwen3_8b_dflash.yaml`

---

## Baseline: 8x H100 (1 Inference + 7 Training FSDP)

200 steps, `micro_batch_size=1`, `draft_accumulation_steps=4`, `dflash_num_anchors=512`, `max_seq_length=2048`.

Global batch size = 1 × 4 × 7 = 28.

### Timing (steady state, steps 50-200)

| Metric | Value |
|--------|-------|
| step_time | 0.87-1.4s (high variance) |
| forward | 260-800ms |
| backward | 465-580ms |
| optimizer | ~15ms |
| data_time | 400-540ms (overlapped) |
| dispatch_wait | high |
| thru (samples/s) | ~20-25 |
| T (train capacity) | ~50 |
| pool | full (24-44 / 64) |

### Analysis

- **Forward variance**: FlexAttention `mask_mod` closure changes every step (different `anchor_positions`), triggering Dynamo recompilation. `Q_LEN = n_blocks × block_size` where `n_blocks` varies per batch.
- **Backward**: 7-way FSDP FULL_SHARD reduce-scatter + all-gather.
- **Data**: Fully overlapped with compute via CPU prefetch (prefetch_depth=8).
- **Bottleneck**: Compute (forward + backward), not data transfer.

---

## Speed Tuning Tests (200 steps each)

All tests use 8x H100 with CLI overrides via `--extra-overrides`. No YAML changes.

### Test A: Reduce Anchors (1 Inference + 7 Training)

```
training.dflash_num_anchors=256
```

Rationale: Halves Q_LEN from ~8K to ~4K. Prior Phase C results showed anchors=256 was the single biggest speedup lever.

| Metric | Value |
|--------|-------|
| Total time | **610s** |
| step_time | **0.55-0.64s** |
| forward | **240-326ms** |
| backward | 256-303ms |
| optimizer | ~13ms |
| thru (samples/s) | 17-19 |
| T (train capacity) | 48-51 |
| pool | full (8-12 / 64) |
| dispatch_wait | 0.7-1.7s |

**TIMING samples (every 50 steps)**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.649s | 0.212s | 0.630s | 0.344s | 0.274s | 0.012s | 0.903s |
| 100 | 0.639s | 0.219s | 0.628s | 0.312s | 0.303s | 0.013s | 0.798s |
| 150 | 0.630s | 0.224s | 0.613s | 0.326s | 0.273s | 0.013s | 0.900s |
| 200 | 0.558s | 0.224s | 0.544s | 0.240s | 0.290s | 0.014s | 0.714s |

**COMPUTE_BREAKDOWN (CUDA event profiling)**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 272.8 | 256.6 |
| 150 | 266.8 | 269.9 |

### Test B: Larger Micro-Batch + Fewer Anchors (1 Inference + 7 Training)

```
training.micro_batch_size=2 training.draft_accumulation_steps=2 training.dflash_num_anchors=256
```

Global batch size = 2 × 2 × 7 = 28 (same as baseline).

| Metric | Value |
|--------|-------|
| Total time | 637s |
| step_time | 0.62-0.71s |
| forward | 260-455ms |
| backward | 226-255ms |
| optimizer | ~13ms |
| thru (samples/s) | 16-18 |
| T (train capacity) | 33-45 |
| pool | 8-12 / 64 |
| dispatch_wait | 0.6-0.8s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.749s | 0.156s | 0.733s | 0.488s | 0.232s | 0.013s | 0.698s |
| 100 | 0.676s | 0.137s | 0.665s | 0.395s | 0.255s | 0.014s | 0.555s |
| 150 | 0.713s | 0.162s | 0.693s | 0.455s | 0.226s | 0.012s | 0.775s |
| 200 | 0.620s | 0.170s | 0.597s | 0.355s | 0.230s | 0.013s | 0.613s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 502.2 | 230.8 |
| 150 | 260.2 | 230.3 |

### Test C: Maximum Micro-Batch (1 Inference + 7 Training)

```
training.micro_batch_size=4 training.draft_accumulation_steps=1 training.dflash_num_anchors=256
```

Global batch size = 4 × 1 × 7 = 28 (same as baseline).

| Metric | Value |
|--------|-------|
| Total time | 640s |
| step_time | 1.39-1.46s |
| forward | 607-986ms |
| backward | 213-218ms |
| optimizer | ~13ms |
| thru (samples/s) | 16-18 |
| T (train capacity) | 16-20 |
| pool | **16-28 / 64 (starved!)** |
| dispatch_wait | 0.1s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 1.447s | 0.350s | 1.095s | 0.865s | 0.217s | 0.014s | 0.071s |
| 100 | 1.459s | 0.237s | 1.218s | 0.986s | 0.218s | 0.013s | 0.069s |
| 150 | 1.440s | 0.523s | 0.915s | 0.686s | 0.216s | 0.013s | 0.069s |
| 200 | 1.389s | 0.549s | 0.837s | 0.607s | 0.217s | 0.013s | 0.071s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 173.3 | 215.2 |
| 150 | 639.5 | 213.8 |

**Problem**: With `accum=1`, every step consumes 28 samples. The single inference GPU produces ~17 samples/s, which cannot keep the pool full. Data starvation causes the trainer to idle waiting for samples.

### Test C2: Maximum Micro-Batch + 2 Inference GPUs (2 Inference + 6 Training)

```
training.micro_batch_size=4 training.draft_accumulation_steps=1 training.dflash_num_anchors=256
inference.inference_num_gpus=2 training.training_num_gpus_per_node=6
```

Global batch size = 4 × 1 × 6 = 24.

| Metric | Value |
|--------|-------|
| Total time | **476s** |
| step_time | 0.86-0.95s |
| forward | 272-434ms |
| backward | 211-214ms |
| optimizer | ~14ms |
| thru (samples/s) | **22-25** |
| I (inference/s) | **33-35** |
| T (train capacity) | 19-26 |
| pool | **64 / 64 (full!)** |
| dispatch_wait | 0.05s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.857s | 0.263s | 0.593s | 0.361s | 0.217s | 0.014s | 0.053s |
| 100 | 0.953s | 0.312s | 0.640s | 0.414s | 0.211s | 0.015s | 0.054s |
| 150 | 0.913s | 0.251s | 0.660s | 0.434s | 0.212s | 0.014s | 0.054s |
| 200 | 0.943s | 0.301s | 0.641s | 0.412s | 0.212s | 0.016s | 0.053s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 374.9 | 212.1 |
| 150 | 272.1 | 212.2 |
| 195 | 142.4 | 214.4 |

---

## Phase 2: Anchors=512 Speed Tuning (200 steps each)

Motivated by `specforge_dflash_training_reference.md`: z-lab uses `dflash_num_anchors=512` for best
acceptance length (τ). This phase tests whether anchors=512 can match anchors=256 speed when
properly tuned with 2 inference GPUs.

### Test 512-A: Baseline (1 Inference + 7 Training)

```
training.dflash_num_anchors=512
```

Identical to original baseline, re-run for consistent comparison.

| Metric | Value |
|--------|-------|
| Total time | **584s** |
| step_time | 0.83-1.40s |
| forward | 258-837ms |
| backward | 456-539ms |
| optimizer | ~13-14ms |
| thru (samples/s) | 17-20 |
| T (train capacity) | 29-33 |
| pool | **12-28 / 64 (starved!)** |
| dispatch_wait | 1.0-1.5s |

**TIMING samples (every 50 steps)**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.825s | 0.396s | 0.811s | 0.258s | 0.539s | 0.014s | 0.678s |
| 100 | 1.404s | 0.442s | 1.394s | 0.777s | 0.603s | 0.013s | 0.066s |
| 150 | 0.864s | 0.434s | 0.848s | 0.300s | 0.534s | 0.014s | 0.644s |
| 200 | 1.377s | 0.419s | 1.351s | 0.837s | 0.500s | 0.013s | 0.201s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 319.3 | 501.6 |
| 150 | 284.9 | 534.8 |
| 195 | 468.3 | 456.1 |

**Problem**: Single inference GPU cannot keep pool saturated (pool=12-28). Dispatch wait 1.0-1.5s.

### Test 512-B: Larger Micro-Batch (1 Inference + 7 Training)

```
training.dflash_num_anchors=512 training.micro_batch_size=2 training.draft_accumulation_steps=2
```

Global batch size = 2 × 2 × 7 = 28 (same as baseline).

| Metric | Value |
|--------|-------|
| Total time | **574s** |
| step_time | 0.85-1.28s |
| forward | 204-705ms |
| backward | 428-553ms |
| optimizer | ~13-14ms |
| thru (samples/s) | 17-19 |
| T (train capacity) | 29-40 |
| pool | **12-20 / 64 (starved!)** |
| dispatch_wait | 1.1-1.4s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.865s | 0.318s | 0.850s | 0.312s | 0.525s | 0.014s | 0.615s |
| 100 | 1.283s | 0.292s | 1.272s | 0.705s | 0.553s | 0.013s | 0.262s |
| 150 | 0.845s | 0.308s | 0.826s | 0.357s | 0.456s | 0.013s | 0.687s |
| 200 | 0.990s | 0.316s | 0.968s | 0.527s | 0.428s | 0.014s | 0.523s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 377.0 | 447.9 |
| 150 | 239.6 | 495.3 |
| 195 | 204.4 | 456.6 |

**Problem**: Same as 512-A — single inference GPU starves the pool. batch=2 slightly faster total time (574s vs 584s) but still bottlenecked by data.

### Test 512-C: Larger Micro-Batch + 2 Inference GPUs (2 Inference + 6 Training)

```
training.dflash_num_anchors=512 training.micro_batch_size=2 training.draft_accumulation_steps=2
inference.inference_num_gpus=2 training.training_num_gpus_per_node=6
```

Global batch size = 2 × 2 × 6 = 24.

| Metric | Value |
|--------|-------|
| Total time | **446s** |
| step_time | 0.99-1.29s |
| forward | 264-706ms |
| backward | 414-515ms |
| optimizer | ~15-16ms |
| thru (samples/s) | **20-25** |
| I (inference/s) | 36-46 |
| T (train capacity) | 19-26 |
| pool | **64 / 64 (full!)** |
| dispatch_wait | 0.05s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.997s | 0.462s | 0.783s | 0.264s | 0.504s | 0.015s | 0.053s |
| 100 | 1.082s | 0.379s | 1.003s | 0.573s | 0.414s | 0.016s | 0.051s |
| 150 | 1.293s | 0.360s | 1.213s | 0.682s | 0.515s | 0.015s | 0.047s |
| 200 | 0.985s | 0.250s | 0.973s | 0.502s | 0.455s | 0.016s | 0.060s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 438.1 | 435.2 |
| 150 | 323.0 | 496.9 |
| 195 | 705.9 | 479.4 |

### Test 512-D: Baseline + 2 Inference GPUs (2 Inference + 6 Training)

```
training.dflash_num_anchors=512 inference.inference_num_gpus=2 training.training_num_gpus_per_node=6
```

Global batch size = 1 × 4 × 6 = 24.

| Metric | Value |
|--------|-------|
| Total time | **457s** |
| step_time | **0.93-1.09s** |
| forward | 305-448ms |
| backward | 473-540ms |
| optimizer | ~14-15ms |
| thru (samples/s) | **22-25** |
| I (inference/s) | 27-41 |
| T (train capacity) | 27-29 |
| pool | **64 / 64 (full!)** |
| dispatch_wait | 0.05-0.07s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 1.085s | 0.516s | 0.933s | 0.380s | 0.537s | 0.015s | 0.066s |
| 100 | 1.001s | 0.431s | 0.982s | 0.448s | 0.518s | 0.015s | 0.066s |
| 150 | 0.972s | 0.419s | 0.930s | 0.443s | 0.473s | 0.015s | 0.060s |
| 200 | 0.934s | 0.427s | 0.917s | 0.363s | 0.540s | 0.014s | 0.066s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 332.8 | 521.9 |
| 150 | 305.3 | 507.3 |
| 195 | 308.0 | 492.1 |

**Notable**: Most stable forward times of all anchors=512 tests (305-448ms range vs 204-837ms for others). Less FlexAttention recompilation jitter with batch=1.

### Test 512-E: 4 Inference + 4 Training GPUs

```
training.dflash_num_anchors=512 inference.inference_num_gpus=4 training.training_num_gpus_per_node=4
```

Global batch size = 1 × 4 × 4 = 16.

| Metric | Value |
|--------|-------|
| Total time | **368s** |
| step_time | **0.75-0.83s** |
| forward | **230-355ms** |
| backward | 459-517ms |
| optimizer | ~15-16ms |
| thru (samples/s) | 17-20 |
| I (inference/s) | **108-120** |
| T (train capacity) | 18-25 |
| pool | **64-72 / 64 (overfull!)** |
| dispatch_wait | 0.04-0.06s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.763s | 0.404s | 0.744s | 0.230s | 0.498s | 0.016s | 0.044s |
| 100 | 0.751s | 0.403s | 0.741s | 0.258s | 0.468s | 0.016s | 0.046s |
| 150 | 0.834s | 0.369s | 0.825s | 0.292s | 0.517s | 0.016s | 0.045s |
| 200 | 0.781s | 0.346s | 0.767s | 0.292s | 0.459s | 0.015s | 0.061s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 247.4 | 364.7 |
| 150 | 355.6 | 496.4 |
| 195 | 331.8 | 496.4 |

**Notable**: Fastest total time of all tests. 4 inference GPUs massively oversaturate the pool (I=108-120 samples/s, pool overflows to 72). Fewer FSDP ranks (4 vs 6-7) means less allreduce overhead, giving the lowest forward times. Trade-off: global batch size drops to 16, reducing per-step sample throughput but more than compensated by faster steps.

### Test 512-F: Larger Micro-Batch + 4 Inference GPUs (4 Inference + 4 Training)

```
training.dflash_num_anchors=512 training.micro_batch_size=2 training.draft_accumulation_steps=2
inference.inference_num_gpus=4 training.training_num_gpus_per_node=4
```

Global batch size = 2 × 2 × 4 = 16 (same as 512-E).

| Metric | Value |
|--------|-------|
| Total time | **394s** |
| step_time | 0.90-0.93s |
| forward | 231-399ms |
| backward | 360-549ms |
| optimizer | ~16ms |
| thru (samples/s) | 17-18 |
| I (inference/s) | 114-128 |
| T (train capacity) | 18-20 |
| pool | **64 / 64 (full)** |
| dispatch_wait | 0.03-0.04s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.895s | 0.412s | 0.743s | 0.261s | 0.466s | 0.016s | 0.034s |
| 100 | 0.910s | 0.394s | 0.805s | 0.282s | 0.507s | 0.016s | 0.035s |
| 150 | 0.909s | 0.393s | 0.794s | 0.298s | 0.480s | 0.016s | 0.039s |
| 200 | 0.927s | 0.419s | 0.796s | 0.231s | 0.549s | 0.016s | 0.033s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 399.8 | 360.4 |
| 150 | 257.7 | 414.2 |
| 195 | 375.3 | 461.2 |

**Notable**: Slower than 512-E (394s vs 368s) despite same global batch size. batch=2 packs 2 sequences per micro-batch, increasing FlexAttention overhead per forward pass (0.90-0.93s/step vs 0.75-0.83s). batch=1 with accum=4 remains superior for the 4+4 split.

---

## Comparison Summary

### Phase 1: Anchors=256 Speed Tuning

| Config | GPUs (I+T) | batch | accum | anchors | Total Time | step_time | thru (s/s) | pool | Bottleneck |
|--------|-----------|-------|-------|---------|------------|-----------|------------|------|------------|
| Baseline | 1+7 | 1 | 4 | 512 | — | 0.87-1.4s | 20-25 | full | compute (fwd variance) |
| **Test A** | 1+7 | 1 | 4 | **256** | **610s** | **0.55-0.64s** | 17-19 | full | compute |
| Test B | 1+7 | 2 | 2 | 256 | 637s | 0.62-0.71s | 16-18 | full | compute |
| Test C | 1+7 | 4 | 1 | 256 | 640s | 1.39-1.46s | 16-18 | starved | **data (pool empty)** |
| Test C2 | 2+6 | 4 | 1 | 256 | 476s | 0.86-0.95s | 22-25 | full | compute |

### Phase 2: Anchors=512 Speed Tuning

| Config | GPUs (I+T) | batch | accum | anchors | Total Time | step_time | thru (s/s) | pool | Bottleneck |
|--------|-----------|-------|-------|---------|------------|-----------|------------|------|------------|
| 512-A | 1+7 | 1 | 4 | 512 | 584s | 0.83-1.40s | 17-20 | starved (12-28) | data |
| 512-B | 1+7 | 2 | 2 | 512 | 574s | 0.85-1.28s | 17-19 | starved (12-20) | data |
| 512-C | 2+6 | 2 | 2 | 512 | 446s | 0.99-1.29s | 20-25 | full | compute |
| 512-D | 2+6 | 1 | 4 | 512 | 457s | 0.93-1.09s | 22-25 | full | compute |
| **512-E** | **4+4** | **1** | **4** | **512** | **368s** | **0.75-0.83s** | 17-20 | **overfull (72)** | compute |
| 512-F | 4+4 | 2 | 2 | 512 | 394s | 0.90-0.93s | 17-18 | full | compute |

### Key Findings

1. **`dflash_num_anchors` is the biggest lever**: 512→256 halves Q_LEN, cuts forward time from 260-800ms to 240-434ms. Consistent with Phase C results on RunPod.

2. **2+ inference GPUs is essential for anchors=512**: With 1 inference GPU, every anchors=512 config suffered pool starvation (12-28/64) and dispatch waits of 1.0-1.5s. Adding more inference GPUs fully resolves this.

3. **4+4 split (512-E) is the fastest overall**: 368s total, 17% faster than 512-C (446s). Fewer FSDP ranks (4 vs 6-7) reduces allreduce overhead, giving the lowest forward times (230-355ms). Trade-off: global batch size drops to 16 vs 24, but faster steps more than compensate.

4. **Inference is massively oversaturated at 4+4**: I=108-120 samples/s with pool overflowing to 72/64. The 4 inference GPUs produce far more data than 4 training GPUs can consume. This suggests 3+5 may be the sweet spot.

5. **Anchors=512 matches anchors=256 speed when properly tuned**: All 2+ inference GPU configs (512-C/D/E) match or beat the best anchors=256 config (C2, 476s).

6. **512-D has the most stable step times among 2+6 configs**: Forward variance 305-448ms vs 204-837ms for other configs. batch=1 with accum=4 avoids FlexAttention recompilation from varying padded batch shapes.

7. **RDMA is not available on Modal**: The RDMA probe confirmed `/dev/infiniband` and `ibstat` are not present. Mooncake uses TCP only, but CPU prefetch effectively hides the latency.

### 200K × 3 Epoch Estimates

600,000 total samples. Estimates based on steady-state throughput (excluding warmup/compilation).

| Config | samples/s | Est. Time |
|--------|-----------|-----------|
| Baseline (1+7, anchors=512) | ~18 | ~9.3 hr |
| Test C2 (2+6, batch=4, anchors=256) | ~24 | ~6.9 hr |
| 512-C (2+6, batch=2, anchors=512) | ~22 | ~7.6 hr |
| 512-D (2+6, batch=1, anchors=512) | ~23 | ~7.2 hr |
| **512-E (4+4, batch=1, anchors=512)** | **~19** | **~8.8 hr** |

Note: 512-E has the fastest wall-clock for 200 steps (368s) but lower samples/s (19 vs 23) due to
smaller global batch (16 vs 24). For full training, effective time depends on optimizer convergence —
fewer samples per step means more optimizer steps needed, partially offsetting the faster step time.

### Recommended Config for Full Training (Quality-Optimized)

Based on `specforge_dflash_training_reference.md`: z-lab achieves τ=3.95 with `dflash_num_anchors=512`,
`max_seq_length=2048`, and 200K+ samples. Anchors=512 is recommended for best acceptance length.

**512-E (fastest step time, 4+4)**:

```bash
modal run --detach --env sandbox scripts/modal_dflash_train.py \
  --num-epochs 3 --dataset-size 200000 \
  --extra-overrides "training.dflash_num_anchors=512 inference.inference_num_gpus=4 training.training_num_gpus_per_node=4"
```

**512-D (best throughput, most stable, 2+6)**:

```bash
modal run --detach --env sandbox scripts/modal_dflash_train.py \
  --num-epochs 3 --dataset-size 200000 \
  --extra-overrides "training.dflash_num_anchors=512 inference.inference_num_gpus=2 training.training_num_gpus_per_node=6"
```

**Speed-optimized (if τ quality not critical)**:

```bash
modal run --detach --env sandbox scripts/modal_dflash_train.py \
  --num-epochs 3 --dataset-size 200000 \
  --extra-overrides "training.dflash_num_anchors=256 training.micro_batch_size=4 training.draft_accumulation_steps=1 inference.inference_num_gpus=2 training.training_num_gpus_per_node=6"
```

---

## Phase G Convergence Analysis (2026-03-25)

### Observed Training Curves (200K PerfectBlend, 3 Epochs, 23,622 Steps)

See `training_metrics_3epoch.png`.

**Loss (CE):**
- Drops sharply from ~8 to ~3.5 in first ~2K steps (early epoch 1)
- Slow decline from ~3.5 to ~2.0 over remaining 21K steps
- Plateaus around step 5K-7K, then creeps down
- Visible bumps at epoch boundaries (dataset re-shuffle causes partial forgetting)

**Accuracy (Top-1):**
- Rises from 0.05 to ~0.35 in first 5K steps
- Stalls at ~0.35-0.45 for the remaining 18K steps
- Reaches only ~0.45 after 23K steps — low for a draft model
- Learning clearly decelerates well before epoch 3

**Diagnosis:** The model learns fast early but hits a wall around 35-45% accuracy. This indicates the optimizer configuration is preventing deeper convergence.

### Root Cause Analysis

#### 1. Global Batch Size Too Large → Too Few Optimizer Steps

Phase G config: `micro_batch=1, accum=4, dp=6` → **global_batch = 24**.

With 190K samples: only **~7,900 steps per epoch**, 23,700 total for 3 epochs. The cosine LR schedule decays too fast relative to data diversity. The optimizer takes big batch steps but sees relatively few unique gradient directions.

z-lab's 800K × 6 epochs with (likely smaller) batch size would have significantly more optimizer steps. Community result (@jianc99, 1.2M × 3 epochs) likely had 100K+ optimizer steps.

#### 2. Cosine LR Decays Before Convergence

```python
# BF16Optimizer: total_steps=23,622, warmup_ratio=0.04
warmup = 945 steps (reaches peak lr=6e-4)
# Then cosine decay: 6e-4 → 0 over 22,677 steps
# By step ~12K (midpoint): LR ≈ 3e-4
# By step ~18K: LR < 1e-4 — model barely learning
```

This explains the accuracy stall at ~0.40 — the LR decays too aggressively for the amount of data diversity.

#### 3. weight_decay = 0.0 (No Regularization)

`BF16Optimizer` passed `weight_decay=0.0` to `AdamW`, making it equivalent to plain Adam. Adding weight decay (0.01) helps prevent overfitting on repeated data across epochs and improves generalization — which directly maps to inference τ.

#### 4. min_lr = 0.0 (LR Decays to Zero)

The cosine schedule decayed to `min_lr=0.0`. In later training steps, the model was effectively frozen. Setting `min_lr` to ~10% of peak (6e-5) keeps the model learning through later epochs.

### Configuration Changes Implemented

| Parameter | Phase G | Phase H | Rationale |
|-----------|---------|---------|-----------|
| `draft_accumulation_steps` | 4 | **2** | 2x more optimizer steps per epoch |
| `weight_decay` | 0.0 (hardcoded) | **0.01** | AdamW regularization → better generalization |
| `min_lr` | 0.0 (hardcoded) | **6e-5** | 10% of peak → prevents LR death |
| `save_interval` | 1000 | **5000** | Less frequent for longer runs |
| `save_per_epoch` | false | **true** | Compare τ at epoch boundaries |
| `max_checkpoints` | 2 | **3** | Keep all 3 epoch checkpoints |

Code changes (commits TBD):
- `torchspec/config/train_config.py`: Added `min_lr` and `weight_decay` to `TrainingConfig`
- `torchspec/training/optimizer.py`: `BF16Optimizer` accepts `min_lr` and passes to scheduler
- `torchspec/training/dflash_trainer.py`: Plumbs `weight_decay` and `min_lr` from args
- `torchspec/training/eagle3_trainer.py`: Same plumbing (consistent across trainers)
- `scripts/modal_dflash_train.py`: Fixed WandB config prefix (`logging.*` not `training.*`)

### Expected Impact

| Change | Speed Impact | τ Impact |
|--------|-------------|----------|
| `accum=4→2` | Neutral (same samples/s, 2x optimizer steps) | **+0.3-0.5 τ** |
| `min_lr=6e-5` | Neutral | **+0.2-0.3 τ** |
| `weight_decay=0.01` | Neutral | **+0.1-0.2 τ** |
| Combined | No speed change | **+0.5-1.0 τ** (est. τ=3.5-4.0 on 190K, 4.0-4.5 on 800K) |

---

## Phase H: 800K PerfectBlend Training Plan (2026-03-25)

### Goal

Close the τ gap to z-lab (currently 3.06 vs 4.05) by scaling data 4x and fixing convergence.

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | PerfectBlend 800K (~760K after filtering) | 4x Phase G data |
| Epochs | 3 | Start with 3, extend to 6 if τ improving |
| GPU split | 4I + 4T (512-E) | Fastest wall-clock from Modal tuning |
| Global batch | 8 (micro=1, accum=2, dp=4) | 2x more optimizer steps than Phase G |
| Anchors | 512 | z-lab recipe |
| Seq length | 2048 | Keep current; 3072 is Priority 2 |
| LR | 6e-4, cosine, warmup 0.04, min_lr 6e-5 | Slower decay |
| Weight decay | 0.01 | New |
| Optimizer steps | ~285K (3 epochs) | 12x more than Phase G (23.6K) |
| Total sample passes | ~2.28M | 4x Phase G (567K), 48% of z-lab (4.8M) |

### Estimated Training Time & Cost

| Epochs | Steps | Wall Time | GPU-hours | Modal Cost |
|--------|-------|-----------|-----------|------------|
| 3 | ~285K | ~10.5 hr | 84 | ~$332 |
| 6 | ~570K | ~21 hr | 168 | ~$664 |

Based on 512-E throughput: ~20 samples/s steady state, ~8 global_batch → ~2.5 step/s.

### τ Targets

| Milestone | Expected τ | Comparison |
|-----------|-----------|------------|
| Epoch 1 | ~3.3-3.6 | Above Phase G final (3.06) |
| Epoch 2 | ~3.6-4.0 | Approaching z-lab (4.05) |
| Epoch 3 | ~3.8-4.2 | Matching z-lab math benchmarks |

### Launch Command

```bash
modal run --detach scripts/modal_dflash_train.py \
  --num-epochs 3 --dataset-size 800000 \
  --wandb-project dflash-800k \
  --hf-repo Xingh3/dflash-qwen3-8b-800k-3epoch \
  --extra-overrides "training.dflash_num_anchors=512 \
    inference.inference_num_gpus=4 training.training_num_gpus_per_node=4"
```

### Monitoring

- **WandB**: `dflash-800k` project — watch `train/avg_loss`, `train/avg_acc`, `train/lr`, `train/grad_norm`
- **Checkpoints**: Saved at every 5K steps + epoch boundaries → auto-converted + uploaded to `Xingh3/dflash-qwen3-8b-800k-3epoch`
- **τ benchmark**: After each epoch, run `scripts/modal_dflash_benchmark.py` against the HF checkpoint
