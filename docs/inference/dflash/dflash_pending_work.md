# DFlash Pending Work

> Last updated: 2026-03-22 (Phase 3 plan added)

## Completed

- [x] Resume Phase C training — resumed from step 17,001
- [x] Inference benchmark — τ = 1.86 at step 18,001 (below target 5.0)
- [x] SpecForge cross-check — identified 2 training quality bugs
- [x] **Fix Bug 1: Zero-loss dummy on empty anchors** — replaced with `raise ValueError` (commit `f3311e4`)
- [x] **Plumb `min_loss_tokens`** — data filtering for sequences with < 2×block_size supervised tokens (commit `f3311e4`)
- [x] **SpecForge deep diff** — line-by-line comparison of all 3 critical file pairs, all match
- [x] **Resolve PyTorch 2.9.1 speed regression** — `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` (Issue 26)
- [x] **Unit tests** — 54/54 pass (3 new tests added for loss decay, label alignment, anchor exclusion)
- [x] **Smoke test** — 30 steps on pod, loss 12.7→8.55, ~5 step/s after warmup
- [x] **Bug 2 investigation** — matches SpecForge, skipped (not a bug)
- [x] **Speed benchmark S1-S3** (Phase 2.2) — all configs ~1.0 step/s with pipeline overhead
- [x] **Compute sub-breakdown profiling** (Phase 2.5.1) — backward=54%, forward=31%, optimizer=16%
- [x] **200-step stability test** — stable ~2.5 step/s, no degradation
- [x] **Mooncake bypass investigation** — no config option, needs code changes
- [x] **max_concurrent_batches=2 test** — no improvement, not bottlenecked on inference
- [x] **GPU Direct RDMA test** — failed, RunPod lacks RDMA hardware
- [x] **Speed optimizations** — no_sync + bf16 reduce → 2.7 step/s (+8%). torch.compile not viable.
- [x] **FULL_SHARD benchmark (Test 1)** — 2.9 step/s, optimizer halved (41→22ms), best per-step throughput
- [x] **3-GPU scaling benchmark (Test 2)** — 2.2 step/s per-step, but 3.2hr total (50% more data/step)
- [x] **FULL_SHARD + 3-GPU benchmark (Test 3)** — 2.3 step/s, optimizer 16-22ms, ~3.0hr total
- [x] **2-inference GPU benchmark (Test 4)** — 1.1 step/s, strictly worse due to Mooncake/PCIe contention
- [x] **Pod disk cleanup** — freed 61 GB on container disk (81%→20%), removed old speed benchmark checkpoints
- [x] **Commit factory.py timeout fix** — 30s→120s for PyTorch 2.9+ compatibility (commit `cedef38`)
- [x] **Async data pre-fetch** — CPU staging to overlap Mooncake TCP with GPU compute (commits `f75a285`, `3ceb630`, `bb922ba`)
- [x] **Test 5b: CPU prefetch benchmark** — **6.8 step/s** (2.3x over Test 1 baseline), data fully overlapped with compute
- [x] **Test 6: NVLink transport investigation** — Not viable: built from source, patched protocol switch, but NVLink requires GPU memory while mooncake-store uses host memory (fundamental mismatch)
- [x] **Test 7: 3 GPU + FULL_SHARD + CPU prefetch** — 5.3 step/s × 3 samples/step = 15.9 samples/s, ~17% faster than Test 5b (2 GPU)

## Active — Phase 3: τ Benchmark Training (incremental)

- [ ] **Train 5k steps** — perfectblend_50k, 3 GPU + FULL_SHARD + CPU prefetch (~16 min)
- [ ] **τ @ 5k steps** — inference benchmark: measure τ, tok/s vs baseline, verify performance
- [ ] **Eagle3 baseline** — benchmark Eagle3 inference speed on same target model for comparison
- [ ] **Train to 10k steps** — continue from 5k checkpoint (if 5k results are promising)
- [ ] **τ @ 10k steps** — inference benchmark, compare with 5k
- [ ] **Train to 15k steps** — continue from 10k checkpoint
- [ ] **τ @ 15k steps** — inference benchmark, compare progression

## Active — Code Improvements

- [x] **NVLink transport investigation** — Not viable: `nvlink_intra` requires GPU memory but mooncake-store uses host memory. Built from source, patched `client_service.cpp`, but fails at `intranode_nvlink_transport.cpp:298` with "Unsupported memory type". TCP + CPU prefetch is the optimal approach.
- [ ] **Draft KV cache**: Add KV cache support to `DFlashDraftModel.forward()` (currently recomputes full context each cycle — O(n²) scaling).

## Future

- [ ] **Data regeneration**: Generate target-model-regenerated training data for τ ≥ 6.0 (Phase 4.4).
- [ ] **Port remaining SpecForge improvements**: Check for additional commits after `507da3e`.
