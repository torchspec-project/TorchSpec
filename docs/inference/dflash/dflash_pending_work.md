# DFlash Pending Work

> Last updated: 2026-03-22

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
- [x] **Phase 3: 5k step training** — completed, loss ~3.4 at step 5000 (~35 min on perfectblend_50k)
- [x] **τ @ 5k steps** — τ = 1.29, DFlash 45.6 tok/s vs baseline 51.2 tok/s (0.88x, slower)

## Active — Phase 3: τ Benchmark Training (incremental)

**Current status**: 5k checkpoint trained and benchmarked. τ = 1.29 is too low for speedup.

- [x] **Train 5k steps** — completed, loss 12→3.4, ~35 min
- [x] **τ @ 5k steps** — τ = 1.29, 45.6 tok/s, 0.88x baseline (too low)
- [ ] **Train to 10k steps** — continue from 5k checkpoint (~17 min additional)
- [ ] **τ @ 10k steps** — expect ~1.5-2.0 based on prior training trajectory
- [ ] **Train to 15k steps** — continue from 10k checkpoint (~17 min additional)
- [ ] **τ @ 15k steps** — first full epoch, expect best single-epoch τ
- [ ] **Eagle3 baseline** — benchmark Eagle3 inference speed on same target model for comparison

**Key observations**:
- Training speed on perfectblend_50k: ~2.3-2.7 step/s (lower than Test 7's 5.3 due to longer sequences)
- Data pipeline: 20-150ms data_time spikes from large sequences. CPU prefetch still helps but doesn't fully hide transfer time for long samples.
- The `acc_len=0.00` in training logs is NOT the inference τ — it's a training metric. Real τ is measured via `benchmark_dflash_inference.py`.

## Active — Code Improvements

- [ ] **Draft KV cache**: Add KV cache support to `DFlashDraftModel.forward()` (currently recomputes full context each cycle — O(n²) scaling). This is likely the main factor limiting inference speedup at higher τ.
- [ ] **Increase prefetch_depth**: For long-sequence datasets, `prefetch_depth=4` or `8` would better hide Mooncake TCP latency spikes.
- [ ] **Set `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` in code**: Add `os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON")` to `train_entry.py` so it's always active without manual shell setup.

## Known Issues

1. **Low τ at 5k steps**: Expected — only ~1/3 of one epoch. Need more training steps.
2. **Data pipeline spikes**: With perfectblend_50k's longer sequences (up to 4096 tokens), data_time can spike to 150-400ms even with CPU prefetch. This reduces effective throughput from 5.3 to ~2.5 step/s.
3. **No draft KV cache**: The benchmark script recomputes full context each draft cycle. With KV cache, DFlash inference would be significantly faster even at the same τ.
4. **Eval timeout with 3 GPUs**: Eval cache generation times out when only 1 GPU is available for inference. Workaround: `dataset.eval_data_path=null`. Note: `eval_interval` is under `dataset`, not `training`. And perf metrics flag is `debug.enable_perf_metrics`. Eval should be disabled for production training.
5. **Missing `libibverbs.so.1`**: Required by mooncake at runtime. Must be installed on the pod (`apt-get install libibverbs-dev`).
6. **RunPod SCP not supported**: RunPod SSH proxy doesn't support SCP file transfer. Workaround: apply patches locally, commit to git, then `git pull` on the pod.
7. **`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` not in codebase**: The env var `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` is only set manually in the shell. Should be set programmatically in the training entry point to avoid forgetting.

## Future

- [ ] **Data regeneration**: Generate target-model-regenerated training data for τ ≥ 6.0 (Phase 4.4).
- [ ] **Port remaining SpecForge improvements**: Check for additional commits after `507da3e`.
