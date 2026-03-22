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
- [x] **Speed benchmark S1-S3** (Phase 2.2) — all configs ~1.0 step/s, bottleneck is pipeline not compute

## Active — Speed Optimization (Blocking)

- [ ] **Investigate pipeline overhead**: Mooncake KV transfer + Ray dispatch dominates step time (~98%). DFlash compute is only 6-15ms per batch but pipeline cycle takes ~1000ms. Need to reduce pipeline overhead for practical training times.
- [ ] **Try `max_concurrent_batches > 1`**: Overlap inference and training to hide pipeline latency.
- [ ] **Evaluate colocate mode for full training**: 1-GPU colocate achieves ~5 step/s (5x faster) — explore multi-GPU colocate as alternative to Mooncake pipeline.

## Active — Training

- [ ] **Retrain from scratch** with bug fixes applied — fresh training on perfectblend_50k, 6 epochs
- [ ] **τ benchmark matrix** (Phase 3) — measure τ at epoch boundaries, by data size, by domain

## Active — Code Improvements

- [ ] **Commit factory.py timeout fix**: Increase `find_free_port` timeout from 30s to 120s for PyTorch 2.9+ compatibility (currently local pod patch only).
- [ ] **Draft KV cache**: Add KV cache support to `DFlashDraftModel.forward()` (currently recomputes full context each cycle — O(n²) scaling).

## Future

- [ ] **Eagle3 inference comparison**: Side-by-side benchmark of DFlash vs Eagle3.
- [ ] **Data regeneration**: Generate target-model-regenerated training data for τ ≥ 6.0 (Phase 4.4).
- [ ] **Port remaining SpecForge improvements**: Check for additional commits after `507da3e`.
