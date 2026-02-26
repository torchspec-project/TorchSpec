#!/usr/bin/env python3
"""Benchmark different EAGLE3 mask_mod modes for flash attention.

Compares compile time and per-call fwd+bwd latency across five mask_mod strategies:
  closure    - compile-time Q_LEN via closure (one kernel per bucket)
  simple     - simplified closure without valid_seq_len checks
  dynamic    - runtime Q_LEN via aux_tensors (vec_size penalty)
  seqlen     - runtime Q_LEN via seqlen_info (runtime integer modulo)
  seqlen_po2 - runtime Q_LEN via seqlen_info (bitmask, power-of-2)

Results on SM100 (B200), bsz=2, lck=3, heads=32/8, head_dim=128, bf16,
q_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:

  mode           compile(s)  speedup      q=128      q=512     q=1024     q=4096     q=8192    q=16384    q=65536   q=131072
  --------------------------------------------------------------------------------------------------------------------------
  closure             88.38     1.0x      1.00x      1.00x      1.00x      1.00x      1.00x      1.00x      1.00x      1.00x
  simple              62.02     1.4x      0.99x      1.01x      1.00x      0.91x      0.89x      0.90x      0.91x      0.91x
  dynamic             27.85     3.2x      1.73x      1.50x      1.69x      1.92x      1.95x      1.96x      1.87x      1.94x
  seqlen              27.35     3.2x      1.03x      1.56x      1.78x      1.91x      1.91x      1.91x      1.84x      1.95x
  seqlen_po2          16.29     5.4x      1.52x      1.04x      1.04x      0.97x      0.94x      0.95x      0.91x      0.96x

Key observations:
  - seqlen_po2 gives the best compile speedup (5.4x) with a single compiled kernel.
    All q_lens here are already power-of-2 so no snap waste; with non-po2 q_lens
    (e.g. 384->512, 640->1024) the speedup is even higher due to fewer buckets.
  - seqlen_po2 per-call latency is within ~5% of closure at q>=512; the small-q
    overhead (1.5x at q=128) is negligible in absolute terms (0.26 vs 0.17 ms).
  - dynamic and seqlen compile 3.2x faster but are ~1.9x slower per-call at large
    q_len due to vec_size penalty (dynamic) or runtime integer modulo (seqlen).
  - simple compiles only 1.4x faster (still one kernel per bucket) but per-call
    is ~9% faster than closure at large q_len (fewer SSA ops).

Usage:
  python tools/bench_eagle3_mask_modes.py
  python tools/bench_eagle3_mask_modes.py --modes closure,seqlen_po2 --q-lens 128,256,512,1024
  python tools/bench_eagle3_mask_modes.py --modes seqlen_po2 --q-lens 128,512,2048,8192,32768,131072
"""

import argparse
import math
import time

import torch

from torchspec.models.draft.llama3_eagle import (
    _build_eagle3_mask_pair,
    _EagleMaskedFlashAttnFunc,
    _snap_q_len,
    set_eagle3_mask_mode,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

ALL_MODES = ["closure", "simple", "dynamic", "seqlen", "seqlen_po2"]


def _clear_compile_caches():
    """Clear all flash_attn compile caches + mask_mod caches."""
    from torchspec.models.draft import llama3_eagle as mod

    # mask_mod caches
    mod._flash_mask_mod_cache.clear()
    mod._flash_mask_mod_simple_cache.clear()
    mod._eagle3_flash_mask_mod_dynamic = None
    mod._eagle3_flash_mask_mod_seqlen = None
    mod._eagle3_flash_mask_mod_seqlen_po2 = None
    # block_sparse caches (SM90)
    mod._bwd_bm_raw_cache.clear()
    mod._bwd_block_sparse_cache.clear()
    # flash_attn compile caches
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    if hasattr(_flash_attn_fwd, "compile_cache"):
        _flash_attn_fwd.compile_cache.clear()
    if hasattr(_flash_attn_bwd, "compile_cache"):
        _flash_attn_bwd.compile_cache.clear()


def run_fwd_bwd(q, k, v, mask_mod_cute, mask_mod_flex, softmax_scale, max_seq_len, aux_tensors):
    """Run forward + backward and synchronize."""
    out = _EagleMaskedFlashAttnFunc.apply(
        q, k, v, mask_mod_cute, mask_mod_flex, softmax_scale, max_seq_len, aux_tensors
    )
    out.sum().backward()
    return out


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


def bench_mode(
    mode: str,
    q_lens: list[int],
    lck: int = 3,
    bsz: int = 2,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype=torch.bfloat16,
    n_warmup: int = 3,
    n_iter: int = 10,
):
    device = torch.cuda.current_device()
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Clear caches and set mode
    _clear_compile_caches()
    set_eagle3_mask_mode(mode)
    torch.cuda.synchronize()

    # --- Compile phase: time building all buckets ---
    snapped = sorted(set(_snap_q_len(ql, mode) for ql in q_lens))
    print(f"  [{mode}] {len(snapped)} bucket(s): {snapped}")

    t0 = time.perf_counter()
    for sq in snapped:
        kv_len = sq * (1 + lck)
        mask_mod_cute, mask_mod_flex, aux_tensors = _build_eagle3_mask_pair(
            sq, kv_len, bsz, lck, device
        )
        q = torch.randn(
            bsz, sq, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        k = torch.randn(bsz, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(bsz, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        run_fwd_bwd(q, k, v, mask_mod_cute, mask_mod_flex, softmax_scale, sq, aux_tensors)
        torch.cuda.synchronize()
        del q, k, v
    compile_time = time.perf_counter() - t0
    print(f"  [{mode}] compile: {compile_time:.2f}s")

    # --- Per-call latency per q_len ---
    results = {}
    for ql in q_lens:
        sq = _snap_q_len(ql, mode)
        kv_len = sq * (1 + lck)
        mask_mod_cute, mask_mod_flex, aux_tensors = _build_eagle3_mask_pair(
            sq, kv_len, bsz, lck, device
        )
        q = torch.randn(
            bsz, sq, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        k = torch.randn(bsz, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(bsz, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(n_warmup):
            q.grad = None
            run_fwd_bwd(q, k, v, mask_mod_cute, mask_mod_flex, softmax_scale, sq, aux_tensors)
            torch.cuda.synchronize()

        # Timed
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for i in range(n_iter):
            q.grad = None
            start_events[i].record()
            run_fwd_bwd(q, k, v, mask_mod_cute, mask_mod_flex, softmax_scale, sq, aux_tensors)
            end_events[i].record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_ms = sum(times) / len(times)
        results[ql] = avg_ms
        print(f"  [{mode}] q_len={ql} (snap={sq}): {avg_ms:.3f} ms  fwd+bwd")
        del q, k, v

    return compile_time, results


def main():
    parser = argparse.ArgumentParser(description="Benchmark EAGLE3 mask_mod modes")
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(ALL_MODES),
        help="Comma-separated list of modes to benchmark",
    )
    parser.add_argument(
        "--q-lens",
        type=str,
        default="128,256,512,1024,2048,4096,8192,16384,32768,65536,131072",
        help="Comma-separated list of q_len values",
    )
    parser.add_argument("--lck", type=int, default=3, help="Number of cached KV blocks")
    parser.add_argument("--bsz", type=int, default=2, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=10)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    q_lens = [int(x.strip()) for x in args.q_lens.split(",")]

    print(f"Benchmarking modes: {modes}")
    print(f"Q_LEN values: {q_lens}")
    print(
        f"lck={args.lck}, bsz={args.bsz}, heads={args.num_heads}/{args.num_kv_heads}, "
        f"head_dim={args.head_dim}"
    )
    print()

    all_results = {}
    for mode in modes:
        print(f"=== Mode: {mode} ===")
        compile_time, results = bench_mode(
            mode,
            q_lens,
            lck=args.lck,
            bsz=args.bsz,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        all_results[mode] = {"compile": compile_time, "per_call": results}
        print()

    # --- Summary table ---
    baseline_mode = "closure" if "closure" in all_results else modes[0]
    baseline = all_results.get(baseline_mode, {})

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'mode':<14} {'compile(s)':>10} {'speedup':>8}", end="")
    for ql in q_lens:
        print(f" {'q=' + str(ql):>10}", end="")
    print()
    print("-" * (34 + 11 * len(q_lens)))

    for mode in modes:
        r = all_results[mode]
        ct = r["compile"]
        base_ct = baseline.get("compile", ct)
        speedup = base_ct / ct if ct > 0 else float("inf")
        print(f"{mode:<14} {ct:>10.2f} {speedup:>7.1f}x", end="")
        for ql in q_lens:
            ms = r["per_call"].get(ql, float("nan"))
            base_ms = baseline.get("per_call", {}).get(ql, ms)
            ratio = ms / base_ms if base_ms > 0 else float("nan")
            print(f" {ratio:>9.2f}x", end="")
        print()

    # Restore default mode
    set_eagle3_mask_mode("seqlen_po2")


if __name__ == "__main__":
    main()
