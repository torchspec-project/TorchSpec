# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Eval-specific setup and runtime helpers for the controller loop."""

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from torchspec.training.checkpoint import _read_checkpoint_metadata, _write_checkpoint_metadata
from torchspec.utils.logging import logger

EVAL_CACHE_IDLE_TIMEOUT = 300.0


@dataclass
class EvalSetupState:
    eval_interval: int
    eval_enabled: bool
    eval_cache_loaded: bool
    eval_cache_path: str | None
    best_eval_score: float
    num_dispatches: int
    eval_samples_per_rank: int


def _check_idle_timeout(
    dispatched_count: int, last_progress_at: float, num_dispatches: int
) -> None:
    idle_for = time.monotonic() - last_progress_at
    if idle_for >= EVAL_CACHE_IDLE_TIMEOUT:
        raise TimeoutError(
            "Timed out while waiting for eval cache generation "
            f"(no progress during eval for {idle_for:.1f}s, "
            f"idle_timeout={EVAL_CACHE_IDLE_TIMEOUT:.1f}s, "
            f"dispatched={dispatched_count}/{num_dispatches})"
        )


def update_checkpoint_eval_meta(
    checkpoint_dir: str | None,
    step: int,
    eval_metrics: dict,
    current_best: float,
) -> float:
    """Append eval metrics to checkpoint meta.json and track best checkpoint."""
    if not checkpoint_dir or not eval_metrics:
        return current_best

    base_dir = Path(checkpoint_dir).expanduser()
    step_id = step + 1
    meta_path = base_dir / f"iter_{step_id:07d}" / "meta.json"

    metadata = _read_checkpoint_metadata(meta_path)
    if not metadata:
        logger.warning(f"Checkpoint meta.json not found at {meta_path}, skipping eval meta update")
        return current_best

    for key in ("eval/avg_loss", "eval/avg_acc", "eval/simulated_acc_len"):
        if key in eval_metrics:
            metadata[key] = eval_metrics[key]
    _write_checkpoint_metadata(meta_path, metadata)

    score = eval_metrics.get("eval/simulated_acc_len")
    if score is not None and score > current_best:
        current_best = score
        (base_dir / "best_checkpointed_iteration.txt").write_text(str(step_id))
        _write_checkpoint_metadata(base_dir / "best_meta.json", metadata)
        logger.info(f"New best checkpoint: iter_{step_id:07d} (simulated_acc_len={score:.2f})")

    return current_best


def generate_eval_cache(
    controller,
    train_group,
    num_dispatches: int,
    samples_per_rank: int,
    eval_cache_path: str | None,
) -> None:
    """Drain eval results from inference and cache individual samples on each trainer."""
    total_samples = num_dispatches * samples_per_rank
    last_progress_at = time.monotonic()
    logger.info(
        f"Caching eval hidden states from inference engine "
        f"({num_dispatches} dispatches, {total_samples} samples per rank)..."
    )
    dispatched = 0
    eval_progress = tqdm(total=num_dispatches, desc="Eval caching", unit="dispatch")
    while dispatched < num_dispatches:
        ok = ray.get(controller.try_dispatch_eval_batch.remote())
        if ok:
            train_group.cache_eval_samples(samples_per_rank)
            dispatched += 1
            last_progress_at = time.monotonic()
            eval_progress.update(1)
        else:
            _check_idle_timeout(dispatched, last_progress_at, num_dispatches)
            time.sleep(0.5)
    eval_progress.close()

    # Wait until all eval samples have finished inference before clearing eval IDs.
    # Otherwise, late eval results can be misrouted into the training pool.
    while True:
        if ray.get(controller.is_eval_dispatch_complete.remote()):
            break
        ok = ray.get(controller.try_dispatch_eval_batch.remote())
        if ok:
            train_group.cache_eval_samples(samples_per_rank)
            dispatched += 1
            last_progress_at = time.monotonic()
            logger.warning(
                "Eval: dispatched extra full batch while waiting for completion; "
                "check eval_dispatch_batch_size consistency"
            )
        else:
            _check_idle_timeout(dispatched, last_progress_at, num_dispatches)
            time.sleep(0.1)

    ray.get(controller.finalize_eval_dispatch.remote())
    logger.info(f"Eval caching complete ({dispatched * samples_per_rank} samples per rank)")
    if eval_cache_path:
        train_group.async_save_eval_cache(eval_cache_path)
        logger.info(f"Eval cache save started (async) to {eval_cache_path}")


def run_eval(step: int, train_group, eval_enabled: bool) -> dict:
    """Run forward-only eval from cache. Assumes eval cache is already populated."""
    if not eval_enabled:
        return {}
    eval_results = train_group.run_eval()
    eval_metrics = eval_results[0] if eval_results else {}
    if eval_metrics:
        eval_metrics["eval/step"] = step
        if wandb.run is not None:
            wandb.log(eval_metrics)
        logger.info(
            f"Step {step} eval: "
            f"loss={eval_metrics.get('eval/avg_loss', 0):.4f}, "
            f"acc={eval_metrics.get('eval/avg_acc', 0):.4f}, "
            f"sim_acc_len={eval_metrics.get('eval/simulated_acc_len', 0):.2f}"
        )
    return eval_metrics


def setup_eval(controller, train_group, args, eval_dataset_size: int) -> EvalSetupState:
    """Prepare eval runtime settings and optionally load/submit eval cache input."""
    eval_interval = args.eval_interval
    eval_enabled = eval_dataset_size > 0
    eval_cache_path: str | None = None
    eval_cache_loaded = False
    num_dispatches = 0

    dispatch_bs = args.per_dp_rank_batch_size * args.dp_size
    max_pool = getattr(args, "max_sample_pool_size", 0) or dispatch_bs
    eval_samples_per_rank = max_pool // args.dp_size

    if eval_enabled:
        num_dispatches = eval_dataset_size // max_pool
        if num_dispatches == 0:
            logger.warning(
                f"Eval dataset ({eval_dataset_size} samples) is smaller than "
                f"max_sample_pool_size ({max_pool}). Disabling eval."
            )
            eval_enabled = False
        else:
            dropped = eval_dataset_size - num_dispatches * max_pool
            if dropped > 0:
                logger.info(
                    f"Eval: {dropped} samples will be dropped (not enough for a full dispatch)"
                )

    best_eval_score = -float("inf")
    if eval_enabled and args.checkpoint_dir:
        best_meta_path = Path(args.checkpoint_dir).expanduser() / "best_meta.json"
        if best_meta_path.exists():
            existing = _read_checkpoint_metadata(best_meta_path)
            if "eval/simulated_acc_len" in existing:
                best_eval_score = existing["eval/simulated_acc_len"]
                logger.info(f"Resumed best eval score: {best_eval_score:.2f}")

    if eval_enabled:
        cache_dir = os.path.abspath(getattr(args, "cache_dir", "./cache"))
        cache_key = hashlib.md5(
            f"{getattr(args, 'eval_data_path', '')}|"
            f"{getattr(args, 'target_model_path', '')}|"
            f"{getattr(args, 'max_seq_length', 0)}|"
            f"{max_pool}".encode()
        ).hexdigest()[:12]
        eval_cache_path = os.path.join(cache_dir, "eval_cache", cache_key)

        loaded = train_group.load_eval_cache(eval_cache_path)
        if all(n > 0 for n in loaded):
            eval_cache_loaded = True
            logger.info(
                f"Eval: loaded cached tensors from {eval_cache_path} ({loaded[0]} batches per rank)"
            )
        else:
            ray.get(controller.submit_eval_dataset.remote())
            logger.info(
                f"Eval: {eval_dataset_size} samples, "
                f"{num_dispatches} dispatches, "
                f"{eval_samples_per_rank} samples/rank"
            )

    return EvalSetupState(
        eval_interval=eval_interval,
        eval_enabled=eval_enabled,
        eval_cache_loaded=eval_cache_loaded,
        eval_cache_path=eval_cache_path,
        best_eval_score=best_eval_score,
        num_dispatches=num_dispatches,
        eval_samples_per_rank=eval_samples_per_rank,
    )
