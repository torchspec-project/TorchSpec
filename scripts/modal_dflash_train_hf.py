"""
DFlash / Eagle3 training on Modal — HF inference backend (1-2 GPUs).

Lightweight variant of modal_dflash_train.py that only provisions 1-2 H100 GPUs
using the HuggingFace inference backend. Use this for small-scale testing or
when SGLang is not needed.

For 4+ GPU SGLang training, use modal_dflash_train.py instead.

GPU auto-config:
    1 GPU  -> HF backend + colocate mode (inference & training share GPU)
    2 GPUs -> HF backend (1 inference + 1 training)

Usage:
    modal run scripts/modal_dflash_train_hf.py                         # 2x H100 (default)
    modal run scripts/modal_dflash_train_hf.py --gpu-count 1           # 1x H100 colocate
    modal run --detach scripts/modal_dflash_train_hf.py                # detached

    To change GPU type, edit HF_GPU constant at the top of this file.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import modal

# =============================================================================
# Constants (shared with modal_dflash_train.py)
# =============================================================================

TORCHSPEC_REPO = "https://github.com/zhubohao911/TorchSpec.git"
TORCHSPEC_BRANCH = "feature/dflash-training"

REPO_DIR = "/workspace/TorchSpec"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/workspace/outputs"
DATA_DIR = "/workspace/data"

HF_GPU = "H100:2"

# =============================================================================
# Modal app + volumes
# =============================================================================

app = modal.App("torchspec-dflash-training-hf")

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("torchspec-outputs", create_if_missing=True)

# =============================================================================
# Container image — base layer (no SGLang needed for HF backend)
# =============================================================================

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "vim", "htop",
        "libibverbs-dev", "librdmacm-dev", "libnuma-dev",
        "libcurl4-openssl-dev",
    )
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        f"git clone {TORCHSPEC_REPO} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {TORCHSPEC_BRANCH}",
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "transformers==4.57.1",
        "datasets",
        "tqdm",
        "wandb",
        "accelerate",
        "pydantic",
        "omegaconf",
        "ray",
        "mooncake-transfer-engine",
        "sglang-router",
        "openai",
        "openai-harmony",
        "qwen-vl-utils",
        "psutil",
        "numpy<2.4",
        "pyzmq",
        "numba",
        "cmake",
        "ninja",
        "packaging",
        "setuptools",
    )
    .run_commands(f"cd {REPO_DIR} && pip install -e '.[dev]'")
    .run_commands(
        "MOONCAKE_DIR=$(python3 -c \"import mooncake, os; print(os.path.dirname(mooncake.__file__))\") && "
        "chmod 755 \"$MOONCAKE_DIR/mooncake_master\" 2>/dev/null || true && "
        "sed -i 's/os.chmod(bin_path, 0o755)/pass/' \"$MOONCAKE_DIR/cli.py\" 2>/dev/null || true",
    )
    .run_commands(
        "mkdir -p /root/.cache && ln -sf /root/.cache/huggingface /root/.cache/huggingface || true",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
            "TORCHSPEC_LOG_LEVEL": "INFO",
            "HF_HOME": HF_CACHE_DIR,
        }
    )
)

# =============================================================================
# GPU configuration — HF backend only (1-2 GPUs)
# =============================================================================

@dataclass
class GPUConfig:
    mode: str
    eagle3_config: str
    dflash_config: str
    overrides: list[str]


def _gpu_config(gpu_count: int) -> GPUConfig:
    if gpu_count == 2:
        return GPUConfig(
            mode="2gpu (HF backend, 1 inference + 1 training)",
            eagle3_config="configs/hf_qwen3_8b.yaml",
            dflash_config="configs/hf_qwen3_8b_dflash_1gpu.yaml",
            overrides=[
                "training.training_num_gpus_per_node=1",
                "inference.inference_num_gpus=1",
                "inference.inference_num_gpus_per_engine=1",
                "inference.inference_num_gpus_per_node=2",
            ],
        )
    return GPUConfig(
        mode="1gpu colocate (HF backend, shared GPU — needs 80GB)",
        eagle3_config="configs/hf_qwen3_8b_1gpu.yaml",
        dflash_config="configs/hf_qwen3_8b_dflash_1gpu.yaml",
        overrides=[
            "training.colocate=true",
            "training.training_num_gpus_per_node=1",
            "inference.inference_num_gpus=1",
            "inference.inference_num_gpus_per_engine=1",
            "inference.inference_num_gpus_per_node=1",
        ],
    )


# =============================================================================
# Training runner
# =============================================================================

def _run_training(
    name: str,
    config: str,
    run_id: str,
    max_steps: int,
    num_epochs: Optional[int],
    gpu_overrides: list[str],
    wandb_project: Optional[str],
    extra_args: list[str],
):
    import os
    import time

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    log_path = f"{OUTPUTS_DIR}/{run_id}.log"

    wandb_args = []
    if os.environ.get("WANDB_API_KEY"):
        wandb_args = [
            "training.report_to=wandb",
            f"training.wandb_project={wandb_project or 'dflash-vs-eagle3'}",
            f"training.wandb_run_id={run_id}",
        ]

    epoch_args = []
    if num_epochs is not None:
        epoch_args = [f"training.num_epochs={num_epochs}"]

    cmd = [
        sys.executable, "-m", "torchspec.train_entry",
        "--config", config,
        f"training.num_train_steps={max_steps}",
        *epoch_args,
        *gpu_overrides,
        *wandb_args,
        *extra_args,
    ]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  TRAINING: {name}")
    print(f"  Config:   {config}")
    print(f"  Run ID:   {run_id}")
    print(f"  Steps:    {max_steps}")
    print(f"  Log:      {log_path}")
    print(sep)
    print(f"  Command:  {' '.join(cmd)}")
    print()

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  {name} completed in {int(elapsed)}s ({mins}m {secs}s)")

    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


# =============================================================================
# Modal function — HF backend (1-2 GPUs)
# =============================================================================

_common_kwargs = dict(
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        OUTPUTS_DIR: outputs_vol,
    },
    timeout=24 * 3600,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)


@app.function(image=base_image, gpu=HF_GPU, **_common_kwargs)
def train_hf(
    gpu_count: int,
    max_steps: int,
    num_epochs: Optional[int],
    run_eagle3: bool,
    run_dflash: bool,
    wandb_project: Optional[str],
    dataset_path: Optional[str],
    dataset_size: int,
    extra_overrides: Optional[str] = None,
):
    """Training entry point for 1-2 GPU configs (HF inference backend)."""
    import os
    import shutil

    import torch

    detected = torch.cuda.device_count()
    print(f"  GPUs detected: {detected}")
    for i in range(detected):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"    GPU {i}: {name} ({mem_gb:.1f} GB)")

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(detected))
    )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{REPO_DIR}/cache/compiled_kernels"

    for stale_dir in [f"{REPO_DIR}/cache", "/tmp/torchinductor_root"]:
        if os.path.isdir(stale_dir):
            shutil.rmtree(stale_dir, ignore_errors=True)
            print(f"  Cleared stale cache: {stale_dir}")

    data_file = dataset_path
    if data_file is None:
        data_file = f"{DATA_DIR}/perfectblend_{dataset_size // 1000}k.jsonl"
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.isfile(data_file):
            print(f"\n  Preparing PerfectBlend dataset ({dataset_size} samples)...")
            subprocess.run(
                [
                    sys.executable, f"{REPO_DIR}/scripts/prepare_perfectblend.py",
                    "--output", data_file,
                    "--sample-size", str(dataset_size),
                    "--seed", "42",
                ],
                check=True,
            )
        print(f"  Dataset: {data_file}")

    cfg = _gpu_config(gpu_count)
    print(f"\n  GPU mode: {cfg.mode}")

    dataset_overrides = [f"dataset.train_data_path={data_file}"]

    user_overrides = []
    if extra_overrides:
        user_overrides = extra_overrides.split()
        print(f"  Extra overrides: {user_overrides}")

    if run_eagle3:
        _run_training(
            name="Eagle3 (baseline)",
            config=cfg.eagle3_config,
            run_id="eagle3-qwen3-8b",
            max_steps=max_steps,
            num_epochs=num_epochs,
            gpu_overrides=cfg.overrides,
            wandb_project=wandb_project,
            extra_args=dataset_overrides + user_overrides,
        )

    if run_dflash:
        _run_training(
            name="DFlash",
            config=cfg.dflash_config,
            run_id="dflash-qwen3-8b",
            max_steps=max_steps,
            num_epochs=num_epochs,
            gpu_overrides=cfg.overrides,
            wandb_project=wandb_project,
            extra_args=dataset_overrides + user_overrides,
        )

    outputs_vol.commit()

    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  GPU mode: {cfg.mode}")
    if run_eagle3:
        print(f"  Eagle3 log: {OUTPUTS_DIR}/eagle3-qwen3-8b.log")
    if run_dflash:
        print(f"  DFlash log: {OUTPUTS_DIR}/dflash-qwen3-8b.log")
    if os.environ.get("WANDB_API_KEY"):
        proj = wandb_project or "dflash-vs-eagle3"
        print(f"  WandB dashboard: https://wandb.ai/{proj}")
    print()
    print("  Key metrics to compare:")
    print("    train/avg_acc   — top-1 token prediction accuracy")
    print("    train/avg_loss  — CE (DFlash) vs KL (Eagle3)")
    print("    train/grad_norm — gradient health")


# =============================================================================
# CLI entry point
# =============================================================================

@app.local_entrypoint()
def main(
    gpu_count: int = 2,
    max_steps: int = 200,
    num_epochs: int = 0,
    run_eagle3: bool = False,
    run_dflash: bool = True,
    wandb_project: Optional[str] = None,
    dataset_path: Optional[str] = None,
    dataset_size: int = 50000,
    extra_overrides: str = "",
):
    if gpu_count > 2:
        print(f"Error: This script supports 1-2 GPUs (got {gpu_count}).")
        print("  For 4+ GPU SGLang mode, use: modal run scripts/modal_dflash_train.py")
        return

    epochs_override = num_epochs if num_epochs > 0 else None
    train_gpus = 1 if gpu_count == 2 else 0

    print("=" * 60)
    print("  DFlash Training on Modal (HF Backend)")
    print("=" * 60)
    print(f"  GPU:          {HF_GPU} ({'1 infer + 1 train' if gpu_count == 2 else 'colocate'})")
    print(f"  Eagle3:       {'YES' if run_eagle3 else 'SKIP'}")
    print(f"  DFlash:       {'YES' if run_dflash else 'SKIP'}")
    print(f"  Max steps:    {max_steps}")
    print(f"  Num epochs:   {epochs_override or '(from YAML config)'}")
    print(f"  Dataset:      {dataset_path or f'PerfectBlend ({dataset_size} samples)'}")
    print(f"  WandB:        {wandb_project or '(disabled / auto)'}")
    if extra_overrides:
        print(f"  Overrides:    {extra_overrides}")
    print("=" * 60)

    if not run_eagle3 and not run_dflash:
        print("Nothing to run. Pass --run-eagle3 and/or --run-dflash.")
        return

    train_hf.spawn(
        gpu_count=gpu_count,
        max_steps=max_steps,
        num_epochs=epochs_override,
        run_eagle3=run_eagle3,
        run_dflash=run_dflash,
        wandb_project=wandb_project,
        dataset_path=dataset_path,
        dataset_size=dataset_size,
        extra_overrides=extra_overrides or None,
    ).get()
