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

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class DatasetConfig:
    chat_template: str = "llama3"
    defer_tokenization: bool = False
    eval_data_path: Optional[str] = None
    eval_interval: int = 50
    eval_micro_batch_size: Optional[int] = None
    last_turn_loss_only: bool = False
    prompt_key: str = "conversations"
    train_data_path: str = ""


@dataclass
class DebugConfig:
    debug_inference_only: bool = False
    debug_train_only: bool = False
    enable_perf_metrics: bool = True
    max_dump_steps: int = 5
    memory_recorder: str = "torch"
    memory_snapshot_dir: str = "."
    memory_snapshot_num_steps: Optional[int] = None
    memory_snapshot_path: str = ""
    profile_dir_name: Optional[str] = "/tmp/torchspec_profiles"
    profile_step_end: int = 0
    profile_step_start: int = 0
    profile_target: list = field(default_factory=lambda: ["train_overall"])
    record_memory_history: bool = False
    save_debug_train_data: Optional[str] = None
    use_pytorch_profiler: bool = False


@dataclass
class InferenceConfig:
    aux_hidden_states_layers: Optional[list] = None
    inference_batch_size: int = 1
    inference_buffer_threshold: int = 32
    inference_engine_type: str = "hf"
    inference_fetch_batch: int = 1
    inference_num_gpus: Optional[int] = None
    inference_num_gpus_per_engine: int = 1
    inference_num_gpus_per_node: int = 8
    max_sample_pool_size: int = 0


@dataclass
class LoggingConfig:
    report_to: str = "none"
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_dir: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_host: Optional[str] = None
    wandb_key: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_random_suffix: bool = True
    wandb_run_id: Optional[str] = None
    wandb_team: Optional[str] = None


@dataclass
class ModelConfig:
    draft_model_config: Optional[str] = None
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    target_model_backend: str = "sglang"
    target_model_path: str = ""
    trust_remote_code: bool = False


@dataclass
class SGLangConfig:
    sglang_additional_ports: int = 4
    sglang_attention_backend: str = "flashinfer"
    sglang_base_gpu_id: int = 0
    sglang_chunked_prefill_size: Optional[int] = None
    sglang_constrained_json_whitespace_pattern: Optional[str] = None
    sglang_context_length: Optional[int] = None
    sglang_data_parallel_size: int = 1
    sglang_disable_cuda_graph: bool = True
    sglang_disable_cuda_graph_padding: bool = False
    sglang_disable_custom_all_reduce: bool = False
    sglang_disable_disk_cache: bool = False
    sglang_disable_flashinfer: bool = False
    sglang_disable_flashinfer_autotune: bool = False
    sglang_disable_flashinfer_sampling: bool = False
    sglang_disable_mla: bool = False
    sglang_disable_nan_detection: bool = False
    sglang_disable_outlines_disk_cache: bool = False
    sglang_disable_radix_cache: bool = False
    sglang_disable_regex_jump_forward: bool = False
    sglang_dist_init_addr: Optional[str] = None
    sglang_dist_timeout: int = 20
    sglang_download_dir: Optional[str] = None
    sglang_dp_size: int = 1
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_enable_metrics: bool = False
    sglang_enable_mixed_chunk: bool = False
    sglang_enable_mla: bool = False
    sglang_enable_multimodal: bool = False
    sglang_enable_nan_detection: bool = False
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_overlap_schedule: bool = False
    sglang_enable_piecewise_cuda_graph: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = True
    sglang_ep_size: int = 1
    sglang_expert_parallel_size: int = 1
    sglang_kv_cache_dtype: Optional[str] = None
    sglang_init_timeout: int = 300
    sglang_log_level: str = "warning"
    sglang_log_level_http: Optional[str] = None
    sglang_log_requests: bool = False
    sglang_log_requests_level: int = 0
    sglang_max_running_requests: Optional[int] = None
    sglang_max_total_tokens: Optional[int] = None
    sglang_mem_fraction_static: float = 0.8
    sglang_model_loader_extra_config: Any = None
    sglang_moe_runner_backend: Optional[str] = None
    sglang_nnodes: int = 1
    sglang_piecewise_cuda_graph_max_tokens: int = 4096
    sglang_piecewise_cuda_graph_tokens: Optional[str] = None
    sglang_pipeline_parallel_size: int = 1
    sglang_port: int = 30000
    sglang_pp_size: int = 1
    sglang_quantization: Optional[str] = None
    sglang_random_seed: Optional[int] = None
    sglang_router_ip: Optional[str] = None
    sglang_router_port: Optional[int] = None
    sglang_router_request_timeout_secs: int = 14400
    sglang_schedule_policy: str = "lpm"
    sglang_speculative_draft_attention_backend: Optional[str] = None
    sglang_server_concurrency: int = 512
    sglang_show_time_cost: bool = False
    sglang_tensor_parallel_size: Optional[int] = None
    sglang_tp_size: int = 8
    sglang_watchdog_timeout: Optional[float] = None


@dataclass
class TrainingConfig:
    attention_backend: str = "sdpa"
    colocate: bool = False
    distributed_backend: str = "nccl"
    distributed_timeout_minutes: int = 10
    draft_accumulation_steps: int = 1
    fsdp_strategy: str = "REPLICATE"

    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    load_path: Optional[str] = None
    lr_decay_style: str = "cosine"
    lr_total_steps: Optional[int] = None
    max_concurrent_batches: int = 1
    max_grad_norm: float = 0.5
    max_seq_length: int = 8192
    no_save_optim: bool = False
    num_epochs: int = 10
    num_train_steps: Optional[int] = None
    micro_batch_size: int = 2
    prefetch_depth: int = 4
    save_interval: int = 5000
    save_per_epoch: bool = False
    seed: int = 0
    train_backend: str = "fsdp"
    train_env_vars: str = "{}"
    training_num_gpus_per_node: int = 1
    training_num_nodes: int = 1
    ttt_length: int = 7
    warmup_ratio: float = 0.015


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mooncake: dict[str, Any] = field(default_factory=dict)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cache_dir: str = "./cache"
    cache_key: Optional[str] = None
    model_download_dir: Optional[str] = None
    output_dir: str = ""


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[list] = None,
    base_config: Optional[DictConfig] = None,
) -> DictConfig:
    schema = OmegaConf.structured(Config)

    configs_to_merge = [schema]

    if base_config is not None:
        configs_to_merge.append(base_config)

    if config_path is not None:
        file_config = OmegaConf.load(config_path)
        configs_to_merge.append(file_config)

    if cli_args:
        cli_config = OmegaConf.from_dotlist(cli_args)
        configs_to_merge.append(cli_config)

    config = OmegaConf.merge(*configs_to_merge)
    return config


def parse_args_and_config() -> DictConfig:
    parser = argparse.ArgumentParser(
        description="Train Eagle3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config configs/llama3.yaml
  python train.py --config configs/llama3.yaml training.micro_batch_size=4
  python train.py --config configs/base.yaml --config configs/experiment.yaml
  python train.py training.learning_rate=1e-5 model.target_model_path=/path/to/model
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        action="append",
        dest="configs",
        help="Path to YAML config file(s). Can be specified multiple times to merge configs.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the final merged config and exit",
    )

    args, unknown = parser.parse_known_args()

    schema = OmegaConf.structured(Config)
    configs_to_merge = [schema]

    if args.configs:
        for config_path in args.configs:
            file_config = OmegaConf.load(config_path)
            configs_to_merge.append(file_config)

    if unknown:
        cli_config = OmegaConf.from_dotlist(unknown)
        configs_to_merge.append(cli_config)

    config = OmegaConf.merge(*configs_to_merge)

    if args.print_config:
        print(OmegaConf.to_yaml(config))
        sys.exit(0)

    return config


# Sections whose fields get a name prefix when flattened to args.
_PREFIXED_SECTIONS = {
    "mooncake": "mooncake_",
}


def config_to_flat_args(config: DictConfig) -> argparse.Namespace:
    flat: dict[str, Any] = {}

    for section_name, section in config.items():
        if isinstance(section, DictConfig):
            prefix = _PREFIXED_SECTIONS.get(section_name, "")
            for key, val in section.items():
                flat_key = f"{prefix}{key}"
                if flat_key in flat:
                    raise ValueError(
                        f"Duplicate config key '{flat_key}' (from section '{section_name}.{key}')"
                    )
                flat[flat_key] = val
        else:
            # Top-level fields (output_dir, cache_dir, etc.)
            if section_name in flat:
                raise ValueError(f"Duplicate config key '{section_name}'")
            flat[section_name] = section

    # --- Computed / alias fields ---
    flat["world_size"] = flat["training_num_nodes"] * flat["training_num_gpus_per_node"]
    flat["rank"] = 0
    flat["dynamic_loss_mask"] = flat["defer_tokenization"]
    flat["use_wandb"] = flat.get("use_wandb", False) or flat.get("report_to") == "wandb"
    flat["use_tensorboard"] = (
        flat.get("use_tensorboard", False) or flat.get("report_to") == "tensorboard"
    )
    save_enabled = flat.get("save_interval", 0) > 0 or flat.get("save_per_epoch", False)
    flat["checkpoint_dir"] = (
        str(Path(flat["output_dir"]) / "checkpoints")
        if flat.get("output_dir") and save_enabled
        else None
    )

    return argparse.Namespace(**flat)


def save_config(config: DictConfig, path: str) -> None:
    OmegaConf.save(config, path)


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
