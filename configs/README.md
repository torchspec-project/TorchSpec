# Configs

Training configuration files for TorchSpec Eagle3 distillation.

## Config hierarchy

```
default.yaml          Base defaults (all fields with sane values)
  └── specific.yaml   Model/setup-specific overrides
        └── CLI args   Runtime overrides (highest priority)
```

Any value in a config file can be overridden via CLI:

```bash
python -m torchspec.train_entry --config configs/sglang_qwen3_8b.yaml training.learning_rate=5e-5
```

## Config files

| Config | Backend | Model | Use case |
|--------|---------|-------|----------|
| `default.yaml` | — | — | Base template with all defaults |
| `hf_qwen3_8b.yaml` | HuggingFace | Qwen3-8B | Single-node, 3 GPUs |
| `sglang_qwen3_8b.yaml` | SGLang | Qwen3-8B | Single-node, 4+ GPUs |
| `sglang_kimi_k25_2node.yaml` | SGLang | Kimi-K2.5 | 2-node H200, 16 GPUs |
| `sglang_kimi_k25_3node.yaml` | SGLang | Kimi-K2.5 | 3-node H100, 24 GPUs |

## Key sections

| Section | Key fields | Description |
|---------|-----------|-------------|
| `model` | `target_model_path`, `draft_model_config` | Target model and optional draft architecture |
| `dataset` | `train_data_path`, `chat_template` | Training data and tokenization |
| `training` | `learning_rate`, `micro_batch_size`, `ttt_length` | Training hyperparameters |
| `inference` | `inference_engine_type`, `inference_num_gpus` | Inference backend configuration |
| `inference.sglang` | `tp_size`, `mem_fraction_static`, `extra_args` | SGLang engine settings (nested under inference) |
| `mooncake` | `protocol`, `device_name` | Mooncake transfer engine settings |

## SGLang engine configuration

SGLang settings live under `inference.sglang` and are split into two tiers:

### Essential fields

These are the fields TorchSpec directly uses. Set them in YAML or via CLI:

```yaml
inference:
  sglang:
    tp_size: 8
    mem_fraction_static: 0.6
    context_length: 262144
    nnodes: 2
    dist_timeout: 600
    enable_multimodal: true
```

| Field | Default | Description |
|-------|---------|-------------|
| `tp_size` | 8 | Tensor parallel degree (validated against `nnodes × gpus_per_engine`) |
| `pp_size` | 1 | Pipeline parallel degree (must be 1) |
| `nnodes` | 1 | Nodes per inference replica (>1 enables multi-node TP) |
| `mem_fraction_static` | 0.8 | GPU memory fraction for KV cache |
| `context_length` | null | Override model's default context length |
| `attention_backend` | flashinfer | Attention backend |
| `quantization` | null | Quantization method (e.g. `awq`, `gptq`) |
| `kv_cache_dtype` | null | KV cache dtype override |
| `moe_runner_backend` | null | MoE runner backend |
| `model_loader_extra_config` | null | Extra config dict for model loading |
| `disable_flashinfer_autotune` | false | Disable FlashInfer autotuning |
| `enable_multimodal` | false | Enable multimodal input support |
| `enable_metrics` | false | Forward SGLang metrics to W&B |
| `dist_init_addr` | null | Distributed init address (auto-negotiated if unset) |
| `dist_timeout` | 20 | Distributed init timeout (seconds) |
| `init_timeout` | 300 | Engine initialization timeout (seconds) |
| `log_level` | warning | SGLang log level |
| `log_requests` | false | Log individual requests |
| `log_requests_level` | 0 | Request log verbosity |

### `extra_args`: power-user passthrough

Any additional `sgl.Engine` keyword argument can be passed via `extra_args`. These are forwarded as-is to the engine constructor:

```yaml
inference:
  sglang:
    tp_size: 8
    extra_args:
      watchdog_timeout: 1800
      enable_torch_compile: true
      enable_nccl_nvls: false
```

Or via CLI:

```bash
python -m torchspec.train_entry --config my.yaml inference.sglang.extra_args.watchdog_timeout=1800
```

### Protected keys

Certain `sgl.Engine` parameters are managed internally by TorchSpec and cannot be set via `extra_args`. If attempted, they are silently dropped with a warning. These include topology keys (`tp_size`, `pp_size`, `base_gpu_id`, `nnodes`, `node_rank`), auto-selected ports (`port`, `nccl_port`), networking keys (`dist_init_addr`, `dist_timeout`), spec-training invariants (`disable_radix_cache`, `enable_return_hidden_states`, `enable_aux_hidden_states`, `enable_spec_training_mooncake`, `chunked_prefill_size`, `disable_cuda_graph`), and keys derived from other config sections (`model_path`, `trust_remote_code`, `mem_fraction_static`).

## Draft model configs

Architecture configs for Eagle3 draft models are in [`draft_models/`](draft_models/). These define the draft model architecture (hidden size, layers, attention heads, etc.) and are referenced via the `model.draft_model_config` field.

## Creating a custom config

1. Copy the closest existing config
2. Update `model.target_model_path` to your model
3. Adjust GPU counts in `training` and `inference` sections
4. If your model needs a custom draft architecture, add a config in `draft_models/`
