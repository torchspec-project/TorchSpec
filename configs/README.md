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
| `sglang` | `sglang_tp_size`, `sglang_mem_fraction_static` | SGLang-specific settings |
| `mooncake` | `protocol`, `device_name` | Mooncake transfer engine settings |

## Draft model configs

Architecture configs for Eagle3 draft models are in [`draft_models/`](draft_models/). These define the draft model architecture (hidden size, layers, attention heads, etc.) and are referenced via the `model.draft_model_config` field.

## Creating a custom config

1. Copy the closest existing config
2. Update `model.target_model_path` to your model
3. Adjust GPU counts in `training` and `inference` sections
4. If your model needs a custom draft architecture, add a config in `draft_models/`
