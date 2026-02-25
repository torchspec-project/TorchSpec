# HF Quickstart

Simplest entry point: trains an Eagle3 draft model using the HuggingFace backend for target model inference.

## Prerequisites

- 3 GPUs (1 inference + 2 training)
- Model access to `Qwen/Qwen3-8B`

## Config

Uses [`configs/hf_qwen3_8b.yaml`](../../configs/hf_qwen3_8b.yaml):
- **Backend:** HFEngine with HFTargetModel (no SGLang required)
- **Training:** 2 GPUs with FSDP, flex_attention
- **Inference:** 1 GPU

## How to run

```bash
./examples/hf-quickstart/run.sh
```
## Common customizations

```bash
# Adjust training steps
./examples/hf-quickstart/run.sh training.num_train_steps=50

# Change learning rate
./examples/hf-quickstart/run.sh training.learning_rate=5e-5
```
