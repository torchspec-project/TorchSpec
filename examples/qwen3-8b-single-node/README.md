# SGLang Single-Node

Demo of disaggregated training with SGLang async inference engine.

## Prerequisites

- 4+ GPUs (2 inference + 2 training by default)
- Model access to `Qwen/Qwen3-8B`
- SGLang installed (included in the `torchspec` conda environment)

## Config

Uses [`configs/sglang_qwen3_8b.yaml`](../../configs/sglang_qwen3_8b.yaml):
- **Backend:** SGLang engine with async inference
- **Training:** 2 GPUs with FSDP, flex_attention
- **Inference:** 2 GPUs in duplicate mode (each engine has full model copy)

## How to run

```bash
./examples/qwen3-8b-single-node/run.sh
```

With a custom config:

```bash
./examples/qwen3-8b-single-node/run.sh configs/sglang_qwen3_8b.yaml
```

Override settings:

```bash
./examples/qwen3-8b-single-node/run.sh configs/sglang_qwen3_8b.yaml training.num_train_steps=10
```

## What to expect

Training launches with SGLang serving the target model for inference. Loss should decrease steadily. Logs are printed to stdout.

## Common customizations

```bash
# Use all 8 GPUs (4 inference + 4 training)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/qwen3-8b-single-node/run.sh \
    training.training_num_gpus_per_node=4 \
    inference.inference_num_gpus=4
```
