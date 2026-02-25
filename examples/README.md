# Examples

Training examples ordered from simplest to most advanced.

| Example | GPUs | Backend | Model | Nodes | Difficulty |
|---------|------|---------|-------|-------|------------|
| [hf-quickstart](hf-quickstart/) | 3 | HuggingFace | Qwen3-8B | 1 | Easiest |
| [qwen3-8b-single-node](qwen3-8b-single-node/) | 4+ | SGLang | Qwen3-8B | 1 | Easy |
| [kimi-k25-2node-h200](kimi-k25-2node-h200/) | 16 | SGLang | Kimi-K2.5 | 2 | Advanced |
| [kimi-k25-3node-h100](kimi-k25-3node-h100/) | 24 | SGLang | Kimi-K2.5 | 3 | Advanced |

## Quick start

If you just want to try TorchSpec locally, start with **hf-quickstart** (3 GPUs, no SGLang dependency):

```bash
./examples/hf-quickstart/run.sh
```

For production workloads with SGLang async inference, use **qwen3-8b-single-node**:

```bash
./examples/qwen3-8b-single-node/run.sh
```

## Data

Sample training data is in [`data/sample_conversations.jsonl`](data/sample_conversations.jsonl). All examples that use local data point to this file by default.
