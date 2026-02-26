# Kimi-K2.5 3-Node Training

Production 3-node setup for training an Eagle3 draft model for Kimi-K2.5 (MoE, 256K context).

## Prerequisites

- 3 nodes with 24 GPUs total:
  - Node 0 (head): 8 GPUs (GPU 0-7) for training
  - Node 1-2 (workers): 8 GPUs each for inference (TP=16)
- Model access to `moonshotai/Kimi-K2.5`
- RDMA network (Mooncake with `mlx5_0` device)
- Running launch_sglang_server.sh to validate local env setup.

## Config

Uses [`configs/sglang_kimi_k25_3node.yaml`](../../configs/sglang_kimi_k25_3node.yaml) with draft model config [`configs/draft_models/kimi_k25_eagle3.json`](../../configs/draft_models/kimi_k25_eagle3.json).

## Dataset format

The dataset is a JSONL file where each row has a `conversations` field containing an OpenAI-style conversation. See [`examples/data/sample_kimi_k25_conversations.jsonl`](../data/sample_kimi_k25_conversations.jsonl) for complete examples covering text-only, multimodal (single/multi-image), system messages, multi-turn, and tool calls.

Example (multimodal with image):

```json
{
  "id": "kimi_mm_001",
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/...jpg"}},
        {"type": "text", "text": "What animal is in this image?"}
      ]
    },
    {
      "role": "assistant",
      "content": "<think>The image shows a domestic cat.</think>This is a cat."
    }
  ]
}
```

Key points:

- Set `chat_template=kimi-k25-vlm` and `sglang_enable_multimodal=True` in your config.
- Assistant responses should include `<think>...</think>` blocks.
- For multimodal samples, use OpenAI-style list content with `image_url` or `image` fields — the pipeline extracts URLs for the SGLang engine and replaces image items with `<|image|>` placeholders for tokenization.
- Plain-string `<|image|>` placeholders tokenize correctly but won't send pixel data to the engine — always use list-of-dicts content for multimodal samples.
- For images in remote storage (S3, GCS, etc.), use pre-signed URLs so the engine can fetch them without credentials.

## How to run

### 1. Start the Ray cluster

On the **head node** (node 0):

```bash
NODE_ROLE=head bash examples/kimi-k25-3node-h100/setup_ray_cluster.sh
```

On each **worker node** (nodes 1-2):

```bash
HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/kimi-k25-3node-h100/setup_ray_cluster.sh
```

### 2. Launch training

On the **head node**:

```bash
bash examples/kimi-k25-3node-h100/run.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_ray_cluster.sh` | Start Ray head/worker nodes |
| `run.sh` | Launch training (run on head node after cluster is ready) |
| `launch_sglang_server.sh` | Reference: standalone SGLang server launch for Kimi-K2.5 |

## Common customizations

```bash
# Override GPU counts
TRAIN_GPUS=8 INFERENCE_GPUS=16 bash examples/kimi-k25-3node-h100/run.sh

# Override config
CONFIG_FILE=path/to/custom.yaml bash examples/kimi-k25-3node-h100/run.sh

# Pass extra overrides
bash examples/kimi-k25-3node-h100/run.sh training.learning_rate=1e-5
```
