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
