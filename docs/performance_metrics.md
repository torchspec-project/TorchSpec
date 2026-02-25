# Performance Metrics

TorchSpec includes an opt-in performance metrics system designed to identify
pipeline bottlenecks between the asynchronous inference and synchronous training
stages.

## Enabling

Set `debug.enable_perf_metrics: true` in your YAML config, or pass the CLI
override:

```bash
python train.py --config configs/default.yaml debug.enable_perf_metrics=true
```

When disabled (default), **zero overhead** is added to the training loop.

## Metrics Reference

All metrics live under the `perf/` namespace in wandb (tied to `train/step`).

### Training Metrics (per optimizer step)

| Metric | Unit | Source | Description |
|---|---|---|---|
| `perf/step_time` | seconds | `trainer.py` | Wall-clock time of `train_from_queue`, covering data fetch + compute + optimizer step. Accurate because `_aggregate_metrics` ends with `.item()` which syncs CUDA. |
| `perf/data_time` | seconds | `trainer.py` | Wall-clock time spent in the data iterator (Ray queue get + Mooncake RDMA fetch + collation + H2D copy). Measured with `time.time()` since this is CPU/network-bound work. |
| `perf/compute_time` | seconds | `trainer.py` | GPU execution time for forward + backward + optimizer step. Measured with **CUDA events** (`torch.cuda.Event(enable_timing=True)`), extracted at the existing `.item()` sync point in `_aggregate_metrics`. No additional `torch.cuda.synchronize()` calls. |
| `perf/train_capacity` | samples/s | `loop.py` | `global_batch_size / step_time`. The effective training throughput if data were always available. |

### Inference Metrics (aggregated between training steps)

| Metric | Unit | Source | Description |
|---|---|---|---|
| `perf/infer_capacity` | samples/s | `inference_manager.py` | System-level inference capacity: `per_slot_rate * max_concurrent_slots`. Per-slot rate is computed from recent `engine.generate()` batch times. |
| `perf/infer_batch_time` | seconds | `inference_manager.py` | Average wall-clock time of a single `engine.generate.remote()` call. Measured inside the semaphore (excludes engine slot wait time). |

### Pipeline Health Metrics

| Metric | Unit | Source | Description |
|---|---|---|---|
| `perf/dispatch_wait` | seconds | `loop.py` | Wall-clock time the main loop spent waiting for the sample pool to accumulate enough samples for dispatch. High values indicate inference is the bottleneck. |

## tqdm Progress Bar

When perf metrics are enabled, the progress bar shows additional fields:

```
Training: 10%|== | loss=0.350, acc=0.830, thru=12.1, I=18.5, T=13.2, wait=0.1s, pool=24, epoch=1/10
```

| Key | Meaning |
|---|---|
| `thru` | Actual pipeline throughput (samples/s). This is the realized end-to-end speed of the system — how many samples are actually flowing through the pipeline per second. In steady state, `thru ≈ min(I, T)`. Derived from the controller's 10-second sliding window on sample pool inflow. |
| `I` | Inference capacity (samples/s) — how fast inference *could* produce if training were infinitely fast. |
| `T` | Training capacity (samples/s) — how fast training *could* consume if inference were infinitely fast. |
| `wait` | Dispatch wait time (seconds) |
| `pool` | Current sample pool size |

When perf metrics are disabled, the bar shows only `thru` (pipeline throughput).

## Bottleneck Identification Guide

### Quick diagnosis

```
if dispatch_wait >> 0:
    bottleneck = "inference"    # training is starved for data
elif pool grows over time:
    bottleneck = "training"     # inference outpaces training
```

### Drill-down within training

```
if data_time >> compute_time:
    bottleneck = "mooncake transfer / data loading"
elif compute_time >> data_time:
    bottleneck = "GPU compute (forward/backward/optimizer)"
```

### Interpreting capacity metrics

- **`I > T`**: Inference can produce data faster than training can consume.
  Training is the bottleneck. The pool will tend to grow.
- **`T > I`**: Training can consume faster than inference produces.
  Inference is the bottleneck. You'll see high `dispatch_wait`.
- **`I ~ T`**: System is balanced. Pipeline throughput (`thru`) approaches
  both capacity values.

### Example scenarios

**Scenario 1: Inference bottleneck**
```
thru=8.2, I=8.5, T=15.3, wait=2.1s, pool=0
```
Inference capacity (8.5) is much lower than training capacity (15.3).
Dispatch wait is high (2.1s per step). Pool is empty. Consider adding more
inference engines or increasing `inference_batch_size`.

**Scenario 2: Training bottleneck**
```
thru=12.1, I=25.0, T=12.3, wait=0.0s, pool=48
```
Training capacity (12.3) is much lower than inference capacity (25.0).
Dispatch wait is near zero. Pool is large and growing. Check whether
`data_time` or `compute_time` dominates to decide between optimizing Mooncake
transfer or GPU computation.

**Scenario 3: Mooncake transfer bottleneck**
```
step_time=1.2s, data_time=0.9s, compute_time=0.25s
```
Data loading takes 75% of step time. Mooncake RDMA transfer or Ray queue
latency is the sub-bottleneck. Consider increasing Mooncake buffer sizes or
reducing sequence length.

## Implementation Notes

### Why CUDA events instead of `torch.cuda.synchronize()`?

`torch.cuda.synchronize()` forces the CPU to wait for all GPU work to
complete, which **stalls the CUDA execution pipeline** and can measurably
degrade training throughput. CUDA events (`torch.cuda.Event(enable_timing=True)`)
insert non-blocking markers into the GPU stream. `event.record()` is
essentially free (~500ns). Timing is extracted later at an existing sync point
(`_aggregate_metrics` calls `.item()` which naturally syncs), so no extra
synchronization is introduced.

### Why `time.time()` for data loading?

Data loading involves CPU/network operations (Ray queue get, Mooncake RDMA
fetch, tensor collation, H2D memcpy). The GPU stream may be idle during these
operations, so CUDA events would report near-zero time. Wall-clock time
correctly captures the actual wait.

### Concurrency model for inference capacity

Multiple inference engines run in parallel, controlled by an asyncio semaphore
(`max_concurrent_per_engine * num_engines` slots). Each `engine.generate.remote()`
call is timed individually **after** the semaphore is acquired (excluding slot
wait time). The system capacity is:

```
infer_capacity = (total_samples / total_engine_time) * max_concurrent_slots
```

This represents the theoretical maximum throughput if all engine slots are
fully utilized.
