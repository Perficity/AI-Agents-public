# FSDP vs DeepSpeed ZeRO: Comparison Reference

## Table of Contents

- [Overview](#overview)
- [When to Pick FSDP](#when-to-pick-fsdp)
- [When to Pick DeepSpeed ZeRO](#when-to-pick-deepspeed-zero)
- [Stage / Strategy Mapping](#stage--strategy-mapping)
- [Memory Model Comparison](#memory-model-comparison)
- [Configuration Quick-Start](#configuration-quick-start)
- [Known Pitfalls](#known-pitfalls)
- [Canonical Sources](#canonical-sources)

## Overview

FSDP (Fully Sharded Data Parallel) is PyTorch-native sharding. DeepSpeed ZeRO is a separate library with more options and more configuration surface. Both achieve similar sharding semantics but differ in integration complexity, debugging experience, and ecosystem compatibility.

As of 2026, FSDP2 (the redesigned version in torchtitan) is the recommended starting point for new PyTorch-based pre-training projects. DeepSpeed ZeRO remains the reference for Hugging Face TRL/Accelerate workflows and when ZeRO-Infinity (NVMe offload) is needed.

## When to Pick FSDP

- Pure PyTorch stack; want to avoid additional dependencies.
- Using torchtitan, litgpt, or a custom training loop.
- Need tight integration with `torch.compile`.
- Model is large but fits on GPU cluster without NVMe offload.
- Team is comfortable debugging PyTorch distributed primitives.

## When to Pick DeepSpeed ZeRO

- Using Hugging Face Trainer / Accelerate (native DeepSpeed integration).
- Need ZeRO-Infinity (NVMe offload for extremely large models).
- Need ZeRO-Offload (CPU offload for optimizer state).
- Existing config management via `deepspeed_config.json`.
- Model or optimizer state genuinely does not fit in GPU memory even with full parameter sharding.

## Stage / Strategy Mapping

| DeepSpeed ZeRO Stage | FSDP2 equivalent (`fully_shard`) | What is Sharded |
|---------------------|-----------------------|-----------------|
| ZeRO-1 | No native `fully_shard` mode — use DDP + `ZeroRedundancyOptimizer` (or a distributed-optimizer wrapper) for the same semantics | Optimizer state |
| ZeRO-2 | `reshard_after_forward=False` | Optimizer state + gradients |
| ZeRO-3 | `reshard_after_forward=True` (default) | Optimizer state + gradients + parameters |
| ZeRO-Infinity | No FSDP equivalent | ZeRO-3 + NVMe offload |

(FSDP1 names `NO_SHARD` / `SHARD_GRAD_OP` / `FULL_SHARD` map to the same semantics but are deprecated since PyTorch 2.11.)

## Memory Model Comparison

**Units: bytes per parameter.** A dimensionless multiple of "M" cannot answer an OOM question — convert to bytes first. Assumptions below: `P` = parameter count, `N` = GPUs (shard degree), bf16 mixed precision with AdamW, which is the recipe the parent skill recommends. Activations are excluded — they are a separate, often larger term (see the Megatron activation formula in SKILL.md).

Per-parameter baseline, unsharded:

```text
bf16 params        2 B
bf16 gradients     2 B
fp32 master copy   4 B   <- mandatory under bf16 mixed precision; do not omit
fp32 Adam m        4 B
fp32 Adam v        4 B
                  ----
                  16 B/param
```

| Strategy | Params | Gradients | Optimizer state (master + m + v) | Bytes/param on each GPU |
|----------|--------|-----------|----------------------------------|-------------------------|
| DDP | 2 | 2 | 12 | 16 |
| ZeRO-1 | 2 | 2 | 12/N | `4 + 12/N` |
| ZeRO-2 / FSDP2 (`reshard_after_forward=False`) | 2 | 2/N | 12/N | `2 + 14/N` |
| ZeRO-3 / FSDP2 (`reshard_after_forward=True`) | 2/N | 2/N | 12/N | `16/N` |

Multiply by `P` for total state. **Worked case — 7B, bf16 + AdamW:** 7e9 × 16 B ≈ **112 GB** of model + optimizer state before a single activation, so it does not fit on one 80 GB H100. At N=8 with ZeRO-3 that is 112/8 = **14 GB per GPU**, leaving roughly 65 GB for activations, fragmentation, and the allocator's slack.

Note that the reductions are not constants: ZeRO-1 and ZeRO-2 approach 4× and 8× only as N→∞ (at N=8 they are 2.9× and 4.3×), while ZeRO-3 is linear in N with no ceiling.

**Legacy fp32 training** (fp32 params 4 + fp32 grads 4 + 8 optimizer = 16 B/param) happens to land on the same total, but the per-row split differs — the ZeRO-3 advantage is identical, the ZeRO-1/2 rows are not. Do not reuse the bf16 rows for an fp32 run.

The 12 B optimizer term assumes Adam/AdamW. Muon carries less state per matmul parameter, so that column shrinks under a Muon/AdamW hybrid — the sharding mechanics are unchanged.

## Configuration Quick-Start

### FSDP2 (PyTorch native, >=2.11)

FSDP1 (`FullyShardedDataParallel` + `ShardingStrategy`) is deprecated as of PyTorch 2.11. Use `fully_shard`, which shards each parameter as a DTensor and composes with TP/PP/CP via DeviceMesh:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.layers:                 # shard each transformer block
    fully_shard(block, mp_policy=mp)        # reshard_after_forward=True ≈ ZeRO-3
fully_shard(model, mp_policy=mp)            # shard the root module last
```

Set `reshard_after_forward=False` on a block for ZeRO-2-like behavior (keep params gathered after forward, fewer all-gathers, more memory). The deprecated FSDP1 equivalent was `ShardingStrategy.FULL_SHARD` / `SHARD_GRAD_OP`.

### DeepSpeed ZeRO-3 config excerpt

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "reduce_bucket_size": "auto"
  },
  "bf16": {"enabled": true}
}
```

## Known Pitfalls

- Do not mix FSDP and DeepSpeed in the same training run. They conflict at the distributed primitive level.
- FSDP2 full sharding (`reshard_after_forward=True`, ZeRO-3 equivalent) adds all-gather communication before each forward pass. This can become a bottleneck with high latency inter-node links — profile `dist.all_gather` overhead before scaling beyond one node.
- ZeRO-3 with Hugging Face `generate()` requires special handling; consult DeepSpeed docs on ZeRO-3 inference mode.
- FSDP checkpoint format differs from standard `state_dict` — use `FullStateDictConfig` / `StateDictType` for saving in a format compatible with non-FSDP loading.
- DeepSpeed ZeRO-Infinity NVMe offload is only beneficial if your NVMe bandwidth exceeds ~3 GB/s per GPU; otherwise it is a bottleneck.

## Canonical Sources

- PyTorch FSDP2 Tutorial: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- DeepSpeed ZeRO docs: https://www.deepspeed.ai/docs/config-json/
- ZeRO paper (Rajbhandari et al. 2019): https://arxiv.org/abs/1910.02054
- FSDP2 design + docs in torchtitan: https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
- torchtitan paper (FSDP2 + CP + compile in production): https://arxiv.org/abs/2410.06511
