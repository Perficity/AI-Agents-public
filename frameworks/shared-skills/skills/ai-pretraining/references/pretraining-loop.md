# Pretraining Loop

Reference for the GPT pretraining training loop, covering mixed precision, gradient accumulation, learning rate scheduling, and checkpointing.

## Table of Contents

- [Canonical Sources](#canonical-sources)
- [Loop Anatomy](#loop-anatomy)
- [Mixed Precision](#mixed-precision)
- [Gradient Accumulation](#gradient-accumulation)
- [Cosine LR with Warmup](#cosine-lr-with-warmup)
- [WSD / Trapezoidal Schedule](#wsd--trapezoidal-schedule)
- [Throughput: TF32, Vocab Padding, MFU](#throughput-tf32-vocab-padding-mfu)
- [Gradient Clipping](#gradient-clipping)
- [Checkpointing](#checkpointing)
- [Baseline Loss Check](#baseline-loss-check)
- [Common Mistakes](#common-mistakes)

## Canonical Sources

- Karpathy "Let's reproduce GPT-2 (124M)" — [youtube.com/watch?v=l8pRSuU81PU](https://www.youtube.com/watch?v=l8pRSuU81PU)
- nanoGPT `train.py` — [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/train.py)
- PyTorch AMP docs — [pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
- GPT-3 paper (Brown et al. 2020) for hyperparameter reference — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

## Loop Anatomy

```python
model = GPT(config).to(device)
optimizer = model.configure_optimizers(weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95))

for step in range(max_steps):
    optimizer.zero_grad(set_to_none=True)  # already the default in current PyTorch

    # Gradient accumulation micro-steps
    loss_accum = torch.zeros((), device=device)
    for micro_step in range(grad_accum_steps):
        x, y = get_batch('train')
        if ddp:
            # DDP resets this to True inside every forward() — re-assign each micro-step,
            # before the forward, so the guard covers forward AND backward.
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # normalize
        loss_accum += loss.detach()     # .detach(), not .item() — .item() forces a GPU sync
        loss.backward()  # accumulates into .grad

    if ddp:
        # loss_accum is per-rank; average it before logging or you print one rank's view.
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # LR schedule
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if step % eval_interval == 0:
        val_loss = estimate_loss(model, 'val')  # see below — no periodic eval means
        # a shard-repeat or overfit can hide behind a healthy-looking train curve
```

Held-out eval — the loop above is not complete without it (the checkpoint dict below stores
`val_loss`, which nothing else computes):

```python
@torch.no_grad()
def estimate_loss(model, split, eval_iters=20):
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        x, y = get_batch(split)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[k] = loss.detach()
    model.train()
    return losses.mean()
```

## Mixed Precision

Use `torch.autocast` with `bfloat16` on Ampere+ GPUs (A100, 3090, 4090):

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

- `bfloat16`: same exponent range as float32, lower mantissa precision. Stable without loss scaling.
- `float16`: narrower exponent range — requires `GradScaler` to prevent NaN/inf in gradients.
- Prefer `bfloat16` when hardware supports it; fall back to `float16` + `GradScaler` on older GPUs (V100, T4).
- The autocast context wraps the forward pass only; backward accumulates in float32.

## Gradient Accumulation

Purpose: simulate a large batch size across multiple small micro-batches to fit GPU memory.

```python
# Effective batch = batch_size * seq_len * grad_accum_steps * num_gpus
# GPT-3 used ~0.5M tokens per step
# nanoGPT target: total_batch_size = 524288 tokens
# Example (1 GPU): batch_size=16, seq_len=1024, grad_accum_steps=32 -> 16*1024*32 = 524288

assert total_batch_size % (batch_size * seq_len * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (batch_size * seq_len * ddp_world_size)
```

The divisor must include `ddp_world_size`: under DDP every rank contributes to a single optimizer step. Omitting it on 8 GPUs gives an effective batch 8× the intended one, with a peak LR tuned for the smaller batch — a silent scaling error. nanoGPT does the equivalent (`gradient_accumulation_steps //= ddp_world_size`) and carries the assert; without the assert the off-by-N goes unnoticed.

Critical: divide `loss` by `grad_accum_steps` inside the micro-batch loop. Failing to do this means each micro-batch contributes at full scale and the effective gradient is `grad_accum_steps` times too large.

Equal-token caveat: dividing by N gives a mean-of-means, which equals the true batch mean **only when every micro-batch contains the same number of loss-contributing tokens**. True for fixed `(B, T)` packed pretraining batches. With padding or variable-length sequences (an SFT loop, for example), accumulate the token count and normalize by the total, not by N.

### DDP gradient sync

Wrap the model with `torch.nn.parallel.DistributedDataParallel`, then suppress the all-reduce on every micro-step except the last — otherwise you pay `grad_accum_steps`× the communication for the same gradient.

`model.require_backward_grad_sync` is **one-shot**: DDP's reducer resets it to `True` inside every `forward()`. Setting it once before the micro-loop does nothing after the first iteration. Re-assign it per micro-step, keyed on the index, before that micro-step's forward (as in the loop above):

```python
model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
```

`model.no_sync()` is the supported public API and the equivalent; `require_backward_grad_sync` is the private-but-conventional nanoGPT shortcut. Either way the guard must enclose the **forward as well as the backward**:

```python
ctx = model.no_sync() if micro_step < grad_accum_steps - 1 else contextlib.nullcontext()
with ctx:
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)
    (loss / grad_accum_steps).backward()
```

Both failure modes here are silent at small scale: sync-every-step is a throughput regression with correct gradients; never-restoring-sync means each rank steps on its own local gradient and the run silently stops being data-parallel.

`loss_accum` is per-rank. All-reduce it (`op=ReduceOp.AVG`) before logging, or the printed loss is one rank's view.

## Cosine LR with Warmup

GPT-2/GPT-3 training schedule:

```python
def get_lr(it):
    # Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # After max_iters: minimum LR
    if it > max_iters:
        return min_lr
    # Cosine decay between warmup and max
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

Typical values for GPT-2 (124M) reproduction:

- `max_lr = 6e-4`, `min_lr = 6e-5` (10% of peak)
- `warmup_iters = 715` (~3.75% of 19073 total steps for FineWeb-Edu — 375M of 10B tokens)
- Adam `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.1`

AdamW weight decay: apply only to 2D+ tensors (weight matrices), not to biases or LayerNorm parameters. Configure two parameter groups:

```python
decay_params = [p for p in params if p.dim() >= 2]
nodecay_params = [p for p in params if p.dim() < 2]
```

The GPT-3-derived warmup is a conservative default, not a law — shorter warmups often work; treat 3.75% as the safe starting point, not the target.

## WSD / Trapezoidal Schedule

Cosine requires committing to `max_iters` up front. WSD (warmup–stable–decay, a.k.a. trapezoidal) does not: warm up, hold a constant plateau, then decay over the final ~10–20% of whatever budget you end up with.

```python
def get_lr_wsd(it, total_iters, warmup_iters, decay_frac=0.1):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    decay_start = total_iters - int(decay_frac * total_iters)
    if it < decay_start:
        return max_lr                                    # stable plateau
    r = (it - decay_start) / max(1, total_iters - decay_start)
    return min_lr + (1 - r) * (max_lr - min_lr)          # linear decay (1-sqrt also used)
```

Decision rule: **cosine** when the token budget is fixed and known — it remains the GPT-2/GPT-3 reproduction default and is what the hyperparameters above are tuned for. **WSD** when you may extend the run, want annealed intermediate checkpoints, or want to branch data-mixture experiments off one plateau. Consistent with [ai-scaling-laws](../../ai-scaling-laws/SKILL.md) ("cosine decay or trapezoidal schedule over D tokens") and with the Kimi K2 recipe recorded in [ai-distributed-training](../../ai-distributed-training/SKILL.md) (MuonClip + WSD).

## Throughput: TF32, Vocab Padding, MFU

Three cheap wins that a from-scratch loop usually leaves on the floor:

- **TF32 matmuls.** `torch.set_float32_matmul_precision('high')` lets fp32 matmuls run on TensorFloat-32 tensor cores on Ampere+. Karpathy's GPT-2 video treats it as the *first* optimization to apply; it is one line and needs no other change. Verify the speedup on your own hardware.
- **Pad the vocab to a multiple of 128.** GPT-2's `vocab_size = 50257` is a bad number for tensor-core tiling; setting `vocab_size = 50304` (50257 rounded up to a multiple of 128) adds unused rows that the model learns to never predict, and buys a few percent throughput for free. Measure on your setup.
- **MFU (model FLOP utilization)** is the throughput metric to track, not tokens/sec alone. Estimate training FLOPs as `6 * N * D` (N = non-embedding params, D = tokens; forward ≈ 2ND, backward ≈ 4ND), divide the per-step FLOPs by step wall-clock, then divide by the GPU's peak dense FLOP/s for your dtype. Without MFU you have no way to tell whether the loop is leaving 3× on the table — the usual outcome of a from-scratch reproduction. Also check you are not CPU-bound in `get_batch` before blaming the model.

## Gradient Clipping

```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clip global gradient norm to 1.0 before the optimizer step. This caps the update magnitude when the loss landscape has sharp curvature (common early in training and at LR peaks).

Log `norm` every step: a consistently high norm (>1.0) suggests the model is struggling; a sudden spike often indicates a bad batch.

## Checkpointing

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': asdict(config),   # plain dict — see note below
    'step': step,
    'data_pos': loader.position,  # shard index + offset, so the data stream resumes too
    'val_loss': val_loss,
}
torch.save(checkpoint, f'ckpt_{step:05d}.pt')

# Resume — weights_only=True prevents arbitrary code execution via pickle
ckpt = torch.load('ckpt_05000.pt', weights_only=True)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
step = ckpt['step']
loader.seek(ckpt['data_pos'])
```

- **Resume the data position, not just the weights.** Restoring `model` and `optimizer` but restarting the loader at shard 0 silently re-trains on data the model has already seen. Nothing errors; the loss curve looks fine. Store the shard index and within-shard offset in the checkpoint and seek on resume.
- **`weights_only=True` and `config`.** The flag is right and should stay, but it will *raise* on load if `config` is a dataclass or custom object — only plain types round-trip unless the class is registered via `torch.serialization.add_safe_globals`. Store the config as a plain dict.
- **`torch.compile` changes state_dict keys.** A compiled model's parameters are prefixed `_orig_mod.`; save from (or strip back to) the uncompiled module, or loading into an uncompiled model fails on every key.

Gradient checkpointing (activation checkpointing) — trades compute for memory by recomputing activations during backward instead of storing them:

```python
from torch.utils.checkpoint import checkpoint
# Wrap each block's forward in checkpoint(): the backward recomputes one extra forward,
# so ~30-40% compute overhead, and stored activations drop from every layer's
# intermediates to roughly one block's worth plus the saved block boundaries.
```

The classic O(√n) activation-memory result is the bound for *optimally placed* checkpoints, not for wrapping every block — don't quote it for per-block wrapping. Modern practice is **selective / op-level checkpointing** (`torch.utils.checkpoint` with an SAC policy, or `checkpoint_wrapper`): recompute the cheap ops, keep the expensive matmuls stored. It dominates all-or-nothing per-block wrapping. Multi-GPU specifics stay with [ai-distributed-training](../../ai-distributed-training/SKILL.md).

## Baseline Loss Check

Before training more than ~100 steps, verify the initial loss matches theory:

- For `vocab_size=50257` (GPT-2 tokenizer): expected initial loss ≈ `ln(50257) ≈ 10.82`
- For a character-level model with 65 chars: ≈ `ln(65) ≈ 4.17`

If step-0 loss is far from this baseline, likely causes:

- Weight initialization is wrong (check init scaling)
- LM head weights are not tied to embeddings
- Loss function is computing something unexpected (shape mismatch)

## Common Mistakes

- **Not zeroing gradients**: call `optimizer.zero_grad()` at the start of each outer step (not after `.step()`). `set_to_none=True` is faster and is already the default in current PyTorch — pass it explicitly only for clarity.
- **Setting `require_backward_grad_sync` once, outside the micro-loop**: DDP resets it on every forward, so it must be re-assigned per micro-step (or use `model.no_sync()`).
- **Resuming weights without resuming the data position**: silently re-trains on the same shard.
- **Dividing loss outside the micro-step loop**: the division by `grad_accum_steps` must happen inside the loop, per micro-batch.
- **Using default Adam betas**: GPT-2/3 used `betas=(0.9, 0.95)`, not PyTorch's default `(0.9, 0.999)`.
- **Skipping LR warmup**: the loss will spike at the start without warmup, especially with large LRs.
- **Checkpoint includes stale optimizer state**: when resuming, restore both `model` and `optimizer` state, and set the LR scheduler to the correct step.
- **`torch.compile` on PyTorch < 2.0**: `torch.compile` requires PyTorch 2.0+. On eligible hardware, it can give 2-3x throughput improvement via kernel fusion.
