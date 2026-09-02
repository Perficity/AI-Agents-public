#!/usr/bin/env python3
"""Executable versions of this skill's three central loop claims.

1. Step-0 loss of a freshly initialized LM is approximately ln(vocab_size).
2. Gradients accumulated over N micro-batches (each loss divided by N) equal the
   gradient of one single large batch.
3. Causal-masked attention rows sum to 1 and never attend to future positions.

Run: python3 check_loop.py   (requires torch; CPU is fine)
"""
import math
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("check_loop.py requires PyTorch. Install it (pip install torch) and re-run.")


class TinyLM(nn.Module):
    """Smallest model that exercises embedding -> transform -> tied LM head."""

    def __init__(self, vocab_size, n_embd=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.emb.weight  # weight tying
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx, targets):
        logits = self.head(self.proj(self.emb(idx)))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def check_baseline_loss(vocab_size=50257, tol=0.10):
    torch.manual_seed(0)
    model = TinyLM(vocab_size)
    x = torch.randint(0, vocab_size, (4, 32))
    y = torch.randint(0, vocab_size, (4, 32))
    with torch.no_grad():
        _, loss = model(x, y)
    expected = math.log(vocab_size)
    rel = abs(loss.item() - expected) / expected
    assert rel < tol, f"step-0 loss {loss.item():.4f} vs ln(V)={expected:.4f} (rel {rel:.3%})"
    return f"step-0 loss {loss.item():.4f} ~= ln({vocab_size}) = {expected:.4f}"


def check_grad_accumulation(vocab_size=257, n_micro=4, micro_b=8, seq=16):
    torch.manual_seed(0)
    model = TinyLM(vocab_size)
    x = torch.randint(0, vocab_size, (n_micro * micro_b, seq))
    y = torch.randint(0, vocab_size, (n_micro * micro_b, seq))

    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    full = [p.grad.detach().clone() for p in model.parameters()]

    model.zero_grad(set_to_none=True)
    for i in range(n_micro):
        xb = x[i * micro_b:(i + 1) * micro_b]
        yb = y[i * micro_b:(i + 1) * micro_b]
        _, l = model(xb, yb)
        (l / n_micro).backward()
    accum = [p.grad.detach().clone() for p in model.parameters()]

    for a, b in zip(full, accum):
        assert torch.allclose(a, b, atol=1e-6), "accumulated grad != single large-batch grad"
    return f"{n_micro} micro-batches of {micro_b} == one batch of {n_micro * micro_b} (allclose)"


def check_causal_attention(T=12, d=16):
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, T, d), torch.randn(1, 1, T, d)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    att = att.masked_fill(mask == 0, float("-inf")).softmax(dim=-1)
    rows = att.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-6), "attention rows do not sum to 1"
    upper = att.squeeze()[torch.triu(torch.ones(T, T), diagonal=1) == 1]
    assert torch.all(upper == 0), "future positions received nonzero attention"
    return f"causal attention rows sum to 1; strict upper triangle exactly 0 (T={T})"


def main():
    failures = 0
    for name, fn in (
        ("baseline loss ~ ln(V)", check_baseline_loss),
        ("grad accumulation == large batch", check_grad_accumulation),
        ("causal attention rows sum to 1", check_causal_attention),
    ):
        try:
            print(f"PASS  {name}: {fn()}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
