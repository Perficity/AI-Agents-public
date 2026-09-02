# Over-Optimization and Evaluation

Preference RL optimizes a *proxy* (the reward model or a preference dataset) for what you
actually want. Optimize any proxy hard enough and it stops tracking the true objective — the
model games the reward while real quality degrades. This is the default failure mode of
post-training, not an edge case. This file covers detecting it and controlling it.

## Table of Contents

- [Why It Happens (Goodhart)](#why-it-happens-goodhart)
- [The Symptoms](#the-symptoms)
- [Controls](#controls)
- [Evaluating the True Objective](#evaluating-the-true-objective)
- [Pick the Serving Budget Before the Run](#pick-the-serving-budget-before-the-run)
- [Fail-Loud Checklist](#fail-loud-checklist)
- [Routing to Depth](#routing-to-depth)

## Why It Happens (Goodhart)

"When a measure becomes a target, it ceases to be a good measure." A reward model is a learned,
imperfect approximation of human preference. Early in training, raising the reward also raises
true quality. Past a point, the policy discovers regions of output space where the reward model
is *wrong* — high predicted reward, low actual quality — and exploits them. The reward curve
keeps rising; the model gets worse. With RLVR the proxy is tighter (a real checker), but it can
still be gamed if the checker is incomplete (e.g. tests that pass on degenerate solutions).

## The Symptoms

- **Length inflation / verbosity** — responses balloon because longer correlated with preferred
  in the data.
- **Sycophancy** — the model agrees with the user to win the preference signal.
- **Over-refusal** — safety training generalizes too far; the model refuses safe requests.
- **Reward-model exploitation** — outputs that score high on the RM but read as worse to humans.
- **Format/keyword gaming** — the policy learns surface features the reward correlated with.
- **Mode collapse / diversity loss** — outputs converge to a narrow high-reward template.

## Controls

- **KL regularization — but scoped by reward source.** Penalizing KL divergence between the
  policy and the reference (SFT) model keeps the policy near its trusted starting behavior and
  bounds how far it can chase the proxy. Whether that is *the* control depends on what produces
  the reward:
  - **Learned reward model (RM+PPO, rubric graders, DPO's implicit β).** KL is the primary
    trust-region control and the thing standing between you and a hacked proxy. Tune it; do not
    omit it. Too high → no learning; too low → reward hacking and drift.
  - **Verifiable checker (RLVR).** `beta=0` is the 2026 standard. There is no learned proxy to
    hack, and KL mainly caps the achievable reasoning gain. TRL's `GRPOConfig` ships
    `beta: float = 0.0` with the docstring "KL coefficient. If `0.0` (default), the reference
    model is not loaded, reducing memory usage and improving training speed" (verified
    2026-08-31); DAPO drops the KL penalty (§2.3) and GSPO sets it to zero. The trust region is
    carried instead by **PPO-style clipping** — plus DAPO's decoupled clipping and dynamic
    sampling. Reach for a nonzero β here only on evidence: observed drift off SFT behavior, or a
    capability regression outside the RLVR task distribution.
  - **The checker is still a proxy.** A verifiable reward is a *much tighter* proxy, not the
    true objective — tests that pass on degenerate solutions are still hackable. β=0 is a
    statement about trust regions, not about invulnerability.
  Anchor values per method (they differ by orders of magnitude and mean different things) are in
  [methods-and-pipeline.md](methods-and-pipeline.md).
- **Early stopping on a held-out true-objective eval** — stop when *real* quality peaks, not
  when reward peaks (they diverge).
- **Reward-model ensembles / uncertainty** — penalize high-variance regions where the RM is
  unsure, shrinking the exploitable surface.
- **On-policy data refresh** — retrain the RM on the current policy's outputs so it doesn't go
  stale exactly where the policy is exploring.
- **Pretraining-gradient / SFT mixing** — mix in SFT or pretraining loss to counter catastrophic
  forgetting and distribution collapse.
- **Length normalization / explicit length penalties** — directly counter the length bias.

## Evaluating the True Objective

The reward curve is *not* the evaluation. Measure the thing you actually want:

- **Held-out human eval** (or a calibrated LLM-judge with known biases) on a fixed prompt set.
- **Verifiable benchmarks** for reasoning (math/code pass-rate) — these *are* the true objective
  for RLVR.
- **Safety / over-refusal pair**: track harmful-prompt refusal *and* benign-prompt acceptance
  together, so safety gains don't hide a usefulness regression.
- **Capability regression suite**: confirm post-training didn't degrade general capabilities the
  preference set didn't cover.

Calibrate any judge model and set thresholds via [ai-evals](../../ai-evals/SKILL.md).

## Pick the Serving Budget Before the Run

Test-time compute is not only a *route-away* from training ("raise the thinking budget instead").
It is an input to the training design, because what you train for is conditioned on the inference
budget you will actually serve at. Decide the target budget first; it constrains three things:

- **`max_completion_length`** — training a long-CoT policy you will then serve under a short
  budget wastes the training, and the served model is out of distribution relative to what it
  learned.
- **Length-bias controls.** How hard you fight GRPO's length inflation (Dr. GRPO, DAPO, explicit
  penalties) depends on whether long outputs are affordable at serving time or a cost you refuse.
- **How you read `avg_response_len`** in a live run. It is a training-budget metric with a direct
  serving consequence: rising length is only "the model reasoning more" if you will pay for that
  length in production. See [grpo-run-diagnostics.md](grpo-run-diagnostics.md).

## Fail-Loud Checklist

Post-training is exactly the setting where success and failure look alike on the training
dashboard. Surface uncertainty explicitly:

- "Reward improved" is **not** "the model improved" — report the held-out true-objective number.
- Name what the eval did *not* cover (capabilities outside the preference distribution).
- If over-refusal or length inflation rose, say so even if the headline metric improved.
- If the verifiable checker could be gamed by degenerate outputs, flag it.

## Routing to Depth

- Reading a **live** GRPO run's metrics (advantage mean/std, entropy, reward exhaustion,
  degenerate groups) -> [grpo-run-diagnostics.md](grpo-run-diagnostics.md)
- Reward model and preference data quality -> [reward-and-data.md](reward-and-data.md)
- Which algorithm and the KL-penalty mechanics -> [methods-and-pipeline.md](methods-and-pipeline.md)
- Eval methodology, judge calibration, thresholds -> [ai-evals](../../ai-evals/SKILL.md)
