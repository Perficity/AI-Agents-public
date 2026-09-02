# Post-Training Methods and the Pipeline

How the post-training stage maps a **reward signal** to an **algorithm**. This file owns the
*selection logic and pipeline*; per-algorithm operational depth (hyperparameters, loss forms,
implementation notes) lives in
[ai-llm/references/post-training.md](../../ai-llm/references/post-training.md) — link there
rather than duplicating it.

## Table of Contents

- [The Pipeline, Stage by Stage](#the-pipeline-stage-by-stage)
- [The Two Orthogonal Axes](#the-two-orthogonal-axes)
- [Reward Signal -> Algorithm Map](#reward-signal---algorithm-map)
- [Method Capsules](#method-capsules)
- [GRPO Variants and Known Biases](#grpo-variants-and-known-biases)
- [Rejection Sampling: the Cheap Rung](#rejection-sampling-the-cheap-rung)
- [RLOO: the Simpler REINFORCE Alternative](#rloo-the-simpler-reinforce-alternative)
- [On-Policy Distillation: the On-Policy/Dense Cell](#on-policy-distillation-the-on-policydense-cell)
- [Agentic and Multi-Turn RL](#agentic-and-multi-turn-rl)
- [Rubrics as Rewards: the Fourth Reward Source](#rubrics-as-rewards-the-fourth-reward-source)
- [Beta Means Different Things: Published Anchors](#beta-means-different-things-published-anchors)
- [Routing to Depth](#routing-to-depth)

## The Pipeline, Stage by Stage

Post-training is a *sequence*; each stage is reached only when the previous one is exhausted
and a measurable gap remains.

1. **SFT (supervised fine-tuning / instruction tuning).** Teach the format and base behavior
   from labeled demonstrations. Cross-entropy on (prompt, ideal-response) pairs. This is the
   floor — most "alignment" gaps that a few hundred good demonstrations can close should be
   closed here, not with RL. (Depth: [ai-llm](../../ai-llm/SKILL.md).)
2. **Preference optimization.** Close the gap SFT can't express in demonstrations — tone,
   helpfulness, harmlessness, "which of two answers is better." Either *offline* (DPO/DAAs on a
   fixed preference set) or *online* (reward model + PPO/GRPO).
3. **Reasoning RL (RLVR).** When the remaining gap is multi-step correctness on tasks with a
   *verifiable* answer (math, code, unit tests). Usually GRPO over a deterministic checker.
4. **Continuous evaluation** against over-optimization runs alongside stages 2–3, not after.

## The Two Orthogonal Axes

Inside stages 2–3, independent choices determine the method:

| Axis | Option A | Option B |
|---|---|---|
| **Where the data comes from** | **Offline** — fixed preference dataset (DPO/DAAs). Simple, stable, no sampling loop, no reward model. | **Online** — sample from the current policy and score live (PPO/GRPO). Higher ceiling, more compute and moving parts. |
| **What produces the reward** | **Human** preferences → reward model · **AI** preferences → RLAIF/Constitutional AI · **rubric** grader → rubrics-as-rewards | **Verifiable** checker (compiler/tests/math) → RLVR (no reward model) |
| **Is a frozen reference model in the loop** | **Reference-based** — DPO, KTO. A frozen copy of the SFT model anchors the implicit KL trust region. Costs a second model resident in memory. | **Reference-free** — ORPO, SimPO. No frozen copy: cheaper and simpler, but the drift bound goes with it. |

Default: **start offline (DPO); promote to online (PPO/GRPO) only when offline plateaus** or
you need a reward model's generalization to unseen prompts.

### Reference-based vs reference-free is a real tradeoff, not an adjective

The reference model is not a stylistic detail of DPO-family methods. It does three things:
holds a second frozen model in memory (the concrete cost ORPO and SimPO exist to remove);
defines the implicit KL anchor that bounds how far the policy may drift from trusted SFT
behavior; and, in SimPO's case, is replaced by a *length-normalized* reward that targets the
same length bias GRPO carries. Choosing reference-free is choosing to give up the drift bound —
so pair it with a capability regression suite (see
[over-optimization-and-eval.md](over-optimization-and-eval.md)), because nothing else in the
objective will tell you the policy wandered.

### Sampling source × reward density

A third framing, which the on-policy-distillation literature made explicit, sorts the same
methods by *where the trained-on tokens come from* and *how dense the learning signal is*:

| | Dense signal (per token) | Sparse signal (per sequence/trajectory) |
|---|---|---|
| **Off-policy** (teacher's or dataset's own text) | **SFT** | rejection sampling / best-of-N |
| **On-policy** (the model's own rollouts) | **on-policy distillation** | **RL** (GRPO/PPO/RLVR) |

The bottom-left cell is the one this skill used to have empty. TRL's paper index carries the
same 2×2 and labels on-policy distillation "on-policy / dense" (verified 2026-08-31).

## Reward Signal -> Algorithm Map

| The reward signal you can actually produce | Algorithm | Why |
|---|---|---|
| Labeled demonstrations (no preference yet) | **SFT** | Not RL; teach behavior directly |
| Pairwise human preferences, want simplicity | **DPO** | Closed-form, no reward model or sampling loop |
| Binary good/bad or unpaired signal | **KTO** | Handles non-paired feedback |
| Want to merge SFT + preference in one stage | **ORPO** | Single-stage, reference-model-free |
| A stronger teacher model exists; student is small | **On-policy distillation** | Teacher scores the student's *own* rollouts token-by-token — dense signal, on-policy |
| Reward model + online RL (historical RLHF default) | **PPO** | Clipped policy gradient + value model; `trl.experimental` as of 2026-08 |
| Reward model + online RL, 2026 default | **GRPO / RLOO family** | Critic-free group baseline; a reward model and a critic-free method are orthogonal choices |
| Many samples scorable per prompt; drop the critic | **GRPO** | Group-relative advantage; removes the value model's weights and optimizer state |
| Non-verifiable task, no real checker | **Rubrics as rewards** | A structured rubric grades the response; the fourth reward source (see below) |
| Multi-turn / tool-using agent in an environment | **Agentic RL** (GRPO-family over trajectories) | Reward attaches to a trajectory, not a single completion |
| Verifiable checker available (math/code/tests) | **RLVR** (often via GRPO) | Reward = correctness; no reward model to hack |
| RL on a large **MoE** model; GRPO won't converge | **GSPO** | Sequence-level ratio is robust to expert-routing volatility |
| Want critic-free RL simpler than GRPO's std-normalized advantage | **RLOO** | Leave-one-out group mean as baseline; no learned value model, no GRPO std bias |
| Human labels are the bottleneck | **RLAIF / Constitutional AI** | Model + written constitution generate the signal |
| Want a quick lift, no RL loop | **Rejection sampling** | best-of-N → SFT on the winners |

## Method Capsules

One-paragraph orientation each; full detail in
[ai-llm/references/post-training.md](../../ai-llm/references/post-training.md).

- **PPO** — the original RLHF workhorse (InstructGPT lineage): a learned reward model scores
  samples; a clipped policy-gradient update with a value/critic model and a KL penalty to the
  SFT reference keeps the policy from drifting. Four models in memory. **Status matters**: TRL's
  paper index lists PPO's trainer as `experimental.ppo.PPOTrainer` while `GRPOTrainer` and
  `RLOOTrainer` are first-class (verified 2026-08-31). Treat PPO as the reference algorithm you
  must understand — the clipped surrogate is the trust region the whole family inherits — not as
  the 2026 production default. **A reward model does not imply PPO**: reward source and
  critic-free-vs-actor-critic are orthogonal, and with a learned RM the current default online
  method is a group-baseline one (GRPO/RLOO).
- **On-policy distillation (OPD)** — see the dedicated section below.
- **DPO** — reframes preference learning as a classification loss directly on the policy,
  eliminating the reward model and the RL loop. The default first reach for pairwise
  preferences. **DAAs** (Direct Alignment Algorithms) generalize it: **KTO** (unpaired),
  **ORPO** (single-stage, no reference model), **SimPO** (length-normalized, reference-free).
- **GRPO** — drops PPO's value model by estimating advantage from the *group* of samples drawn
  per prompt (their relative rewards). This removes one of four models — its weights *and* its
  optimizer state — from the training footprint entirely; DeepSeekMath's abstract frames the
  benefit qualitatively ("optimizing the memory usage of PPO") and gives no percentage, so do
  not quote one. The memory lever behind DeepSeek-R1-style training;
  pairs naturally with verifiable rewards. Carries known biases — see *GRPO Variants and Known
  Biases* below.
- **GSPO** (Group Sequence Policy Optimization, Qwen 2025) — GRPO computes the importance ratio
  at the *token* level; GSPO computes it at the *sequence* level (sequence-likelihood ratio +
  sequence-level clipping). The payoff is stability: token-level ratios interact badly with
  **MoE expert-routing volatility** (≈10% of activated experts change per update), which can
  stall GRPO convergence and forces hacks like Routing Replay. GSPO is robust to that and
  trained Qwen3-235B-A22B. Reach for it when doing **RL on a large MoE** model.
- **RLVR** — RL with Verifiable Rewards: the reward is a deterministic checker (does the code
  pass tests? is the math answer correct?), so there is no reward model to over-optimize. The
  standard 2026 reasoning recipe; cheaper and less hackable where a checker exists.
- **RLAIF / Constitutional AI** — replace human preference labels with a model judging against
  a written constitution; scales the preference signal when human labeling is the bottleneck.

## GRPO Variants and Known Biases

GRPO is the dominant reasoning-RL method, but its vanilla objective carries documented biases.
Know them before scaling a run — each has a named workaround.

| Bias / failure mode | What it does | Workaround |
|---|---|---|
| **Length-normalization bias** | Per-response length normalization attenuates gradients on longer outputs → the policy inflates response length to game it | **Dr. GRPO** removes the per-response length normalization; **DAPO** normalizes by total token count instead |
| **Advantage-std bias** | Dividing the group advantage by its std over-weights easy/hard prompts (low variance) → miscalibrated gradients across difficulty | **Dr. GRPO** removes the std normalization in the advantage |
| **MoE routing volatility** | Token-level importance ratios are unstable when expert routing shifts per update → GRPO may not converge on MoE | **GSPO** (sequence-level ratio); see the capsule above |
| **Entropy collapse / exploration loss** | The policy narrows too fast and stops exploring | **DAPO**'s decoupled-clip (higher upper clip) + dynamic sampling (drop all-correct/all-wrong groups) |

Detecting these in a live run — including how to tell entropy collapse from healthy exploration,
and why all-correct groups are as gradient-dead as all-wrong ones — is in
[grpo-run-diagnostics.md](grpo-run-diagnostics.md).

**Dr. GRPO** = "Done Right" GRPO: strips the three bias sources (per-response length norm, advantage
std norm, and the KL term). **DAPO** (open large-scale RL recipe) keeps GRPO's group structure but
swaps in token-level loss, decoupled clipping, dynamic sampling, and drops the KL penalty. Pick the
fix by the symptom; don't stack all of them blindly. GRPO is now a **family, not one fixed
objective** — as of mid-2026 DAPO is TRL's default `GRPOTrainer` loss type, and the shipped
`loss_type` roster on `main` is `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo`,
`vespo` (verified 2026-08-31 against `trl/trainer/grpo_config.py`). Treat any single fix as
provisional and re-check TRL's current default and roster before quoting one as *the* answer —
this list turned over between 2026-07 and 2026-08.

## Rejection Sampling: the Cheap Rung

Before any RL loop, **best-of-N rejection sampling** often captures much of the gain: generate
N candidates per prompt, score them (reward model, verifier, or LLM-judge), keep the best, and
SFT on those winners. No policy-gradient machinery, no KL tuning, easy to reason about. Use it
as the first rung above SFT and as a baseline any heavier RL method must beat.

## RLOO: the Simpler REINFORCE Alternative

**RLOO** (REINFORCE Leave-One-Out) is a lighter-weight critic-free alternative to GRPO/PPO: it
uses the mean reward of the *other* samples in a group as the baseline instead of a learned value
model or GRPO's group-std normalization. TRL ships an `RLOOTrainer` alongside `GRPOTrainer` as of
2026 — reach for it when GRPO's std-normalization biases are the concern and you want a simpler
critic-free baseline without adopting the full Dr. GRPO/DAPO fix set. TRL's GRPO/RLOO trainers
also gained **environment-owned rewards** in 2026 (e.g. Harbor/OpenEnv integration), letting a
sandboxed task suite compute the reward directly instead of a hand-rolled scoring function —
useful when the verifiable checker is itself a multi-step environment (agentic/tool-use tasks),
not a single-shot grader.

## On-Policy Distillation: the On-Policy/Dense Cell

If a **stronger teacher model already exists** and the student is small, try this *before*
reaching for GRPO. The mechanism: the student generates the rollouts, and the teacher scores
the student's *own* tokens — the student is optimized to minimize the KL divergence between its
token distributions and the teacher's along its own trajectories. That is what puts it in the
on-policy/dense cell: on-policy like RL (the model trains on what it actually produces, so there
is no train/inference distribution mismatch), dense like SFT (a signal on every token, not one
scalar per sequence).

TRL's paper index (verified 2026-08-31) reports that on-policy distillation "has been shown to
outperform SFT, GRPO and can be used to restore generalization capabilities lost during SFT,"
and that it "is more compute efficient and is less prone to overfitting when trained with
limited data." Circulating magnitudes (a claimed ~30x cost reduction vs off-policy distillation;
fewer gradient steps than RL) trace to a vendor blog and secondary coverage — **cite the
direction, not the numbers**, until read in a primary source.

Trainers, per TRL's paper index (names verified 2026-08-31; TRL moves fast, re-check before
writing code):

- `DistillationTrainer` / `DistillationConfig` — first-class, the always-on-policy case.
- `experimental.gkd.GKDTrainer` — Generalized Knowledge Distillation, exposes on/off-policy
  mixing via `lmbda`.
- `experimental.async_distillation.AsyncDistillationTrainer` — MOPD multi-teacher fusion of
  per-domain experts into one student.
- `experimental.iw_opd.IWOPDTrainer` — importance-weighted OPD, addressing position bias.

Where it does *not* apply: no teacher stronger than the student exists, or the capability you
want is not in any teacher — that is the case RL is for. Note also that DeepSeek-R1's own paper
found distillation more effective than pure RL for small dense models.

## Agentic and Multi-Turn RL

When the policy acts in an **environment** over several turns — calling tools, reading results,
retrying — the single-completion framing above stops fitting. What changes:

- **The unit of reward.** Reward attaches to a *trajectory* (did the task get done?), not a
  completion. **Trajectory-level** reward is easy to define and gives one sparse scalar for many
  turns; **turn-level** reward is denser but requires deciding what a good intermediate turn is,
  which is the same proxy-design problem as a PRM.
- **Credit assignment across turns.** With trajectory-level reward, every turn in a successful
  trajectory is reinforced, including the wasteful ones. Group-relative methods help here for the
  same reason they help elsewhere — the baseline is other trajectories on the same task — but the
  within-trajectory attribution problem is not solved by the advantage estimator.
- **Rollout infrastructure becomes the bottleneck.** Multi-turn rollouts are long, ragged, and
  latency-dominated. Production setups run a separate inference engine (vLLM) for generation,
  **colocated** with the trainer (shares GPUs, simple, serialized) or **disaggregated** (separate
  rollout pool, better utilization, asynchronous, and a train/rollout policy mismatch to correct
  for). TRL exposes this correction as `vllm_importance_sampling_correction` /
  `vllm_importance_sampling_mode`; **verl** is the common open backbone for large agentic RL runs.
  Depth: [ai-distributed-training](../../ai-distributed-training/SKILL.md).
- **Environment-owned rewards.** TRL's GRPO/RLOO trainers gained these in 2026 (Harbor/OpenEnv),
  letting the task suite compute reward directly — see the RLOO section above.

Survey for design-choice tradeoffs: Wang and Ammanabrolu, *A Practitioner's Guide to Multi-turn
Agentic Reinforcement Learning* (arXiv 2510.01132, title and authors verified 2026-08-31); it
states existing frameworks and definitions are "fragmented" with no systematic analysis of which
design choices matter across tasks. Read it before assuming a single-turn recipe transfers.

## Rubrics as Rewards: the Fourth Reward Source

The reward-source axis is usually taught as three options — human preferences, AI preferences,
verifiable checker — which leaves "the task is not verifiable and I don't trust a bare
LLM-judge" as a dead end. **Rubrics as rewards (RaR)** is the fourth: instead of a scalar RM or
a binary checker, a *structured, multi-criteria rubric* grades the response, and the aggregated
rubric score is the reward.

Gunjal et al., *Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains* (arXiv
2507.17746, title and authors verified 2026-08-31) states the motivation directly: RLVR "has
proven effective for complex reasoning tasks with clear correctness signals such as math and
coding," but "extending it to real-world reasoning tasks is challenging, as evaluation depends
on nuanced, multi-criteria judgments rather than binary correctness."

Why this is not just LLM-as-judge with extra steps: the rubric is written *before* training and
is inspectable, so the proxy is legible and auditable — you can read what the policy is being
paid for. It is still a learned/model-mediated proxy, so it is hackable in the way a checker is
not, and the over-optimization controls apply in full: it sits on the RM side of the
KL-by-reward-source rule below, not the RLVR side.

Where it fits the decision: reach for rubrics when the task is real work with no mechanical
checker (writing, analysis, advice, multi-criteria judgment) — instead of either forcing a fake
verifier or falling back to opaque pairwise preferences.

## Beta Means Different Things: Published Anchors

`beta` is overloaded across methods and the overload spans orders of magnitude, which makes
"tune beta" useless advice without an anchor. Values below are from TRL's paper index, which
carries per-paper configs with section references (fetched 2026-08-31):

| Method | What `beta` is | Anchor | Effect of raising it |
|---|---|---|---|
| **DPO** | Implicit-reward temperature on the reference model | `0.1` (paper, Appendix B; also TRL's default) | Tighter to the reference; weaker preference fit |
| **ORPO** | λ, the odds-ratio loss weight | `0.1` (Table 7, Mistral-ORPO-β) | More preference weight vs the SFT term |
| **GRPO (RLVR)** | KL coefficient to the reference policy | `0.0` = TRL's shipped default; DeepSeek-R1 used `0.001` | Caps achievable reasoning gain |
| **DAPO / GSPO** | KL coefficient | `0.0` — both drop it deliberately | n/a; they are the beta=0 case |
| **RLOO** | KL coefficient | `0.03` (paper, section C) | Tighter trust region |
| **Distillation (GKD/OPD)** | Generalized-JSD interpolation, **not** a KL penalty | `0.5` = JSD; `1.0` = reverse KL | Different object entirely |

Two lessons: DPO-family `beta` and RL-family `beta` differ by roughly two orders of magnitude
and are not the same quantity; and in distillation `beta` is not a regularizer at all. Never
carry a value across method families.

## Routing to Depth

- Per-algorithm hyperparameters, loss equations, decision tree, comparison table ->
  [ai-llm/references/post-training.md](../../ai-llm/references/post-training.md)
- Code-level TRL/verl implementation -> `huggingface-skills:` plugin (TRL); check TRL's current
  docs for the default `GRPOTrainer` loss type and trainer roster before quoting specifics —
  these move fast. **GSPO is neither a separate trainer nor a `loss_type`**: TRL's paper index
  states "GSPO is a GRPO variant that computes importance sampling weights at the sequence level
  instead of per-token" and configures it with `importance_sampling_level="sequence"`, a
  parameter orthogonal to `loss_type` (allowed values `"token"` or `"sequence"`). Writing
  `loss_type="gspo"` is a validation error. A `GSPO-token` variant exists under
  `trl.experimental` (verified 2026-08-31 against `trl/trainer/grpo_config.py` on `main` and the
  TRL paper index)
- Scaling the RL run (FSDP, vLLM rollouts, async RL) ->
  [ai-distributed-training](../../ai-distributed-training/SKILL.md)
- Reward modeling and preference data -> [reward-and-data.md](reward-and-data.md)
- Keeping it from over-optimizing -> [over-optimization-and-eval.md](over-optimization-and-eval.md)
