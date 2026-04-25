# OpenEnv Hackathon — Judging Criteria (extracted from official PDF)

Source: `OpenEnv.pdf`, slides "What Judges Look For" through "Final Note."

## Weights

| Criterion | Weight | What it means |
|---|---|---|
| Environment Innovation | **40%** | Is the environment novel, creative, or genuinely challenging? Does it meaningfully test agent behavior in a way that hasn't been done before? |
| Storytelling & Presentation | **30%** | Can you clearly explain the problem, the environment, and what the agent learned? Is the demo engaging and easy to follow for a non-technical audience? |
| Showing Improvement in Rewards | **20%** | Is there observable evidence of training progress? Reward curves, before/after behavior, comparison against a baseline — anything that proves the agent learned something. |
| Reward & Training Pipeline | **10%** | Is the reward logic coherent? Does the pipeline produce meaningful improvement in the trained agent's behavior? |

## Minimum submission requirements (non-negotiable)

1. Use OpenEnv (latest release). Build on top of the framework.
2. **A working training script using Unsloth or Hugging Face TRL**, ideally as a Colab notebook so judges can re-run it.
3. **Loss and reward plots from a real run.**
4. A short writeup: mini-blog on HF, or <2 min YouTube video, or slide deck.
5. Push environment to a Hugging Face Space.
6. README that motivates the problem, explains the env, shows results, and links all materials.
7. Do not include big video files in the HF Hub submission — link out.

## What makes a submission stand out

- **Pick an ambitious, original problem.** Judges have seen chess/snake/grid-world clones.
- **Reward signal that actually teaches:** rich and informative (not 0/1 terminal); composable rubrics; hard to game.
- **Real training, end-to-end:**
  - Training loop connects to the **environment** (not a static dataset).
  - Train long enough that curves mean something.
  - Compare trained agent vs random/untrained baseline.
  - Plots and numbers in README and writeup.
- **Readable plots:** labeled axes with units, committed as PNG/JPG (not just in a Colab cell), embedded in README with one-line captions, ablations on same axes for comparison.
- **Tell a story, not an API doc.** README should answer in 3-5 minutes:
  1. Problem — what capability gap?
  2. Environment — what does the agent see/do/get rewarded for?
  3. Results — what changed after training?
  4. Why does it matter — who would care?

## Engineering table-stakes

- Use OpenEnv `Environment` / `MCPEnvironment` base classes properly.
- Respect client/server separation (clients should never import server internals).
- Follow Gym-style API (`reset`, `step`, `state`).
- Valid `openenv.yaml` manifest.
- Don't use reserved tool names (`reset`, `step`, `state`, `close`) for MCP tools.

## Algorithm choice — explicit guidance

The criteria mandate **TRL or Unsloth**, not any specific algorithm. **SFT, DPO, GRPO, RFT, RewardTrainer — all qualify.** The bar is "the agent learned something, you can show it." Choose the algorithm most likely to produce a clean improvement curve given your env's reward shape and your model's capability.

For envs with structured action spaces and small base models where format-failure dominates baseline rollouts: **SFT-on-oracle warm-start, then optional RL polish**, is the standard recipe. This satisfies "training loop connects to environment" because the oracle runs in the env to generate trajectories, and any subsequent RL stage runs live rollouts.

## Submission deadline reminder

- Day-2 5-hour reminder, then 2-hour reminder, then submission deadline.
- Top-100 finalists announced May 1.
- Winners livestream May 8.
- **Changes/commits after submission deadline are not considered.**

## HF compute notes (from PDF, "How to Access Infrastructure")

- HF credit: $30/person via `huggingface.co/coupons/claim/hf-openenv-community`.
- HF Jobs: pay-as-you-go GPU (CPU → H100 → TPU). Submit via `hf` CLI, `huggingface_hub` Python client, or Jobs HTTP API.
- Recommended GPU: T4 (small/medium) is fine for a 3B with LoRA. Larger only if needed.
- Train + run env on HF — no Colab requirement except as the judges' reproducible artifact.
