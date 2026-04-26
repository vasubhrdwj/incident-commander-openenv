---
title: Incident Commander (OpenEnv)
emoji: 🚨
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Teaching a 3B model to be an on-call engineer

### *(and what it took to actually train it)*

*A submission for the OpenEnv Hackathon — what worked, what didn't, and why most of the work happened **before** training.*

— **Team VBxAG**

> **Quick links:**
> [HF Space (env)](https://huggingface.co/spaces/vasubhrdwj/incident-commander-openenv) ·
> [Live API](https://vasubhrdwj-incident-commander-openenv.hf.space) ·
> [Demo replay UI](https://vasubhrdwj-incident-commander-openenv.hf.space/replay) ·
> [Trained adapter (SFT)](https://huggingface.co/vasubhrdwj/incident-commander-sft-lora) ·
> [Trained adapter (RFT-on-SFT)](https://huggingface.co/vasubhrdwj/incident-commander-rft-on-sft-lora) ·
> [Reproducible Colab](training/train_colab.ipynb) ·
> **[Technical documentation → `blog.md`](blog.md)**

---

## Why we built this

When a payments outage hits at 3 AM, the fix isn't more code — it's judgment. Triage the alerts. Read the dashboards. Decide between `rollback`, `hold`, `restart`, `partial_rollback`. Communicate to thousands of customers without overcommitting. Write the post-mortem the next morning so the same thing doesn't happen again. The on-call **Incident Commander** role is one of the highest-leverage human jobs in modern SaaS — and one of the most thankless. Most engineers learn it by being thrown in. Companies pay enormous costs for the hours those decisions take and the mistakes they sometimes produce.

We've watched the LLM space chase coding evals, math benchmarks, and reasoning leaderboards. Operational judgment under uncertainty — the actual job description for site reliability, security response, and ops engineering — has gone almost entirely unmeasured. And yet models are already in the loop on real production systems: observability copilots that summarize alerts, runbook agents that suggest remediations, AI-assisted post-incident reviews. If we're going to trust them with operational decisions, someone needs to test whether they can actually hold the seat.

The idea for this env crystallized over a late-night Teams call before the hackathon kicked off. We'd been firefighting P1s in production and reading the Cloudflare and AWS post-mortems from earlier in the month. The shared observation was simple: the cognitive shape of the IC role — *observe → decide → commit → communicate → reflect*, all under a clock — looked exactly like a long-horizon RL environment that no one had built yet.

![Incident Commander web UI — alerts, dashboard, NPC chat, and live rubric meter](https://github.com/vasubhrdwj/incident-commander-openenv/blob/main/incident_commander/blog_assets/incident_commander_UI.png)

> The web UI at `/web` on a running env. Left rail: alerts and dashboard health. Center: action picker and observation stream. Right rail: NPC chat with the four specialists. Bottom: live rubric meter that fills as components are credited.

The Incident Commander OpenEnv environment puts an agent into a fintech outage with a 6-service graph, four specialist NPCs (SRE, Security, Comms, Eng Lead), and three different fault templates. The agent gets logs, metrics, traces, audit events, and external-status feeds — but only when it asks for them, and asking burns a step. The reward is a six-component rubric that scores containment, MTTR, root-cause attribution, mitigation correctness, comms SLA, and post-mortem quality. **No LLM judge anywhere.** Everything is programmatic.

What makes the env interesting (to us) is that the three tasks each test a different cognitive ability:

- **Easy** — a canary regression. *Reactive deduction:* read what's in front of you, roll back the bad deploy.
- **Medium** — a third-party outage with three seed-keyed variants. *Discriminative attribution:* tell apart things that look alike, because rolling back **our** service when the upstream provider is down makes things worse.
- **Hard** — a silent data corruption. *Inverted reasoning under no signal:* the dashboard is green, no alerts fire, and the only evidence is a tag in the audit log. The model has to know to query an unprompted observability surface and apply `partial_rollback` to a specific cohort, not a full rollback.

We wanted an env where doing the easy-task playbook on the hard task would **fail**, hard, even when the actions look reasonable. That's where most of the design time went.

## The structural anti-gaming demo

The single thing we're proudest of in this project is what the **[`/replay` UI](https://vasubhrdwj-incident-commander-openenv.hf.space/replay)** shows. Three deterministic episodes, animated step-by-step, with the rubric bars filling live. Same env, same six rubric components, same level of effort, completely different outcomes:

- **Episode 1 — Easy oracle:** 0.8716. Calibrates the rubric ceiling on the easy task.
- **Episode 2 — Hard oracle:** 0.8546. Hard task done **right** (`query_audit` → `partial_rollback orders` → `customer_email` cohort).
- **Episode 3 — Hard task, easy-task playbook:** 0.4792. **FAIL.** Same agent shape, same effort, different choices.

The Δ=0.375 between Episodes 2 and 3 lives entirely in three structural anti-gaming guards baked into the rubric:

1. `status_page` on a silent-corruption incident earns **0** comms — the customer cohort needs `customer_email`, not a public banner.
2. Full `rollback` is rejected by the fault matcher on data corruption; only `partial_rollback` credits.
3. MTTR follows from mitigation: an unfixed fault has no time-to-mitigate, so the MTTR component zeros too.

The agent in Episode 3 *knew what the problem was* — it diagnosed correctly. It just chose the wrong response. The rubric punishes that with no LLM-as-judge anywhere. That's the kind of structural bite we think these envs need if we want to use them to actually train models, not just evaluate them.

## The training journey — five attempts, two wins

Here's where we'll be honest. **We tried to train a Llama-3.2-3B-Instruct LoRA on this env four times before anything worked.** This is the unromantic part of RL on small models.

### Attempt 1 — GRPO directly on base. Regressed by −0.356.

We started with TRL's `GRPOTrainer` because most of the OpenEnv tutorials use GRPO. The training script ran. The reward curve looked plausible. The eval came back: pre 0.736 → **post 0.380**.

Diagnosis: Pydantic validation errors dominated the rollout group. When 3 of 4 rollouts in a group emit malformed JSON, the group mean is near zero, and `(0 − 0)/(0 + ε)` is a huge spurious advantage that gets multiplied through the policy gradient. Combine that with no KL penalty (omitted to fit a 4-bit reference into a free T4) and you get policy drift toward whatever degenerate format the model emits most often.

### Attempt 2 — RFT directly on base. Regressed by −0.310.

OK, GRPO was unstable. Rejection-sampling FT (sample N, keep top K, SFT on those) is supposed to be the safe alternative — the loss curve is clean SFT NLL, not a finicky advantage estimator.

It regressed by basically the same amount. Same root cause: when the **base model can't reliably emit valid `ICAction` JSON**, there's nothing for either RFT or GRPO to amplify. Filtering the model's own bad rollouts gives you bad rollouts. Mode collapse onto safe-but-empty actions.

### What we should have done first

This is the lesson we wish we'd internalized day one. Both GRPO and RFT depend on the model occasionally producing format-valid, non-trivial-reward trajectories. **Our 3B model couldn't.** Pre-training error rate: 2 out of 3 baseline rollouts crashed on parse/validation.

The fix isn't a different RL algorithm. It's stepping back to **supervised fine-tuning on a deterministic teacher** — the standard SFT-then-RL recipe behind every modern instruction-tuned LLM (InstructGPT, Llama-2-Chat, DeepSeek-R1, Qwen-Chat). Skipping SFT is a known failure mode for small base models on structured-output envs. We hit it. Twice.

### Attempt 3 — SFT on all 3 tasks with the full oracle. Regressed by −0.094.

We generated ~570 labeled `(observation, action)` pairs from deterministic scripted oracle policies across 30 seeds × 3 tasks × ~6 actions per episode. Trained 1 epoch of LoRA SFT.

Macro-mean: pre 0.693 → post 0.599. Regression. Caught by the gate, no adapter saved.

Per-task breakdown told the real story:

- **Easy:** 0.96 → 0.46 (catastrophic regression — already near-optimal from the prompt)
- **Medium:** 0.78 → 0.77 (flat)
- **Hard:** 0.35 → 0.58 (real gain, but with JSON parse failures on the postmortem)

Two things broke:

- SFT on the easy task **memorized a fixed action sequence**, including the long postmortem JSON. The trained model emitted identical actions on every seed regardless of observation — rigid imitation that scored worse than the flexible prompt-only version.
- The oracle's postmortem JSON was 250-400 tokens of nested fields. Loss bottomed at 1.53 (not 0.5) — the model couldn't memorize it cleanly. On hard, 2 of 3 seeds parse-failed mid-postmortem and lost the credit entirely.

### Attempt 4 — Focused SFT (medium + hard, simplified postmortem). +0.122.

Two changes:

- Drop easy from training data. It was already at 0.96; the headroom was zero and the downside was the catastrophic regression we'd just seen.
- Trim oracle postmortems from 4-item timeline + 4-item actions_taken to 2 + 2. Grader minimums are `timeline ≥ 2`, `actions_taken ≥ 1`; oracle still scores 0.10/0.10. The shorter target became memorizable.

The result on the third try: macro-mean 0.693 → 0.815 (+0.122). **Hard task more than doubled — 0.347 → 0.895.** Went from worst-scoring task to best-scoring task in the project. Three eval seeds, all 0.895, deterministic, zero parse errors. That is the headline.

The trade we made: spend the easy task's 0.20 of headroom (still left at 0.76, comfortably above oracle on most criteria) to unlock +0.55 on the task that mattered. That's a strictly positive trade in macro-mean terms and a *much* better story.

The loss curve was monotonic and clean — no spikes, no plateaus, just successful memorization of a deliberately-simplified target. Component-level breakdown showed the gain came from exactly where it should: on hard, RCA went from 0.00 → 0.20 (full credit), mitigation 0.00 → 0.15 (full credit), postmortem held at full 0.10 — the simplified-postmortem fix did exactly what it was supposed to.

The full plots, per-task tables, and component breakdowns live in **[`blog.md`](blog.md)** under "Training results."

### Attempt 5 — RFT polish on top of SFT-warm. +0.190.

After Attempt 4 landed, we tried online rejection-sampling fine-tuning *on top of* the SFT-warm checkpoint. The setup: 2 iterations × 12 rollouts × keep top 2, LR 2e-5, score-floor 0.55, **and a critical new safeguard — `--require-done 1`** that filters out any rollout that didn't complete the episode cleanly.

The safeguard mattered. An earlier RFT-on-SFT attempt collapsed catastrophically (−0.703) because step=25-without-finish trajectories that scored 0.45 from partial credit passed the score floor and got reinforced. By iter 3 of that previous run, the model had learned to "query forever, never mitigate." The require_done filter eliminates that failure mode at the source: trajectories that didn't actually finish can't enter the kept set, even if their score passes the floor.

This time the run worked.

**The narrative twist:** the SFT checkpoint scored 0.815 under greedy decoding (T=0.0), but only **0.620** when sampled at T=0.7 (which is what RFT's pre-eval uses for honest variance estimation). The +0.190 gain from RFT polish is, in effect, the model becoming *robust to higher-temperature sampling*. The trained model scores 0.810 at T=0.7 — nearly matching its own greedy ceiling. RFT didn't push absolute capability much; it pushed reliability under noisy decoding. That's a smaller but real win.

Combined journey from base to final:

| Stage | Macro-mean | Cumulative Δ vs raw base |
|---|---:|---:|
| Llama-3.2-3B base, broken prompt | 0.31 | — |
| + Phase-1 prompt fixes (no training) | 0.69 | +0.38 |
| + SFT focused (Attempt 4) | 0.815 | +0.51 |
| + RFT polish (Attempt 5) at T=0.7 | 0.810 | (matches SFT under noisy sampling) |

> **Reproducibility note:** Exact numbers per re-run may vary by ±0.05 on a different T4 due to GPU non-determinism in matmul kernels and sampling RNG state; the magnitude and direction of the gains are stable. The committed `training/sft_metrics.json` and `training/rft_on_sft_metrics.json` are the canonical references; the Colab notebook reproduces the pipeline, not the bit-identical floats.

## What we actually learned

**Prompt design is the cheapest 0.38 macro-mean you'll ever buy.** Before any training, fixing the system prompt — adding a services list, a per-fault mitigation table, a parse-retry mechanism — moved baseline from 0.31 to 0.69. That's larger than every training delta in this project combined. *Prompt is data.* Don't skip it.

**SFT first, RL second isn't paper jargon.** It's there because every published recipe runs into the same wall: a base model that can't sample valid actions can't be improved by sample-filtering RL methods. SFT on a teacher is what makes the rollouts good enough for RL to find anything to amplify. We learned this the expensive way.

**Predictions about training outcomes are mostly wrong.** We (and the AI tooling we leaned on) consistently mispredicted per-task deltas. Macro-mean predictions were closer because aggregate cancellations are forgiving. The honest practice: forecast direction and rough magnitude, not specific numbers; let the regression gate be the authority; treat each run as new evidence rather than confirmation of a precise model.

**The regression gate is the single most useful piece of training infra in the project.** Three of five runs regressed. The gate refused to overwrite our best adapter on every one of them. Without it we would have shipped a worse model under the impression we'd improved.

**A `--require-done` filter is what made RFT-on-SFT actually work.** This is a one-line code change with outsized impact. RFT's filter-and-amplify mechanism is brittle because a trajectory's *score* doesn't tell you whether it *finished* — partial credit from observability-only behavior can match real wins numerically. Filtering on actual completion was the difference between a +0.19 polish and a −0.70 collapse.

## Honest negative results — and why we're keeping them

The failed scripts (`training/train_grpo.py`, the original direct-RFT path, the multi-task SFT run, the previous unfiltered RFT-on-SFT collapse) are still in the repo. The metrics JSONs of every regressed run are still in `training/`. We want a curious judge to be able to re-run them and see the regressions for themselves. Deleting the misses to make the README cleaner would weaken the actual story, which is *how* the working pipeline got to working — through repeatable, recorded failure.

| Attempt | Pre | Post | Δ | Failure mechanism |
| --- | ---: | ---: | ---: | --- |
| GRPO direct on base | 0.736 | 0.380 | **−0.356** | Format-failure dominated rollouts → group advantage degenerate → policy collapsed to a single mode at greedy decode. |
| RFT direct on base | 0.310 | 0.000 | **−0.310** | `score_floor=0.30` on a baseline of 0.31 filtered noise into the training set; mode collapse onto safe-but-empty actions. |
| SFT on all 3 tasks | 0.693 | 0.599 | **−0.094** | Memorized full oracle postmortem failed to generalize → JSON parse errors on hard, mode collapse on easy. Gate caught it; no save. |
| RFT-on-SFT (no `--require-done`) | 0.815 | 0.112 | **−0.703** | Step=25-without-finish trajectories passed the score floor with partial credit; reinforced "query forever, never mitigate". |

## Try it

| | URL |
| --- | --- |
| **HF Space (env)** | [vasubhrdwj/incident-commander-openenv](https://huggingface.co/spaces/vasubhrdwj/incident-commander-openenv) |
| **Live API** | [vasubhrdwj-incident-commander-openenv.hf.space](https://vasubhrdwj-incident-commander-openenv.hf.space) — `/docs`, `/health`, `/replay` |
| **Demo replay UI** | [/replay on the live Space](https://vasubhrdwj-incident-commander-openenv.hf.space/replay) |
| **Trained adapter (SFT)** | [vasubhrdwj/incident-commander-sft-lora](https://huggingface.co/vasubhrdwj/incident-commander-sft-lora) |
| **Trained adapter (RFT-on-SFT)** | [vasubhrdwj/incident-commander-rft-on-sft-lora](https://huggingface.co/vasubhrdwj/incident-commander-rft-on-sft-lora) |
| **Reproducible Colab** | [`training/train_colab.ipynb`](training/train_colab.ipynb) |
| **Technical documentation** | [`blog.md`](blog.md) — action/observation/reward spec, all training plots, full benchmark numbers, reproducibility commands |
| **2-minute pitch script** | [`demo/PITCH.md`](demo/PITCH.md) |
| **Full design narrative** | [`PROJECT.md`](../PROJECT.md) (repo root) |

**Smoke-test the live env (no install required):**

```bash
openenv validate --url https://vasubhrdwj-incident-commander-openenv.hf.space
```

The agent doesn't outscore the oracle. The oracle is hand-written and near-optimal by construction. The point isn't to match it — it's that the env, the rubric, and the training pipeline together produce a clean improvement signal that survives independent re-runs and can't be gamed by a model that *knows what the problem is* but answers wrong.

Incidents teach the humans who survive them. We wanted the agents to learn too.

— **Team VBxAG**
