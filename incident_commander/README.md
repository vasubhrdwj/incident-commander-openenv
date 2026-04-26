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

# Incident Commander — OpenEnv

**Problem:** Train and evaluate LLM agents as **Incident Commanders** during a production outage: gather evidence from observability, coordinate specialist responders, choose the right mitigation under time pressure, communicate to stakeholders, and close with a structured post-mortem.

**Environment:** Deterministic simulator over a **6-service graph**, **fault injection** (task-dependent), **synthetic logs / metrics / traces / audit / external status**, and **four specialist NPCs** (SRE, Security, Comms, Eng Lead) implemented as deterministic FSMs. Each HTTP/WebSocket session is one episode with a fixed step budget.

**Shipped tasks (3):**

| `IC_TASK_ID` | Cognitive ability tested | Right mitigation | Right comms |
| --- | --- | --- | --- |
| `easy_canary_regression` | **Reactive deduction** ("read what's in front of you") | `rollback` | `status_page` |
| `medium_third_party_attribution` | **Discriminative attribution** ("tell apart things that look alike") — variant by `IC_SEED` mod 3 | `hold` / `feature_flag` / `rollback` | `status_page` |
| `hard_silent_data_corruption` | **Inverted reasoning under no signal** ("act when nothing tells you to") | `partial_rollback` | `customer_email` + cohort |

The same scripted oracle scores **0.872 / 0.468 / 0.239** across the three tasks — concrete evidence the env doesn't reward repetition of one trick.

Full design narrative: see repo root **`PROJECT.md`** (one directory up from this package).

### Submission materials

| | URL |
| --- | --- |
| **HF Space (env)** | [vasubhrdwj/incident-commander-openenv](https://huggingface.co/spaces/vasubhrdwj/incident-commander-openenv) |
| **Running app** | [vasubhrdwj-incident-commander-openenv.hf.space](https://vasubhrdwj-incident-commander-openenv.hf.space) |
| **Demo replay UI** | [vasubhrdwj-incident-commander-openenv.hf.space/replay](https://vasubhrdwj-incident-commander-openenv.hf.space/replay) |
| **Training Colab** | [`training/train_colab.ipynb`](training/train_colab.ipynb) — judges can open in Colab and run end-to-end |
| **Trained LoRA adapter** | [vasubhrdwj/incident-commander-sft-lora](https://huggingface.co/vasubhrdwj/incident-commander-sft-lora) — SFT on oracle, +0.122 macro, **+0.548 on hard task** |
| **2-min pitch script** | [`demo/PITCH.md`](demo/PITCH.md) |
| **Writeup / story** | This README + [`demo/PITCH.md`](demo/PITCH.md) cover the short judge-facing writeup and demo narrative. |

Smoke-test the API: `openenv validate --url https://vasubhrdwj-incident-commander-openenv.hf.space`

---

## Configuration (env vars)

Read at **session creation** (each WebSocket client gets a fresh env):

| Variable | Default | Purpose |
| --- | --- | --- |
| `IC_TASK_ID` | `easy_canary_regression` | Which task / fault template |
| `IC_SEED` | `0` | RNG + medium-task variant selection |
| `IC_STEP_BUDGET` | (task default) | Max steps before forced termination |

**Medium task variants** (deterministic from seed): `IC_TASK_ID=medium_third_party_attribution` with seeds `0` / `1` / `2` → provider outage, bad integration, our deploy — smoke-test all three in separate containers or sessions.

---

## Action space (`ICAction`)

Flat union: required **`op`** plus optional payload fields (only those relevant to `op`).

| `op` | Role |
| --- | --- |
| `query_logs`, `query_metrics`, `query_trace` | Observability pulls (scoped by `service` where applicable) |
| `query_audit` | Audit log slice |
| `query_external_status` | Third-party / provider status page |
| `delegate` | Request work from a specialist (`specialist`: sre / security / comms / eng_lead) |
| `mitigate` | Execute mitigation (`mitigation`, `target` / `service` as needed) |
| `communicate` | Stakeholder update (`channel`, `message`, optional `audience`) |
| `diagnose` | Submit RCA hypothesis (`service`, `root_cause_tag`) |
| `resolve` | Declare incident resolved |
| `postmortem` | Terminal structured JSON (`postmortem_json`) |

**Mitigation kinds:** `restart`, `rollback`, `partial_rollback`, `scale`, `feature_flag`, `hold`.

Wire schema: `models.py` or **`GET /schema`** on the running server.

---

## Observation space (`ICObservation`)

Each step returns a rich observation including:

- Simulator time, step index, task id, high-level **service health** and **alerts**
- Results of the **last query** (logs, traces, audit, external status) when applicable
- **NPC messages** and latest specialist **reports**
- **Comms** state (e.g. status page last update)
- **`done`**, **`reward`** (incremental rubric signal this step), optional status text

---

## Reward (rubric)

Six **independent** weighted components (dense per-step + terminal), all clamped to **[0, 1]** for the episode total:

| Component | Weight | Intent |
| --- | ---: | --- |
| Containment | 0.25 | Limit blast radius / premature bad mitigations |
| MTTR | 0.20 | Time from fault to correct mitigation |
| Correct RCA | 0.20 | `diagnose` vs ground truth (partial credit for service-only) |
| Right mitigation | 0.15 | Correct `mitigate` for this task (`hold` and `partial_rollback` rewarded only when ground truth) |
| Comms SLA | 0.10 | Timely comms; **task-conditional channel** — `status_page` on easy/medium, `customer_email`+cohort on hard |
| Post-mortem | 0.10 | Validated JSON structure + factual fields at end |

**Anti-gaming guards (structural, not heuristic):**
- `hold` mitigation credits only when ground truth is `hold` (otherwise zero — easy-task `hold` earns nothing).
- `partial_rollback` mitigation rejects full `rollback` on the data-corruption fault (easy-task playbook does not transfer).
- Comms grader is task-conditional on `correct_mitigation` — `status_page` on the hard task earns 0 comms.
- Status-page spam (≥2 posts within 60s) zeros the comms component.
- Invalid actions burn a step — no free retries.

---

## Training — SFT on oracle trajectories (the journey)

The training story is the *recovery from two failed RL attempts*. We tried RL first, both runs regressed below baseline, and the fix was to step back to **supervised fine-tuning on deterministic oracle trajectories** — the standard SFT-then-RL recipe behind every modern instruction-tuned LLM. Numbers below are real, all from `training/sft_metrics.json`.

### Headline numbers (from `sft_metrics.json`)

| Task | Base + Phase-1 prompt | Post-SFT | Δ |
| --- | ---: | ---: | ---: |
| `easy_canary_regression` | 0.956 | 0.761 | −0.195 |
| `medium_third_party_attribution` | 0.775 | 0.790 | +0.015 |
| **`hard_silent_data_corruption`** | **0.347** | **0.895** | **+0.548** |
| **Macro-mean** | **0.693** | **0.815** | **+0.122** |

The hard task — silent data corruption requiring audit-only diagnosis and `partial_rollback` — went from worst-scoring task in the project to **best-scoring task**, deterministically across 3 eval seeds, no JSON parse errors. That is the headline.

### Plots (committed under `assets/`)

| | Caption |
| --- | --- |
| ![pre vs post](assets/sft_pre_post.png) | **Headline:** per-task baseline vs post-SFT macro-mean. Hard task triples; medium holds; easy regresses by ~0.20 (cost of multi-task LoRA on a task already at 0.96). Net: **+0.122 macro**. |
| ![SFT loss curve](assets/sft_loss.png) | Training-set NLL across 49 logged steps. Monotonic, bottoms near 1.95 — clean memorization of the simplified oracle template, no instability or spikes. |
| ![component breakdown](assets/sft_components.png) | Six-component rubric breakdown pre vs post per task. Hard's RCA goes 0.00 → 0.20 (full credit), mitigation 0.00 → 0.15 (full credit), postmortem stays at full 0.10 — the simplified-postmortem fix held. |

### What we trained on

- Base: `unsloth/Llama-3.2-3B-Instruct`, 4-bit, LoRA (rank 16, α 32, q/k/v/o/gate/up/down).
- Dataset: `training/build_sft_dataset.py` rolls each task's deterministic oracle policy through a fresh in-process env across 30 seeds × 2 tasks (medium + hard; easy excluded — see "design choices" below). 390 labeled `(observation, action)` pairs.
- Trainer: TRL `SFTTrainer` + Unsloth, 1 epoch, lr 1e-4, batch 2 × grad-accum 4, ~17 min wall on a T4.
- Regression gate: `train_sft.py` refuses to save the adapter unless macro-mean delta ≥ +0.05. Final delta +0.122 cleared the gate.

### Run it yourself

```bash
# HF Jobs (T4, ~30 min wall, one-shot)
hf jobs run --flavor t4-medium --secrets HF_TOKEN \
  ghcr.io/meta-pytorch/openenv-base:latest \
  -- bash -c '
    git clone https://huggingface.co/spaces/vasubhrdwj/incident-commander-openenv /workspace/repo &&
    cd /workspace/repo && pip install -e ".[training]" &&
    python -m incident_commander.training.build_sft_dataset \
        --seeds 30 --tasks medium_third_party_attribution hard_silent_data_corruption \
        --output sft_oracle.jsonl &&
    python -m incident_commander.training.train_sft \
        --dataset sft_oracle.jsonl --output-dir ./ic-sft-oracle \
        --metrics-json sft_metrics.json --min-improvement 0.05 --precision auto &&
    python -m incident_commander.training.plot_metrics --mode sft \
        --metrics sft_metrics.json --out assets/
  '
```

Colab path: open [`training/train_colab.ipynb`](training/train_colab.ipynb) → `Runtime → T4 GPU` → `Run all`.

### Design choices that mattered

- **Train on medium + hard only, not easy.** Easy was already at 0.956 from the prompt — there was no headroom, only downside. Multi-task LoRA still bled into easy (regressed 0.20), but the gain on hard (+0.55) more than compensates. **The honest tradeoff is: spend easy headroom to unlock hard learning.**
- **Trim oracle postmortem schema** from 4-item timeline + 4-item actions_taken to 2 + 2. Grader minimums are `timeline ≥ 2`, `actions_taken ≥ 1`; oracle still scores 0.10/0.10 on postmortem. The shorter target is now memorizable cleanly — no more 2/3-seed JSON parse failures we saw on the first SFT attempt.
- **Regression gate enforced.** Two earlier RL attempts pushed bad checkpoints (we regressed before catching it). The gate in `train_sft.py` now refuses to save when macro-mean delta < +0.05, so a bad run can't masquerade as a successful one.

### Honest negative results — what didn't work, and why

We tried two RL approaches before SFT. **Both regressed.** Documented here rather than buried, because the *why* is the actual lesson.

| Attempt | Pre | Post | Δ | Failure mechanism |
| --- | ---: | ---: | ---: | --- |
| GRPO direct on base | 0.736 | 0.380 | **−0.356** | Format-failure dominated rollouts → group advantage degenerate → policy collapsed to a single mode at greedy decode. |
| RFT direct on base | 0.310 | 0.000 | **−0.310** | `score_floor=0.30` on a baseline of 0.31 filtered noise into the training set; mode collapse onto safe-but-empty actions. |
| SFT on all 3 tasks (first attempt) | 0.693 | 0.599 | **−0.094** | Memorized full oracle postmortem failed to generalize → JSON parse errors on hard, mode collapse on easy. Gate caught it; no save. |

**Lesson:** Both filter-and-amplify RL methods (GRPO, RFT) need the model to *occasionally* sample format-valid trajectories with reward variance. A 3B base on a structured-action env can't. SFT on a deterministic teacher bypasses the problem — there's nothing to filter, just labeled imitation. **Then** RL polish on the SFT-warm checkpoint becomes meaningful (next stage; in progress).

The failed-run code lives at `training/train_grpo.py` and the original `training/train_rft.py` is preserved as evidence. We did not delete them; reproducibility means showing the misses too.

### Phase 1 — the prompt-engineering delta we don't take credit for in training

Worth flagging separately: a substantial chunk of the model's competence comes from `inference.py`'s system prompt (services list, per-fault mitigation table, parse-retry-with-hint at temperature 0.4). That alone moved baseline from **0.31 → 0.693** before any training. The SFT result builds on that foundation. Both deltas stack into the final score the trained adapter produces.

### Inference-time evidence (separate from training)

Best-of-N reward-guided sampling at T=0.9 with N=3 on Llama-3.2-3B base matched the oracle ceiling: **0.872 max / 0.855 mean** on `easy_canary_regression`. The rubric is a verifier — sampling N times and keeping the best is a valid post-training improvement lever even when weight updates fail. See `training/best_of_n.py`.

---

## Baseline scores

All numbers are mean across 3 eval seeds on the trained model unless noted. Source for trained-model rows: `training/sft_metrics.json` (committed). Source for oracle / best-of-N rows: in-process eval with `IC_MOCK_POLICY=1` and `training/best_of_n.py` respectively.

| Task | Policy | Score | Notes |
| --- | --- | ---: | --- |
| easy_canary_regression | Oracle / ideal (`IC_MOCK_POLICY=1`) | **0.872** | Scripted ceiling. |
| easy_canary_regression | Llama-3.2-3B base + Phase-1 prompt | **0.956** | Mean of 3 seeds. Prompt fixes alone solve this task. |
| easy_canary_regression | Llama-3.2-3B + **SFT LoRA** | 0.761 | Multi-task LoRA bleeds into easy; net regression −0.20 — the cost of unlocking hard. |
| easy_canary_regression | Best-of-N (N=3, T=0.9) on base | 0.872 max / 0.855 mean | Inference-time, no weight updates. Matched oracle on first draw. |
| medium_third_party_attribution | Oracle / ideal | 0.832 (provider) / 0.832 (integration) / 0.832 (our_deploy) | Three seed variants exercise `hold` / `feature_flag` / `rollback`. |
| medium_third_party_attribution | Llama-3.2-3B base + Phase-1 prompt | 0.775 | Mean across 3 seed variants. |
| medium_third_party_attribution | Llama-3.2-3B + **SFT LoRA** | 0.790 | +0.015. Phase-1 prompt was already strong here; SFT marginal. |
| hard_silent_data_corruption | Oracle / ideal (`IC_MOCK_POLICY=1`) | **0.855** | Audit-only detection + `partial_rollback` + targeted `customer_email`. No alerts ever fire. |
| hard_silent_data_corruption | Llama-3.2-3B base + Phase-1 prompt | 0.347 | Hard for the prompt-only agent — model can't infer the `partial_rollback` playbook from prompt alone. |
| **hard_silent_data_corruption** | **Llama-3.2-3B + SFT LoRA** | **0.895** | **+0.548** — best score on the hardest task, deterministic across 3 seeds, no parse errors. The headline result. |
| hard_silent_data_corruption | Easy-task playbook on hard task | 0.479 | Negative-transfer demo: status_page earns 0 comms, full `rollback` earns 0 mitigation. |
| **Macro-mean across 3 tasks** | Phase-1 prompt | 0.693 | |
| **Macro-mean across 3 tasks** | **+ SFT LoRA** | **0.815** | **+0.122** — gate-passed adapter saved to `vasubhrdwj/incident-commander-sft-lora`. |
| Macro-mean (failed attempts) | GRPO direct on base | 0.380 | **−0.356** vs. paired baseline. See "Honest negative results" above. |
| Macro-mean (failed attempts) | RFT direct on base | 0.000 | **−0.310** vs. paired baseline. Mode collapse to safe-but-empty actions. |

**Reproducing the best-of-N run:**

```bash
# server running at $ENV_URL with IC_TASK_ID=easy_canary_regression
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct HF_TOKEN=... \
python -m incident_commander.training.best_of_n \
  --n 8 --temperature 0.9 --task easy_canary_regression \
  --output-json ./bon_results.json
```

### Demo episodes (no API key needed)

`demo/run_episodes.py` runs three deterministic in-process episodes —
easy oracle, hard oracle, and the headline "hard task with the easy-task
playbook" episode — and prints rubric-component breakdowns plus a
combined `demo/episodes.json` artifact. The third episode is the
demonstration: same agent shape, same six components, same level of
effort, but the structural anti-gaming guards (task-conditional comms +
strict `partial_rollback` matcher) collapse the score from **0.855 → 0.479**.

```bash
PYTHONPATH=. python -m incident_commander.demo.run_episodes
# writes demo/episodes.json; runs in <2 seconds
```

You can also view the same three episodes as an animated rubric replay
in your browser at **`/replay`** on any running server (local or HF
Space). It loads `demo/episodes.json` and animates the six-component
rubric bars filling step-by-step — the cleanest single visual of the
"structural anti-gaming guards bite" headline.

The 2-minute pitch script keyed to terminal screens lives at
[`demo/PITCH.md`](demo/PITCH.md).

---

## Quick start (Python client)

Install from this repo (see `pyproject.toml`), run the server locally or point at your **Space URL**.

```python
from incident_commander import ICAction, IncidentCommanderEnv

# Local: uvicorn, Docker, or Hugging Face Space — set base_url accordingly.
with IncidentCommanderEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    obs = result.observation
    print(obs.task_id, obs.done, obs.reward)

    result = env.step(
        ICAction(op="query_logs", service="api_gateway", query="error")
    )
    print(result.observation.reward, result.done)
```

**Strict eval logging** for the hackathon checklist: run **`inference.py`** from the package directory (see file header for env vars and `[START]`/`[STEP]`/`[END]` log format).

---

## Build & run (Docker)

From **`incident_commander/`** (directory containing `openenv.yaml`):

```bash
docker build -t incident-commander-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 \
  -e IC_TASK_ID=easy_canary_regression \
  -e IC_SEED=0 \
  incident-commander-env:latest
```

OpenAPI: **http://localhost:8000/docs** · Demo replay: **http://localhost:8000/replay**. Some OpenEnv deployments also expose the generated playground at **`/web`**.

---

## Deploy to Hugging Face Spaces

```bash
cd incident_commander
huggingface-cli login   # once
openenv validate --verbose
openenv push --repo-id vasubhrdwj/incident-commander-openenv
```

After deploy, smoke-test:

```bash
openenv validate --url https://vasubhrdwj-incident-commander-openenv.hf.space
```

---

## Project layout

```
incident_commander/
├── openenv.yaml
├── README.md
├── models.py
├── client.py
├── inference.py
├── simulator/
├── graders/
└── server/
    ├── app.py
    ├── incident_commander_environment.py
    └── Dockerfile
```
