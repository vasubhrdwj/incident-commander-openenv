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
| **Trained LoRA adapter** | [vasubhrdwj/incident-commander-llama3.2-3b-rft](https://huggingface.co/vasubhrdwj/incident-commander-llama3.2-3b-rft) |
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

## Training — RFT (rejection-sampling fine-tuning)

The submission's training pipeline is **online RFT**: each iteration samples N=6 fresh rollouts from a live env server, keeps the top K=2 by reward, and SFTs Llama-3.2-3B-Instruct (LoRA, rank 16, 4-bit base via Unsloth) on those trajectories. The training loop **connects to the live `IncidentCommanderEnvironment`** every iteration, not a static dataset.

**Run it yourself**: open [`training/train_colab.ipynb`](training/train_colab.ipynb) in Colab → `Runtime → T4 GPU` → `Run all`. ~30–40 min wall time on a T4 when Colab allocates one. If Colab cannot attach a GPU, run the same command on Hugging Face Jobs with a T4/small-medium GPU; the notebook remains the judge-rerunnable recipe, and the HF Job is the compute path used to produce the submitted artifacts.

**HF Jobs fallback** (same RFT config, recommended when Colab is CPU-only):

```bash
cd incident_commander
HF_TOKEN=... python -m incident_commander.training.launch_hf_rft_job \
  --repo-url https://github.com/vasubhrdwj/incident-commander-openenv.git \
  --model-repo-id vasubhrdwj/incident-commander-llama3.2-3b-rft \
  --flavor t4-medium --timeout 2h
```

The token must allow Hugging Face Jobs (`job.write`) and writes to the target model repo. The job uploads the LoRA adapter plus `rft_metrics.json` and the four plot PNGs under `training_artifacts/` in the adapter repo.

### Plots generated by the RFT run

| | Caption |
| --- | --- |
| ![score summary](assets/score_summary.png) | **Headline:** pre vs post-training total score, error bars from 3 eval episodes. |
| ![training reward](assets/training_reward.png) | Mean rollout score across all sampled rollouts (grey), top-K kept (green), best-of-batch (amber). Trends up over 8 RFT iterations. |
| ![training loss](assets/training_loss.png) | SFT NLL loss against rejection-sampled top-K trajectories. Trends down. |
| ![component breakdown](assets/component_comparison.png) | Pre/post totals against the six rubric component weight ceilings. |

### Honest ablation: GRPO regressed

We tried GRPO first (`training/train_grpo.py`, with the original Colab notebook at `training/train_grpo_colab.ipynb`). The run regressed: **pre = 0.736 → post = 0.380, Δ = –0.356**. Diagnosed causes (so future work can fix):
- **Parse failures dominated the advantage**: when 3 of 4 rollouts in a group emit malformed JSON, group mean ≈ 0 and `(0 - 0)/(0 + ε)` is a huge unsigned advantage that gets multiplied through the policy gradient.
- **No KL penalty to the reference model**: omitted to fit a 4-bit reference into a free T4. Without it the policy drifts arbitrarily far from base.
- **Eval at T=0.1**: greedy decoding reads whatever mode the policy collapsed into. Even a partially-improved policy looks regressed at low temp.

We documented this as a known ablation rather than papering over it. The RFT pipeline above replaces it as the primary training story.

### Inference-time evidence (separate from RFT)

Best-of-N reward-guided sampling at T=0.9 with N=3 on Llama-3.2-3B base matched the oracle ceiling: **0.872 max / 0.855 mean** on `easy_canary_regression`. The rubric is a verifier — sampling N times and keeping the best is a valid post-training improvement lever even when weight updates fail. See `training/best_of_n.py`.

---

## Baseline scores

| Task | Policy | Seed | Final score | Notes |
| --- | --- | --- | --- | --- |
| easy_canary_regression | Oracle / ideal (`IC_MOCK_POLICY=1`) | 0 | 0.872 | Scripted ceiling, used to calibrate rubric |
| easy_canary_regression | Llama-3.2-3B-Instruct (base, T=0.1) | 0 | 0.736 | Zero-shot, mean across 3 eval episodes |
| easy_canary_regression | Llama-3.2-3B-Instruct + GRPO LoRA (50 iter) | 0 | 0.380 | **Regressed** — archived as failed ablation (parse-error-dominated advantage, no KL, mode collapse). Not shipped. |
| easy_canary_regression | Llama-3.2-3B-Instruct best-of-N (N=3, T=0.9) | 0 | **0.872** (max) / 0.855 (mean) | Reward-guided sampling — inference-time, not trained weights. Scores: 0.872, 0.846, 0.846. Matched the oracle ceiling on the first draw. Run truncated at N=3 by HF free-tier quota. |
| easy_canary_regression | Llama-3.2-3B-Instruct **+ RFT LoRA** (8 iter × 6 rollouts) | 0 | Produced by `rft_metrics.json` | Online rejection-sampling fine-tuning — see [`training/train_colab.ipynb`](training/train_colab.ipynb). Plots in `assets/`. |
| medium_third_party_attribution | Oracle / ideal | 0,1,2 | 0.468 | Seed variants exercise `hold`, `feature_flag`, and `rollback`; used as a transfer/generalisation check, not the training target. |
| hard_silent_data_corruption | Oracle / ideal (`IC_MOCK_POLICY=1`) | 0 | **0.855** | Audit-only detection + `partial_rollback` + targeted `customer_email`. No alerts ever fire on this task. |
| hard_silent_data_corruption | Easy-task playbook on hard task | 0 | 0.479 | Negative-transfer demo: status_page earns 0 comms, full `rollback` earns 0 mitigation, and the agent must `query_audit` despite a green dashboard. |

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
