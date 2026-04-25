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

**Shipped tasks (2):**

| `IC_TASK_ID` | Focus | Notes |
| --- | --- | --- |
| `easy_canary_regression` | Deductive canary regression | Ground-truth mitigation: **rollback** |
| `medium_third_party_attribution` | Third-party vs integration vs bad deploy | Variant from **`IC_SEED`**; mitigation may be **hold**, **feature_flag**, or **rollback** |

**Stretch (stub in repo):** `hard_silent_data_corruption` — low-signal corruption + **partial_rollback** (not required for HF demo).

Full design narrative: see repo root **`PROJECT.md`** (one directory up from this package).

### Live Space (early deploy)

| | URL |
| --- | --- |
| **Hub** | [vasubhrdwj/incident-commander-openenv](https://huggingface.co/spaces/vasubhrdwj/incident-commander-openenv) |
| **Running app** | [vasubhrdwj-incident-commander-openenv.hf.space](https://vasubhrdwj-incident-commander-openenv.hf.space) |

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
| Right mitigation | 0.15 | Correct `mitigate` for this task (`hold` rewarded only when ground truth) |
| Comms SLA | 0.10 | Timely status-page update; anti-spam |
| Post-mortem | 0.10 | Validated JSON structure + factual fields at end |

---

## Baseline scores (fill as you run eval)

| Task | Policy | Seed | Final score | Notes |
| --- | --- | --- | --- | --- |
| easy_canary_regression | Oracle / ideal (`IC_MOCK_POLICY=1`) | 0 | 0.872 | Scripted ceiling, used to calibrate rubric |
| easy_canary_regression | Llama-3.2-3B-Instruct (base, T=0.1) | 0 | 0.736 | Zero-shot, mean across 3 eval episodes |
| easy_canary_regression | Llama-3.2-3B-Instruct + GRPO LoRA (50 iter) | 0 | 0.380 | **Regressed** — archived as failed ablation (parse-error-dominated advantage, no KL, mode collapse). Not shipped. |
| easy_canary_regression | Llama-3.2-3B-Instruct best-of-N (N=3, T=0.9) | 0 | **0.872** (max) / 0.855 (mean) | Plan §4.5.7 reward-guided sampling — inference-time, not trained weights. Scores: 0.872, 0.846, 0.846. Matched the oracle ceiling on the first draw. Run truncated at N=3 by HF free-tier quota. |
| medium_third_party_attribution | Oracle / ideal | 0,1,2 | *TBD* | One row per variant |
| medium_third_party_attribution | LLM baseline | * | *TBD* | |
| hard_silent_data_corruption | Oracle / ideal (`IC_MOCK_POLICY=1`) | 0 | **0.855** | Audit-only detection + `partial_rollback` + targeted `customer_email`. No alerts ever fire on this task. |
| hard_silent_data_corruption | LLM baseline | 0 | *TBD* | Expect a steep drop vs easy: status_page earns 0 comms here, full `rollback` earns 0 mitigation, and the agent must `query_audit` despite a green dashboard. |

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

OpenAPI: **http://localhost:8000/docs** · Web UI: **http://localhost:8000/web**

---

## Deploy to Hugging Face Spaces

```bash
cd incident_commander
huggingface-cli login   # once
openenv validate --verbose
openenv push --repo-id YOUR_USERNAME/incident-commander-openenv
```

After deploy, smoke-test:

```bash
openenv validate --url https://YOUR_USERNAME-incident-commander-openenv.hf.space
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
