# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Incident Commander OpenEnv — a deterministic, multi-actor, long-horizon RL environment that puts an LLM in the seat of an on-call IC during a simulated outage, plus a GRPO training scaffold. Full design narrative: `PROJECT.md` (root) and `incident_commander/README.md` (HF Space header). Time-phased plan: `.cursor/plans/incident_commander_openenv_8deacc01.plan.md`.

## Common commands

All commands below assume you are in `incident_commander/` unless noted. The package uses `uv` + a Python 3.11 venv at repo root (`.venv/`).

```bash
# Install (editable, env-server deps only; training extras are GPU-only)
uv pip install -e .
uv pip install -e ".[dev]"       # + pytest
uv pip install -e ".[training]"  # GPU-only: torch, trl, unsloth, peft, bnb

# Run the env server locally (three equivalent ways)
IC_TASK_ID=easy_canary_regression uvicorn server.app:app --host 0.0.0.0 --port 8000
IC_TASK_ID=easy_canary_regression python -m server.app
IC_TASK_ID=easy_canary_regression uv run server

# Build + run the Docker image (matches HF Space)
docker build -t incident-commander-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 -e IC_TASK_ID=easy_canary_regression -e IC_SEED=0 \
  incident-commander-env:latest
# OpenAPI: http://localhost:8000/docs · Web UI: http://localhost:8000/web

# OpenEnv CLI (build/validate/push — run from incident_commander/)
openenv build
openenv validate --verbose
openenv validate --url https://YOUR-USER-incident-commander-openenv.hf.space
openenv push --repo-id YOUR_USER/incident-commander-openenv

# Baseline inference (server must already be running at $ENV_URL, default http://localhost:8000)
IC_MOCK_POLICY=1 python inference.py            # scripted oracle, no API key, scores ~0.872
python inference.py                              # uses HF_TOKEN + MODEL_NAME from workspace .env

# Training (GPU; after installing [training])
python -m training.eval  --model unsloth/Qwen2.5-7B-Instruct --task easy_canary_regression --seeds 3
python -m training.train --task easy_canary_regression --steps 1500
python -m training.eval  --adapters ./checkpoints/latest --task easy_canary_regression --seeds 3
# Colab workflow: training/train_grpo_colab.ipynb
```

Tests: pytest is a dev extra but `incident_commander/tests/` is currently empty. Run `pytest` from `incident_commander/` once tests land; use `pytest tests/test_smoke.py::test_name -v` for a single test.

## Architecture

### One package, mapped into subdirectories
`incident_commander/pyproject.toml` declares the package `incident_commander` with `package-dir = { "incident_commander" = "." }` plus subpackages (`simulator`, `graders`, `server`, `training`) each also mapped to their top-level directory. Consequence: `server/app.py` and `server/incident_commander_environment.py` import models via a try/except that falls back from relative (`from ..models`) to flat (`from models`) — this lets the same files work both as an installed package and when run directly inside the Docker container's working dir. Preserve both import paths when editing server code.

### Request path (OpenEnv server)
`server/app.py` calls `openenv.core.env_server.http_server.create_app(_make_env, ICAction, ICObservation, ...)`. `_make_env` reads `IC_TASK_ID` / `IC_SEED` / `IC_STEP_BUDGET` **at session creation**, not at process start — but because `max_concurrent_envs=1`, a single process serves one task at a time. **To evaluate a different task, run a separate container/process**, not a second session.

### Determinism is a hard contract
All random draws flow through **one** seeded `random.Random` stored on the `Simulator`. No wall-clock time is read anywhere in the sim. NPCs (`simulator/npcs.py`) are deterministic FSMs, never LLM calls. The same action sequence produces bit-identical observations and bit-identical rewards — replay asserts this in smoke tests. Any change that introduces `time.time()`, `random.random()` (unseeded), or live network calls in the env/grader will break the submission's reproducibility requirement.

### Reward model (`graders/rubric.py` + `graders/postmortem_check.py`)
Six weighted components summing to `[0, 1]`: containment 0.25, MTTR 0.20, correct RCA 0.20, right mitigation 0.15, comms SLA 0.10, post-mortem 0.10. Fully programmatic — **no LLM-as-judge**. Anti-gaming guards are structural, not heuristic:
- `hold` credits mitigation only when ground-truth mitigation is `hold` (required for the medium third-party-outage variant).
- `status_page` has a 60s cooldown — a second update too soon **zeroes** the comms component.
- `diagnose` keeps only the best partial-credit score across the episode; service-only match = 0.5, service+tag = 1.0.
- Wrong mitigation first + right one later never credits.
- Invalid actions burn a step via `Simulator.advance_with_noop` — no free retries.

When adding tasks or faults, preserve these invariants or the grader becomes exploitable.

### Action/observation schema (`models.py`)
`ICAction` is a flat discriminated union on `op` (11 kinds). `ICObservation` carries always-on fields (alerts, dashboard, sim time, blast radius) plus query-scoped fields (log_samples, trace_spans, audit_events, external_status) that are only populated when the matching `query_*` op was called — this is intentional: information-gathering has a real step cost. Schema is also discoverable at `GET /schema` on a running server.

### inference.py is the contract boundary
`incident_commander/inference.py` is both the judges' eval harness and the single source of truth for `render_observation()` / `dict_to_action()`. The training rollout driver (`training/rollout.py`) is expected to reuse these helpers — do not fork them. The log format (`[START]` / `[STEP]×N` / `[END]`, exact decimal places, `done`/`success` lowercase, `error` literal `null`) is validated by the submission checklist regex and must not drift. `load_dotenv(override=True)` means workspace `.env` values beat empty shell exports.

### Config comes from env vars only
Env server session: `IC_TASK_ID`, `IC_SEED`, `IC_STEP_BUDGET`. Inference: `ENV_URL`, `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` (must have **"Make calls to Inference Providers"** scope), `MAX_STEPS`, `MY_ENV_TASK`, `IC_MOCK_POLICY`. No config files — adding one would fork config surface from the Docker/Space deployment path.

### Training stack placement
`training/` is **not** installed on the env server (separate `[training]` extra, GPU-only). The RL loop is TRL `GRPOTrainer` + Unsloth 4-bit LoRA base, trained against `easy_canary_regression` only (curriculum: train easy, eval all). Reward-hack defense in the loop: log each of the six components as a separate column; reject any checkpoint where RCA / mitigation / post-mortem did not improve, even if `total` did.
