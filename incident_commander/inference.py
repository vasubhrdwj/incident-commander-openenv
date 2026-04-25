#!/usr/bin/env python3
"""
Baseline ``inference.py`` for the Incident Commander OpenEnv environment.

Contract (from the OpenEnv submission checklist):

* Uses the ``openai`` client only — no ``requests``, ``httpx``, ``anthropic``
  SDK, or ``transformers`` pipeline for LLM calls.
* Reads ``API_BASE_URL`` / ``MODEL_NAME`` / ``HF_TOKEN`` (falls back to
  ``API_KEY``) / ``ENV_URL`` from environment variables. Loads ``.env`` from
  the repository root (parent of this package) and from this package directory
  if those files exist.
* Emits exactly one ``[START]`` line, one ``[STEP]`` line per env step, and
  exactly one ``[END]`` line — emitted even when an exception occurs (the
  ``log_end`` call is wrapped in a ``finally`` block). Decimal precision,
  lowercase bools, and field order all follow the checklist verbatim.

Additional knobs:

* ``MY_ENV_TASK`` / ``IC_TASK_ID`` — which of the three task scenarios to run.
* ``MAX_STEPS`` — hard cap on env steps this episode (default 25).
* ``IC_MOCK_POLICY=1`` — swap the LLM call for a scripted oracle policy. The
  oracle scores ~0.87 on the canary task and is handy for validating the
  log-format pipeline without spending API credits. Disabled by default; the
  real submission always uses the LLM path.

The server must already be running at ``ENV_URL`` with matching ``IC_TASK_ID``
(one container per task — the standard OpenEnv multi-task pattern).
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
# explicit paths: find_dotenv() can miss a repo-root .env; override=True lets
# file values win over e.g. an empty ``export HF_TOKEN=`` in the shell.
# python-dotenv is a soft dep — if missing, fall through to plain env vars.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    for _env_path in (os.path.join(_PARENT, ".env"), os.path.join(_HERE, ".env")):
        if os.path.isfile(_env_path):
            load_dotenv(_env_path, override=True)
except ImportError:
    pass
if _PARENT and _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from openai import OpenAI  # noqa: E402

from incident_commander import ICAction, ICObservation, IncidentCommanderEnv  # noqa: E402

ENV_NAME = "incident_commander"
DEFAULT_TASK = "easy_canary_regression"
DEFAULT_MAX_STEPS = 25
DEFAULT_ENV_URL = "http://localhost:8000"
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_field = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={_bool(done)} error={error_field}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool(success)} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _bool(b: bool) -> str:
    return "true" if b else "false"


SYSTEM_PROMPT = """You are an on-call Incident Commander (IC) for a fintech microservices product.
Job: detect → diagnose → mitigate → communicate → write a post-mortem. The episode
ends when you call resolve, postmortem, or run out of step budget.

VALID SERVICES (use exactly these names — the topology never changes):
  api_gw, auth, payments, orders, inventory, notifications

Topology shape (so you don't blame the wrong service):
  api_gw      → fans out to auth, payments, orders
  orders      → depends on inventory and payments  (payments outages cascade here)
  payments    → talks to the external provider 'stripe'
  auth, inventory, notifications  — leaves with no downstream IC concerns

VALID FAULT TAGS (use exactly one of these for diagnose / postmortem):
  bad_deploy           — recent canary or deploy spiked latency/error rate.
  upstream_third_party — external provider (e.g. stripe) is degraded.
  bad_integration      — our integration to an external provider is misconfigured.
  data_corruption      — silent: dashboard healthy, audit shows anomalous migrations.

ACTIONS (emit ONE as JSON; include exactly the fields listed):
- {"op": "query_metrics", "service": "<svc>"}
- {"op": "query_logs",    "service": "<svc>", "since_sec": 0}
- {"op": "query_trace",   "service": "<svc>"}                 // or {"trace_id": "..."}
- {"op": "query_audit"}                                       // optional: "service", "since_sec"
- {"op": "query_external_status", "target": "stripe"}         // 'stripe' is the only provider
- {"op": "delegate", "role": "sre|security|comms|eng_lead", "task": "<one sentence>"}
- {"op": "mitigate", "mitigation": "rollback|partial_rollback|restart|scale|feature_flag|hold", "target": "<svc-or-flag>"}
- {"op": "communicate", "channel": "status_page|customer_email|exec_update", "message": "<short>", "cohort": "<optional cohort id>"}
- {"op": "diagnose", "root_cause_service": "<svc>", "root_cause_tag": "<tag>"}
- {"op": "resolve"}
- {"op": "postmortem", "postmortem_json": {
    "summary": "<20+ char summary>",
    "root_cause_service": "<svc>",
    "root_cause_tag": "<tag>",
    "timeline": ["event 1", "event 2"],
    "actions_taken": ["action 1", "action 2"]
  }}

PLAYBOOK (task-agnostic):
 1. Pull the relevant signal — query_metrics for visible outages,
    query_audit for silent / dashboard-green incidents.
 2. Delegate to a specialist (sre for deploys, eng_lead for integrations,
    security for data anomalies) to correlate the signal with recent changes.
 3. Communicate within the SLA window using the right channel:
      - status_page  → customer-visible outages (bad_deploy, upstream_third_party,
                        bad_integration).
      - customer_email + cohort → silent incidents that affect a specific cohort
                        (data_corruption).
 4. Diagnose with both root_cause_service AND root_cause_tag.
 5. Mitigate using the mitigation appropriate to the fault tag:
      - bad_deploy           → rollback   on the affected service.
      - upstream_third_party → hold       (no target needed; wait for provider).
      - bad_integration      → feature_flag to the backup processor (target name
                                like "<service>_backup_processor", or just the
                                service name — the env accepts either).
      - data_corruption      → partial_rollback on the affected service.
 6. Postmortem with all five required fields filled.

Output STRICTLY one JSON object. No prose, no markdown fences, no commentary.
"""


def render_observation(obs: ICObservation, history: list[str]) -> str:
    """Compact text summary of the observation the LLM sees each turn."""
    dashboard = "\n".join(
        f"  {name}: latency_p99={h.latency_p99_ms}ms error_rate={h.error_rate:.3%} "
        f"rps={h.requests_per_sec:.0f} healthy={h.healthy}"
        for name, h in obs.dashboard.items()
    )
    alerts = (
        "\n".join(f"  [{a.severity}] {a.service}: {a.message}" for a in obs.alerts)
        if obs.alerts
        else "  (none)"
    )
    logs = (
        "\n".join(
            f"  t={l.timestamp_sec}s [{l.level}] {l.service}: {l.message}"
            for l in obs.log_samples[:8]
        )
        if obs.log_samples
        else "  (none)"
    )
    traces = (
        "\n".join(
            f"  {s.service}.{s.operation} dur={s.duration_ms}ms error={s.error}"
            for s in obs.trace_spans[:6]
        )
        if obs.trace_spans
        else "  (none)"
    )
    audit = (
        "\n".join(
            f"  t={e.timestamp_sec}s {e.actor} {e.action} -> {e.resource} anomalous={e.anomalous}"
            for e in obs.audit_events[:6]
        )
        if obs.audit_events
        else "  (none)"
    )
    external = (
        f"  {obs.external_status.provider}: {obs.external_status.status} — {obs.external_status.message}"
        if obs.external_status
        else "  (not queried)"
    )
    npc = (
        "\n".join(
            f"  [{r.role} @ t={r.received_at_sec}s] {r.findings}" for r in obs.npc_reports[-3:]
        )
        if obs.npc_reports
        else "  (none)"
    )
    chat = (
        "\n".join(f"  [{m.channel} @ t={m.timestamp_sec}s] {m.body}" for m in obs.chat_feed[-3:])
        if obs.chat_feed
        else "  (none)"
    )
    hist = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history[-8:])) or "  (none)"

    return f"""Task: {obs.task_id}
Sim time: {obs.sim_time_sec}s  Blast radius: {obs.blast_radius_pct:.1%}  \
Revenue loss: ${obs.revenue_loss_usd:.0f}  Steps left: {obs.step_budget_remaining}

Last action result: {obs.last_action_result}

Alerts:
{alerts}

Service dashboard:
{dashboard}

Recent logs (only populated when queried):
{logs}

Recent traces (only populated when queried):
{traces}

Audit events (only populated when queried):
{audit}

External status (only populated when queried):
{external}

NPC reports so far:
{npc}

Chat feed:
{chat}

Your recent action history:
{hist}

Now output the next JSON action."""


_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")
_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_action_json(text: str) -> dict:
    """Extract the first well-formed JSON object from the LLM's response.

    Handles three common small-model failure modes:
      1. raw JSON (the happy path);
      2. JSON wrapped in a ``\u0060\u0060\u0060json … \u0060\u0060\u0060`` markdown fence;
      3. JSON with prose preamble — fall back to first ``{`` … last ``}`` slice.
    """
    text = text.strip()

    if text.startswith("```"):
        text = _FENCE_OPEN_RE.sub("", text)
        text = _FENCE_CLOSE_RE.sub("", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = _FENCE_BLOCK_RE.search(text)
    if fence:
        return json.loads(fence.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


def dict_to_action(raw: dict) -> ICAction:
    """Turn a parsed JSON dict into an :class:`ICAction`."""
    if "op" not in raw:
        raise ValueError("model output missing 'op'")
    return ICAction(**raw)


def action_str(action: ICAction) -> str:
    """Compact single-token representation for ``[STEP] action=...`` logging."""
    op = action.op
    bits: list[str] = []
    if action.service:
        bits.append(f"svc={action.service}")
    if action.role:
        bits.append(f"role={action.role}")
    if action.mitigation:
        bits.append(f"mit={action.mitigation}")
    if action.target and action.target != action.service:
        bits.append(f"target={action.target}")
    if action.channel:
        bits.append(f"ch={action.channel}")
    if action.root_cause_service or action.root_cause_tag:
        bits.append(
            f"rc={action.root_cause_service or '?'}:{action.root_cause_tag or '?'}"
        )
    suffix = f"[{','.join(bits)}]" if bits else ""
    return f"{op}{suffix}"


@dataclass
class PolicyFn:
    """Callable producing the next :class:`ICAction` given obs + history."""

    fn: Callable[[ICObservation, list[str]], ICAction]


def llm_policy(llm: OpenAI, model: str) -> PolicyFn:
    """Return a policy that calls the LLM for every step.

    Each step is allowed two attempts: a deterministic greedy decode first,
    then a higher-temperature retry if the greedy output failed to parse or
    validate. The retry is fed the parser's error message so the model can
    self-correct (a single re-prompt closes most format-failure modes on
    small base models, per the Phase 1 fix in the recovery plan).
    """

    def _call(obs: ICObservation, history: list[str]) -> ICAction:
        user = render_observation(obs, history)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            resp = llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0 if attempt == 0 else 0.4,
                max_tokens=1024,
                stream=False,
            )
            content = resp.choices[0].message.content or ""
            try:
                return dict_to_action(parse_action_json(content))
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                f"Your previous output was not a valid action: {exc}. "
                                "Output exactly one JSON object matching the action grammar. "
                                "No markdown, no commentary."
                            ),
                        },
                    ]
        # Both attempts failed — surface the last parse error to the caller.
        raise last_exc if last_exc is not None else RuntimeError("llm_policy: unreachable")

    return PolicyFn(fn=_call)


def mock_oracle_policy() -> PolicyFn:
    """Scripted policy for pipeline validation. Mirrors an ideal canary-task run."""
    script: list[ICAction] = [
        ICAction(op="query_metrics", service="payments"),
        ICAction(op="delegate", role="sre", task="investigate payments regression after canary deploy"),
        ICAction(op="communicate", channel="status_page", message="Investigating elevated errors on payments; update in 15m."),
        ICAction(op="diagnose", root_cause_service="payments", root_cause_tag="bad_deploy"),
        ICAction(op="mitigate", mitigation="rollback", target="payments"),
        ICAction(
            op="postmortem",
            postmortem_json={
                "summary": (
                    "Canary deploy canary-v2.4.1 on payments caused 2x baseline error rate; "
                    "rolled back and traffic restored."
                ),
                "root_cause_service": "payments",
                "root_cause_tag": "bad_deploy",
                "timeline": [
                    "t=30s canary canary-v2.4.1 rolled out to 5% traffic",
                    "t=30s error rate on payments breached SLO",
                    "t=120s rollback issued",
                    "t=180s metrics back to baseline",
                ],
                "actions_taken": [
                    "Queried metrics on payments",
                    "Delegated to SRE for deploy correlation",
                    "Posted status_page update",
                    "Executed rollback on payments",
                ],
                "sub_agent_summary": (
                    "SRE confirmed canary regression; Eng Lead confirmed blast radius contained."
                ),
            },
        ),
    ]
    idx = [0]

    def _call(obs: ICObservation, history: list[str]) -> ICAction:
        i = idx[0]
        idx[0] = min(i + 1, len(script) - 1)
        return script[i]

    return PolicyFn(fn=_call)


def run_episode(
    *,
    policy: PolicyFn,
    env_url: str,
    task: str,
    model: str,
    max_steps: int,
) -> tuple[bool, int, float, list[float], Optional[str]]:
    """Run a single episode under ``policy``; returns (success, steps, score, rewards, error)."""
    rewards: list[float] = []
    history: list[str] = []
    success = False
    steps_done = 0
    error_message: Optional[str] = None

    with IncidentCommanderEnv(base_url=env_url).sync() as client:
        result = client.reset()
        obs = result.observation

        for step_idx in range(1, max_steps + 1):
            steps_done = step_idx
            try:
                action = policy.fn(obs, history)
            except Exception as action_err:
                err = f"action_selection_error: {type(action_err).__name__}: {action_err}"
                log_step(step_idx, "parse_error", 0.0, False, err)
                error_message = err
                break

            action_tag = action_str(action)
            try:
                result = client.step(action)
            except Exception as step_err:
                err = f"step_error: {type(step_err).__name__}: {step_err}"
                log_step(step_idx, action_tag, 0.0, False, err)
                error_message = err
                break

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            history.append(
                f"{action_tag} -> reward={reward:.3f} t={result.observation.sim_time_sec}s "
                f"last={result.observation.last_action_result}"
            )
            log_step(step_idx, action_tag, reward, result.done, None)

            obs = result.observation
            if result.done:
                success = True
                break

    score = sum(rewards)
    return success, steps_done, score, rewards, error_message


def main() -> int:
    api_base = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model = os.getenv("MODEL_NAME", " ")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    env_url = os.getenv("ENV_URL", DEFAULT_ENV_URL)
    task = os.getenv("MY_ENV_TASK") or os.getenv("IC_TASK_ID") or DEFAULT_TASK
    max_steps = int(os.getenv("MAX_STEPS", str(DEFAULT_MAX_STEPS)))
    use_mock = os.getenv("IC_MOCK_POLICY", "").lower() in {"1", "true", "yes"}

    if not use_mock and not api_key:
        print(
            "[ERROR] HF_TOKEN (or API_KEY) must be set when IC_MOCK_POLICY is off",
            file=sys.stderr,
            flush=True,
        )
        log_start(task, ENV_NAME, model)
        log_end(False, 0, 0.0, [])
        return 2

    if use_mock:
        policy = mock_oracle_policy()
        log_model = f"mock:{model or 'oracle'}"
    else:
        llm = OpenAI(base_url=api_base, api_key=api_key)
        policy = llm_policy(llm, model)
        log_model = model or "(unset)"

    log_start(task, ENV_NAME, log_model)

    success = False
    steps_done = 0
    score = 0.0
    rewards: list[float] = []
    error_message: Optional[str] = None
    try:
        success, steps_done, score, rewards, error_message = run_episode(
            policy=policy,
            env_url=env_url,
            task=task,
            model=log_model,
            max_steps=max_steps,
        )
    except Exception:
        error_message = "unhandled_exception"
        traceback.print_exc(file=sys.stderr)
    finally:
        log_end(success, steps_done, score, rewards)
        if error_message is not None:
            print(f"[DEBUG] error: {error_message}", file=sys.stderr, flush=True)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
