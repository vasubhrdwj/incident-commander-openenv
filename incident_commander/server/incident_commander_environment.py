"""
Incident Commander Environment — thin OpenEnv adapter over :class:`Simulator`.

The server-side class is intentionally small: it implements the OpenEnv
``reset`` / ``step`` / ``state`` surface and delegates every piece of world
logic to :class:`~incident_commander.simulator.Simulator`. Per-op payload
validation lives here so that clearly-malformed actions surface a helpful
message without poking the simulator.

Per-session task selection:
    Each WebSocket session creates a fresh :class:`IncidentCommanderEnvironment`
    (see ``SUPPORTS_CONCURRENT_SESSIONS``), so the server-side ``app`` factory
    reads ``IC_TASK_ID`` / ``IC_SEED`` / ``IC_STEP_BUDGET`` from the process
    environment and threads them through the constructor. For multi-task
    evaluation from ``inference.py`` we spin up one container per task — the
    standard OpenEnv pattern.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..graders import RubricGrader  # type: ignore[import-not-found]
    from ..models import ICAction, ICObservation  # type: ignore[import-not-found]
    from ..simulator import Simulator, get_task_config  # type: ignore[import-not-found]
except ImportError:
    from graders import RubricGrader  # type: ignore[import-not-found, no-redef]
    from models import ICAction, ICObservation  # type: ignore[import-not-found, no-redef]
    from simulator import Simulator, get_task_config  # type: ignore[import-not-found, no-redef]


DEFAULT_TASK_ID: str = "easy_canary_regression"
DEFAULT_SEED: int = 0


class IncidentCommanderEnvironment(Environment):
    """On-call IC coordinating a simulated microservices outage.

    The env itself owns almost no logic — it holds a :class:`Simulator`,
    forwards actions to it, and lifts the rendered observations back out.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        task_id: str = DEFAULT_TASK_ID,
        seed: int = DEFAULT_SEED,
        step_budget: Optional[int] = None,
    ) -> None:
        task = get_task_config(task_id)
        if step_budget is not None:
            task = _with_step_budget(task, step_budget)
        self._task_id = task.task_id
        self._seed = seed
        self._sim = Simulator(task=task, seed=seed)
        self._grader = RubricGrader()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> ICObservation:
        """Start a fresh episode with a deterministic initial state."""
        self._sim.reset()
        self._grader.reset()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return self._sim.initial_observation()

    def step(self, action: ICAction) -> ICObservation:  # type: ignore[override]
        """Advance one step and attach the rubric-derived incremental reward.

        Pipeline:
            1. Validate per-op payload (cheap, env-level).
            2. If invalid, no-op the simulator clock forward so wasted steps
               still cost step budget.
            3. Otherwise delegate to :class:`Simulator.step`.
            4. Ask the grader for the incremental reward since the last step;
               if the episode is ending, also collect the terminal components.
        """
        self._state.step_count += 1

        err = _validate_action_payload(action)
        if err is not None:
            obs = self._sim.advance_with_noop(status=f"invalid {action.op}: {err}")
        else:
            obs, _ = self._sim.step(action)

        step_reward = self._grader.observe_step(self._sim)
        if obs.done:
            step_reward += self._grader.observe_terminal(self._sim)
        obs.reward = round(step_reward, 6)
        return obs

    @property
    def state(self) -> State:
        """Current environment state (episode id + step count)."""
        return self._state


def _with_step_budget(task, step_budget: int):
    """Return a copy of ``task`` with an overridden step budget."""
    from dataclasses import replace

    return replace(task, step_budget=step_budget)


def _validate_action_payload(action: ICAction) -> Optional[str]:
    """Cheap env-level payload validation — returns an error string or ``None``."""
    op = action.op
    if op in ("query_logs", "query_metrics"):
        if not action.service:
            return "missing 'service'"
        return None
    if op == "query_trace":
        if not action.service and not action.trace_id:
            return "must provide at least one of 'service' or 'trace_id'"
        return None
    if op == "query_audit":
        return None
    if op == "query_external_status":
        if not action.target:
            return "missing 'target' (provider name)"
        return None
    if op == "delegate":
        if not action.role:
            return "missing 'role'"
        if not action.task:
            return "missing 'task'"
        return None
    if op == "mitigate":
        if not action.mitigation:
            return "missing 'mitigation'"
        if action.mitigation != "hold" and not action.target:
            return "missing 'target' for non-hold mitigation"
        return None
    if op == "communicate":
        if not action.channel:
            return "missing 'channel'"
        if not action.message:
            return "missing 'message'"
        return None
    if op == "diagnose":
        if not action.root_cause_service:
            return "missing 'root_cause_service'"
        if not action.root_cause_tag:
            return "missing 'root_cause_tag'"
        return None
    if op == "resolve":
        return None
    if op == "postmortem":
        if not action.postmortem_json:
            return "missing 'postmortem_json'"
        return None
    return f"unknown op '{op}'"
