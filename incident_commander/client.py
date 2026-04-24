"""
Incident Commander Environment — typed client.

``IncidentCommanderEnv`` is the thin Pydantic / WebSocket client that agents
(and the baseline ``inference.py``) use to drive the environment.

Usage::

    from incident_commander import IncidentCommanderEnv, ICAction

    with IncidentCommanderEnv(base_url="http://localhost:8000") as client:
        result = client.reset()
        result = client.step(ICAction(op="query_metrics", service="payments"))
        result = client.step(
            ICAction(op="delegate", role="sre", task="correlate payments latency to recent deploys")
        )
"""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ICAction, ICObservation


class IncidentCommanderEnv(EnvClient[ICAction, ICObservation, State]):
    """WebSocket client for the Incident Commander environment.

    Each instance holds its own server-side session (see
    ``IncidentCommanderEnvironment.SUPPORTS_CONCURRENT_SESSIONS``), so
    independent agents/evaluations never cross-contaminate episodes.
    """

    def _step_payload(self, action: ICAction) -> Dict:
        """Serialize an :class:`ICAction` for transport.

        ``exclude_none=True`` keeps the wire payload compact — the env's
        ``step`` validates per-op required fields and defaults the rest.
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ICObservation]:
        """Parse a ``step`` or ``reset`` server response into a typed result."""
        obs_data = dict(payload.get("observation", {}))
        if "done" in payload:
            obs_data["done"] = payload["done"]
        if "reward" in payload:
            obs_data["reward"] = payload["reward"]
        observation = ICObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse a ``state`` server response into the OpenEnv :class:`State`."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
