"""
FastAPI application for the Incident Commander Environment.

Each WebSocket session gets its own :class:`IncidentCommanderEnvironment`
(see ``SUPPORTS_CONCURRENT_SESSIONS``), so server-side task selection must
happen in a factory. We read ``IC_TASK_ID`` / ``IC_SEED`` / ``IC_STEP_BUDGET``
from the process environment at session creation time.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover - surfaced during dependency resolution
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'."
    ) from e

from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

try:
    from ..models import ICAction, ICObservation
    from .incident_commander_environment import (
        DEFAULT_SEED,
        DEFAULT_TASK_ID,
        IncidentCommanderEnvironment,
    )
except ImportError:
    from models import ICAction, ICObservation
    from server.incident_commander_environment import (
        DEFAULT_SEED,
        DEFAULT_TASK_ID,
        IncidentCommanderEnvironment,
    )


def _make_env() -> IncidentCommanderEnvironment:
    """Factory called once per WebSocket session.

    Each session picks up ``IC_TASK_ID`` / ``IC_SEED`` / ``IC_STEP_BUDGET`` at
    creation time. For multi-task evaluation from ``inference.py``, run one
    container per task (the standard OpenEnv pattern).
    """
    task_id = os.getenv("IC_TASK_ID", DEFAULT_TASK_ID)
    seed = int(os.getenv("IC_SEED", str(DEFAULT_SEED)))
    step_budget_raw = os.getenv("IC_STEP_BUDGET")
    step_budget = int(step_budget_raw) if step_budget_raw else None
    return IncidentCommanderEnvironment(task_id=task_id, seed=seed, step_budget=step_budget)


app = create_app(
    _make_env,
    ICAction,
    ICObservation,
    env_name="incident_commander",
    max_concurrent_envs=1,
)


# --- /replay demo replay UI ----------------------------------------------
#
# A self-contained dark-theme HTML viewer that loads
# ``demo/episodes.json`` and plays back the three demo episodes step-by-step,
# animating the six-component rubric bars as actions land.
#
# Why ``/replay`` and not ``/web``: openenv-core's runtime base image ships a
# built-in playground UI at ``/web`` (auto-generated from the action schema).
# Our custom route would be shadowed by that mount in production deployments
# even though it works in older local installs. We use ``/replay`` to stay
# out of the framework's path namespace.
#
# Path resolution: ``Path(__file__).parent.parent`` resolves to the package
# root both when installed (``incident_commander/``) and in the Docker
# image (``/app/env``), because the Dockerfile copies the whole tree to
# ``/app/env`` and runs from there.

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEB_INDEX = _REPO_ROOT / "web" / "index.html"
_DEMO_EPISODES = _REPO_ROOT / "demo" / "episodes.json"


@app.get("/replay", response_class=HTMLResponse, include_in_schema=False)
def replay_ui() -> HTMLResponse:
    """Serve the demo replay viewer."""
    if not _WEB_INDEX.exists():
        return HTMLResponse(
            "<h1>UI not bundled</h1><p>web/index.html missing from this build.</p>",
            status_code=500,
        )
    return HTMLResponse(_WEB_INDEX.read_text(encoding="utf-8"))


@app.get("/replay/episodes.json", include_in_schema=False)
def replay_episodes() -> JSONResponse:
    """Serve the precomputed demo episodes for the replay viewer."""
    if not _DEMO_EPISODES.exists():
        return JSONResponse(
            {"error": "demo/episodes.json missing — run `python -m incident_commander.demo.run_episodes`"},
            status_code=404,
        )
    return FileResponse(_DEMO_EPISODES, media_type="application/json")


def main() -> None:
    """Entry point for `python -m server.app` and `uv run server`.

    Host and port come from ``HOST`` / ``PORT`` env vars so the same binary
    runs identically under `docker run -e PORT=...`, HF Spaces, and local
    dev. The validator matches the literal string ``main()`` to confirm the
    script has a callable entry point.
    """
    import os

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
