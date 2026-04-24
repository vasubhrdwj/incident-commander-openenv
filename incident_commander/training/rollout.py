"""Closed-loop episode rollout driver.

Single source of truth for "how an LLM plays one Incident Commander episode".
Reused by eval (pre/post-training scoring) and by the GRPO training loop
(trajectory collection). Imports prompt + parsing from ``inference`` so
training and eval never drift on prompt format.

The ``generate`` callable is the only policy-specific piece. It takes an
OpenAI-style ``messages`` list and returns the assistant's completion string.
- For eval via the HF Router, pass a wrapper around ``openai.OpenAI``.
- For on-GPU training, pass a wrapper around ``model.generate()`` (Unsloth /
  transformers).

The env server is expected to be running at ``env_url``. For training on
Colab, that's a uvicorn subprocess on ``http://localhost:8000``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from incident_commander import IncidentCommanderEnv
from incident_commander.inference import (
    SYSTEM_PROMPT,
    action_str,
    dict_to_action,
    parse_action_json,
    render_observation,
)

GenerateFn = Callable[[list[dict]], str]


@dataclass
class StepRecord:
    """One step's worth of training data."""

    messages: list[dict]
    completion: str
    reward: float
    action_tag: str


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    score: float
    rewards: list[float]
    trajectory: list[StepRecord] = field(default_factory=list)
    error: Optional[str] = None


def run_rollout(
    *,
    generate: GenerateFn,
    env_url: str,
    max_steps: int = 25,
) -> EpisodeResult:
    """Run one episode under ``generate``; return full trajectory + score.

    Trajectory records preserve the exact ``messages`` / ``completion`` pairs
    so a training loop can re-tokenize them for log-prob computation without
    re-rendering observations.
    """
    rewards: list[float] = []
    history: list[str] = []
    trajectory: list[StepRecord] = []
    success = False
    error: Optional[str] = None
    steps_done = 0

    with IncidentCommanderEnv(base_url=env_url).sync() as client:
        result = client.reset()
        obs = result.observation

        for step_idx in range(1, max_steps + 1):
            steps_done = step_idx
            user = render_observation(obs, history)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            try:
                completion = generate(messages)
                action = dict_to_action(parse_action_json(completion))
            except Exception as e:
                error = f"action_selection_error: {type(e).__name__}: {e}"
                break

            try:
                result = client.step(action)
            except Exception as e:
                error = f"step_error: {type(e).__name__}: {e}"
                break

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            tag = action_str(action)
            history.append(
                f"{tag} -> reward={reward:.3f} t={result.observation.sim_time_sec}s "
                f"last={result.observation.last_action_result}"
            )
            trajectory.append(
                StepRecord(
                    messages=messages,
                    completion=completion,
                    reward=reward,
                    action_tag=tag,
                )
            )
            obs = result.observation
            if result.done:
                success = True
                break

    return EpisodeResult(
        success=success,
        steps=steps_done,
        score=sum(rewards),
        rewards=rewards,
        trajectory=trajectory,
        error=error,
    )
