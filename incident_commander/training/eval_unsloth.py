"""In-process evaluation harness for Unsloth-loaded models.

What it does
------------
Given a HuggingFace ``model`` + ``tokenizer`` (typically loaded with
:func:`unsloth.FastLanguageModel.from_pretrained` and optionally a LoRA
adapter on top), runs N seeded episodes per task and returns per-task
aggregate stats — including the same six rubric components the grader
exposes, so the regression gate in ``train_sft.py`` can detect "total went
up but RCA went down" gaming.

Why in-process
--------------
Same reason ``build_sft_dataset.py`` is in-process: we drive
:class:`IncidentCommanderEnvironment` directly. Eliminates uvicorn /
port-management overhead, and most importantly avoids loading the model
weights into a subprocess (we keep them in the trainer's GPU memory). The
single seeded RNG inside the simulator gives bit-identical observations to
the HTTP path, and the trainer never needs to leave the Python process.

Why a parse-retry
-----------------
The Phase-2 SFT recipe still inherits Phase 1's ``parse_action_json`` +
``llm_policy`` retry logic. We mirror that here: greedy decode first, then
one stochastic retry with the parser's error fed back as a corrective user
turn. This is what makes a partially-trained model recoverable instead of
zero-scoring on the first malformed token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from incident_commander.inference import (
    SYSTEM_PROMPT,
    action_str,
    dict_to_action,
    parse_action_json,
    render_observation,
)
from incident_commander.models import ICAction, ICObservation
from incident_commander.server.incident_commander_environment import (
    IncidentCommanderEnvironment,
)


TASKS = (
    "easy_canary_regression",
    "medium_third_party_attribution",
    "hard_silent_data_corruption",
)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class EvalEpisode:
    """One ``(task_id, seed)`` rollout's worth of measurements."""

    task: str
    seed: int
    score: float
    steps: int
    done: bool
    error: Optional[str] = None
    components: Dict[str, float] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)


@dataclass
class TaskAggregate:
    """Per-task aggregate over ``N`` seeded episodes."""

    task: str
    n: int
    mean: float
    std: float
    scores: List[float]
    component_means: Dict[str, float]
    errors: int
    done_rate: float


# --------------------------------------------------------------------------- #
# Generation: tokenizer chat template + Unsloth model.generate
# --------------------------------------------------------------------------- #


def _generate_once(
    model: Any,
    tokenizer: Any,
    messages: List[dict],
    *,
    do_sample: bool,
    temperature: float,
    max_new_tokens: int,
) -> str:
    """One forward pass of ``model.generate`` from ``messages``."""
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    gen_kwargs: Dict[str, Any] = {
        "input_ids": inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
    # Unsloth / transformers: generate returns prompt + completion concatenated.
    output = model.generate(**gen_kwargs)
    completion_ids = output[0, inputs.shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def _policy_step(
    model: Any,
    tokenizer: Any,
    obs: ICObservation,
    history: List[str],
    *,
    max_new_tokens: int,
    retry_temperature: float,
) -> ICAction:
    """Decode one action from ``model``; retry once on parse failure.

    Mirrors ``inference.llm_policy`` so trained behaviour transfers cleanly
    to the judges' eval harness without re-prompting differences.
    """
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_observation(obs, history)},
    ]
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        completion = _generate_once(
            model,
            tokenizer,
            messages,
            do_sample=(attempt > 0),
            temperature=retry_temperature,
            max_new_tokens=max_new_tokens,
        )
        try:
            return dict_to_action(parse_action_json(completion))
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": completion},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous output was not a valid action: {exc}. "
                            "Output exactly one JSON object matching the action grammar. "
                            "No markdown, no commentary."
                        ),
                    },
                ]
    raise last_exc if last_exc is not None else RuntimeError("eval: unreachable")


# --------------------------------------------------------------------------- #
# Episode-level runner
# --------------------------------------------------------------------------- #


def eval_episode(
    model: Any,
    tokenizer: Any,
    *,
    task_id: str,
    seed: int,
    max_steps: int = 25,
    max_new_tokens: int = 512,
    retry_temperature: float = 0.4,
) -> EvalEpisode:
    """Run one episode and return its score + rubric breakdown."""
    env = IncidentCommanderEnvironment(task_id=task_id, seed=seed)
    obs = env.reset()
    history: List[str] = []
    actions: List[str] = []
    score = 0.0
    steps = 0
    done = False
    error: Optional[str] = None

    for step_idx in range(1, max_steps + 1):
        steps = step_idx
        try:
            action = _policy_step(
                model,
                tokenizer,
                obs,
                history,
                max_new_tokens=max_new_tokens,
                retry_temperature=retry_temperature,
            )
        except Exception as exc:
            error = f"action_selection_error: {type(exc).__name__}: {exc}"
            break

        try:
            next_obs = env.step(action)
        except Exception as exc:
            error = f"step_error: {type(exc).__name__}: {exc}"
            break

        reward = float(next_obs.reward or 0.0)
        score += reward
        tag = action_str(action)
        actions.append(tag)
        history.append(f"{tag} -> reward={reward:.3f}")
        if next_obs.done:
            done = True
            break
        obs = next_obs

    components = env._grader.score.as_dict() if hasattr(env, "_grader") else {}
    return EvalEpisode(
        task=task_id,
        seed=seed,
        score=score,
        steps=steps,
        done=done,
        error=error,
        components=components,
        actions=actions,
    )


# --------------------------------------------------------------------------- #
# Aggregation across (task, seed) grid
# --------------------------------------------------------------------------- #


def _stddev(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def eval_task(
    model: Any,
    tokenizer: Any,
    task_id: str,
    *,
    seeds: int = 3,
    max_steps: int = 25,
    max_new_tokens: int = 512,
    retry_temperature: float = 0.4,
    label: str = "",
) -> TaskAggregate:
    """Run ``seeds`` episodes (seed=0..seeds-1) for one task and aggregate."""
    episodes: List[EvalEpisode] = []
    for s in range(seeds):
        ep = eval_episode(
            model,
            tokenizer,
            task_id=task_id,
            seed=s,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            retry_temperature=retry_temperature,
        )
        episodes.append(ep)
        prefix = f"  [{label}] " if label else "  "
        print(
            f"{prefix}{task_id} seed={s} score={ep.score:.3f} steps={ep.steps} "
            f"done={ep.done} err={ep.error}",
            flush=True,
        )

    scores = [ep.score for ep in episodes]
    component_keys = ("containment", "mttr", "rca", "mitigation", "comms", "postmortem")
    component_means: Dict[str, float] = {
        k: (
            sum(ep.components.get(k, 0.0) for ep in episodes) / max(1, len(episodes))
        )
        for k in component_keys
    }
    return TaskAggregate(
        task=task_id,
        n=len(episodes),
        mean=sum(scores) / max(1, len(scores)),
        std=_stddev(scores),
        scores=scores,
        component_means=component_means,
        errors=sum(1 for ep in episodes if ep.error is not None),
        done_rate=(sum(1 for ep in episodes if ep.done) / max(1, len(episodes))),
    )


def eval_all_tasks(
    model: Any,
    tokenizer: Any,
    *,
    seeds: int = 3,
    max_steps: int = 25,
    max_new_tokens: int = 512,
    retry_temperature: float = 0.4,
    label: str = "",
) -> Dict[str, dict]:
    """Run ``seeds`` episodes per task across all 3 tasks. Returns plain dicts."""
    out: Dict[str, dict] = {}
    for task_id in TASKS:
        agg = eval_task(
            model,
            tokenizer,
            task_id,
            seeds=seeds,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            retry_temperature=retry_temperature,
            label=label,
        )
        out[task_id] = {
            "n": agg.n,
            "mean": agg.mean,
            "std": agg.std,
            "scores": agg.scores,
            "components": agg.component_means,
            "errors": agg.errors,
            "done_rate": agg.done_rate,
        }
    return out


__all__ = [
    "TASKS",
    "EvalEpisode",
    "TaskAggregate",
    "eval_episode",
    "eval_task",
    "eval_all_tasks",
]
