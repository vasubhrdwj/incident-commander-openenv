"""Multi-episode evaluation — shared pre-training / post-training harness.

Runs N episodes through the env server and reports mean ± std of episode
score. Episodes are *semantically independent* (each starts from
``client.reset()``), but because ``IC_SEED`` is a process-level env var on
the server, same-server runs replay the same scenario. Variance comes from
sampling temperature on the generator side.

Used by:
- ``train_grpo.py`` to record pre- and post-training baselines inside the
  same Colab process (single source of truth — no drift between eval and
  training generator).
- Standalone CLI for quick local sanity checks (``python -m
  incident_commander.training.eval``).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Optional

from .rollout import GenerateFn, run_rollout


@dataclass
class EvalSummary:
    label: str
    n: int
    scores: list[float]
    mean: float
    std: float
    errors: int

    def __str__(self) -> str:
        return (
            f"{self.label}: n={self.n} mean={self.mean:.3f} ± {self.std:.3f} "
            f"errors={self.errors} scores={[round(s, 3) for s in self.scores]}"
        )


def evaluate(
    *,
    generate: GenerateFn,
    env_url: str,
    n_episodes: int = 3,
    max_steps: int = 25,
    label: str = "eval",
    verbose: bool = True,
) -> EvalSummary:
    """Run ``n_episodes`` rollouts and aggregate scores.

    Episodes that error out (``result.error is not None``) count as 0.0 in the
    mean but are also tallied separately under ``errors`` so a run of all
    parse-errors doesn't silently look like "the model scored 0.0".
    """
    scores: list[float] = []
    errors = 0
    for i in range(n_episodes):
        result = run_rollout(generate=generate, env_url=env_url, max_steps=max_steps)
        if result.error is not None:
            errors += 1
        scores.append(result.score)
        if verbose:
            print(
                f"[{label}] ep={i + 1}/{n_episodes} steps={result.steps} "
                f"score={result.score:.3f} done={result.success} "
                f"error={result.error}",
                flush=True,
            )

    m = mean(scores) if scores else 0.0
    s = stdev(scores) if len(scores) > 1 else 0.0
    summary = EvalSummary(
        label=label, n=len(scores), scores=scores, mean=m, std=s, errors=errors
    )
    if verbose:
        print(f"[{label}] SUMMARY {summary}", flush=True)
    return summary
