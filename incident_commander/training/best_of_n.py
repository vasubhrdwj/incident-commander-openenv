"""Best-of-N reward-guided sampling — post-training evidence path.

Plan §4.5.7 fallback. GRPO on `easy_canary_regression` regressed (see plan
`training-run` note): the 3B base already scored 0.736 / 0.872 oracle, leaving
almost no RL headroom, and the advantage signal got dominated by JSON schema
errors. Best-of-N runs N whole episodes at higher temperature and picks the
max-reward one — the rubric is the verifier. It is NOT trained weights; it is
inference-time reward-guided selection against a verifiable reward, which is
the exact framing plan §4.5.7 sanctioned.

Reuses ``training.rollout.run_rollout`` so the prompt / parsing / log format
never drift from eval or from ``inference.py``.

Usage::

    # server must already be running at $ENV_URL (default http://localhost:8000)
    python -m incident_commander.training.best_of_n \\
        --n 8 --temperature 0.9 --task easy_canary_regression

Env vars match ``inference.py``: ``API_BASE_URL``, ``MODEL_NAME``, ``HF_TOKEN``
(or ``API_KEY``), ``ENV_URL``, ``IC_TASK_ID``, ``MAX_STEPS``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from statistics import mean, stdev
from typing import Optional

from openai import OpenAI

from .rollout import EpisodeResult, run_rollout

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_ENV_URL = "http://localhost:8000"
DEFAULT_TASK = "easy_canary_regression"
DEFAULT_MAX_STEPS = 25
DEFAULT_N = 8
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 600


@dataclass
class BoNSummary:
    task: str
    model: str
    n: int
    temperature: float
    scores: list[float]
    mean: float
    std: float
    max_score: float
    max_index: int
    errors: int

    def pretty(self) -> str:
        return (
            f"task={self.task} model={self.model} n={self.n} T={self.temperature}\n"
            f"  scores = {[round(s, 3) for s in self.scores]}\n"
            f"  mean = {self.mean:.3f}  std = {self.std:.3f}\n"
            f"  best = {self.max_score:.3f}  (episode index {self.max_index})\n"
            f"  errors = {self.errors}"
        )


def _build_generate(client: OpenAI, model: str, temperature: float):
    def generate(messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_TOKENS,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    return generate


def run_best_of_n(
    *,
    client: OpenAI,
    model: str,
    env_url: str,
    task: str,
    n: int,
    temperature: float,
    max_steps: int,
    verbose: bool = True,
) -> tuple[BoNSummary, list[EpisodeResult]]:
    """Run ``n`` independent episodes, return summary + each episode's result."""
    generate = _build_generate(client, model, temperature)
    results: list[EpisodeResult] = []
    scores: list[float] = []
    errors = 0

    for i in range(n):
        r = run_rollout(generate=generate, env_url=env_url, max_steps=max_steps)
        results.append(r)
        scores.append(r.score)
        if r.error is not None:
            errors += 1
        if verbose:
            print(
                f"[bon] ep={i + 1}/{n} steps={r.steps} score={r.score:.3f} "
                f"done={r.success} error={r.error}",
                flush=True,
            )

    max_score = max(scores) if scores else 0.0
    max_index = scores.index(max_score) if scores else -1
    summary = BoNSummary(
        task=task,
        model=model,
        n=n,
        temperature=temperature,
        scores=scores,
        mean=mean(scores) if scores else 0.0,
        std=stdev(scores) if len(scores) > 1 else 0.0,
        max_score=max_score,
        max_index=max_index,
        errors=errors,
    )
    if verbose:
        print(f"[bon] SUMMARY\n{summary.pretty()}", flush=True)
    return summary, results


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (
        os.path.join(os.path.dirname(os.path.dirname(here)), ".env"),
        os.path.join(os.path.dirname(here), ".env"),
    ):
        if os.path.isfile(p):
            load_dotenv(p, override=True)


def main() -> int:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="Best-of-N reward-guided sampling")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--task", default=os.getenv("IC_TASK_ID", DEFAULT_TASK))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", DEFAULT_MAX_STEPS)))
    parser.add_argument("--env-url", default=os.getenv("ENV_URL", DEFAULT_ENV_URL))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", ""))
    parser.add_argument("--api-base", default=os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL))
    parser.add_argument(
        "--output-json",
        default=None,
        help="If set, write the summary + per-episode scores to this JSON file",
    )
    args = parser.parse_args()

    api_key: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not api_key:
        print("[ERROR] HF_TOKEN or API_KEY must be set", file=sys.stderr, flush=True)
        return 2
    if not args.model:
        print("[ERROR] MODEL_NAME env var or --model flag required", file=sys.stderr, flush=True)
        return 2

    client = OpenAI(base_url=args.api_base, api_key=api_key)

    summary, results = run_best_of_n(
        client=client,
        model=args.model,
        env_url=args.env_url,
        task=args.task,
        n=args.n,
        temperature=args.temperature,
        max_steps=args.max_steps,
    )

    if args.output_json:
        payload = {
            "summary": asdict(summary),
            "episodes": [
                {
                    "index": i,
                    "steps": r.steps,
                    "score": r.score,
                    "rewards": r.rewards,
                    "success": r.success,
                    "error": r.error,
                }
                for i, r in enumerate(results)
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[bon] wrote {args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
