"""Build the SFT-on-oracle labeled dataset.

For each ``(task_id, seed)`` we step the corresponding deterministic oracle
through a fresh env and record one labeled example per env step:

    {
      "messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": render_observation(obs, history)},
        {"role": "assistant", "content": "<oracle action JSON>"}
      ],
      "task":   <task_id>,
      "seed":   <seed>,
      "step":   <1-indexed env step>,
      "reward": <env-returned reward for this step, float>
    }

Output is JSONL — one example per line, ready for HuggingFace
``datasets.load_dataset("json", data_files=...)`` and TRL ``SFTTrainer``.

Why in-process and not via HTTP
-------------------------------
The plan's draft spawned a uvicorn subprocess per ``(task, seed)``. Doing it
in-process via :class:`IncidentCommanderEnvironment` (the same class the smoke
tests use) is strictly equivalent for our purposes:

* The simulator is the same single seeded RNG either way.
* Pydantic serialization is identical — HTTP wraps it but doesn't change the
  payload bytes.
* No port-management headaches; runs fine on a laptop with no network.

We save ~10s × ``3 tasks × N seeds`` of subprocess startup, which adds up
quickly at the 30-seed default.

Usage
-----
::

    python -m incident_commander.training.build_sft_dataset \\
        --seeds 30 --output sft_oracle.jsonl

The default 30 seeds × 3 tasks × ~6 actions = ~540 examples — plenty for a
single-epoch SFT pass on a 3B model with a small LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

from incident_commander.inference import SYSTEM_PROMPT, action_str, render_observation
from incident_commander.models import ICAction
from incident_commander.server.incident_commander_environment import (
    IncidentCommanderEnvironment,
)
from incident_commander.training.oracle_policies import script_for


TASKS = (
    "easy_canary_regression",
    "medium_third_party_attribution",
    "hard_silent_data_corruption",
)


def _action_to_json(action: ICAction) -> str:
    """Serialise an :class:`ICAction` exactly the way we want the model to emit it.

    ``exclude_none=True`` drops every irrelevant field — the env validator
    already enforces per-op required fields, and trimming nulls keeps the
    label tokens lean. ``separators=(",", ":")`` removes whitespace so the
    model learns the most compact valid form.
    """
    return json.dumps(
        action.model_dump(exclude_none=True),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def collect_pairs(task_id: str, seed: int) -> Iterator[dict]:
    """Yield one training example per oracle step for ``(task_id, seed)``."""
    env = IncidentCommanderEnvironment(task_id=task_id, seed=seed)
    obs = env.reset()
    script = script_for(task_id, seed=seed)
    history: list[str] = []

    for step_idx, action in enumerate(script, start=1):
        rendered = render_observation(obs, history)
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rendered},
                {"role": "assistant", "content": _action_to_json(action)},
            ],
            "task": task_id,
            "seed": seed,
            "step": step_idx,
        }
        # Step the env so the next observation reflects the action's effect.
        # We discard the returned reward in the label (the trainer doesn't
        # need it; SFT uses the assistant tokens as supervision), but we keep
        # it in history so downstream inspection lines up with eval logs.
        next_obs = env.step(action)
        reward = float(next_obs.reward or 0.0)
        history.append(f"{action_str(action)} -> reward={reward:.3f}")
        if next_obs.done:
            break
        obs = next_obs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of seeds to roll out per task (default: 30).",
    )
    p.add_argument(
        "--output",
        default="sft_oracle.jsonl",
        help="Output JSONL path (default: sft_oracle.jsonl in CWD).",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        help=f"Override which tasks to include (default: {' '.join(TASKS)}).",
    )
    args = p.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    n_pairs = 0
    with out_path.open("w") as fh:
        for task_id in args.tasks:
            for seed in range(args.seeds):
                for pair in collect_pairs(task_id, seed):
                    fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    counts[task_id] += 1
                    n_pairs += 1
            print(
                f"[build] task={task_id} seeds=0..{args.seeds - 1} "
                f"pairs={counts[task_id]}",
                flush=True,
            )

    print(
        f"\n[build] wrote {n_pairs} pairs → {out_path}\n"
        f"[build] per-task counts: "
        + ", ".join(f"{t}={c}" for t, c in counts.items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
