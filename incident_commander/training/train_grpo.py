"""GRPO training loop for the Incident Commander environment.

Closed-loop, multi-step RL. Each training iteration:

 1. Collect ``group_size`` episode rollouts against a live env server, using
    the current LoRA-adapted policy with temperature > 0 for exploration.
 2. Compute group-relative advantages over the episode totals:
    ``adv_i = (score_i - group_mean) / (group_std + eps)``. No value function
    (that's the "GR" in GRPO).
 3. For every ``(observation, action)`` step inside each trajectory, compute
    the policy NLL of the action under the current model. Loss is
    ``adv * nll`` summed over the group, mean-normalised.
 4. One Adam step through the LoRA parameters.

Pragmatic simplifications vs. canonical GRPO:

* No importance-ratio clipping. We take one optimiser step per rollout
  batch, so the ratio is 1 and clipping is a no-op. If you want multi-epoch
  updates later, add a ratio + ``torch.clamp(...)``.
* KL penalty to the frozen reference model is **optional** and off by default.
  Unsloth's 4-bit reference would double the VRAM footprint; a 48h hackathon
  run is better served by a tight time-boxed schedule + per-component reward
  auditing (plan §4.5.5) than by KL-regularising a small model.

Expected environment:
* Env server running at ``ENV_URL`` (default ``http://localhost:8000``) with
  ``IC_TASK_ID=easy_canary_regression``. In Colab this is a uvicorn
  subprocess; locally it's Docker.
* GPU with >= 12 GB VRAM (Llama-3.2-3B in 4-bit LoRA comfortably fits a free
  Colab T4; Qwen2.5-7B needs ~14 GB and is tight).

Run directly::

    python -m incident_commander.training.train_grpo \\
        --base-model unsloth/Llama-3.2-3B-Instruct \\
        --iterations 50 --group-size 4 \\
        --output-dir ./ic-grpo-lora

Typical total runtime on a T4: ~30–90 min for 50 iterations with 3B + 4 rollouts
per iteration (episode length ~7 steps).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Optional

# These imports are heavy (torch + unsloth + transformers). We keep them at
# module scope because any sensible caller is in a GPU context already, but
# a fast-fail message beats a cryptic ImportError.
try:
    import torch
    from unsloth import FastLanguageModel
except ImportError as e:  # pragma: no cover - hackathon scaffolding
    raise ImportError(
        "train_grpo requires the [training] extra. On Colab, install with:\n"
        "  pip install 'openenv-incident_commander[training] @ .'\n"
        "or:\n"
        "  pip install unsloth trl transformers accelerate torch"
    ) from e

from .eval import evaluate
from .rollout import EpisodeResult, GenerateFn, run_rollout


# --------------------------------------------------------------------------- #
# Generator factory
# --------------------------------------------------------------------------- #


def make_unsloth_generator(
    model,
    tokenizer,
    *,
    max_new_tokens: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> GenerateFn:
    """Wrap an Unsloth model so it implements the :data:`GenerateFn` protocol.

    The returned callable is deterministic iff ``temperature == 0``. For
    training we use temperature=0.7 (plan §4.5.4); for eval, 0.1.
    """

    def _generate(messages: list[dict]) -> str:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion_ids = output[0, inputs.shape[1]:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True)

    return _generate


# --------------------------------------------------------------------------- #
# Loss computation
# --------------------------------------------------------------------------- #


def _completion_nll(model, tokenizer, messages: list[dict], completion: str) -> torch.Tensor:
    """Mean negative log-likelihood of ``completion`` given ``messages``.

    Masks the prompt tokens to ``-100`` so ``F.cross_entropy`` averages over
    just the completion tokens. Returns a scalar tensor on the model's device
    with autograd enabled.
    """
    # Prompt-only (no completion) to find the split point.
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    full_messages = messages + [{"role": "assistant", "content": completion}]
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).to(model.device)

    prompt_len = prompt_ids.shape[1]
    # Guard: if the chat template re-renders the full sequence shorter than
    # the prompt-only one (very rare — some templates re-flow whitespace),
    # fall back to mean-NLL over the whole thing and log a warning.
    if full_ids.shape[1] <= prompt_len:
        labels = full_ids.clone()
    else:
        labels = full_ids.clone()
        labels[0, :prompt_len] = -100  # ignore in loss

    outputs = model(input_ids=full_ids, labels=labels)
    return outputs.loss


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


@dataclass
class IterationMetrics:
    iteration: int
    group_scores: list[float]
    mean_score: float
    std_score: float
    loss: float
    n_steps: int


def run_grpo(
    *,
    model,
    tokenizer,
    env_url: str,
    iterations: int = 50,
    group_size: int = 4,
    lr: float = 5e-6,
    max_steps_per_episode: int = 25,
    rollout_temperature: float = 0.7,
    advantage_eps: float = 1e-4,
) -> list[IterationMetrics]:
    """Run the main GRPO loop. Returns per-iteration metrics."""
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    generator = make_unsloth_generator(
        model, tokenizer, temperature=rollout_temperature
    )

    metrics: list[IterationMetrics] = []
    wall_start = time.time()

    for it in range(1, iterations + 1):
        it_start = time.time()
        # 1. Collect group of rollouts.
        # Rollouts drive env via WebSocket — must be no_grad to keep VRAM
        # sane; gradient computation happens in phase 3.
        FastLanguageModel.for_inference(model)  # Unsloth fast generation mode
        rollouts: list[EpisodeResult] = []
        for g in range(group_size):
            traj = run_rollout(
                generate=generator,
                env_url=env_url,
                max_steps=max_steps_per_episode,
            )
            rollouts.append(traj)
            print(
                f"[iter {it}] rollout {g + 1}/{group_size} "
                f"steps={traj.steps} score={traj.score:.3f} "
                f"error={traj.error}",
                flush=True,
            )

        # 2. Group-relative advantages over episode totals.
        scores = [t.score for t in rollouts]
        g_mean = mean(scores)
        g_std = stdev(scores) if len(scores) > 1 else 0.0
        denom = g_std + advantage_eps
        advantages = [(s - g_mean) / denom for s in scores]

        # 3. Build training batch and compute loss.
        FastLanguageModel.for_training(model)  # back to grad-enabled mode
        total_loss = torch.zeros((), device=model.device)
        n_steps = 0
        for traj, adv in zip(rollouts, advantages):
            if not traj.trajectory:
                continue
            for step in traj.trajectory:
                nll = _completion_nll(
                    model, tokenizer, step.messages, step.completion
                )
                # REINFORCE with group-relative advantage: maximize
                # ``adv * log_prob`` == minimize ``adv * nll``.
                total_loss = total_loss + adv * nll
                n_steps += 1

        if n_steps == 0:
            print(
                f"[iter {it}] no valid steps collected (all rollouts errored); "
                f"skipping update",
                flush=True,
            )
            continue

        loss = total_loss / n_steps
        # 4. Optimizer step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        m = IterationMetrics(
            iteration=it,
            group_scores=scores,
            mean_score=g_mean,
            std_score=g_std,
            loss=float(loss.detach().item()),
            n_steps=n_steps,
        )
        metrics.append(m)
        elapsed = time.time() - it_start
        print(
            f"[iter {it}] mean={g_mean:.3f} std={g_std:.3f} loss={m.loss:.4f} "
            f"n_steps={n_steps} wall={elapsed:.1f}s "
            f"total_wall={(time.time() - wall_start) / 60:.1f}min",
            flush=True,
        )

    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Closed-loop GRPO training for Incident Commander"
    )
    p.add_argument(
        "--base-model",
        default=os.getenv("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct"),
        help="HF Hub id for the Unsloth-prepared base model",
    )
    p.add_argument("--env-url", default=os.getenv("ENV_URL", "http://localhost:8000"))
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-steps-per-episode", type=int, default=25)
    p.add_argument("--rollout-temperature", type=float, default=0.7)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "./ic-grpo-lora"),
        help="Where to save the merged LoRA adapter at the end",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Rollouts for pre- and post-training eval",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    print(f"[setup] loading {args.base_model} in 4-bit via Unsloth", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("[eval] pre-training baseline", flush=True)
    FastLanguageModel.for_inference(model)
    pre = evaluate(
        generate=make_unsloth_generator(
            model, tokenizer, temperature=0.1
        ),
        env_url=args.env_url,
        n_episodes=args.eval_episodes,
        max_steps=args.max_steps_per_episode,
        label="pre_training",
    )

    print("[train] starting GRPO", flush=True)
    metrics = run_grpo(
        model=model,
        tokenizer=tokenizer,
        env_url=args.env_url,
        iterations=args.iterations,
        group_size=args.group_size,
        lr=args.lr,
        max_steps_per_episode=args.max_steps_per_episode,
        rollout_temperature=args.rollout_temperature,
    )

    print("[eval] post-training baseline", flush=True)
    FastLanguageModel.for_inference(model)
    post = evaluate(
        generate=make_unsloth_generator(
            model, tokenizer, temperature=0.1
        ),
        env_url=args.env_url,
        n_episodes=args.eval_episodes,
        max_steps=args.max_steps_per_episode,
        label="post_training",
    )

    delta = post.mean - pre.mean
    print(
        f"\n=========================================\n"
        f"pre:  {pre}\n"
        f"post: {post}\n"
        f"delta: {delta:+.3f}\n"
        f"iterations: {len(metrics)}\n"
        f"=========================================",
        flush=True,
    )

    print(f"[save] LoRA adapter -> {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
